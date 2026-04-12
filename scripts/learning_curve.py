#!/usr/bin/env python3
"""
Suggestion B — Courbe d'apprentissage zero-shot → QLoRA frugal.

Entraîne Qwen2.5-7B-Instruct avec QLoRA sur {10, 50, 200, 500, 1000, 3000}
échantillons de Re-DocRED / HotpotQA, évalue chaque checkpoint, et trace
la courbe F1/EM = f(n_train).

Usage :
    python learning_curve.py --task extraction
    python learning_curve.py --task multihop
    python learning_curve.py --task all
"""
from __future__ import annotations

import gc
import json
import os
import random
import sys
import time

# ── ajouter scripts/ au PYTHONPATH si nécessaire ────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from synsynth_config import (
    WORKSPACE_ROOT, RESULTS_DIR, logger, safe_path, RANDOM_SEED,
)
from qlora_finetune import (
    QLORA_DATA_DIR, QLORA_MODELS_DIR, HF_CACHE,
    DEFAULT_BASE_MODEL, LORA_R, LORA_ALPHA, LORA_DROPOUT,
    LORA_TARGET_MODULES, MAX_SEQ_LENGTH, WARMUP_RATIO, WEIGHT_DECAY,
    _unload_ollama_models,
)

# ── Points de la courbe d'apprentissage ─────────────────────────────────
CURVE_POINTS = [10, 50, 200, 500, 1000, 3000]

# ── Répertoire de sortie ────────────────────────────────────────────────
LC_RESULTS_DIR = safe_path("results", "learning_curve")
os.makedirs(LC_RESULTS_DIR, exist_ok=True)


def _subsample_jsonl(src_path: str, n: int, seed: int = RANDOM_SEED) -> str:
    """Créé un fichier JSONL sous-échantillonné de n lignes. Renvoie le chemin."""
    dst_path = os.path.join(QLORA_DATA_DIR, f"{os.path.basename(src_path).replace('.jsonl', '')}_n{n}.jsonl")
    if os.path.exists(dst_path):
        with open(dst_path) as f:
            existing = sum(1 for _ in f)
        if existing == n:
            logger.info("Sous-échantillon déjà prêt : %s (%d lignes)", dst_path, n)
            return dst_path

    with open(src_path) as f:
        all_lines = f.readlines()

    rng = random.Random(seed)
    sampled = rng.sample(all_lines, min(n, len(all_lines)))
    with open(dst_path, "w") as f:
        f.writelines(sampled)
    logger.info("Sous-échantillon créé : %s (%d lignes)", dst_path, len(sampled))
    return dst_path


def _train_at_n(task: str, n: int, base_model: str = DEFAULT_BASE_MODEL) -> str:
    """Entraîne QLoRA sur n échantillons. Renvoie le chemin de l'adaptateur."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    adapter_dir = os.path.join(QLORA_MODELS_DIR, f"{task}_n{n}", "adapter")
    if os.path.isfile(os.path.join(adapter_dir, "adapter_config.json")):
        logger.info("Adaptateur déjà entraîné : %s", adapter_dir)
        return adapter_dir

    # Données source complètes
    full_jsonl = os.path.join(QLORA_DATA_DIR, f"{task}_train.jsonl")
    if not os.path.exists(full_jsonl):
        raise FileNotFoundError(f"Données manquantes : {full_jsonl}")

    # Sous-échantillonner
    sub_jsonl = _subsample_jsonl(full_jsonl, n)

    samples = []
    with open(sub_jsonl, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    dataset = Dataset.from_list(samples)

    logger.info("=" * 60)
    logger.info("Learning curve — tâche=%s, n_train=%d", task, n)
    logger.info("=" * 60)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, cache_dir=HF_CACHE, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantification 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=HF_CACHE,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Chat template → texte brut
    def _apply_template(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(_apply_template, remove_columns=["messages"])

    # Adapter les hyperparamètres : epochs = max(3, 10 si n <= 50)
    num_epochs = 3 if n >= 200 else 10
    lr = 2e-4 if n <= 200 else 1e-4  # lr réduit pour grands n (évite overfitting)
    grad_accum = max(1, min(8, n // 2))  # Éviter grad_accum > n_samples/batch
    batch_size = min(2, n)

    ckpt_dir = os.path.join(QLORA_MODELS_DIR, f"{task}_n{n}", "checkpoints")
    training_args = SFTConfig(
        output_dir=ckpt_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_steps=max(1, n // 20),
        save_strategy="no",
        bf16=True,
        optim="paged_adamw_8bit",
        seed=RANDOM_SEED,
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        args=training_args,
    )

    t0 = time.time()
    logger.info("Entraînement n=%d (epochs=%d, lr=%.0e, batch=%d×%d)...",
                n, num_epochs, lr, batch_size, grad_accum)
    trainer.train()
    train_time = time.time() - t0
    logger.info("Entraînement terminé en %.0f s.", train_time)

    os.makedirs(adapter_dir, exist_ok=True)
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    with open(os.path.join(adapter_dir, "qlora_meta.json"), "w") as f:
        json.dump({
            "base_model": base_model,
            "task": task,
            "n_train": n,
            "num_epochs": num_epochs,
            "learning_rate": lr,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "train_samples": len(dataset),
            "elapsed_seconds": round(train_time, 1),
        }, f, indent=2)

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return adapter_dir


def _evaluate_extraction(adapter_dir: str, n_train: int) -> dict:
    """Évalue un adaptateur extraction sur les 500 échantillons DocRED."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    _unload_ollama_models()

    with open(os.path.join(adapter_dir, "qlora_meta.json")) as f:
        meta = json.load(f)
    base_model = meta["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir, cache_dir=HF_CACHE, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_config,
        device_map="auto", cache_dir=HF_CACHE,
        trust_remote_code=True, dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    # Monkey-patch synsynth_model pour utiliser HF
    import synsynth_model
    from qlora_finetune import (
        _hf_generate, _hf_generate_structured,
        TEMPERATURE, TOP_P,
    )
    import qlora_finetune
    qlora_finetune._qlora_model = model
    qlora_finetune._qlora_tokenizer = tokenizer

    orig_gen = synsynth_model.generate
    orig_gen_struct = synsynth_model.generate_structured
    synsynth_model.generate = _hf_generate
    synsynth_model.generate_structured = _hf_generate_structured

    try:
        from exp_extraction import run as run_extraction
        logger.info("Évaluation extraction (n_train=%d)...", n_train)
        result = run_extraction()
        result["n_train"] = n_train
        result["adapter_dir"] = adapter_dir
    finally:
        synsynth_model.generate = orig_gen
        synsynth_model.generate_structured = orig_gen_struct
        qlora_finetune._qlora_model = None
        qlora_finetune._qlora_tokenizer = None
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return result


def _evaluate_multihop(adapter_dir: str, n_train: int) -> dict:
    """Évalue un adaptateur multihop sur les 500 échantillons HotpotQA."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    _unload_ollama_models()

    with open(os.path.join(adapter_dir, "qlora_meta.json")) as f:
        meta = json.load(f)
    base_model = meta["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir, cache_dir=HF_CACHE, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_config,
        device_map="auto", cache_dir=HF_CACHE,
        trust_remote_code=True, dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    import synsynth_model
    import qlora_finetune
    from qlora_finetune import _hf_generate, _hf_generate_structured
    qlora_finetune._qlora_model = model
    qlora_finetune._qlora_tokenizer = tokenizer

    orig_gen = synsynth_model.generate
    orig_gen_struct = synsynth_model.generate_structured
    synsynth_model.generate = _hf_generate
    synsynth_model.generate_structured = _hf_generate_structured

    try:
        from exp_multihop import run as run_multihop
        logger.info("Évaluation multihop (n_train=%d)...", n_train)
        result = run_multihop()
        result["n_train"] = n_train
        result["adapter_dir"] = adapter_dir
    finally:
        synsynth_model.generate = orig_gen
        synsynth_model.generate_structured = orig_gen_struct
        qlora_finetune._qlora_model = None
        qlora_finetune._qlora_tokenizer = None
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return result


def run_learning_curve(task: str) -> list[dict]:
    """Exécute la courbe d'apprentissage complète pour une tâche."""
    results = []
    output_path = os.path.join(LC_RESULTS_DIR, f"learning_curve_{task}.json")

    # Charger les résultats existants
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        done_ns = {r["n_train"] for r in results}
        logger.info("Résultats existants pour %s : n_train=%s", task, sorted(done_ns))
    else:
        done_ns = set()

    evaluate_fn = _evaluate_extraction if task == "extraction" else _evaluate_multihop
    # multihop_v2 utilise le même évaluateur que multihop (même jeu de test)

    for n in CURVE_POINTS:
        if n in done_ns:
            logger.info("Point n=%d déjà évalué — skip.", n)
            continue

        logger.info("━" * 60)
        logger.info("COURBE D'APPRENTISSAGE — %s — n_train=%d", task, n)
        logger.info("━" * 60)

        # Entraîner
        adapter_dir = _train_at_n(task, n)

        # Évaluer
        result = evaluate_fn(adapter_dir, n)
        results.append(result)

        # Sauvegarder incrémentalement
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Résultat sauvegardé → %s", output_path)

    # Ajouter le point zero-shot (n=0) à partir des résultats existants
    if 0 not in {r.get("n_train") for r in results}:
        # multihop_v2 utilise le même zero-shot que multihop
        zs_task = "multihop" if task in ("multihop_v2", "multihop_v3", "multihop_v4") else task
        zs_path = os.path.join(RESULTS_DIR, f"{zs_task}.json")
        if os.path.exists(zs_path):
            with open(zs_path) as f:
                zs = json.load(f)
            zs["n_train"] = 0
            zs["method"] = "zero-shot"
            results.insert(0, zs)

    # Trier par n_train
    results.sort(key=lambda r: r.get("n_train", 0))

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Courbe d'apprentissage complète → %s", output_path)
    return results


def plot_learning_curve(task: str):
    """Trace la courbe d'apprentissage."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    lc_path = os.path.join(LC_RESULTS_DIR, f"learning_curve_{task}.json")
    with open(lc_path) as f:
        results = json.load(f)

    ns = [r["n_train"] for r in results]
    if task == "extraction":
        scores = [r.get("f1_score", 0) for r in results]
        metric_label = "F1 Score"
    else:  # multihop, multihop_v2, multihop_v3, multihop_v4
        scores = [r.get("exact_match", r.get("accuracy", 0)) for r in results]
        metric_label = "Exact Match"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, scores, "o-", color="#2196F3", linewidth=2, markersize=8, label="QLoRA 4-bit (Qwen2.5-7B)")

    # Baseline zero-shot Gemma-4-27B
    if task == "extraction":
        ax.axhline(y=0.7023, color="#FF9800", linestyle="--", linewidth=1.5,
                    label="Zero-shot Gemma-4-27B (F1=0.70)")
        ax.axhline(y=0.802, color="#4CAF50", linestyle=":", linewidth=1.5,
                    label="DREEAM supervisé (F1=0.80)")
    elif task in ("multihop", "multihop_v2", "multihop_v3", "multihop_v4"):
        ax.axhline(y=0.462, color="#FF9800", linestyle="--", linewidth=1.5,
                    label="Zero-shot Phi-4-14B (EM=0.46)")

    ax.set_xlabel("Nombre d'exemples d'entraînement", fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    task_label = task.replace("_v4", " V4").replace("_v3", " V3").replace("_v2", " V2").replace("_", " ").capitalize()
    ax.set_title(f"Courbe d'apprentissage — {task_label}\n"
                 f"QLoRA 4-bit sur RTX 3090 (24 Go)", fontsize=13)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig_path = os.path.join(LC_RESULTS_DIR, f"learning_curve_{task}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure → %s", fig_path)
    return fig_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Courbe d'apprentissage zero-shot → QLoRA")
    parser.add_argument("--task", choices=["extraction", "multihop", "multihop_v2", "multihop_v3", "multihop_v4", "all"], default="all")
    parser.add_argument("--plot-only", action="store_true",
                        help="Tracer les courbes sans relancer les entraînements.")
    args = parser.parse_args()

    tasks = ["extraction", "multihop"] if args.task == "all" else [args.task]  # multihop_v2 must be requested explicitly

    for task in tasks:
        if not args.plot_only:
            run_learning_curve(task)
        plot_learning_curve(task)


if __name__ == "__main__":
    main()
