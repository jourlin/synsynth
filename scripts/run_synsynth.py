#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║                   SYNSYNTH+ — Pipeline Expérimental                ║
║                                                                      ║
║  Modèle  : unsloth/gemma-4-26B-A4B-it-GGUF  (UD-Q4_K_XK)          ║
║  Objectif : Évaluation 4 axes + Génération d'article scientifique   ║
║  Sandbox  : Toutes les I/O sont confinées dans PJKG4/               ║
╚══════════════════════════════════════════════════════════════════════╝

Usage :
    python run_synsynth.py                   # pipeline complet
    python run_synsynth.py --self-improve    # pipeline + boucle d'auto-amélioration
    python run_synsynth.py --exp extraction  # une seule expérience
    python run_synsynth.py --article-only    # rédaction seule (résultats existants)
    python run_synsynth.py --n-samples 20    # nombre d'échantillons réduit (test rapide)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# ── Sécurité : s'assurer que le CWD est bien dans le workspace ─────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(_SCRIPT_DIR)  # projet = parent de scripts/
os.chdir(WORKSPACE)

# Ajouter le dossier scripts/ au path pour les imports internes
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Empêcher toute évasion par variable d'environnement
os.environ["HF_HOME"] = os.path.join(WORKSPACE, "cache", "huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(WORKSPACE, "cache", "huggingface")
os.environ["TORCH_HOME"] = os.path.join(WORKSPACE, "cache", "torch")
os.environ["XDG_CACHE_HOME"] = os.path.join(WORKSPACE, "cache")

from synsynth_config import (
    logger, RESULTS_DIR, safe_path, DATA_DIR, ARTICLE_DIR,
    TASK_MODELS, DEFAULT_MODEL,
)
from synsynth_io import write_json, read_json, write_text
import synsynth_model

# Flag global pour le mode reprise
_RESUME_MODE = False


# ============================================================================
#  Fonctions orchestrateur
# ============================================================================

def run_experiment(name: str, n_samples: int | None = None) -> dict:
    """Importe et exécute dynamiquement une expérience par nom."""
    EXPERIMENTS = {
        "extraction":  "exp_extraction",
        "query":       "exp_query",
        "multihop":    "exp_multihop",
        "rag":         "exp_rag",
    }
    if name not in EXPERIMENTS:
        raise ValueError(f"Expérience inconnue : {name!r}. Choix : {list(EXPERIMENTS)}")

    # Sélection du meilleur modèle pour cette tâche
    model = TASK_MODELS.get(name, DEFAULT_MODEL)
    synsynth_model.OLLAMA_MODEL = model
    logger.info("Modèle sélectionné pour '%s' : %s", name, model)

    mod = __import__(EXPERIMENTS[name])
    return mod.run(n_samples=n_samples)


def run_all_experiments(n_samples: int | None = None) -> dict[str, dict]:
    """Exécute les 4 expériences séquentiellement.

    En mode --resume, les expériences déjà terminées (fichier résultat
    existant avec le bon n_samples) sont sautées.
    """
    results = {}
    exp_names = ["extraction", "query", "multihop", "rag"]
    exp_result_keys = {
        "extraction": "extraction",
        "query": "text_to_query",
        "multihop": "multihop_reasoning",
        "rag": "rag_faithfulness",
    }

    for name in exp_names:
        logger.info("━" * 60)

        # En mode resume, vérifier si un résultat complet existe déjà
        if _RESUME_MODE:
            result_key = exp_result_keys[name]
            result_path = os.path.join(RESULTS_DIR, f"{result_key}.json")
            if os.path.exists(result_path):
                existing = read_json(f"results/{result_key}.json")
                if existing and "error" not in existing:
                    logger.info("⏭  Expérience '%s' déjà terminée (résultat existant) — sautée.", name)
                    results[result_key] = existing
                    continue

        try:
            res = run_experiment(name, n_samples=n_samples)
            results[res.get("experiment", name)] = res
            # Sauvegarde incrémentale
            write_json(
                f"results/{res.get('experiment', name)}.json",
                res,
            )
        except Exception as e:
            logger.error("Échec de l'expérience '%s' : %s", name, e, exc_info=True)
            results[name] = {"experiment": name, "error": str(e)}

    return results


def generate_article(all_results: dict) -> str:
    """Appelle le module de rédaction."""
    from synsynth_article import generate_article as _gen
    return _gen(all_results)


def generate_visualizations(all_results: dict) -> list[str]:
    """Appelle le module de visualisation."""
    try:
        from synsynth_viz import plot_summary
        return plot_summary(all_results)
    except ImportError as e:
        logger.warning("Visualisation indisponible (matplotlib?) : %s", e)
        return []


def load_existing_results() -> dict:
    """Charge les résultats précédemment sauvegardés."""
    p = os.path.join(RESULTS_DIR, "all_results.json")
    if os.path.exists(p):
        rel = os.path.relpath(p, WORKSPACE)
        return read_json(rel)
    # Charger les fichiers individuels
    results = {}
    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith(".json") and fname != "all_results.json":
            rel = os.path.relpath(os.path.join(RESULTS_DIR, fname), WORKSPACE)
            data = read_json(rel)
            key = data.get("experiment", fname.replace(".json", ""))
            results[key] = data
    return results


# ============================================================================
#  Point d'entrée
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SYNSYNTH+ — Pipeline expérimental avec Gemma-4-26B",
    )
    parser.add_argument(
        "--exp", type=str, default=None,
        choices=["extraction", "query", "multihop", "rag"],
        help="Exécuter une seule expérience (par défaut : toutes).",
    )
    parser.add_argument(
        "--article-only", action="store_true",
        help="Générer l'article à partir de résultats existants.",
    )
    parser.add_argument(
        "--viz-only", action="store_true",
        help="Générer uniquement les visualisations.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="Nombre d'échantillons par expérience (défaut : config).",
    )
    parser.add_argument(
        "--skip-article", action="store_true",
        help="Exécuter les expériences sans générer l'article.",
    )
    parser.add_argument(
        "--self-improve", action="store_true",
        help="Activer la boucle d'auto-amélioration (Gemma-4 corrige ses scripts).",
    )
    parser.add_argument(
        "--max-improve-iter", type=int, default=3,
        help="Nombre max d'itérations d'auto-amélioration par expérience (défaut : 3).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Reprendre le pipeline : saute les expériences terminées et reprend les checkpoints.",
    )
    parser.add_argument(
        "--gbnf", action="store_true",
        help="Activer le décodage contraint par JSON Schema (GBNF) via PJKG5.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Graine aléatoire (défaut : 42). Permet de mesurer la variance inter-runs.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Forcer un modèle Ollama pour toutes les tâches (ex: llama3.1:8b). "
             "Utile pour les baselines comparatives.",
    )
    parser.add_argument(
        "--qlora", action="store_true",
        help="Activer le fine-tuning QLoRA (Phase 2b). Entraîne les modèles "
             "sur Re-DocRED/HotpotQA puis évalue avec inférence HuggingFace. "
             "Résultats sauvegardés avec suffixe _qlora.",
    )
    parser.add_argument(
        "--qlora-base-model", type=str, default=None,
        help="Modèle HuggingFace de base pour QLoRA "
             "(défaut : Qwen/Qwen2.5-7B-Instruct).",
    )
    args = parser.parse_args()

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║           SYNSYNTH+ — Démarrage du pipeline             ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info("Workspace : %s", WORKSPACE)

    # ── Seed aléatoire ─────────────────────────────────────────────────
    if args.seed is not None:
        import synsynth_config
        synsynth_config.RANDOM_SEED = args.seed
        import random
        random.seed(args.seed)
        logger.info("Seed aléatoire fixée à %d", args.seed)

    # ── Modèle forcé (baselines) ───────────────────────────────────────
    if args.model:
        for key in TASK_MODELS:
            TASK_MODELS[key] = args.model
        logger.info("Modèle forcé pour toutes les tâches : %s", args.model)

    # Activer le mode reprise si demandé
    global _RESUME_MODE
    if args.resume:
        _RESUME_MODE = True
        logger.info("Mode REPRISE activé — les expériences terminées seront sautées.")

    # Activer le décodage contraint GBNF si demandé
    if args.gbnf:
        pjkg5 = os.path.join(os.path.dirname(WORKSPACE), "PJKG5")
        if pjkg5 not in sys.path:
            sys.path.insert(0, pjkg5)
        from gbnf_patch import patch_model_module, set_task
        patch_model_module()
        # Envelopper run_experiment pour injecter set_task() avant chaque exp
        _original_run_experiment = run_experiment
        def _gbnf_run_experiment(name, n_samples=None):
            set_task(name)
            return _original_run_experiment(name, n_samples=n_samples)
        globals()['run_experiment'] = _gbnf_run_experiment
        logger.info("Décodage contraint GBNF activé (JSON Schema strict).")

    # Activer le QLoRA si demandé
    if args.qlora:
        from qlora_finetune import (
            finetune, has_finetuned_model, patch_inference, unpatch_inference,
            QLORA_TASKS,
        )

        # 1. Entraîner les modèles si nécessaire
        for _task in QLORA_TASKS:
            if not has_finetuned_model(_task):
                logger.info("Entraînement QLoRA pour '%s'...", _task)
                finetune(_task, base_model=args.qlora_base_model)

        # 2. Envelopper run_experiment pour patcher l'inférence par tâche
        _prev_run_experiment_qlora = globals()['run_experiment']

        def _qlora_run_experiment(name, n_samples=None):
            if name in QLORA_TASKS and has_finetuned_model(name):
                patch_inference(name)
                try:
                    res = _prev_run_experiment_qlora(name, n_samples=n_samples)
                finally:
                    unpatch_inference()
                # Tagguer le résultat comme QLoRA
                original_exp = res.get("experiment", name)
                res["experiment"] = original_exp + "_qlora"
                res["method"] = "qlora"
                res["base_experiment"] = original_exp
                return res
            else:
                return _prev_run_experiment_qlora(name, n_samples=n_samples)

        globals()['run_experiment'] = _qlora_run_experiment
        logger.info("QLoRA activé pour : %s", QLORA_TASKS)

    t_global = time.time()

    # ── Mode article seul ──────────────────────────────────────────────
    if args.article_only:
        all_results = load_existing_results()
        if not all_results:
            logger.error("Aucun résultat trouvé. Lancez d'abord les expériences.")
            sys.exit(1)
        generate_article(all_results)
        logger.info("Article généré → article/SYNSYNTH_article.md")
        return

    # ── Mode visualisation seule ───────────────────────────────────────
    if args.viz_only:
        all_results = load_existing_results()
        if not all_results:
            logger.error("Aucun résultat trouvé.")
            sys.exit(1)
        paths = generate_visualizations(all_results)
        for p in paths:
            logger.info("Figure → %s", p)
        return

    # ── Expérience unique ──────────────────────────────────────────────
    if args.exp:
        if args.self_improve:
            from synsynth_selfimprove import self_improve
            logger.info("Mode AUTO-AMÉLIORATION pour '%s' (max %d iter).",
                        args.exp, args.max_improve_iter)
            res = self_improve(
                args.exp,
                run_experiment,
                n_samples=args.n_samples,
                max_iterations=args.max_improve_iter,
            )
        else:
            res = run_experiment(args.exp, n_samples=args.n_samples)
        write_json(f"results/{res.get('experiment', args.exp)}.json", res)
        if not args.skip_article:
            all_results = load_existing_results()
            all_results[res.get("experiment", args.exp)] = res
            generate_article(all_results)
        return

    # ── Pipeline complet ───────────────────────────────────────────────
    if args.self_improve:
        from synsynth_selfimprove import self_improve_all
        logger.info("Mode AUTO-AMÉLIORATION activé (max %d iter/exp).",
                    args.max_improve_iter)
        all_results = self_improve_all(
            run_experiment,
            n_samples=args.n_samples,
            max_iterations=args.max_improve_iter,
        )
    else:
        all_results = run_all_experiments(n_samples=args.n_samples)

    # Sauvegarde globale
    write_json("results/all_results.json", all_results)

    # Visualisations
    fig_paths = generate_visualizations(all_results)

    # Rédaction de l'article
    if not args.skip_article:
        article = generate_article(all_results)
        logger.info("Article sauvegardé → article/SYNSYNTH_article.md")

    # ── Rapport final ──────────────────────────────────────────────────
    elapsed_total = time.time() - t_global
    logger.info("━" * 60)
    logger.info("Pipeline terminé en %.1f s.", elapsed_total)
    logger.info("Résultats       → results/")
    logger.info("Figures         → %s", ", ".join(fig_paths) if fig_paths else "(aucune)")
    logger.info("Article         → article/SYNSYNTH_article.md")
    logger.info("Logs            → logs/synsynth.log")

    # Résumé chiffré
    for key, res in all_results.items():
        if "error" in res:
            logger.warning("  %-25s  ERREUR : %s", key, res["error"])
        else:
            score = (
                res.get("f1_score")
                or res.get("accuracy")
                or res.get("exact_accuracy")
                or res.get("avg_faithfulness")
                or res.get("avg_token_f1")
                or "?"
            )
            logger.info("  %-25s  score = %s", key, score)


if __name__ == "__main__":
    main()
