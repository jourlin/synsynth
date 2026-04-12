#!/usr/bin/env python3
"""
Benchmark multi-modèles pour SYNSYNTH+.

Teste chaque modèle Ollama sur un petit échantillon de chaque tâche :
  1. Extraction de relations (JSON parse + F1)
  2. Text-to-Query (Cypher generation + accuracy)
  3. Raisonnement multi-hop (exact match)
  4. RAG fidélité

Usage :
    python benchmark_models.py [--samples N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# ── Empêcher l'import initial de synsynth_model (qui fait _check_ollama) ──
# On doit importer APRÈS avoir patché le modèle
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKSPACE)

from synsynth_config import logger, RESULTS_DIR

# ── Modèles candidats par tâche ──────────────────────────────────────────
# (model_name, note_pourquoi)
EXTRACTION_MODELS = [
    "gemma4:26b",
    "qwen3.5:27b",
    "qwen3-deep:latest",
    "phi4:latest",
    "mistral-small:latest",
    "nuextract:latest",
    "llama3.1:8b",
]

QUERY_MODELS = [
    "gemma4:26b",
    "qwen3.5:27b",
    "qwen3-deep:latest",
    "qwen2.5-coder:32b",
    "phi4:latest",
    "mistral-small:latest",
]

MULTIHOP_MODELS = [
    "gemma4:26b",
    "qwen3.5:27b",
    "qwen3-deep:latest",
    "deepseek-r1:32b",
    "phi4:latest",
    "mistral-small:latest",
]

RAG_MODELS = [
    "gemma4:26b",
    "qwen3.5:27b",
    "qwen3-deep:latest",
    "command-r7b:latest",
    "phi4:latest",
    "mistral-small:latest",
    "llama3.1:8b",
]


def set_model(model_name: str):
    """Change le modèle actif dans synsynth_model (monkey-patch)."""
    import synsynth_model
    synsynth_model.OLLAMA_MODEL = model_name
    logger.info("── Modèle changé → %s ──", model_name)


def warm_up(model_name: str) -> bool:
    """Charge le modèle en mémoire avec un prompt trivial. Retourne True si OK."""
    import synsynth_model
    try:
        set_model(model_name)
        resp = synsynth_model.generate("Réponds 'ok'.", max_new_tokens=8)
        logger.info("  warm-up OK: %s", resp[:50])
        return True
    except Exception as e:
        logger.error("  warm-up ÉCHEC pour %s: %s", model_name, e)
        return False


# ── Benchmark Extraction ────────────────────────────────────────────────
def bench_extraction(n_samples: int) -> dict:
    """Teste l'extraction sur n_samples avec le modèle courant."""
    import exp_extraction
    return exp_extraction.run(n_samples)


# ── Benchmark Query ─────────────────────────────────────────────────────
def bench_query(n_samples: int) -> dict:
    """Teste text-to-query sur n_samples avec le modèle courant."""
    import exp_query
    return exp_query.run(n_samples)


# ── Benchmark Multihop ──────────────────────────────────────────────────
def bench_multihop(n_samples: int) -> dict:
    """Teste le raisonnement multi-hop sur n_samples."""
    import exp_multihop
    return exp_multihop.run(n_samples)


# ── Benchmark RAG ───────────────────────────────────────────────────────
def bench_rag(n_samples: int) -> dict:
    """Teste RAG fidélité sur n_samples."""
    import exp_rag
    return exp_rag.run(n_samples)


# ── Fonctions utilitaires ──────────────────────────────────────────────
def extract_metrics(result: dict, task: str) -> dict:
    """Extrait les métriques clés d'un résultat d'expérience."""
    m = {"task": task}
    if task == "extraction":
        m["f1"] = result.get("f1_score", 0)
        m["precision"] = result.get("precision", 0)
        m["recall"] = result.get("recall", 0)
        # Calculer parse_fail rate
        details = result.get("details", [])
        pf = sum(1 for d in details if d.get("status") == "parse_fail")
        m["parse_fail_rate"] = round(pf / len(details), 4) if details else 0
    elif task == "query":
        m["accuracy"] = result.get("accuracy", 0)
        m["cypher_valid"] = result.get("cypher_syntax_valid_rate", 0)
    elif task == "multihop":
        m["exact_match"] = result.get("exact_match_accuracy", 0)
        # Fallback : calculer depuis details
        if m["exact_match"] == 0:
            details = result.get("details", [])
            if details:
                em = sum(1 for d in details if d.get("exact_match"))
                m["exact_match"] = round(em / len(details), 4)
    elif task == "rag":
        m["faithfulness"] = result.get("avg_faithfulness", 0)
        m["relevance"] = result.get("avg_answer_relevance", 0)
        m["context_precision"] = result.get("avg_context_precision", 0)
    m["elapsed_s"] = result.get("elapsed_seconds", 0)
    m["n_samples"] = result.get("n_samples", 0)
    return m


def run_task_benchmark(task: str, models: list[str], n_samples: int,
                       bench_fn) -> list[dict]:
    """Exécute un benchmark pour une tâche sur tous les modèles candidats."""
    results = []
    for model in models:
        logger.info("=" * 60)
        logger.info("BENCHMARK %s — modèle: %s — %d samples", task.upper(), model, n_samples)
        logger.info("=" * 60)

        if not warm_up(model):
            results.append({
                "model": model, "task": task, "status": "error",
                "error": "warm-up failed"
            })
            continue

        try:
            t0 = time.time()
            raw_result = bench_fn(n_samples)
            elapsed = time.time() - t0

            metrics = extract_metrics(raw_result, task)
            metrics["model"] = model
            metrics["status"] = "ok"
            results.append(metrics)

            logger.info("RÉSULTAT %s / %s: %s [%.1fs]",
                        task, model, json.dumps(metrics, ensure_ascii=False), elapsed)
        except Exception as e:
            logger.error("ERREUR %s / %s: %s", task, model, e)
            results.append({
                "model": model, "task": task, "status": "error",
                "error": str(e)
            })

    return results


def print_leaderboard(all_results: list[dict]):
    """Affiche un tableau récapitulatif par tâche."""
    tasks = {}
    for r in all_results:
        task = r.get("task", "?")
        tasks.setdefault(task, []).append(r)

    print("\n" + "=" * 80)
    print("LEADERBOARD — Benchmark multi-modèles SYNSYNTH+")
    print("=" * 80)

    best_per_task = {}

    for task, results in tasks.items():
        print(f"\n{'─' * 60}")
        print(f"  TÂCHE: {task.upper()}")
        print(f"{'─' * 60}")

        # Tri par métrique principale
        if task == "extraction":
            key = "f1"
            results.sort(key=lambda r: r.get(key, 0), reverse=True)
            print(f"  {'Modèle':<30} {'F1':>6} {'Prec':>6} {'Rec':>6} {'PFail%':>7} {'Time':>7}")
            for r in results:
                if r.get("status") == "error":
                    print(f"  {r['model']:<30} {'ERROR':>6} {r.get('error', '')}")
                else:
                    print(f"  {r['model']:<30} {r.get('f1', 0):>6.3f} {r.get('precision', 0):>6.3f} "
                          f"{r.get('recall', 0):>6.3f} {r.get('parse_fail_rate', 0)*100:>6.1f}% "
                          f"{r.get('elapsed_s', 0):>6.0f}s")
        elif task == "query":
            key = "accuracy"
            results.sort(key=lambda r: r.get(key, 0), reverse=True)
            print(f"  {'Modèle':<30} {'Acc':>6} {'Cypher':>7} {'Time':>7}")
            for r in results:
                if r.get("status") == "error":
                    print(f"  {r['model']:<30} {'ERROR':>6} {r.get('error', '')}")
                else:
                    print(f"  {r['model']:<30} {r.get('accuracy', 0):>6.3f} "
                          f"{r.get('cypher_valid', 0):>7.3f} "
                          f"{r.get('elapsed_s', 0):>6.0f}s")
        elif task == "multihop":
            key = "exact_match"
            results.sort(key=lambda r: r.get(key, 0), reverse=True)
            print(f"  {'Modèle':<30} {'EM':>6} {'Time':>7}")
            for r in results:
                if r.get("status") == "error":
                    print(f"  {r['model']:<30} {'ERROR':>6} {r.get('error', '')}")
                else:
                    print(f"  {r['model']:<30} {r.get('exact_match', 0):>6.3f} "
                          f"{r.get('elapsed_s', 0):>6.0f}s")
        elif task == "rag":
            key = "faithfulness"
            results.sort(key=lambda r: r.get(key, 0), reverse=True)
            print(f"  {'Modèle':<30} {'Faith':>6} {'Relev':>6} {'CtxP':>6} {'Time':>7}")
            for r in results:
                if r.get("status") == "error":
                    print(f"  {r['model']:<30} {'ERROR':>6} {r.get('error', '')}")
                else:
                    print(f"  {r['model']:<30} {r.get('faithfulness', 0):>6.3f} "
                          f"{r.get('relevance', 0):>6.3f} "
                          f"{r.get('context_precision', 0):>6.3f} "
                          f"{r.get('elapsed_s', 0):>6.0f}s")
        else:
            key = None

        # Meilleur modèle
        ok_results = [r for r in results if r.get("status") == "ok"]
        if ok_results and key:
            best = ok_results[0]
            best_per_task[task] = best["model"]
            print(f"\n  ★ MEILLEUR: {best['model']} ({key}={best.get(key, 0):.3f})")

    print(f"\n{'=' * 80}")
    print("RÉSUMÉ — Meilleur modèle par tâche :")
    for task, model in best_per_task.items():
        print(f"  • {task:15s} → {model}")
    print("=" * 80)

    return best_per_task


def main():
    parser = argparse.ArgumentParser(description="Benchmark multi-modèles SYNSYNTH+")
    parser.add_argument("--samples", "-n", type=int, default=5,
                        help="Nombre d'échantillons par tâche (défaut: 5)")
    parser.add_argument("--tasks", "-t", nargs="+",
                        choices=["extraction", "query", "multihop", "rag", "all"],
                        default=["all"],
                        help="Tâches à benchmarker (défaut: all)")
    args = parser.parse_args()

    n = args.samples
    do_all = "all" in args.tasks

    all_results = []
    t_global = time.time()

    if do_all or "extraction" in args.tasks:
        all_results.extend(
            run_task_benchmark("extraction", EXTRACTION_MODELS, n, bench_extraction))

    if do_all or "query" in args.tasks:
        all_results.extend(
            run_task_benchmark("query", QUERY_MODELS, n, bench_query))

    if do_all or "multihop" in args.tasks:
        all_results.extend(
            run_task_benchmark("multihop", MULTIHOP_MODELS, n, bench_multihop))

    if do_all or "rag" in args.tasks:
        all_results.extend(
            run_task_benchmark("rag", RAG_MODELS, n, bench_rag))

    total_time = time.time() - t_global

    # Afficher le leaderboard
    best_per_task = print_leaderboard(all_results)

    # Sauvegarder les résultats
    out_file = os.path.join(RESULTS_DIR, "benchmark_models.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "samples_per_task": n,
            "total_elapsed_seconds": round(total_time, 1),
            "best_per_task": best_per_task,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)

    logger.info("Résultats sauvegardés dans %s", out_file)
    logger.info("Temps total: %.1f min", total_time / 60)


if __name__ == "__main__":
    main()
