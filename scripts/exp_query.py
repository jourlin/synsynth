"""
Expérience 2 — Text-to-Graph Query (WebQuestionsSP style).

Le modèle reçoit une question en langage naturel et doit produire :
  1. Une réponse factuelle courte.
  2. (Optionnel) une requête Cypher correspondante.

On mesure l'Accuracy (correspondance exacte / inclusion de la réponse).
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from synsynth_config import logger, TARGET_ACCURACY_QUERY
from synsynth_model import generate_structured
from synsynth_data import load_query_data
from synsynth_stats import bootstrap_ci
from synsynth_checkpoint import save_checkpoint, load_checkpoint, clear_checkpoint

SYSTEM_PROMPT = (
    "Tu es un moteur de recherche de connaissances. "
    "Pour chaque question, réponds avec un JSON : "
    "{\"answer\": \"…\", \"cypher\": \"…\"} "
    "où 'answer' est la réponse courte et 'cypher' est la requête Cypher "
    "pour retrouver l'information dans un graphe de connaissances."
)


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


_NUM_RE = re.compile(r"[\d]+(?:[.,]\d+)?")


def _extract_number(s: str) -> float | None:
    """Extraire la première valeur numérique d'une chaîne."""
    m = _NUM_RE.search(s.replace("\u202f", "").replace(" ", ""))
    if m:
        return float(m.group().replace(",", "."))
    return None


def _answer_match(pred_answer: str, gold_answer: str) -> bool:
    """Match souple : inclusion bi-directionnelle + tolérance numérique ±10 %."""
    p, g = _normalize(pred_answer), _normalize(gold_answer)
    if p in g or g in p or p == g:
        return True
    # Tolérance numérique
    pn, gn = _extract_number(p), _extract_number(g)
    if pn is not None and gn is not None and gn != 0:
        return abs(pn - gn) / abs(gn) <= 0.10
    return False


def _strip_markdown(raw: str) -> str:
    """Retire les blocs ```json...``` ou ```...``` qui entourent la réponse."""
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    return m.group(1).strip() if m else raw


def _parse_response(raw: str) -> dict | None:
    cleaned = _strip_markdown(raw)
    # Tenter json.loads sur la chaîne entière d'abord (gère les accolades imbriquées)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "answer" in obj:
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    # Fallback regex : trouver le bloc JSON le plus externe
    depth = 0
    start = None
    for i, c in enumerate(cleaned):
        if c == '{':
            if depth == 0:
                start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(cleaned[start:i+1])
                    if isinstance(obj, dict) and "answer" in obj:
                        return obj
                except (json.JSONDecodeError, ValueError):
                    pass
                start = None
    # Fallback : toute la réponse est la answer
    return {"answer": cleaned.strip(), "cypher": ""}


# ── Point d'entrée ─────────────────────────────────────────────────────────

def run(n_samples: int | None = None) -> dict[str, Any]:
    data = load_query_data(n_samples) if n_samples else load_query_data()
    logger.info("=== Exp 2 : Text-to-Query — %d échantillons ===", len(data))

    # ── Reprise depuis checkpoint ──────────────────────────────────────
    ckpt = load_checkpoint("query")
    if ckpt:
        start_idx = ckpt["next_idx"]
        correct = ckpt["correct"]
        cypher_valid = ckpt["cypher_valid"]
        details = ckpt["details"]
        elapsed_prev = ckpt.get("elapsed", 0.0)
        logger.info("Reprise query à idx=%d", start_idx)
    else:
        start_idx = 0
        correct = 0
        cypher_valid = 0
        details = []
        elapsed_prev = 0.0

    t0 = time.time()
    for i in range(start_idx, len(data)):
        sample = data[i]
        prompt = f"Question : {sample['question']}"
        raw = generate_structured(prompt, system=SYSTEM_PROMPT, json_mode=True, max_new_tokens=512)
        pred = _parse_response(raw)

        ans_ok = _answer_match(pred["answer"], sample["answer"]) if pred else False

        # Vérification syntaxique basique du Cypher
        cypher_ok = False
        if pred and pred.get("cypher"):
            cypher_upper = pred["cypher"].upper()
            cypher_ok = "MATCH" in cypher_upper or "RETURN" in cypher_upper

        if ans_ok:
            correct += 1
        if cypher_ok:
            cypher_valid += 1

        details.append({
            "idx": i,
            "question": sample["question"],
            "gold_answer": sample["answer"],
            "pred_answer": pred["answer"] if pred else "",
            "answer_correct": ans_ok,
            "cypher_valid": cypher_ok,
        })

        if (i + 1) % 10 == 0:
            logger.info("  … %d/%d traités", i + 1, len(data))
            save_checkpoint("query", {
                "next_idx": i + 1, "correct": correct,
                "cypher_valid": cypher_valid, "details": details,
                "elapsed": elapsed_prev + (time.time() - t0),
            })

    elapsed = elapsed_prev + (time.time() - t0)
    accuracy = correct / len(data) if data else 0.0
    cypher_rate = cypher_valid / len(data) if data else 0.0

    # Bootstrap CI sur accuracy
    per_sample_correct = [
        1.0 if d.get("answer_correct") else 0.0 for d in details
    ]
    accuracy_ci = bootstrap_ci(per_sample_correct)

    results = {
        "experiment": "text_to_query",
        "n_samples": len(data),
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "accuracy_ci": accuracy_ci,
        "target_accuracy": TARGET_ACCURACY_QUERY,
        "target_met": accuracy >= TARGET_ACCURACY_QUERY,
        "cypher_syntax_valid_rate": round(cypher_rate, 4),
        "elapsed_seconds": round(elapsed, 1),
        "details": details,
    }
    logger.info(
        "Text-to-Query — Accuracy=%.2f  (cible %.2f)  Cypher-valid=%.2f  [%.1fs]",
        accuracy, TARGET_ACCURACY_QUERY, cypher_rate, elapsed,
    )
    clear_checkpoint("query")
    return results
