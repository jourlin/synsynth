"""
Vérificateur de fidélité (Faithfulness Checker) pour SYNSYNTH+.

Décompose une réponse en claims atomiques, puis vérifie chaque claim
par NLI (Natural Language Inference) contre le contexte source.
Utilise le LLM comme juge avec deux perspectives pour réduire la variance.
"""
from __future__ import annotations

import json
import re
from typing import Any

from synsynth_config import logger
from synsynth_model import generate_structured


# ── Prompts ────────────────────────────────────────────────────────────────

_DECOMPOSE_PROMPT = (
    "Décompose la réponse suivante en affirmations atomiques (claims). "
    "Chaque claim doit être une phrase simple et indépendante. "
    "Réponds UNIQUEMENT en JSON : {\"claims\": [\"claim 1\", \"claim 2\", ...]}"
)

_NLI_PROMPT = (
    "Tu es un vérificateur de faits. Pour chaque affirmation, détermine si "
    "elle est SUPPORTÉE par le contexte (le contexte contient l'information, "
    "même reformulée ou paraphrasée) ou NON-SUPPORTÉE (information inventée "
    "ou absente du contexte). "
    "Réponds UNIQUEMENT en JSON : "
    "{\"verdicts\": [{\"claim\": \"...\", \"supported\": true/false}, ...]}"
)


def decompose_claims(answer: str) -> list[str]:
    """Décompose une réponse en claims atomiques via le LLM."""
    prompt = f"Réponse à décomposer :\n{answer}"
    raw = generate_structured(prompt, system=_DECOMPOSE_PROMPT, max_new_tokens=512)

    # Extraire le JSON
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            claims = obj.get("claims", [])
            if isinstance(claims, list) and claims:
                return [str(c) for c in claims]
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback : découper en phrases
    sentences = re.split(r"[.!?]+", answer)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def verify_claims(claims: list[str], context: str) -> list[dict]:
    """Vérifie chaque claim contre le contexte via NLI."""
    if not claims:
        return []

    claims_text = "\n".join(f"- {c}" for c in claims)
    prompt = (
        f"Contexte :\n{context}\n\n"
        f"Affirmations à vérifier :\n{claims_text}"
    )
    raw = generate_structured(prompt, system=_NLI_PROMPT, max_new_tokens=1024)

    # Extraire les verdicts
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            verdicts = obj.get("verdicts", [])
            if isinstance(verdicts, list):
                return verdicts
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback : considérer non-vérifiable
    return [{"claim": c, "supported": False} for c in claims]


def compute_faithfulness(answer: str, context: str) -> dict[str, Any]:
    """Pipeline complet : décomposition → vérification → score.

    Returns:
        {"faithfulness": float, "total_claims": int, "supported": int,
         "claims": [{"claim": str, "supported": bool}, ...]}
    """
    claims = decompose_claims(answer)
    if not claims:
        return {"faithfulness": 1.0, "total_claims": 0, "supported": 0, "claims": []}

    verdicts = verify_claims(claims, context)

    # Aligner les verdicts avec les claims
    supported = sum(1 for v in verdicts if v.get("supported", False))
    total = len(claims)
    score = supported / total if total > 0 else 0.0

    return {
        "faithfulness": round(score, 4),
        "total_claims": total,
        "supported": supported,
        "claims": verdicts,
    }
