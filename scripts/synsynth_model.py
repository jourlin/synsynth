"""
Interface d'inférence vers Gemma-4-26B via Ollama (API HTTP locale).

Le modèle doit avoir été préalablement tiré :
    ollama pull gemma4:26b

Fournit les mêmes fonctions generate() / generate_structured() que la
version unsloth, mais s'appuie sur l'API REST Ollama (localhost:11434).
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Optional

from synsynth_config import (
    MODEL_REPO, MODEL_QUANT, MAX_SEQ_LEN,
    TEMPERATURE, TOP_P, CACHE_DIR, WORKSPACE_ROOT, logger,
)

# ── Configuration Ollama ────────────────────────────────────────────────────
OLLAMA_BASE   = "http://127.0.0.1:11434"
OLLAMA_MODEL  = "gemma4:26b"          # nom du modèle dans ollama list
_TIMEOUT      = 600                    # secondes — les réponses longues prennent du temps


def _ollama_chat(
    messages: list[dict],
    *,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_tokens: int = 2048,
    json_format: bool = False,
) -> str:
    """Appel bas-niveau à POST /api/chat (mode non-streaming).

    Gemma-4 utilise un mode « thinking » : le raisonnement va dans
    message.thinking et la réponse finale dans message.content.
    On alloue suffisamment de tokens pour les deux phases.
    """
    # Budget généreux pour thinking + content
    # Gemma-4 utilise facilement 2000-3000 tokens en thinking,
    # il faut prévoir au minimum 4096 tokens pour éviter les troncatures.
    effective_tokens = max(max_tokens * 4, 4096)

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": effective_tokens,
            "num_ctx": MAX_SEQ_LEN,
        },
    }
    if json_format:
        payload["format"] = "json"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            msg = body.get("message", {})
            content = msg.get("content", "").strip()
            thinking = msg.get("thinking", "").strip()

            # Détecter un content corrompu (que des espaces/newlines/accolades)
            if content and json_format:
                import re as _re
                if not _re.search(r'[a-zA-Z0-9]', content):
                    logger.debug("Content corrompu (pas de caractères utiles) — fallback thinking.")
                    content = ""

            # Si content est vide mais thinking contient la réponse
            if not content and thinking:
                logger.debug("Content vide — extraction depuis thinking.")
                if json_format:
                    content = _extract_last_json_from_thinking(thinking)
                else:
                    content = _extract_answer_from_thinking(thinking)

            return content
    except urllib.error.URLError as e:
        logger.error("Ollama injoignable (%s). Le serveur tourne-t-il ?", e)
        raise RuntimeError(f"Ollama injoignable : {e}") from e


def _extract_last_json_from_thinking(thinking: str) -> str:
    """Extrait le DERNIER objet JSON valide depuis le bloc thinking.

    Gemma-4 met souvent un écho du template en début de thinking,
    et sa conclusion (le vrai JSON de réponse) à la fin.
    On parcourt tous les candidats JSON et on prend le dernier valide.
    """
    import re
    candidates = list(re.finditer(r'\{[^{}]*\}', thinking, re.DOTALL))
    # Parcourir en ordre inverse pour trouver le dernier JSON valide
    for m in reversed(candidates):
        try:
            json.loads(m.group())
            return m.group()
        except (json.JSONDecodeError, ValueError):
            continue
    # Aucun JSON valide trouvé — fallback textuel
    return _extract_answer_from_thinking(thinking)


def _extract_answer_from_thinking(thinking: str) -> str:
    """Extrait la réponse finale depuis le bloc thinking de Gemma-4.

    Cherche des patterns comme "Direct answer:", "Réponse:", "Answer:", etc.
    Sinon renvoie la dernière ligne non vide.
    """
    import re
    # Patterns courants où Gemma met sa réponse finale dans le thinking
    for pattern in [
        r"(?:direct answer|réponse finale?|answer|réponse)\s*:\s*[\"«]?(.+?)[\"»]?\s*$",
        r"(?:donc|thus|so)\s*[,:]\s*(.+)$",
    ]:
        m = re.search(pattern, thinking, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip().strip('"').strip("«»").strip()

    # Fallback : dernières lignes significatives (pas juste des puces)
    lines = [l.strip() for l in thinking.strip().splitlines() if l.strip()]
    if lines:
        # Prendre la dernière ligne qui ressemble à une réponse
        for line in reversed(lines):
            cleaned = re.sub(r"^[\*\-\•\d\.]+\s*", "", line).strip()
            if len(cleaned) > 5:
                return cleaned
        return lines[-1]
    return thinking


def _check_ollama():
    """Vérifie qu'Ollama est accessible et que le modèle est disponible."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            models = [m["name"] for m in body.get("models", [])]
            if not any(OLLAMA_MODEL in m for m in models):
                logger.warning(
                    "Modèle '%s' absent d'Ollama. Disponibles : %s",
                    OLLAMA_MODEL, models,
                )
            else:
                logger.info("Ollama OK — modèle '%s' trouvé.", OLLAMA_MODEL)
    except Exception as e:
        logger.error("Impossible de contacter Ollama : %s", e)
        raise


# Vérification au premier import
_check_ollama()


def generate(
    prompt: str,
    *,
    system: str = "",
    max_new_tokens: int = 2048,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    json_format: bool = False,
) -> str:
    """Génère une complétion à partir d'un prompt via Ollama."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    return _ollama_chat(
        messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        json_format=json_format,
    )


def generate_structured(
    prompt: str,
    *,
    system: str = "",
    json_mode: bool = False,
    max_new_tokens: int = 4096,
) -> str:
    """Génère avec un prompt orienté sortie structurée (JSON/Markdown)."""
    if json_mode:
        system = (system + "\n" if system else "") + (
            "Tu dois répondre UNIQUEMENT avec un objet JSON valide, "
            "sans texte avant ni après."
        )
    return generate(
        prompt,
        system=system,
        max_new_tokens=max_new_tokens,
        temperature=0.1,   # quasi-déterministe pour les sorties structurées
        json_format=json_mode,
    )
