"""
Sandbox I/O — Toutes les écritures et lectures passent par ce module
afin de garantir le confinement dans WORKSPACE_ROOT.
"""
from __future__ import annotations

import json
import os
from typing import Any

from synsynth_config import WORKSPACE_ROOT, safe_path, logger


def read_text(relpath: str) -> str:
    """Lit un fichier texte à l'intérieur du workspace."""
    p = safe_path(relpath)
    with open(p, encoding="utf-8") as f:
        return f.read()


def write_text(relpath: str, content: str) -> str:
    """Écrit un fichier texte. Crée les répertoires intermédiaires."""
    p = safe_path(relpath)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Fichier écrit → %s (%d car.)", relpath, len(content))
    return p


def write_json(relpath: str, obj: Any) -> str:
    """Sérialise un objet en JSON dans le workspace."""
    return write_text(relpath, json.dumps(obj, ensure_ascii=False, indent=2))


def read_json(relpath: str) -> Any:
    """Lit un fichier JSON depuis le workspace."""
    return json.loads(read_text(relpath))


def append_text(relpath: str, content: str) -> str:
    """Ajoute du texte à un fichier existant."""
    p = safe_path(relpath)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(content)
    return p
