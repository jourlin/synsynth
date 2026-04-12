"""
Mécanisme de checkpoint / reprise pour les expériences SYNSYNTH+.

Permet de sauvegarder l'état intermédiaire d'une expérience et de
reprendre là où elle s'est arrêtée en cas d'interruption.
"""
from __future__ import annotations

import json
import os

from synsynth_config import RESULTS_DIR, logger

CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")


def save_checkpoint(exp_name: str, data: dict) -> None:
    """Sauvegarde atomique (write-then-rename) d'un checkpoint."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"ckpt_{exp_name}.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)  # atomique sur POSIX
    logger.debug("Checkpoint sauvegardé : %s (next_idx=%d)", exp_name, data.get("next_idx", -1))


def load_checkpoint(exp_name: str) -> dict | None:
    """Charge un checkpoint existant, ou renvoie None."""
    path = os.path.join(CHECKPOINT_DIR, f"ckpt_{exp_name}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Checkpoint trouvé pour '%s' — reprise à idx=%d", exp_name, data.get("next_idx", 0))
        return data
    return None


def clear_checkpoint(exp_name: str) -> None:
    """Supprime le checkpoint une fois l'expérience terminée."""
    path = os.path.join(CHECKPOINT_DIR, f"ckpt_{exp_name}.json")
    if os.path.exists(path):
        os.remove(path)
        logger.info("Checkpoint supprimé : %s (expérience terminée)", exp_name)
