"""
Configuration centralisée pour SYNSYNTH+ — Sandboxée dans PJKG4.
"""
import os
import sys

# ── Racine absolue du workspace ─────────────────────────────────────────────
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Garde-fou de confinement ────────────────────────────────────────────────
def safe_path(*parts: str) -> str:
    """Renvoie un chemin absolu garanti SOUS WORKSPACE_ROOT.
    Lève RuntimeError si le chemin résolu tente de remonter."""
    candidate = os.path.normpath(os.path.join(WORKSPACE_ROOT, *parts))
    if not candidate.startswith(WORKSPACE_ROOT):
        raise RuntimeError(
            f"Accès interdit : {candidate!r} est hors de {WORKSPACE_ROOT!r}"
        )
    return candidate


# ── Répertoires de travail ─────────────────────────────────────────────────
DATA_DIR       = safe_path("data")
RESULTS_DIR    = safe_path("results")
MODELS_DIR     = safe_path("models")
CACHE_DIR      = safe_path("cache")
ARTICLE_DIR    = safe_path("article")
LOGS_DIR       = safe_path("logs")

for d in (DATA_DIR, RESULTS_DIR, MODELS_DIR, CACHE_DIR, ARTICLE_DIR, LOGS_DIR):
    os.makedirs(d, exist_ok=True)

# ── Modèle ──────────────────────────────────────────────────────────────────
MODEL_REPO    = "unsloth/gemma-4-26B-A4B-it-GGUF"
MODEL_QUANT   = "UD-Q4_K_XK"
MAX_SEQ_LEN   = 8192
TEMPERATURE   = 0.3
TOP_P         = 0.9

# ── Meilleurs modèles par tâche (benchmark du 5 avril 2026) ────────────────
TASK_MODELS = {
    "extraction": "gemma4:26b",        # F1=0.75, P=1.00, R=0.60
    "query":      "qwen3-deep:latest",  # Acc=1.00, Cypher=1.00, 20s
    "multihop":   "phi4:latest",        # EM=0.80, 124s
    "rag":        "mistral-small:latest",# Faith=1.00, Relev=0.92
}
DEFAULT_MODEL = "gemma4:26b"  # modèle par défaut si tâche non listée

# ── Seuils d'évaluation (tirés du cahier des charges) ──────────────────────
TARGET_F1_EXTRACTION   = 0.85   # F1 ≥ 85 % sur DocRED/TACRED
TARGET_ACCURACY_QUERY  = 0.90   # Accuracy ≥ 90 % WebQuestionsSP
TARGET_FAITHFULNESS    = 1.00   # Fidélité RAGAS ≈ 100 %

# ── Paramètres d'expérimentation ───────────────────────────────────────────
RANDOM_SEED        = 42
BATCH_SIZE         = 4
NUM_EVAL_SAMPLES   = 200       # nombre d'échantillons pour chaque benchmark
NUM_MULTIHOP_HOPS  = 3         # profondeur max pour le raisonnement multi-hop

# Tailles par expérience — benchmarks complets pour publication
NUM_EXTRACTION_SAMPLES = 500   # Re-DocRED validation (téléchargement HF)
NUM_QUERY_SAMPLES      = 200   # WebQuestionsSP-style (hardcoded + synthétique)
NUM_MULTIHOP_SAMPLES   = 500   # HotpotQA validation (téléchargement HF)
NUM_RAG_SAMPLES        = 50    # RAGAS-style (hardcoded + synthétique)

# ── Journalisation ─────────────────────────────────────────────────────────
import logging

LOG_FILE = safe_path("logs", "synsynth.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("SYNSYNTH+")
