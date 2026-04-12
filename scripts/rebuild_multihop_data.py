#!/usr/bin/env python3
"""
Reconstruction des données d'entraînement multihop avec des chaînes de
raisonnement construites à partir des supporting_facts gold de HotpotQA.

Problème corrigé :
    Les données V1 contenaient 100% de chaînes dégénérées :
        {"reasoning_chain": ["D'après les faits fournis"], "answer": "..."}
    Le modèle Phi-4 qui les a générées n'a produit aucun raisonnement réel.

Solution :
    On exploite les supporting_facts gold de HotpotQA pour construire des
    chaînes de raisonnement template à 2-3 étapes, sans recourir à un LLM.

Usage :
    python scripts/rebuild_multihop_data.py
    python scripts/rebuild_multihop_data.py --max-samples 5000
"""
from __future__ import annotations

import json
import os
import sys
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from synsynth_config import WORKSPACE_ROOT, logger, RANDOM_SEED

HF_CACHE = os.path.join(WORKSPACE_ROOT, "data", "hf_cache")
QLORA_DATA_DIR = os.path.join(WORKSPACE_ROOT, "data", "qlora")

MULTIHOP_SYSTEM = (
    "Tu es un agent de raisonnement multi-hop. "
    "Tu reçois une question complexe et des faits de support. "
    "Tu dois raisonner étape par étape en reliant les faits, puis fournir "
    "ta réponse finale COURTE et PRÉCISE (quelques mots seulement). "
    "Réponds en JSON : "
    '{"reasoning_chain": ["...", "..."], "answer": "réponse courte"}'
)

# Nombre de points pour la courbe d'apprentissage
CURVE_POINTS = [10, 50, 200, 500, 1000, 3000]


def extract_supporting_sentences(example: dict) -> list[tuple[str, str]]:
    """Extrait les phrases de support gold de HotpotQA.

    Renvoie une liste de (title, sentence_text) triées par ordre d'apparition.
    """
    titles = example.get("context", {}).get("title", [])
    sentences = example.get("context", {}).get("sentences", [])
    sf_titles = example.get("supporting_facts", {}).get("title", [])
    sf_sent_ids = example.get("supporting_facts", {}).get("sent_id", [])

    # Index title -> sentences
    title_to_sents = {}
    for t, s in zip(titles, sentences):
        title_to_sents[t] = s

    # Extraire les phrases de support
    support_pairs = []
    seen = set()
    for sf_title, sf_idx in zip(sf_titles, sf_sent_ids):
        key = (sf_title, sf_idx)
        if key in seen:
            continue
        seen.add(key)
        sents = title_to_sents.get(sf_title, [])
        if 0 <= sf_idx < len(sents):
            support_pairs.append((sf_title, sents[sf_idx]))

    return support_pairs


def build_reasoning_chain(
    support_pairs: list[tuple[str, str]],
    question: str,
    answer: str,
    q_type: str,
) -> list[str]:
    """Construit une chaîne de raisonnement à partir des faits de support gold.

    Produit 2-4 étapes selon le nombre de faits et le type de question.
    """
    if not support_pairs:
        return [f"La réponse à la question est {answer}."]

    chain = []

    # Étape(s) factuelle(s) : une par fait de support
    for i, (title, sent) in enumerate(support_pairs):
        sent_clean = sent.strip().rstrip(".")
        chain.append(f"D'après l'article « {title} » : {sent_clean}.")

    # Étape de synthèse
    if q_type == "comparison" and len(support_pairs) >= 2:
        t1 = support_pairs[0][0]
        t2 = support_pairs[1][0]
        chain.append(
            f"En comparant les informations sur {t1} et {t2}, "
            f"la réponse est {answer}."
        )
    elif len(support_pairs) >= 2:
        chain.append(
            f"En reliant ces {len(support_pairs)} faits, "
            f"la réponse est {answer}."
        )
    else:
        chain.append(f"Donc la réponse est {answer}.")

    return chain


def format_multihop_sample_v2(example: dict) -> dict | None:
    """Formate un exemple HotpotQA en sample chat avec chaîne de raisonnement.

    Version 2 : utilise les supporting_facts gold au lieu d'un placeholder.
    """
    question = example.get("question", "")
    answer = example.get("answer", "")
    q_type = example.get("type", "bridge")

    if not question or not answer:
        return None

    # Contexte complet (tous les paragraphes, comme en V1)
    titles = example.get("context", {}).get("title", [])
    sentences = example.get("context", {}).get("sentences", [])
    context_parts = []
    for title, sents in zip(titles, sentences):
        context_parts.append(f"{title}: {' '.join(sents)}")
    context = "\n".join(context_parts)  # Tous les paragraphes (gold + distractors)

    if not context:
        return None

    # Chaîne de raisonnement à partir des supporting_facts gold
    support_pairs = extract_supporting_sentences(example)
    reasoning_chain = build_reasoning_chain(support_pairs, question, answer, q_type)

    user_prompt = f"Faits :\n{context}\n\nQuestion : {question}"
    assistant_response = json.dumps(
        {"reasoning_chain": reasoning_chain, "answer": answer},
        ensure_ascii=False,
    )

    return {
        "messages": [
            {"role": "system", "content": MULTIHOP_SYSTEM},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]
    }


def rebuild_multihop_data(max_samples: int = 3000, seed: int = RANDOM_SEED):
    """Reconstruit les données multihop V2 avec chaînes de raisonnement gold."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("pip install datasets requis.")
        return

    logger.info("Chargement de HotpotQA (distractor, train)...")
    ds = load_dataset("hotpot_qa", "distractor", split="train", cache_dir=HF_CACHE)
    logger.info("HotpotQA train : %d questions.", len(ds))

    # Formater tous les exemples
    all_samples = []
    chain_lengths = []
    for ex in ds:
        sample = format_multihop_sample_v2(ex)
        if sample:
            # Compter la longueur de chaîne pour stats
            parsed = json.loads(sample["messages"][2]["content"])
            chain_lengths.append(len(parsed["reasoning_chain"]))
            all_samples.append(sample)
        if len(all_samples) >= max_samples:
            break

    logger.info(
        "V2 : %d samples formatés. Chaîne moyenne : %.1f étapes (min=%d, max=%d).",
        len(all_samples),
        sum(chain_lengths) / len(chain_lengths),
        min(chain_lengths),
        max(chain_lengths),
    )

    # Sauvegarder le fichier principal
    # Nommé multihop_v2_train.jsonl pour compatibilité avec learning_curve.py
    # qui cherche {task}_train.jsonl → task="multihop_v2"
    os.makedirs(QLORA_DATA_DIR, exist_ok=True)
    out_path = os.path.join(QLORA_DATA_DIR, "multihop_v2_train.jsonl")
    with open(out_path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    logger.info("Sauvegardé : %s (%d lignes)", out_path, len(all_samples))

    # Créer les sous-échantillons pour la courbe d'apprentissage
    rng = random.Random(seed)
    for n in CURVE_POINTS:
        if n > len(all_samples):
            logger.warning("n=%d > %d samples disponibles, skip.", n, len(all_samples))
            continue
        sub = rng.sample(all_samples, n) if n < len(all_samples) else all_samples
        sub_path = os.path.join(QLORA_DATA_DIR, f"multihop_v2_train_n{n}.jsonl")
        with open(sub_path, "w") as f:
            for sample in sub:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        # Stats de chaîne pour ce sous-échantillon
        sub_chains = [
            len(json.loads(s["messages"][2]["content"])["reasoning_chain"])
            for s in sub
        ]
        logger.info(
            "  → %s : %d lignes, chaîne moy=%.1f",
            sub_path, n, sum(sub_chains) / len(sub_chains),
        )

    # Afficher un exemple
    ex = all_samples[0]
    parsed = json.loads(ex["messages"][2]["content"])
    logger.info("\n=== Exemple V2 ===")
    logger.info("Question : %s", ex["messages"][1]["content"][-200:])
    logger.info("Chaîne   : %s", parsed["reasoning_chain"])
    logger.info("Réponse  : %s", parsed["answer"])

    # Comparer avec V1
    v1_path = os.path.join(QLORA_DATA_DIR, "multihop_train.jsonl")
    if os.path.exists(v1_path):
        with open(v1_path) as f:
            v1_first = json.loads(f.readline())
        v1_parsed = json.loads(v1_first["messages"][2]["content"])
        logger.info("\n=== Exemple V1 (dégénéré) ===")
        logger.info("Chaîne   : %s", v1_parsed["reasoning_chain"])
        logger.info("Réponse  : %s", v1_parsed["answer"])


# ═══════════════════════════════════════════════════════════════════════
# V3 — Chaînes concises + contexte réduit (2 gold + 3 distracteurs)
# ═══════════════════════════════════════════════════════════════════════

def build_reasoning_chain_v3(
    support_pairs: list[tuple[str, str]],
    answer: str,
    q_type: str,
) -> list[str]:
    """Chaîne de raisonnement concise : fait-clé résumé, pas de citation verbatim."""
    if not support_pairs:
        return [f"Réponse : {answer}."]

    chain = []
    for title, sent in support_pairs:
        # Tronquer les phrases longues à ~100 chars
        s = sent.strip().rstrip(".")
        if len(s) > 120:
            s = s[:117] + "..."
        chain.append(f"{title} : {s}.")

    # Synthèse
    if q_type == "comparison" and len(support_pairs) >= 2:
        chain.append(f"Comparaison → {answer}.")
    elif len(support_pairs) >= 2:
        chain.append(f"Donc → {answer}.")
    else:
        chain.append(f"Donc → {answer}.")

    return chain


def format_multihop_sample_v3(example: dict) -> dict | None:
    """V3 : contexte réduit (2 gold + 3 distracteurs) + chaînes concises.

    Cible : < 1024 tokens total par sample.
    """
    question = example.get("question", "")
    answer = example.get("answer", "")
    q_type = example.get("type", "bridge")

    if not question or not answer:
        return None

    titles = example.get("context", {}).get("title", [])
    sentences = example.get("context", {}).get("sentences", [])
    sf_titles_set = set(example.get("supporting_facts", {}).get("title", []))

    # Séparer paragraphes gold et distracteurs
    gold_parts = []
    distractor_parts = []
    for title, sents in zip(titles, sentences):
        para = f"{title}: {' '.join(sents)}"
        if title in sf_titles_set:
            gold_parts.append(para)
        else:
            distractor_parts.append(para)

    # Garder les 2 paragraphes gold + max 3 distracteurs, mélangés
    selected = gold_parts + distractor_parts[:3]
    random.shuffle(selected)
    context = "\n".join(selected)

    if not context:
        return None

    support_pairs = extract_supporting_sentences(example)
    reasoning_chain = build_reasoning_chain_v3(support_pairs, answer, q_type)

    user_prompt = f"Faits :\n{context}\n\nQuestion : {question}"
    assistant_response = json.dumps(
        {"reasoning_chain": reasoning_chain, "answer": answer},
        ensure_ascii=False,
    )

    return {
        "messages": [
            {"role": "system", "content": MULTIHOP_SYSTEM},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]
    }


def rebuild_multihop_data_v3(max_samples: int = 3000, seed: int = RANDOM_SEED):
    """Reconstruit données multihop V3 : concises et dans le budget tokens."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("pip install datasets requis.")
        return

    random.seed(seed)

    logger.info("Chargement de HotpotQA (distractor, train)...")
    ds = load_dataset("hotpot_qa", "distractor", split="train", cache_dir=HF_CACHE)
    logger.info("HotpotQA train : %d questions.", len(ds))

    all_samples = []
    chain_lengths = []
    for ex in ds:
        sample = format_multihop_sample_v3(ex)
        if sample:
            parsed = json.loads(sample["messages"][2]["content"])
            chain_lengths.append(len(parsed["reasoning_chain"]))
            all_samples.append(sample)
        if len(all_samples) >= max_samples:
            break

    logger.info(
        "V3 : %d samples formatés. Chaîne moyenne : %.1f étapes (min=%d, max=%d).",
        len(all_samples),
        sum(chain_lengths) / len(chain_lengths),
        min(chain_lengths),
        max(chain_lengths),
    )

    os.makedirs(QLORA_DATA_DIR, exist_ok=True)
    out_path = os.path.join(QLORA_DATA_DIR, "multihop_v3_train.jsonl")
    with open(out_path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    logger.info("Sauvegardé : %s (%d lignes)", out_path, len(all_samples))

    rng = random.Random(seed)
    for n in CURVE_POINTS:
        if n > len(all_samples):
            logger.warning("n=%d > %d samples disponibles, skip.", n, len(all_samples))
            continue
        sub = rng.sample(all_samples, n) if n < len(all_samples) else all_samples
        sub_path = os.path.join(QLORA_DATA_DIR, f"multihop_v3_train_n{n}.jsonl")
        with open(sub_path, "w") as f:
            for sample in sub:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        sub_chains = [
            len(json.loads(s["messages"][2]["content"])["reasoning_chain"])
            for s in sub
        ]
        logger.info(
            "  → %s : %d lignes, chaîne moy=%.1f",
            sub_path, n, sum(sub_chains) / len(sub_chains),
        )

    # Exemple
    ex = all_samples[0]
    parsed = json.loads(ex["messages"][2]["content"])
    logger.info("\n=== Exemple V3 ===")
    logger.info("User (last 300): %s", ex["messages"][1]["content"][-300:])
    logger.info("Chaîne   : %s", parsed["reasoning_chain"])
    logger.info("Réponse  : %s", parsed["answer"])


# ═══════════════════════════════════════════════════════════════════════
# V4 — Format aligné sur l'évaluation (exp_multihop.py)
#   Fix 1 : system prompt identique (avec les 2 exemples few-shot)
#   Fix 2 : user prompt = Question → Faits de support (bullet) → instruction
#   Fix 3 : contexte réduit (2 gold + 3 distracteurs) comme V3
# ═══════════════════════════════════════════════════════════════════════

# System prompt identique à exp_multihop.py SYSTEM_PROMPT
MULTIHOP_SYSTEM_V4 = (
    "Tu es un agent de raisonnement multi-hop. "
    "Tu reçois une question complexe et des faits de support. "
    "Tu dois raisonner étape par étape en reliant les faits, puis fournir "
    "ta réponse finale COURTE et PRÉCISE (quelques mots seulement). "
    "Réponds en JSON : "
    '{"reasoning_chain": ["...", "..."], "answer": "réponse courte"}\n\n'
    "Exemple :\n"
    "Question : Le fondateur de l'entreprise basée à Cupertino a étudié où ?\n"
    "Faits : Apple a son siège à Cupertino. Apple a été fondée par Steve Jobs. "
    "Steve Jobs a étudié au Reed College.\n"
    'Réponse : {"reasoning_chain": ["Apple est basée à Cupertino", '
    '"Steve Jobs a fondé Apple", "Jobs a étudié au Reed College"], '
    '"answer": "Reed College"}\n\n'
    "Exemple :\n"
    "Question : Were Scott Derrickson and Ed Wood of the same nationality?\n"
    "Faits : Scott Derrickson is an American director. Ed Wood was an American filmmaker.\n"
    'Réponse : {"reasoning_chain": ["Scott Derrickson is American", '
    '"Ed Wood was American"], "answer": "yes"}'
)


def format_multihop_sample_v4(example: dict) -> dict | None:
    """V4 : format aligné sur exp_multihop.py (system prompt + user prompt).

    - System prompt = SYSTEM_PROMPT de exp_multihop.py (avec exemples few-shot)
    - User prompt = Question → Faits de support (bullets) → instruction
    - Contexte réduit = 2 gold + 3 distracteurs (comme V3)
    - Chaînes concises (comme V3)
    """
    question = example.get("question", "")
    answer = example.get("answer", "")
    q_type = example.get("type", "bridge")

    if not question or not answer:
        return None

    titles = example.get("context", {}).get("title", [])
    sentences = example.get("context", {}).get("sentences", [])
    sf_titles_set = set(example.get("supporting_facts", {}).get("title", []))

    # Séparer paragraphes gold et distracteurs
    gold_parts = []
    distractor_parts = []
    for title, sents in zip(titles, sentences):
        # Chaque phrase comme un fait séparé (format bullet, comme l'éval)
        for s in sents:
            s = s.strip()
            if not s:
                continue
            if title in sf_titles_set:
                gold_parts.append(s)
            else:
                distractor_parts.append(s)

    # Garder toutes les phrases gold + max 10 phrases distracteurs
    selected = gold_parts + distractor_parts[:10]
    random.shuffle(selected)

    if not selected:
        return None

    # Format identique à exp_multihop.py: bullet points
    facts = "\n".join(f"- {s}" for s in selected)

    support_pairs = extract_supporting_sentences(example)
    reasoning_chain = build_reasoning_chain_v3(support_pairs, answer, q_type)

    # Format identique à exp_multihop.py: Question → Faits → instruction
    user_prompt = (
        f"Question : {question}\n\n"
        f"Faits de support :\n{facts}\n\n"
        "Raisonne étape par étape puis donne la réponse."
    )

    assistant_response = json.dumps(
        {"reasoning_chain": reasoning_chain, "answer": answer},
        ensure_ascii=False,
    )

    return {
        "messages": [
            {"role": "system", "content": MULTIHOP_SYSTEM_V4},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]
    }


def rebuild_multihop_data_v4(max_samples: int = 3000, seed: int = RANDOM_SEED):
    """Reconstruit données multihop V4 : format aligné sur l'évaluation."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("pip install datasets requis.")
        return

    random.seed(seed)

    logger.info("Chargement de HotpotQA (distractor, train)...")
    ds = load_dataset("hotpot_qa", "distractor", split="train", cache_dir=HF_CACHE)
    logger.info("HotpotQA train : %d questions.", len(ds))

    all_samples = []
    chain_lengths = []
    for ex in ds:
        sample = format_multihop_sample_v4(ex)
        if sample:
            parsed = json.loads(sample["messages"][2]["content"])
            chain_lengths.append(len(parsed["reasoning_chain"]))
            all_samples.append(sample)
        if len(all_samples) >= max_samples:
            break

    logger.info(
        "V4 : %d samples formatés. Chaîne moyenne : %.1f étapes (min=%d, max=%d).",
        len(all_samples),
        sum(chain_lengths) / len(chain_lengths),
        min(chain_lengths),
        max(chain_lengths),
    )

    os.makedirs(QLORA_DATA_DIR, exist_ok=True)
    out_path = os.path.join(QLORA_DATA_DIR, "multihop_v4_train.jsonl")
    with open(out_path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    logger.info("Sauvegardé : %s (%d lignes)", out_path, len(all_samples))

    rng = random.Random(seed)
    for n in CURVE_POINTS:
        if n > len(all_samples):
            logger.warning("n=%d > %d samples disponibles, skip.", n, len(all_samples))
            continue
        sub = rng.sample(all_samples, n) if n < len(all_samples) else all_samples
        sub_path = os.path.join(QLORA_DATA_DIR, f"multihop_v4_train_n{n}.jsonl")
        with open(sub_path, "w") as f:
            for sample in sub:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        sub_chains = [
            len(json.loads(s["messages"][2]["content"])["reasoning_chain"])
            for s in sub
        ]
        logger.info(
            "  → %s : %d lignes, chaîne moy=%.1f",
            sub_path, n, sum(sub_chains) / len(sub_chains),
        )

    # Exemple
    ex = all_samples[0]
    parsed = json.loads(ex["messages"][2]["content"])
    logger.info("\n=== Exemple V4 ===")
    logger.info("System (100 chars): %s...", ex["messages"][0]["content"][:100])
    logger.info("User (last 400): %s", ex["messages"][1]["content"][-400:])
    logger.info("Chaîne   : %s", parsed["reasoning_chain"])
    logger.info("Réponse  : %s", parsed["answer"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reconstruction données multihop")
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--version", choices=["v2", "v3", "v4"], default="v4",
        help="Version des données à générer (défaut: v4)",
    )
    args = parser.parse_args()
    if args.version == "v2":
        rebuild_multihop_data(max_samples=args.max_samples, seed=args.seed)
    elif args.version == "v3":
        rebuild_multihop_data_v3(max_samples=args.max_samples, seed=args.seed)
    else:
        rebuild_multihop_data_v4(max_samples=args.max_samples, seed=args.seed)
