# SYNSYNTH+ — Reproducibility Package

**Frugal Knowledge Graph Construction with Local LLMs: A Zero-Shot Pipeline, Self-Consistency and Wisdom of Artificial Crowds.**

> Jourlin, P. (2026). *Frugal Knowledge Graph Construction with Local LLMs: A Zero-Shot Pipeline, Self-Consistency and Wisdom of Artificial Crowds.* [arXiv:2604.11104](https://arxiv.org/abs/2604.11104).

This repository contains the scripts, data, pre-computed results and an interactive demo
to **fully reproduce** the experiments described in the paper.

🇫🇷 [Version française](README.FR.md)

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Hardware & Software Requirements](#hardware--software-requirements)
4. [Installation](#installation)
5. [Reproducing Experiments](#reproducing-experiments)
6. [Exploring Pre-computed Results](#exploring-pre-computed-results)
7. [Interactive Demo](#interactive-demo)
8. [Key Results](#key-results)
9. [License & Citation](#license--citation)

---

## Overview

SYNSYNTH+ is a complete pipeline that transforms unstructured text into a
queryable knowledge graph, using lightweight models deployed locally on a
single consumer-grade GPU. The project evaluates four axes:

| Axis | Benchmark | Main Model |
|------|-----------|------------|
| Relation extraction | Re-DocRED (N=500) | gemma4:26b (MoE, 4B active) |
| Text-to-Cypher | WebQuestionsSP | qwen3-deep:latest |
| Multi-hop reasoning | HotpotQA (N=500) | phi4:latest (14B) |
| RAG Faithfulness | RAGAS | mistral-small:latest |

The pipeline also includes **self-consistency** (SC) analyses, a **cascade
pipeline** (phi4 → gpt-oss), **QLoRA learning curves**, and **calibration**
and **variance** studies.

---

## Repository Structure

```
synsynth/
├── README.md                  # This document
├── README.FR.md               # French version
├── LICENSE.md                 # Hippocratic License 3.0
├── requirements.txt           # Python dependencies (109 packages)
├── scripts/
│   ├── run_synsynth.py        # Main CLI entry point
│   ├── synsynth_config.py     # Centralized configuration (auto-detects paths)
│   ├── synsynth_model.py      # Ollama API interface
│   ├── synsynth_data.py       # Dataset loaders (HotpotQA, Re-DocRED, etc.)
│   ├── synsynth_io.py         # Sandboxed I/O
│   ├── synsynth_stats.py      # Bootstrap CI, token F1
│   ├── synsynth_checkpoint.py # Checkpoint management
│   ├── synsynth_viz.py        # Matplotlib visualizations
│   ├── synsynth_article.py    # Markdown article generator
│   ├── exp_extraction.py      # Exp 1 — Relation extraction
│   ├── exp_query.py           # Exp 2 — Text-to-Graph Query
│   ├── exp_multihop.py        # Exp 3 — Multi-hop reasoning
│   ├── exp_rag.py             # Exp 4 — RAG Faithfulness
│   ├── self_consistency.py    # Self-consistency (SC) k passes
│   ├── pipeline_self_consistency.py  # Cascade phi4→gpt-oss with SC
│   ├── variance_self_consistency.py  # Inter-run variance SC (R=5)
│   ├── variance_cascade.py    # Inter-run variance cascade (R=3)
│   ├── benchmark_models.py    # Multi-model benchmark
│   ├── benchmark_model_selection.py  # Per-task model selection
│   ├── benchmark_extraction.py       # Cross-model extraction
│   ├── extraction_cross_model.py     # 8-model extraction comparison
│   ├── learning_curve.py      # QLoRA learning curves
│   ├── qlora_finetune.py      # QLoRA 4-bit fine-tuning
│   ├── faithfulness_checker.py       # NLI faithfulness checker
│   ├── calibration_analysis.py       # Calibration analysis
│   ├── confusion_clustering.py       # Confusion matrix clustering
│   └── rebuild_multihop_data.py      # Multi-hop data generation V2/V3/V4
├── data/                      # Datasets (downloaded automatically)
│   ├── extraction.json        # Re-DocRED cache
│   ├── multihop.json          # HotpotQA cache
│   ├── query.json             # WebQuestionsSP questions
│   └── rag.json               # RAG examples
├── results/                   # Pre-computed results (JSON + PNG)
│   ├── all_results.json       # Consolidated summary of all experiments
│   ├── self_consistency/      # SC k=5 for phi4, gpt-oss, phi4-reasoning
│   │   └── variance/          # Inter-run variance R=5 (15 files)
│   ├── pipeline_self_consistency/  # Cascade V5b + SC V5a
│   ├── extraction_benchmark/  # F1 extraction for 8 models
│   ├── extraction_cross_model/# Cross-model details
│   ├── model_selection/       # Per-task model selection
│   ├── calibration/           # Calibration diagrams
│   ├── confusion_analysis/    # Heatmaps & dendrograms
│   └── learning_curve/        # QLoRA curves (JSON + PNG)
├── article/
│   └── jourlin_2026a_fr.pdf   # Compiled article (22 pages)
└── demo/                      # Interactive Gradio application
    ├── app.py                 # Demo server
    ├── data/all_results.json  # Data for the demo
    ├── requirements.txt       # Gradio dependencies
    └── README.md              # HuggingFace Spaces card
```

---

## Hardware & Software Requirements

### Hardware

| Component | Minimum | Recommended (reference config.) |
|-----------|---------|---------------------------------|
| GPU | NVIDIA 16 GB VRAM | NVIDIA RTX 3090 (24 GB) |
| RAM | 32 GB | 64 GB |
| Storage | 50 GB | 100 GB (quantized models) |
| OS | Linux (Ubuntu 22.04+) | Ubuntu 24.04 |

### Software

| Component | Version |
|-----------|---------|
| Python | ≥ 3.10 |
| Ollama | ≥ 0.6 |
| CUDA | ≥ 12.x |
| NVIDIA driver | ≥ 550 |

---

## Installation

### 1. Install Ollama and download models

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &   # Start the server if not already running

# Main models used in the paper
ollama pull phi4:latest          # 14B, multi-hop reasoning
ollama pull gpt-oss:20b          # MoE 20B, cascade
ollama pull phi4-reasoning:plus  # 14B, chain-of-thought reasoning
ollama pull gemma4:26b           # MoE 26B (4B active), extraction
```

### 2. Clone the repository and install dependencies

```bash
git clone https://github.com/jourlin/synsynth.git
cd synsynth
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Reproducing Experiments

### Full pipeline (4 axes)

```bash
cd scripts
python run_synsynth.py
```

This sequentially runs all 4 experiments (extraction, query, multihop, RAG)
and generates results in `results/`.

### Individual experiments

```bash
# Relation extraction (Re-DocRED, N=50)
python scripts/exp_extraction.py

# Text-to-Cypher (WebQuestionsSP, N=20)
python scripts/exp_query.py

# Multi-hop reasoning (HotpotQA, N=500)
python scripts/exp_multihop.py

# RAG Faithfulness (RAGAS, N=20)
python scripts/exp_rag.py
```

### Self-consistency and cascade

```bash
# SC k=5 with phi4:latest
python scripts/self_consistency.py

# Cascade pipeline phi4→gpt-oss k=5
python scripts/pipeline_self_consistency.py

# Inter-run variance SC (R=5, ~20h GPU)
python scripts/variance_self_consistency.py

# Inter-run variance cascade (R=3, ~15h GPU)
python scripts/variance_cascade.py
```

### Additional analyses

```bash
# Multi-model benchmark
python scripts/benchmark_models.py

# 8-model extraction comparison
python scripts/extraction_cross_model.py

# QLoRA learning curves
python scripts/learning_curve.py

# Confidence calibration
python scripts/calibration_analysis.py

# Confusion matrix clustering
python scripts/confusion_clustering.py
```

---

## Exploring Pre-computed Results

All results are provided as JSON files in `results/`.
You can explore them without a GPU:

```python
import json

# Consolidated summary
with open("results/all_results.json") as f:
    results = json.load(f)

# Self-consistency phi4 k=5
with open("results/self_consistency/sc_phi4_latest_k5.json") as f:
    sc = json.load(f)
    print(f"Voted EM: {sc['summary']['em_voted']}")

# Inter-run variance (15 files, R=5 × 3 models)
import glob
for f in sorted(glob.glob("results/self_consistency/variance/*.json")):
    data = json.load(open(f))
    print(f"{f}: EM={data['summary']['em_voted']:.3f}")
```

---

## Interactive Demo

A Gradio application lets you visually explore the results:

```bash
cd demo
pip install -r requirements.txt
python app.py
```

Open http://localhost:7860 in a browser.

---

## Key Results

### Cascade pipeline V5b (phi4 → gpt-oss, SC k=5)

| Metric | Value | 95% CI |
|--------|-------|--------|
| Exact Match | 0.552 | [0.508; 0.594] |
| Rerouted questions | 45.4% | -- |

### Self-consistency V5a (phi4, SC k=3)

| Metric | Value | 95% CI |
|--------|-------|--------|
| Exact Match | 0.482 | [0.440; 0.522] |

### Inter-run variance (R=5, N=181 hard questions)

| Model | σ(EM) |
|-------|-------|
| phi4:latest | ≤ 0.032 |
| gpt-oss:20b | ≤ 0.032 |
| phi4-reasoning:plus | ≤ 0.032 |

### Extraction (Re-DocRED, QLoRA 4-bit)

| n_train | F1 |
|--------:|---:|
| 0 (zero-shot Gemma-4) | 0.702 |
| 3000 | 0.794 |
| DREEAM (SOTA) | 0.802 |

---

## License & Citation

### Source code

Distributed under **Hippocratic License 3.0** (HL3).
See [LICENSE.md](LICENSE.md) for the full text.

### Article

The article (`article/jourlin_2026a_fr.pdf`) is distributed under
**Creative Commons CC BY-NC-SA 4.0**.
English version on ArXiv: [arXiv:2604.11104](https://arxiv.org/abs/2604.11104).

### Citation

```bibtex
@article{jourlin2026synsynth,
  title   = {Frugal Knowledge Graph Construction with Local {LLMs}:
             A Zero-Shot Pipeline, Self-Consistency and Wisdom
             of Artificial Crowds},
  author  = {Jourlin, Pierre},
  year    = {2026},
  eprint  = {2604.11104},
  archiveprefix = {arXiv},
  primaryclass  = {cs.AI},
  url     = {https://arxiv.org/abs/2604.11104}
}
```

---

## Contact

**Pierre Jourlin** — Laboratoire Informatique d'Avignon (LIA), Avignon Université
# SYNSYNTH+ — Package de Reproductibilité

**Architecture d'IA frugale pour la synthèse et le raisonnement sur graphes de connaissances : de l'extraction relationnelle au RAG structuré.**

> Jourlin, P. (2026). *SYNSYNTH+ : Vers une architecture d'IA frugale pour la synthèse et le raisonnement sur graphes de connaissances.* Avignon Université, LIA.

Ce dépôt contient les scripts, données, résultats pré-calculés et la démo interactive
permettant de **reproduire intégralement** les expériences décrites dans l'article.

---

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Structure du dépôt](#structure-du-dépôt)
3. [Prérequis matériels et logiciels](#prérequis-matériels-et-logiciels)
4. [Installation](#installation)
5. [Reproduction des expériences](#reproduction-des-expériences)
6. [Exploration des résultats pré-calculés](#exploration-des-résultats-pré-calculés)
7. [Démo interactive](#démo-interactive)
8. [Résultats clés](#résultats-clés)
9. [Licence et citation](#licence-et-citation)

---

## Vue d'ensemble

SYNSYNTH+ est un pipeline complet qui transforme du texte non structuré en un graphe
de connaissances interrogeable, en utilisant des modèles légers déployés localement
sur un seul GPU grand public. Le projet évalue quatre axes :

| Axe | Benchmark | Modèle principal |
|-----|-----------|-----------------|
| Extraction de relations | Re-DocRED (N=500) | gemma4:26b (MoE, 4B actifs) |
| Text-to-Cypher | WebQuestionsSP | qwen3-deep:latest |
| Raisonnement multi-hop | HotpotQA (N=500) | phi4:latest (14B) |
| RAG Faithfulness | RAGAS | mistral-small:latest |

Le pipeline inclut également des analyses de **self-consistency** (SC),
un **pipeline cascade** (phi4 → gpt-oss), des **courbes d'apprentissage QLoRA**,
et des études de **calibration** et de **variance**.

---

## Structure du dépôt

```
synsynth/
├── README.md                  # Ce document
├── LICENSE.md                 # Hippocratic License 3.0
├── requirements.txt           # Dépendances Python (109 packages)
├── scripts/
│   ├── run_synsynth.py        # Point d'entrée CLI principal
│   ├── synsynth_config.py     # Configuration centralisée (auto-détecte les chemins)
│   ├── synsynth_model.py      # Interface Ollama API
│   ├── synsynth_data.py       # Chargeurs de datasets (HotpotQA, Re-DocRED, etc.)
│   ├── synsynth_io.py         # I/O sandboxées
│   ├── synsynth_stats.py      # Bootstrap CI, token F1
│   ├── synsynth_checkpoint.py # Gestion des checkpoints
│   ├── synsynth_viz.py        # Visualisations matplotlib
│   ├── synsynth_article.py    # Générateur d'article markdown
│   ├── exp_extraction.py      # Exp 1 — Extraction de relations
│   ├── exp_query.py           # Exp 2 — Text-to-Graph Query
│   ├── exp_multihop.py        # Exp 3 — Raisonnement multi-hop
│   ├── exp_rag.py             # Exp 4 — RAG Faithfulness
│   ├── self_consistency.py    # Self-consistency (SC) k passées
│   ├── pipeline_self_consistency.py  # Cascade phi4→gpt-oss avec SC
│   ├── variance_self_consistency.py  # Variance inter-runs SC (R=5)
│   ├── variance_cascade.py    # Variance inter-runs cascade (R=3)
│   ├── benchmark_models.py    # Benchmark multi-modèles
│   ├── benchmark_model_selection.py  # Sélection de modèle par tâche
│   ├── benchmark_extraction.py       # Extraction cross-modèle
│   ├── extraction_cross_model.py     # Extraction comparée 8 modèles
│   ├── learning_curve.py      # Courbes d'apprentissage QLoRA
│   ├── qlora_finetune.py      # Fine-tuning QLoRA 4-bit
│   ├── faithfulness_checker.py       # Vérificateur NLI de fidélité
│   ├── calibration_analysis.py       # Analyse de calibration
│   ├── confusion_clustering.py       # Clustering matrice de confusion
│   └── rebuild_multihop_data.py      # Génération données multi-hop V2/V3/V4
├── data/                      # Datasets (téléchargés automatiquement)
│   ├── extraction.json        # Cache Re-DocRED
│   ├── multihop.json          # Cache HotpotQA
│   ├── query.json             # Questions WebQuestionsSP
│   └── rag.json               # Exemples RAG
├── results/                   # Résultats pré-calculés (JSON + PNG)
│   ├── all_results.json       # Résumé consolidé de toutes les expériences
│   ├── self_consistency/      # SC k=5 pour phi4, gpt-oss, phi4-reasoning
│   │   └── variance/          # Variance inter-runs R=5 (15 fichiers)
│   ├── pipeline_self_consistency/  # Cascade V5b + SC V5a
│   ├── extraction_benchmark/  # F1 extraction pour 8 modèles
│   ├── extraction_cross_model/# Détails cross-modèle
│   ├── model_selection/       # Sélection de modèle par tâche
│   ├── calibration/           # Diagrammes de calibration
│   ├── confusion_analysis/    # Heatmaps & dendrogrammes
│   └── learning_curve/        # Courbes QLoRA (JSON + PNG)
├── article/
│   └── jourlin_2026a_fr.pdf   # Article compilé (22 pages)
└── demo/                      # Application Gradio interactive
    ├── app.py                 # Serveur de démo
    ├── data/all_results.json  # Données pour la démo
    ├── requirements.txt       # Dépendances Gradio
    └── README.md              # Carte HuggingFace Spaces
```

---

## Prérequis matériels et logiciels

### Matériel

| Composant | Minimum | Recommandé (config. de référence) |
|-----------|---------|----------------------------------|
| GPU | NVIDIA 16 Go VRAM | NVIDIA RTX 3090 (24 Go) |
| RAM | 32 Go | 64 Go |
| Stockage | 50 Go | 100 Go (modèles quantifiés) |
| OS | Linux (Ubuntu 22.04+) | Ubuntu 24.04 |

### Logiciel

| Composant | Version |
|-----------|---------|
| Python | ≥ 3.10 |
| Ollama | ≥ 0.6 |
| CUDA | ≥ 12.x |
| Pilote NVIDIA | ≥ 550 |

---

## Installation

### 1. Installer Ollama et télécharger les modèles

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &   # Démarre le serveur si ce n'est pas déjà fait

# Modèles principaux utilisés dans l'article
ollama pull phi4:latest          # 14B, raisonnement multi-hop
ollama pull gpt-oss:20b          # MoE 20B, cascade
ollama pull phi4-reasoning:plus  # 14B, raisonnement chaîne de pensée
ollama pull gemma4:26b           # MoE 26B (4B actifs), extraction
```

### 2. Cloner le dépôt et installer les dépendances

```bash
git clone https://github.com/jourlin/synsynth.git
cd synsynth
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Reproduction des expériences

### Pipeline complet (4 axes)

```bash
cd scripts
python run_synsynth.py
```

Cela exécute séquentiellement les 4 expériences (extraction, query, multihop, RAG)
et génère les résultats dans `results/`.

### Expériences individuelles

```bash
# Extraction de relations (Re-DocRED, N=50)
python scripts/exp_extraction.py

# Text-to-Cypher (WebQuestionsSP, N=20)
python scripts/exp_query.py

# Raisonnement multi-hop (HotpotQA, N=500)
python scripts/exp_multihop.py

# RAG Faithfulness (RAGAS, N=20)
python scripts/exp_rag.py
```

### Self-consistency et cascade

```bash
# SC k=5 avec phi4:latest
python scripts/self_consistency.py

# Pipeline cascade phi4→gpt-oss k=5
python scripts/pipeline_self_consistency.py

# Variance inter-runs SC (R=5, ≈20h GPU)
python scripts/variance_self_consistency.py

# Variance inter-runs cascade (R=3, ≈15h GPU)
python scripts/variance_cascade.py
```

### Analyses additionnelles

```bash
# Benchmark multi-modèles
python scripts/benchmark_models.py

# Extraction comparée sur 8 modèles
python scripts/extraction_cross_model.py

# Courbes d'apprentissage QLoRA
python scripts/learning_curve.py

# Calibration de la confiance
python scripts/calibration_analysis.py

# Clustering de la matrice de confusion
python scripts/confusion_clustering.py
```

---

## Exploration des résultats pré-calculés

Tous les résultats sont fournis sous forme de fichiers JSON dans `results/`.
Vous pouvez les explorer sans GPU :

```python
import json

# Résumé consolidé
with open("results/all_results.json") as f:
    results = json.load(f)

# Self-consistency phi4 k=5
with open("results/self_consistency/sc_phi4_latest_k5.json") as f:
    sc = json.load(f)
    print(f"EM voté: {sc['summary']['em_voted']}")

# Variance inter-runs (15 fichiers, R=5 × 3 modèles)
import glob
for f in sorted(glob.glob("results/self_consistency/variance/*.json")):
    data = json.load(open(f))
    print(f"{f}: EM={data['summary']['em_voted']:.3f}")
```

---

## Démo interactive

Une application Gradio permet d'explorer visuellement les résultats :

```bash
cd demo
pip install -r requirements.txt
python app.py
```

Ouvrir http://localhost:7860 dans un navigateur.

---

## Résultats clés

### Pipeline cascade V5b (phi4 → gpt-oss, SC k=5)

| Métrique | Valeur | IC 95% |
|----------|--------|--------|
| Exact Match | 0.552 | [0.508 ; 0.594] |
| Questions reroutées | 45.4% | — |

### Self-consistency V5a (phi4, SC k=3)

| Métrique | Valeur | IC 95% |
|----------|--------|--------|
| Exact Match | 0.482 | [0.440 ; 0.522] |

### Variance inter-runs (R=5, N=181 questions difficiles)

| Modèle | σ(EM) |
|--------|-------|
| phi4:latest | ≤ 0.032 |
| gpt-oss:20b | ≤ 0.032 |
| phi4-reasoning:plus | ≤ 0.032 |

### Extraction (Re-DocRED, QLoRA 4-bit)

| n_train | F1 |
|--------:|---:|
| 0 (zero-shot Gemma-4) | 0.702 |
| 3000 | 0.794 |
| DREEAM (SOTA) | 0.802 |

---

## Licence et citation

### Code source

Distribué sous **Hippocratic License 3.0** (HL3).
Voir [LICENSE.md](LICENSE.md) pour le texte complet.

### Article

L'article (`article/jourlin_2026a_fr.pdf`) est diffusé sous
**Creative Commons CC BY-NC-SA 4.0**.

### Citation

```bibtex
@article{jourlin2026synsynth,
  title   = {SYNSYNTH+ : Vers une architecture d'IA frugale pour la synthèse
             et le raisonnement sur graphes de connaissances},
  author  = {Jourlin, Pierre},
  year    = {2026},
  institution = {Avignon Université, Laboratoire d'Informatique d'Avignon}
}
```

---

## Contact

**Pierre Jourlin** — Laboratoire Informatique d'Avignon (LIA), Avignon Université
