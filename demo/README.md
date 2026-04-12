---
title: "SYNSYNTH+ — IA Frugale pour Graphes de Connaissances"
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: other
---

# SYNSYNTH+ — Démo Interactive

**Architecture d'IA frugale pour la synthèse et le raisonnement sur graphes de connaissances : de l'extraction relationnelle au RAG structuré.**

## Description

SYNSYNTH+ est un pipeline complet qui transforme du texte non structuré en un graphe de connaissances interrogeable, en utilisant un modèle léger (Gemma-4-26B-A4B, 4B paramètres actifs/token) déployé sur un seul GPU grand public (RTX 3090).

Cette démo présente les résultats pré-calculés des 4 axes d'évaluation :

| Expérience | Benchmark | Résultat clé |
|---|---|---|
| Extraction de relations | Re-DocRED | F1 = 0.794 (QLoRA 3000 ex.), zero-shot = 0.702 |
| Text-to-Cypher | WebQuestionsSP | Accuracy = 0.80, Cypher valide = 1.00 |
| Raisonnement multi-hop | HotpotQA | EM = 0.406 (QLoRA V4), zero-shot Phi-4 = 0.462 |
| Évaluation RAGAS | RAGAS | Fidélité = 0.50, Pertinence = 1.00 |

## Impact écologique

SYNSYNTH+ consomme **15× moins de CO₂** par requête que DeepSeek-V3 et **17× moins** que Llama-3 405B, grâce à :
- Une architecture MoE (128→8 experts, 4B params actifs)
- Une quantification Q4_K_M (4 bits)
- Un déploiement sur infrastructure locale française (56 gCO₂/kWh)

## Auteur

**Pierre Jourlin** — Laboratoire Informatique d'Avignon (LIA), Avignon Université

## Licences

- Code : [Hippocratic License 3.0](https://firstdonoharm.dev/)
- Article : [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Liens

- 📦 [Code source (GitHub)](https://github.com/jourlin/synsynth)
