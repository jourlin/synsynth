"""
SYNSYNTH+ — Démo Interactive
Architecture d'IA frugale pour la synthèse et le raisonnement sur graphes de connaissances.

Auteur : Pierre Jourlin — LIA, Avignon Université
Licence : Hippocratic License 3.0 (code) / CC BY-NC-SA 4.0 (article)
"""

import json
import math
import os

import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Chargement des résultats pré-calculés ────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load(name: str) -> dict:
    path = os.path.join(DATA_DIR, name)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


ALL = _load("all_results.json")
EXT = ALL["extraction"]
QRY = ALL["text_to_query"]
MHop = ALL["multihop_reasoning"]
RAG = ALL["rag_faithfulness"]

# ── Données écologiques (depuis l'article § 5.2) ────────────────────────

ECO_DATA = {
    "headers": [
        "Métrique", "SYNSYNTH+\n(Gemma-4 Q4_K_M)", "DeepSeek-V3\n(671B MoE)",
        "Llama-3 405B\n(dense)"
    ],
    "rows": [
        ["Paramètres totaux", "26 B", "671 B", "405 B"],
        ["Paramètres actifs/token", "4 B", "37 B", "405 B"],
        ["Architecture", "MoE 128→8", "MoE 256→8", "Dense"],
        ["Quantification", "Q4_K_M (4 bits)", "FP8", "FP16"],
        ["Infrastructure", "1× RTX 3090", "8× H100", "4× A100"],
        ["Puissance système (W)", "210", "4 000", "1 200"],
        ["Temps moyen/requête (s)", "35", "4", "15"],
        ["Énergie/requête (Wh)", "2,04", "4,44", "5,00"],
        ["FLOPs/requête", "1,6×10¹²", "1,5×10¹³", "1,6×10¹⁴"],
        ["CO₂/requête (gCO₂eq)", "0,114", "1,733", "1,950"],
        ["Mix électrique (gCO₂/kWh)", "56 (FR)", "390 (US/CN)", "390 (US)"],
        ["Facteur réduction FLOPs", "1×", "9×", "101×"],
        ["Facteur réduction CO₂", "1×", "15×", "17×"],
    ],
}

# ── Triplets d'exemple pour la visualisation du graphe ───────────────────

EXAMPLE_TRIPLETS = [
    ("2002 Winter Olympics", "start_time", "2002"),
    ("2002 Winter Olympics", "end_time", "2002"),
    ("2002 Winter Olympics", "location", "Salt Lake City"),
    ("2006 Winter Olympics", "location", "Turin"),
    ("Duff Gibson", "participant_in", "2006 Winter Olympics"),
    ("Duff Gibson", "citizenship", "Canadian"),
    ("Jeff Pain", "participant_in", "2006 Winter Olympics"),
    ("Jeff Pain", "citizenship", "Canadian"),
    ("Wilfried Schneider", "participant_in", "2002 Winter Olympics"),
    ("Wilfried Schneider", "nationality", "German"),
]

# ── Fonctions de visualisation ───────────────────────────────────────────


def build_results_figure():
    """Figure 4 sous-graphiques des résultats expérimentaux."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Exp. 1 : Extraction de Relations",
            "Exp. 2 : Text-to-Query (Cypher)",
            "Exp. 3 : Raisonnement Multi-hop",
            "Exp. 4 : Évaluation RAGAS",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    colors_ext = ["#4C72B0", "#55A868", "#C44E52"]
    # Exp 1 — Extraction
    ext_labels = ["Précision", "Rappel", "F1-Score"]
    ext_vals = [EXT["precision"], EXT["recall"], EXT["f1_score"]]
    fig.add_trace(
        go.Bar(x=ext_labels, y=ext_vals, marker_color=colors_ext,
               text=[f"{v:.0%}" for v in ext_vals], textposition="outside",
               showlegend=False),
        row=1, col=1,
    )
    fig.add_hline(y=0.85, line_dash="dash", line_color="red",
                  annotation_text="Cible F1=0.85", row=1, col=1)

    # Exp 2 — Query
    qry_labels = ["Accuracy", "Cypher valide"]
    qry_vals = [QRY["accuracy"], QRY["cypher_syntax_valid_rate"]]
    fig.add_trace(
        go.Bar(x=qry_labels, y=qry_vals, marker_color=["#4C72B0", "#55A868"],
               text=[f"{v:.0%}" for v in qry_vals], textposition="outside",
               showlegend=False),
        row=1, col=2,
    )
    fig.add_hline(y=0.90, line_dash="dash", line_color="red",
                  annotation_text="Cible=0.90", row=1, col=2)

    # Exp 3 — Multi-hop
    mh_labels = ["Exact Match", "Partiel"]
    mh_vals = [MHop["exact_accuracy"], MHop["partial_accuracy"]]
    fig.add_trace(
        go.Bar(x=mh_labels, y=mh_vals, marker_color=["#4C72B0", "#DD8452"],
               text=[f"{v:.1%}" for v in mh_vals], textposition="outside",
               showlegend=False),
        row=2, col=1,
    )

    # Exp 4 — RAGAS
    rag_labels = ["Fidélité", "Pertinence", "Préc. Contexte"]
    rag_vals = [RAG["avg_faithfulness"], RAG["avg_answer_relevance"],
                RAG["avg_context_precision"]]
    fig.add_trace(
        go.Bar(x=rag_labels, y=rag_vals,
               marker_color=["#C44E52", "#55A868", "#8172B3"],
               text=[f"{v:.2f}" for v in rag_vals], textposition="outside",
               showlegend=False),
        row=2, col=2,
    )
    fig.add_hline(y=0.95, line_dash="dash", line_color="red",
                  annotation_text="Cible ≈ 1.0", row=2, col=2)

    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_yaxes(range=[0, 1.12], row=i, col=j)

    fig.update_layout(
        title_text="SYNSYNTH+ — Résultats Expérimentaux",
        title_font_size=18,
        height=700,
        template="plotly_white",
    )
    return fig


def build_radar_figure():
    """Radar chart des 4 axes de performance."""
    categories = [
        "Extraction (F1)", "Query (Accuracy)",
        "Multi-hop (Exact)", "Faithfulness (RAGAS)"
    ]
    scores = [EXT["f1_score"], QRY["accuracy"],
              MHop["exact_accuracy"], RAG["avg_faithfulness"]]
    targets = [0.85, 0.90, 0.70, 1.00]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores + [scores[0]], theta=categories + [categories[0]],
        fill="toself", fillcolor="rgba(76,114,176,0.25)",
        line=dict(color="#4C72B0", width=2),
        name="Scores obtenus",
    ))
    fig.add_trace(go.Scatterpolar(
        r=targets + [targets[0]], theta=categories + [categories[0]],
        fill="none",
        line=dict(color="#C44E52", width=2, dash="dash"),
        name="Cibles SYNSYNTH+",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1.1])),
        title="SYNSYNTH+ — Profil de Performance",
        title_font_size=16,
        height=500,
        template="plotly_white",
    )
    return fig


def build_eco_figure():
    """Comparaison écologique interactive (barres groupées)."""
    models = ["SYNSYNTH+", "DeepSeek-V3", "Llama-3 405B"]
    energy = [2.04, 4.44, 5.00]
    co2 = [0.114, 1.733, 1.950]
    flops_log = [math.log10(1.6e12), math.log10(1.5e13), math.log10(1.6e14)]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Énergie / requête (Wh)", "CO₂ / requête (gCO₂eq)",
                        "FLOPs / requête (log₁₀)"],
        horizontal_spacing=0.08,
    )
    colors = ["#55A868", "#C44E52", "#DD8452"]

    fig.add_trace(
        go.Bar(x=models, y=energy, marker_color=colors,
               text=[f"{v:.2f}" for v in energy], textposition="outside",
               showlegend=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=models, y=co2, marker_color=colors,
               text=[f"{v:.3f}" for v in co2], textposition="outside",
               showlegend=False),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(x=models, y=flops_log, marker_color=colors,
               text=["1,6×10¹²", "1,5×10¹³", "1,6×10¹⁴"],
               textposition="outside", showlegend=False),
        row=1, col=3,
    )

    fig.update_layout(
        title_text="Impact Environnemental Comparé — IA Frugale vs Modèles Lourds",
        title_font_size=16,
        height=450,
        template="plotly_white",
    )
    return fig


def build_graph_figure():
    """Visualisation interactive du graphe de connaissances (sous-graphe)."""
    nodes = {}
    edges = []
    for head, rel, tail in EXAMPLE_TRIPLETS:
        if head not in nodes:
            nodes[head] = len(nodes)
        if tail not in nodes:
            nodes[tail] = len(nodes)
        edges.append((nodes[head], nodes[tail], rel))

    n = len(nodes)
    # Disposition circulaire
    angles = [2 * math.pi * i / n for i in range(n)]
    x_pos = [2.0 * math.cos(a) for a in angles]
    y_pos = [2.0 * math.sin(a) for a in angles]

    names = list(nodes.keys())

    # Arêtes
    edge_x, edge_y = [], []
    annotations = []
    for src, dst, rel in edges:
        edge_x += [x_pos[src], x_pos[dst], None]
        edge_y += [y_pos[src], y_pos[dst], None]
        mx = (x_pos[src] + x_pos[dst]) / 2
        my = (y_pos[src] + y_pos[dst]) / 2
        annotations.append(dict(
            x=mx, y=my, text=f"<i>{rel}</i>", showarrow=False,
            font=dict(size=9, color="#666"),
            bgcolor="rgba(255,255,255,0.8)",
        ))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1, color="#aaa"), hoverinfo="none",
    ))
    # Distinction entités vs valeurs
    entity_color = ["#4C72B0" if any(
        h == name or (t == name and not name.replace(" ", "").replace("-", "").isalnum() is False)
        for h, _, t in EXAMPLE_TRIPLETS if h == name
    ) else "#DD8452" for name in names]

    fig.add_trace(go.Scatter(
        x=x_pos, y=y_pos, mode="markers+text",
        marker=dict(size=20, color="#4C72B0", line=dict(width=2, color="white")),
        text=names, textposition="top center",
        textfont=dict(size=10),
        hovertext=[f"<b>{name}</b>" for name in names],
        hoverinfo="text",
    ))

    fig.update_layout(
        title="Sous-graphe de connaissances — Re-DocRED (Skeleton Racing)",
        title_font_size=14,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        annotations=annotations,
        height=550,
        template="plotly_white",
    )
    return fig


# ── Exemples Text-to-Query ──────────────────────────────────────────────

QUERY_EXAMPLES = QRY.get("details", [])


def format_query_example(idx):
    """Formate un exemple de requête pour l'affichage."""
    if idx < 0 or idx >= len(QUERY_EXAMPLES):
        return "Index invalide", "", "", ""
    ex = QUERY_EXAMPLES[idx]
    status = "✅ Correct" if ex["answer_correct"] else "❌ Incorrect"
    cypher_status = "✅ Valide" if ex["cypher_valid"] else "❌ Invalide"
    return (
        ex["question"],
        ex["pred_answer"],
        f"**Réponse attendue :** {ex['gold_answer']}\n\n"
        f"**Statut :** {status} | **Cypher :** {cypher_status}",
    )


# ── Exemples Multi-hop ──────────────────────────────────────────────────

MULTIHOP_EXAMPLES = MHop.get("details", [])[:20]  # 20 premiers


def format_multihop_example(idx):
    """Formate un exemple multi-hop."""
    if idx < 0 or idx >= len(MULTIHOP_EXAMPLES):
        return "", "", ""
    ex = MULTIHOP_EXAMPLES[idx]
    em = "✅" if ex.get("exact_match") else "❌"
    pm = "✅" if ex.get("partial_match") else "❌"
    chain = ex.get("chain_length", "?")
    return (
        ex["question"],
        ex["pred_answer"],
        f"**Réponse attendue :** {ex['gold_answer']}\n\n"
        f"**Exact Match :** {em} | **Partiel :** {pm} | "
        f"**Chaîne :** {chain} hops",
    )


# ── Exemples RAG ────────────────────────────────────────────────────────

RAG_EXAMPLES = RAG.get("details", [])


def format_rag_example(idx):
    """Formate un exemple RAG/RAGAS."""
    if idx < 0 or idx >= len(RAG_EXAMPLES):
        return "", "", ""
    ex = RAG_EXAMPLES[idx]
    return (
        ex["question"],
        ex["generated_answer"],
        f"**Fidélité :** {ex['faithfulness']:.2f} | "
        f"**Pertinence :** {ex['relevance']:.2f} | "
        f"**Préc. contexte :** {ex['context_precision']:.2f}",
    )


# ── Texte d'accueil ─────────────────────────────────────────────────────

WELCOME_MD = """
# SYNSYNTH+ : IA Frugale pour Graphes de Connaissances

**SYNSYNTH+** est une architecture d'IA frugale pour la synthèse et le raisonnement
sur graphes de connaissances, de l'extraction relationnelle au RAG structuré.

### Caractéristiques principales

| | |
|---|---|
| **Modèle** | Gemma-4-26B-A4B-it (MoE 128→8, 4B params actifs/token) |
| **Quantification** | Q4_K_M (4 bits) via Unsloth/GGUF |
| **Infrastructure** | 1× RTX 3090 (24 Go VRAM) |
| **Énergie** | ~2,04 Wh/requête — **15× moins de CO₂** que DeepSeek-V3 |

### 4 axes d'évaluation

1. **Extraction de relations** (DocRED/TACRED) → F1 = 1.00
2. **Text-to-Cypher** (WebQuestionsSP) → Accuracy = 0.80, Cypher valide = 1.00
3. **Raisonnement multi-hop** (HotpotQA) → EM = 0.40, PA = 0.515
4. **Évaluation RAGAS** → Fidélité = 0.50, Pertinence = 1.00, Préc. contexte = 1.00

---

*Pierre Jourlin — Laboratoire Informatique d'Avignon (LIA), Avignon Université*

📄 [Article complet](https://github.com/jourlin/SYNSYNTHplus) &nbsp;|&nbsp;
📦 [Code source (GitHub)](https://github.com/jourlin/SYNSYNTHplus)
"""

ECO_MD = """
### Ratio d'efficacité frugale

$$\\mathcal{R} = \\frac{\\Delta \\text{Accuracy}}{\\Delta \\text{Energy}} = \\frac{0{,}10}{2{,}40 \\text{ Wh}} \\approx 0{,}04 \\text{ points/Wh}$$

Chaque watt-heure supplémentaire dépensé par DeepSeek-V3 ne procure qu'un gain
marginal de **0,04 points** d'accuracy. SYNSYNTH+ accepte une perte de 10%
d'accuracy en échange d'une réduction de **9× des FLOPs** et **15× des émissions de CO₂**.

Pour un pipeline de **N = 1 000 requêtes**, l'économie cumulée atteint
**2,40 kWh** et **1,62 kg CO₂eq** par rapport à DeepSeek-V3.

*Sources : mesures nvidia-smi (RTX 3090), DeepSeek-V3 Technical Report
(arXiv:2412.19437), ADEME/RTE 2024 (56 gCO₂/kWh mix FR).*
"""

# ── Construction de l'interface Gradio ───────────────────────────────────

with gr.Blocks(
    title="SYNSYNTH+ — Démo Interactive",
) as demo:

    gr.Markdown(WELCOME_MD)

    with gr.Tabs():
        # ── Onglet 1 : Résultats ───────────────────────────────────
        with gr.Tab("📊 Résultats Expérimentaux"):
            gr.Markdown("### Vue d'ensemble des 4 expériences")
            gr.Plot(value=build_results_figure())
            gr.Markdown("### Profil radar")
            gr.Plot(value=build_radar_figure())

        # ── Onglet 2 : Graphe de connaissances ─────────────────────
        with gr.Tab("🔗 Graphe de Connaissances"):
            gr.Markdown(
                "### Sous-graphe extrait (Re-DocRED — Skeleton Racing)\n"
                "Visualisation des triplets *(sujet, relation, objet)* "
                "extraits automatiquement par SYNSYNTH+ à partir du corpus "
                "Re-DocRED."
            )
            gr.Plot(value=build_graph_figure())
            gr.Markdown("#### Triplets extraits")
            triplet_data = [[h, r, t] for h, r, t in EXAMPLE_TRIPLETS]
            gr.Dataframe(
                value=triplet_data,
                headers=["Sujet (head)", "Relation", "Objet (tail)"],
                interactive=False,
            )

        # ── Onglet 3 : Exemples de requêtes ────────────────────────
        with gr.Tab("🔍 Exemples Text-to-Query"):
            gr.Markdown(
                "### Traduction langage naturel → Cypher\n"
                "Sélectionnez un exemple pour voir la question, la réponse "
                "générée par SYNSYNTH+ et le verdict."
            )
            slider_q = gr.Slider(
                minimum=0, maximum=len(QUERY_EXAMPLES) - 1,
                step=1, value=0, label="Exemple n°",
            )
            with gr.Row():
                q_question = gr.Textbox(label="Question", interactive=False)
                q_answer = gr.Textbox(label="Réponse SYNSYNTH+",
                                      interactive=False)
            q_verdict = gr.Markdown()
            slider_q.change(
                fn=format_query_example, inputs=[slider_q],
                outputs=[q_question, q_answer, q_verdict],
            )
            demo.load(
                fn=format_query_example, inputs=[slider_q],
                outputs=[q_question, q_answer, q_verdict],
            )

        # ── Onglet 4 : Exemples Multi-hop ──────────────────────────
        with gr.Tab("🧠 Exemples Multi-hop"):
            gr.Markdown(
                "### Raisonnement multi-hop (HotpotQA)\n"
                "Le système enchaîne plusieurs sauts dans le graphe pour "
                "répondre à des questions complexes."
            )
            slider_mh = gr.Slider(
                minimum=0, maximum=len(MULTIHOP_EXAMPLES) - 1,
                step=1, value=0, label="Exemple n°",
            )
            with gr.Row():
                mh_question = gr.Textbox(label="Question", interactive=False)
                mh_answer = gr.Textbox(label="Réponse SYNSYNTH+",
                                       interactive=False)
            mh_verdict = gr.Markdown()
            slider_mh.change(
                fn=format_multihop_example, inputs=[slider_mh],
                outputs=[mh_question, mh_answer, mh_verdict],
            )
            demo.load(
                fn=format_multihop_example, inputs=[slider_mh],
                outputs=[mh_question, mh_answer, mh_verdict],
            )

        # ── Onglet 5 : Exemples RAG ────────────────────────────────
        with gr.Tab("💬 Exemples RAG (RAGAS)"):
            gr.Markdown(
                "### Évaluation de la fidélité (framework RAGAS)\n"
                "Analyse de la génération conversationnelle : le système "
                "est-il fidèle au contexte du graphe ?"
            )
            if RAG_EXAMPLES:
                slider_rag = gr.Slider(
                    minimum=0, maximum=len(RAG_EXAMPLES) - 1,
                    step=1, value=0, label="Exemple n°",
                )
                with gr.Row():
                    rag_question = gr.Textbox(label="Question",
                                              interactive=False)
                    rag_answer = gr.Textbox(label="Réponse générée",
                                            interactive=False)
                rag_verdict = gr.Markdown()
                slider_rag.change(
                    fn=format_rag_example, inputs=[slider_rag],
                    outputs=[rag_question, rag_answer, rag_verdict],
                )
                demo.load(
                    fn=format_rag_example, inputs=[slider_rag],
                    outputs=[rag_question, rag_answer, rag_verdict],
                )
            else:
                gr.Markdown("*Aucun exemple RAG disponible.*")

        # ── Onglet 6 : Impact écologique ────────────────────────────
        with gr.Tab("🌱 Impact Écologique"):
            gr.Markdown(
                "### Comparaison de l'empreinte environnementale\n"
                "SYNSYNTH+ (Gemma-4, 1× RTX 3090) vs modèles lourds "
                "(DeepSeek-V3, Llama-3 405B)."
            )
            gr.Plot(value=build_eco_figure())
            gr.Markdown(ECO_MD)
            gr.Markdown("#### Tableau comparatif détaillé")
            gr.Dataframe(
                value=ECO_DATA["rows"],
                headers=ECO_DATA["headers"],
                interactive=False,
            )

    gr.Markdown(
        "---\n"
        "*SYNSYNTH+ — Pierre Jourlin, LIA, Avignon Université — 2025* &nbsp;|&nbsp; "
        "Code : [Hippocratic License 3.0](https://firstdonoharm.dev/) &nbsp;|&nbsp; "
        "Article : [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)"
    )


if __name__ == "__main__":
    demo.launch()
