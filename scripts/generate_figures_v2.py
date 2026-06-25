#!/usr/bin/env python3
"""
Generate thesis-ready comparison figures for the FOUR encoding strategies:

    1. Image Baseline     (CLIP vision encoder on rendered tables)
    2. Text Baseline      (BGE embeddings on linearised markdown)
    3. Graph-Augmented    (GraphSAGE on cell/header graph)
    4. Hybrid + Reranker  (text+graph fusion + MiniLM cross-encoder)

This script adds the image baseline row required by the thesis novelty
claim ("compare three encoding strategies: OCR/string vs image vs graph")
to the existing three-pipeline comparison, making the comparison table
complete.

Outputs (in docs/figures/):
    retrieval_comparison_4way.{png,pdf}
    scaling_effect_4way.{png,pdf}
    training_curve.{png,pdf}            (unchanged from v1 — kept for convenience)
    + stdout: a 4-row LaTeX table and plain-text summary

Usage:
    python scripts/generate_figures_v2.py
"""

import io
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("docs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Result file locations — 4 pipelines × 2 scales
PATHS = {
    # n=100
    "image_100":  "data/results/image_wikitq_100/retrieval_metrics.json",
    "text_100":   "data/results/text_wikitq/retrieval_metrics.json",
    "graph_100":  "data/results/graph_wikitq/retrieval_metrics.json",
    "hybrid_100": "data/results/hybrid_wikitq/retrieval_metrics.json",
    # n=500
    "image_500":  "data/results/image_wikitq_500/retrieval_metrics.json",
    "text_500":   "data/results/text_wikitq_500/retrieval_metrics.json",
    "graph_500":  "data/results/graph_500_newmodel/retrieval_metrics.json",
    "hybrid_500": "data/results/hybrid_rerank_500/retrieval_metrics.json",
}

# Colour per pipeline (consistent across all figures)
COLORS = {
    "image":  "#8172B3",   # purple  (visual encoding)
    "text":   "#4C72B0",   # blue    (OCR/string)
    "graph":  "#DD8452",   # orange  (structural)
    "hybrid": "#55A868",   # green   (our best)
}

LABELS = {
    "image":  "Image (CLIP)",
    "text":   "Text Baseline",
    "graph":  "Graph-Augmented",
    "hybrid": "Hybrid + Reranker",
}


def load_metrics(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ────────────────────────────────────────────────
#  Figure 1 — 4-way bar chart at n=100 & n=500
# ────────────────────────────────────────────────

def _bar_panel_4way(ax, metrics_dict, title):
    """metrics_dict: {pipeline_key: metrics_json}."""
    metrics = ["recall_at_1", "recall_at_5", "mrr"]
    labels = ["R@1", "R@5", "MRR"]
    x = np.arange(len(labels))
    w = 0.2

    keys = ["image", "text", "graph", "hybrid"]
    offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]

    bars_all = []
    for key, off in zip(keys, offsets):
        vals = [metrics_dict[key][m] for m in metrics]
        bars = ax.bar(
            x + off, vals, w,
            label=LABELS[key], color=COLORS[key], edgecolor="white",
        )
        bars_all.append(bars)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.3)

    for bars in bars_all:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.012,
                f"{h:.3f}",
                ha="center", va="bottom", fontsize=7.5,
            )


def fig1_retrieval_comparison_4way():
    m100 = {k: load_metrics(PATHS[f"{k}_100"]) for k in ("image", "text", "graph", "hybrid")}
    m500 = {k: load_metrics(PATHS[f"{k}_500"]) for k in ("image", "text", "graph", "hybrid")}

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8))
    _bar_panel_4way(axes[0], m100, "(a) Index Size = 100 Tables")
    _bar_panel_4way(axes[1], m500, "(b) Index Size = 500 Tables")

    fig.suptitle(
        "Retrieval Performance Across Four Encoding Strategies\n"
        "WikiTableQuestions Test Set — image / text / graph / hybrid+reranker",
        fontsize=14, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
    plt.savefig(OUT_DIR / "retrieval_comparison_4way.png", dpi=200, bbox_inches="tight")
    plt.savefig(OUT_DIR / "retrieval_comparison_4way.pdf", bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'retrieval_comparison_4way.png'}")


# ────────────────────────────────────────────────
#  Figure 2 — Scaling effect, 4 lines
# ────────────────────────────────────────────────

def fig2_scaling_effect_4way():
    sizes = [100, 500]
    data = {
        k: (
            load_metrics(PATHS[f"{k}_100"]),
            load_metrics(PATHS[f"{k}_500"]),
        )
        for k in ("image", "text", "graph", "hybrid")
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    markers = {"image": "^", "text": "o", "graph": "s", "hybrid": "D"}
    linewidths = {"image": 1.8, "text": 2.0, "graph": 2.0, "hybrid": 2.5}

    for key in ("image", "text", "graph", "hybrid"):
        r1 = [data[key][0]["recall_at_1"], data[key][1]["recall_at_1"]]
        r5 = [data[key][0]["recall_at_5"], data[key][1]["recall_at_5"]]
        ax1.plot(sizes, r1, marker=markers[key], color=COLORS[key],
                 label=LABELS[key], linewidth=linewidths[key], markersize=9)
        ax2.plot(sizes, r5, marker=markers[key], color=COLORS[key],
                 label=LABELS[key], linewidth=linewidths[key], markersize=9)

    for ax, ylabel, title, ylim in [
        (ax1, "Recall@1", "(a) R@1 vs Index Size", (0, 0.7)),
        (ax2, "Recall@5", "(b) R@5 vs Index Size", (0, 0.95)),
    ]:
        ax.set_xlabel("Index Size (# tables)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sizes)
        ax.set_ylim(*ylim)

    fig.suptitle(
        "Effect of Index Size — All Four Encoding Strategies\nWikiTableQuestions Test Set",
        fontsize=14, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
    plt.savefig(OUT_DIR / "scaling_effect_4way.png", dpi=200, bbox_inches="tight")
    plt.savefig(OUT_DIR / "scaling_effect_4way.pdf", bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'scaling_effect_4way.png'}")


# ────────────────────────────────────────────────
#  LaTeX + text summary (4 pipelines)
# ────────────────────────────────────────────────

def print_latex_table_4way():
    m100 = {k: load_metrics(PATHS[f"{k}_100"]) for k in ("image", "text", "graph", "hybrid")}
    m500 = {k: load_metrics(PATHS[f"{k}_500"]) for k in ("image", "text", "graph", "hybrid")}

    print("\n" + "=" * 74)
    print("LaTeX Table (copy into thesis):")
    print("=" * 74)
    print(r"""
\begin{table}[h]
\centering
\caption{Retrieval performance across four encoding strategies on
WikiTableQuestions test set. The three baselines (image, text, graph)
are evaluated independently; Hybrid + Reranker fuses text + graph scores
via $\alpha=0.7$ linear combination of min-max normalised scores and
re-ranks the top-15 with a MiniLM cross-encoder.}
\label{tab:retrieval-results-4way}
\begin{tabular}{llccc}
\toprule
\textbf{Index Size} & \textbf{Pipeline} & \textbf{R@1} & \textbf{R@5} & \textbf{MRR} \\
\midrule""")
    for n, d in [(100, m100), (500, m500)]:
        print(f"\\multirow{{4}}{{*}}{{{n}}} & Image Baseline (CLIP) & "
              f"{d['image']['recall_at_1']:.3f} & "
              f"{d['image']['recall_at_5']:.3f} & "
              f"{d['image']['mrr']:.3f} \\\\")
        print(f"                          & Text Baseline (BGE) & "
              f"{d['text']['recall_at_1']:.3f} & "
              f"{d['text']['recall_at_5']:.3f} & "
              f"{d['text']['mrr']:.3f} \\\\")
        print(f"                          & Graph-Augmented (GraphSAGE) & "
              f"{d['graph']['recall_at_1']:.3f} & "
              f"{d['graph']['recall_at_5']:.3f} & "
              f"{d['graph']['mrr']:.3f} \\\\")
        print(f"                          & \\textbf{{Hybrid + Reranker}} & "
              f"\\textbf{{{d['hybrid']['recall_at_1']:.3f}}} & "
              f"\\textbf{{{d['hybrid']['recall_at_5']:.3f}}} & "
              f"\\textbf{{{d['hybrid']['mrr']:.3f}}} \\\\")
        if n == 100:
            print(r"\midrule")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")


def print_summary_4way():
    m100 = {k: load_metrics(PATHS[f"{k}_100"]) for k in ("image", "text", "graph", "hybrid")}
    m500 = {k: load_metrics(PATHS[f"{k}_500"]) for k in ("image", "text", "graph", "hybrid")}

    print("\n" + "=" * 78)
    print(f"{'RETRIEVAL RESULTS — FOUR ENCODING STRATEGIES':^78}")
    print("=" * 78)
    header = f"{'n':>5} {'Pipeline':<24} {'R@1':>8} {'R@5':>8} {'MRR':>8}"
    print(header)
    print("-" * 78)
    for n, d in [(100, m100), (500, m500)]:
        for key in ("image", "text", "graph", "hybrid"):
            m = d[key]
            print(f"{n:>5} {LABELS[key]:<24} {m['recall_at_1']:>8.4f} "
                  f"{m['recall_at_5']:>8.4f} {m['mrr']:>8.4f}")
        # Relative improvement of hybrid vs each baseline
        print(f"{'':>5} {'  hybrid vs image':<24} "
              f"{_pct(d['hybrid']['recall_at_1'], d['image']['recall_at_1']):>8} "
              f"{_pct(d['hybrid']['recall_at_5'], d['image']['recall_at_5']):>8} "
              f"{_pct(d['hybrid']['mrr'], d['image']['mrr']):>8}")
        print(f"{'':>5} {'  hybrid vs text':<24} "
              f"{_pct(d['hybrid']['recall_at_1'], d['text']['recall_at_1']):>8} "
              f"{_pct(d['hybrid']['recall_at_5'], d['text']['recall_at_5']):>8} "
              f"{_pct(d['hybrid']['mrr'], d['text']['mrr']):>8}")
        print()


def _pct(new, base):
    if base == 0:
        return "n/a"
    return f"{(new - base) / base * 100:+.1f}%"


if __name__ == "__main__":
    print("Generating 4-way thesis figures...\n")
    fig1_retrieval_comparison_4way()
    fig2_scaling_effect_4way()
    print_latex_table_4way()
    print_summary_4way()
    print("\nAll figures saved to docs/figures/")
