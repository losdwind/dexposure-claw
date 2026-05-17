#!/usr/bin/env python3
"""
Figure 3: DeXposure-Bench -- A six-axis evaluation suite (cover figure)

Two visual zones:
  TOP    : six benchmark cards arranged 3×2; each card shows
           id · capability · primary metric · which framework layer
           it stresses (mapped to Fig 1's L1..L4).
  BOTTOM : the regulator-aligned ground-truth equation as a single
           anchor (the methodological contribution).
  LEFT   : positioning panel — DeXposure-Bench is the only suite
           covering Pred + Anom + Calib + Scenario + Decision.

Style continuity with Fig 1: cream paper, Economist red, serif title.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.10,
})

# ── Palette ─────────────────────────────────────────────
PAPER     = "#FAF6EC"
INK       = "#1A1A1A"
INK_2     = "#2A2A2A"
MUTE      = "#67635A"
DIM       = "#9A9588"
RULE      = "#D8D2C0"
RED       = "#C8102E"
RED_DK    = "#9A0C23"
BLUE      = "#2E5077"
GREEN     = "#4A7C3E"
AMBER     = "#C5A55A"
HIGHLIGHT = "#FFF4D1"
PANEL_BG  = "#FFFFFF"

# Layer colours match Fig 1
L1_C, L2_C, L3_C, L4_C = RED, BLUE, GREEN, AMBER

# ── Canvas ─────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(14, 9.5))
fig.patch.set_facecolor(PAPER)
ax.set_facecolor(PAPER)
ax.set_xlim(0, 14)
ax.set_ylim(0, 9.5)
ax.axis("off")

# ── Masthead ────────────────────────────────────────────
ax.plot([0.4, 13.6], [9.30, 9.30], color=RED, lw=2.4)
ax.text(0.4, 9.42, "DeXposure-Bench",
        fontsize=18, fontweight="bold", color=INK,
        ha="left", va="bottom", fontstyle="italic", family="serif")
ax.text(13.6, 9.42,
        "A standardised six-axis evaluation suite for FM+LLM decision pipelines",
        fontsize=10, color=MUTE, ha="right", va="bottom", fontstyle="italic")

# ── Subtitle band ───────────────────────────────────────
ax.text(0.4, 8.95,
        "CONTRIBUTION II",
        fontsize=10, fontweight="bold", color=RED_DK, ha="left", va="center",
        family="sans-serif")
ax.text(2.55, 8.95,
        "Six independently reportable benchmarks, each mapped to the framework layer it stresses.",
        fontsize=10, color=INK_2, ha="left", va="center",
        fontstyle="italic", family="serif")

# ────────────────────────────────────────────────────────
# SIX BENCHMARK CARDS (3 columns × 2 rows)
# ────────────────────────────────────────────────────────
BENCHES = [
    dict(
        bid="b1_forecast",
        name="Temporal Graph Prediction",
        question=r"Does $\hat G_{t+h}$ track the true future?",
        metrics="PageRank MAE  ·  HHI MAE  ·  Gini MAE\nrank correlation  ·  trend consistency",
        layer="L1",
        layer_c=L1_C,
        n_methods="4 methods × 4 horizons",
    ),
    dict(
        bid="b2_warning",
        name="Streaming Anomaly Detection",
        question="Does the monitor fire ahead of crises?",
        metrics="precision  ·  recall  ·  F1\nlead time (weeks)  ·  alert stability",
        layer="L2",
        layer_c=L2_C,
        n_methods="3 events × 3 alert budgets",
    ),
    dict(
        bid="b3_calibration",
        name="Predictive Uncertainty Quality",
        question="Is the 90\\% PI actually 90\\%?",
        metrics="PI coverage @ 0.90  ·  PI width\nECE  ·  CRPS",
        layer="L1",
        layer_c=L1_C,
        n_methods="2 FM methods · 4 horizons",
    ),
    dict(
        bid="b4_stress",
        name="What-if Scenario Fidelity",
        question="Do scenarios on $\\hat G$ match those on $G_{t+h}$?",
        metrics="loss MAE  ·  distressed-count MAE\npropagation depth  ·  target overlap@$k$",
        layer="L2",
        layer_c=L2_C,
        n_methods="4 methods × 5 scenarios",
    ),
    dict(
        bid="b5_decision",
        name="Supervisory Ticket Quality",
        question="Do tickets identify systemically risky protocols?",
        metrics="precision  ·  recall@$k$  ·  F1\nstability  ·  judge  ·  FIR",
        layer="L3+L4",
        layer_c=GREEN,
        n_methods="5 methods · regulator-aligned",
    ),
    dict(
        bid="b6_robustness",
        name="Data-Quality Sensitivity",
        question="How fast does quality fall under degradation?",
        metrics="per-benchmark metrics under\n5 degradation regimes  ·  rel.~degrad.",
        layer="L1",
        layer_c=L1_C,
        n_methods="4 methods × 5 regimes",
    ),
]

# Grid geometry
GRID_X0 = 0.40
GRID_Y0 = 3.80
GRID_X1 = 13.60
GRID_Y1 = 8.55
COLS, ROWS = 3, 2
CARD_GAP_X = 0.22
CARD_GAP_Y = 0.30
CARD_W = (GRID_X1 - GRID_X0 - (COLS-1)*CARD_GAP_X) / COLS
CARD_H = (GRID_Y1 - GRID_Y0 - (ROWS-1)*CARD_GAP_Y) / ROWS

for i, B in enumerate(BENCHES):
    r = i // COLS
    c = i % COLS
    x = GRID_X0 + c * (CARD_W + CARD_GAP_X)
    y = GRID_Y1 - (r+1) * CARD_H - r * CARD_GAP_Y
    accent = B["layer_c"]

    # Card body
    ax.add_patch(FancyBboxPatch((x, y), CARD_W, CARD_H,
                                boxstyle="round,pad=0.04,rounding_size=0.05",
                                facecolor=PANEL_BG, edgecolor=RULE,
                                linewidth=0.7, zorder=2))
    # Left accent strip (the layer colour)
    ax.add_patch(Rectangle((x, y), 0.08, CARD_H,
                           facecolor=accent, edgecolor="none", zorder=3))
    # Top-right layer tag pill
    pill_x = x + CARD_W - 0.92
    pill_y = y + CARD_H - 0.42
    ax.add_patch(FancyBboxPatch((pill_x, pill_y), 0.78, 0.30,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=accent, alpha=0.18,
                                edgecolor=accent, linewidth=0.8, zorder=3))
    ax.text(pill_x + 0.39, pill_y + 0.16,
            "stresses " + B["layer"],
            fontsize=7.5, fontweight="bold", color=accent,
            ha="center", va="center",
            family="sans-serif", zorder=4)

    # ID
    ax.text(x + 0.24, y + CARD_H - 0.28, B["bid"],
            fontsize=9, fontweight="bold", color=RED_DK,
            ha="left", va="center", family="monospace", zorder=4)
    # Name
    ax.text(x + 0.24, y + CARD_H - 0.65, B["name"],
            fontsize=12, fontweight="bold", color=INK,
            ha="left", va="center", family="serif", zorder=4)
    # Thin red rule under title
    ax.plot([x + 0.24, x + 1.20],
            [y + CARD_H - 0.85, y + CARD_H - 0.85],
            color=RED, lw=1.0, zorder=4)
    # Question
    ax.text(x + 0.24, y + CARD_H - 1.20, B["question"],
            fontsize=10, color=INK_2,
            ha="left", va="center", family="serif",
            fontstyle="italic", zorder=4)
    # Metrics label
    ax.text(x + 0.24, y + 0.92, "METRICS",
            fontsize=7.5, fontweight="bold", color=MUTE,
            ha="left", va="center", family="sans-serif", zorder=4)
    ax.text(x + 0.24, y + 0.46, B["metrics"],
            fontsize=8.8, color=INK, ha="left", va="center",
            family="serif", linespacing=1.35, zorder=4)
    # Scope
    ax.text(x + CARD_W - 0.24, y + 0.22, B["n_methods"],
            fontsize=7.6, color=MUTE, ha="right", va="center",
            family="sans-serif", fontstyle="italic", zorder=4)

# ────────────────────────────────────────────────────────
# GROUND-TRUTH PANEL (bottom)
# ────────────────────────────────────────────────────────
GT_X0, GT_Y0 = 0.40, 1.05
GT_X1, GT_Y1 = 9.10, 3.45
ax.add_patch(FancyBboxPatch((GT_X0, GT_Y0), GT_X1 - GT_X0, GT_Y1 - GT_Y0,
                            boxstyle="round,pad=0.04,rounding_size=0.05",
                            facecolor=HIGHLIGHT, edgecolor=RULE, linewidth=0.7))
ax.text(GT_X0 + 0.20, GT_Y1 - 0.30,
        "REGULATOR-ALIGNED GROUND TRUTH",
        fontsize=9, fontweight="bold", color=RED_DK,
        ha="left", va="center", family="sans-serif")
ax.text(GT_X0 + 0.20, GT_Y1 - 0.62,
        "Methodological contribution: rank protocols by absolute weight loss,\n"
        "not by fractional change. Aligns the metric with systemic impact.",
        fontsize=9.5, color=INK_2, ha="left", va="top",
        family="serif", fontstyle="italic", linespacing=1.4)

# Equation (typeset)
eqn = (r"$\Delta_t^h(v) \;=\; w_t(v) - w_{t+h}(v),"
       r"\quad \mathcal{S}_t^h \;=\; \mathrm{top}_{\pi=0.05}"
       r"\,\{\, v : w_t(v) > 0,\;\; \Delta_t^h(v) > 0 \,\}$")
ax.text((GT_X0 + GT_X1)/2, GT_Y0 + 0.95, eqn,
        fontsize=12, color=INK, ha="center", va="center",
        family="serif")

# Equation footnote
ax.text((GT_X0 + GT_X1)/2, GT_Y0 + 0.40,
        r"$|\mathcal{S}_t^h|\approx 283$ protocols/week  ·  stable across the 33-week test split",
        fontsize=8.5, color=MUTE, ha="center", va="center",
        family="serif", fontstyle="italic")

# ────────────────────────────────────────────────────────
# POSITIONING PANEL (bottom right)
# ────────────────────────────────────────────────────────
PP_X0, PP_Y0 = 9.30, 1.05
PP_X1, PP_Y1 = 13.60, 3.45
ax.add_patch(FancyBboxPatch((PP_X0, PP_Y0), PP_X1 - PP_X0, PP_Y1 - PP_Y0,
                            boxstyle="round,pad=0.04,rounding_size=0.05",
                            facecolor=PANEL_BG, edgecolor=RULE, linewidth=0.7))
ax.text(PP_X0 + 0.20, PP_Y1 - 0.28,
        "POSITIONING",
        fontsize=9, fontweight="bold", color=RED_DK,
        ha="left", va="center", family="sans-serif")
ax.text(PP_X0 + 0.20, PP_Y1 - 0.55,
        "First suite covering all five axes",
        fontsize=10, color=INK, ha="left", va="center",
        family="serif", fontstyle="italic")

# 5-axis comparison table
suites = [
    ("OGB",              [1, 0, 0, 0, 0]),
    ("TGB",              [1, 0, 0, 0, 0]),
    ("HELM",             [0, 0, 0, 0, 1]),
    ("SWE-bench",        [0, 0, 0, 0, 1]),
    ("AgentBench",       [0, 0, 0, 0, 1]),
    ("DeXposure-Bench",  [1, 1, 1, 1, 1]),
]
axes_lbl = ["Pred", "Anom", "Calib", "Scen", "Dec"]
tbl_x0 = PP_X0 + 0.25
tbl_y0 = PP_Y0 + 0.30
col_w = 0.45
row_h = 0.25
name_col_w = 1.50

# Header row
for ai, a in enumerate(axes_lbl):
    ax.text(tbl_x0 + name_col_w + ai * col_w + col_w/2,
            tbl_y0 + 6 * row_h + 0.04,
            a, fontsize=7.5, fontweight="bold", color=MUTE,
            ha="center", va="bottom", family="sans-serif")
ax.plot([tbl_x0, tbl_x0 + name_col_w + len(axes_lbl)*col_w],
        [tbl_y0 + 6 * row_h, tbl_y0 + 6 * row_h],
        color=RULE, lw=0.6)

for ri, (sname, cells) in enumerate(suites):
    yr = tbl_y0 + (5 - ri) * row_h + 0.04
    is_ours = (sname == "DeXposure-Bench")
    if is_ours:
        ax.add_patch(Rectangle((tbl_x0 - 0.05, yr - 0.10),
                               name_col_w + len(axes_lbl)*col_w + 0.05,
                               row_h - 0.02,
                               facecolor=HIGHLIGHT, edgecolor="none"))
    ax.text(tbl_x0, yr, sname,
            fontsize=8.3 if not is_ours else 9,
            fontweight="bold" if is_ours else "normal",
            color=RED_DK if is_ours else INK_2,
            ha="left", va="center",
            family="serif" if is_ours else "sans-serif",
            fontstyle="italic" if is_ours else "normal")
    for ai, on in enumerate(cells):
        xm = tbl_x0 + name_col_w + ai * col_w + col_w/2
        if on:
            ax.text(xm, yr, "●", fontsize=10,
                    color=RED if is_ours else MUTE,
                    ha="center", va="center")
        else:
            ax.text(xm, yr, "—", fontsize=8,
                    color=DIM, ha="center", va="center")

# ── Save ────────────────────────────────────────────────
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
_default = os.path.join(_here, "fig3_evaluation_framework")
OUT = os.environ.get("FIG_OUT", _default)
plt.savefig(OUT + ".pdf", format="pdf", facecolor=PAPER)
plt.savefig(OUT + ".png", format="png", dpi=300, facecolor=PAPER)
print("Fig 3 (DeXposure-Bench taxonomy) saved.")
