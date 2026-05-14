#!/usr/bin/env python3
"""
Figure 3: Three-Layer Evaluation Framework
Financial color palette matching Fig 1 & Fig 2
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# -- Financial palette --
NAVY      = "#1B2A4A"
STEEL     = "#4A6FA5"
SLATE     = "#6B7B8D"
GOLD      = "#C5A55A"
GOLD_LIGHT= "#F5ECD7"
CREAM     = "#FAF8F2"
ICE       = "#E8EDF2"
CHARCOAL  = "#2D3436"
DARK_RED  = "#8B3A3A"
LIGHT_RED = "#F2E0DE"
TEAL      = "#3D7A80"
TEAL_LIGHT= "#DFF0F0"
GREEN_D   = "#3A6B4E"
GREEN_L   = "#E2EDE6"
AMBER     = "#B8860B"
AMBER_L   = "#FDF5E6"
WARM_GRAY = "#E8E4DE"

fig, ax = plt.subplots(1, 1, figsize=(13, 9))
ax.set_xlim(0, 13)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor(CREAM)
ax.set_facecolor(CREAM)

# Title
ax.text(6.5, 8.7, "Three-Layer Evaluation Framework", fontsize=13,
        fontweight="bold", ha="center", color=NAVY)
ax.text(6.5, 8.35, "No standard benchmark exists for DeFi risk monitoring agents -- we propose a layered design",
        fontsize=8.5, ha="center", color=SLATE, fontstyle="italic")

# ==============================================================
#  METHOD LEGEND
# ==============================================================
legend_x = 0.3
ax.text(legend_x + 0.8, 7.7, "Methods", fontsize=10, fontweight="bold",
        ha="center", color=NAVY)

methods = [
    ("m6_fm_llm", "FM + LLM", STEEL, "Proposed"),
    ("m5_fm_rules",     "FM + Rules", GREEN_D, "Ablation"),
    ("m1_persistence_rules",     "Persist + Rules", SLATE, "Baseline"),
    ("m2_snapshot_llm",     "Pure LLM", AMBER, "Ablation"),
    ("m4_fm_only",     "FM only",         TEAL,    "FM baseline"),
    ("m3_evolvegcn",   "EvolveGCN",       TEAL,    "Temporal-GNN"),
]

for i, (mid, desc, color, tag) in enumerate(methods):
    y = 7.1 - i * 0.55
    box = FancyBboxPatch((legend_x, y - 0.18), 2.0, 0.4,
                          boxstyle="round,pad=0.05", facecolor=color,
                          edgecolor=color, linewidth=0.8, alpha=0.15, zorder=3)
    ax.add_patch(box)
    ax.plot([legend_x, legend_x + 0.15], [y + 0.02, y + 0.02],
            color=color, lw=3, zorder=4)
    ax.text(legend_x + 0.25, y + 0.06, f"{mid}", fontsize=8, fontweight="bold",
            va="center", color=color, zorder=4)
    ax.text(legend_x + 1.1, y + 0.06, desc, fontsize=6.5, va="center",
            color=SLATE, zorder=4)
    ax.text(legend_x + 2.1, y + 0.06, tag, fontsize=6, va="center",
            color=SLATE, fontstyle="italic", zorder=4)

# ==============================================================
#  LAYER 1: FM Prediction Quality
# ==============================================================
layer_x = 3.0
layer_w = 9.5
l1_y = 7.0
l1_h = 1.6

l1_bg = FancyBboxPatch((layer_x, l1_y - l1_h/2), layer_w, l1_h,
                         boxstyle="round,pad=0.12", facecolor=GOLD_LIGHT,
                         edgecolor=GOLD, linewidth=1.5, zorder=2)
ax.add_patch(l1_bg)

ax.text(layer_x + 0.2, l1_y + l1_h/2 - 0.2, "Layer 1: FM Prediction Quality",
        fontsize=10, fontweight="bold", color=AMBER, zorder=3)
ax.text(layer_x + 0.2, l1_y + l1_h/2 - 0.45,
        "\"Can the FM produce accurate graph forecasts?\"    [Runs on GPU]",
        fontsize=7.5, color=SLATE, fontstyle="italic", zorder=3)

benchmarks_l1 = [
    ("b1_forecast", "Risk Forecasting", "MAE, Spearman rho,\ntrend consistency"),
    ("b2_warning", "Early Warning", "Lead time, precision,\nrecall on 3 crises"),
    ("b3_calibration", "Uncertainty", "ECE, PI coverage,\nCRPS"),
    ("b6_robustness", "Robustness", "b1_forecast under 5\ndegradation regimes"),
]
bx_start = layer_x + 0.3
bx_w = 2.1
for i, (bid, bname, bmetrics) in enumerate(benchmarks_l1):
    bx = bx_start + i * (bx_w + 0.15)
    by = l1_y - 0.3
    box = FancyBboxPatch((bx, by - 0.35), bx_w, 0.7,
                          boxstyle="round,pad=0.06", facecolor=CREAM,
                          edgecolor=GOLD, linewidth=0.8, zorder=3, alpha=0.95)
    ax.add_patch(box)
    ax.text(bx + bx_w/2, by + 0.15, f"{bid}: {bname}", fontsize=7.5,
            fontweight="bold", ha="center", color=AMBER, zorder=4)
    ax.text(bx + bx_w/2, by - 0.12, bmetrics, fontsize=6, ha="center",
            va="center", color=SLATE, zorder=4, linespacing=1.2)

ax.text(layer_x + layer_w - 0.3, l1_y + l1_h/2 - 0.2,
        "m5_fm_rules  m1_persistence_rules  m4_fm_only  m3_evolvegcn", fontsize=7, ha="right", color=SLATE,
        fontweight="bold", zorder=3)

# ==============================================================
#  LAYER 2: LLM Reasoning Quality
# ==============================================================
l2_y = 4.7
l2_h = 1.6

l2_bg = FancyBboxPatch((layer_x, l2_y - l2_h/2), layer_w, l2_h,
                         boxstyle="round,pad=0.12", facecolor=ICE,
                         edgecolor=STEEL, linewidth=1.5, zorder=2)
ax.add_patch(l2_bg)

ax.text(layer_x + 0.2, l2_y + l2_h/2 - 0.2, "Layer 2: LLM Reasoning Quality",
        fontsize=10, fontweight="bold", color=NAVY, zorder=3)
ax.text(layer_x + 0.2, l2_y + l2_h/2 - 0.45,
        "\"Given FM data, does LLM reason better than rules?\"    [Runs locally, calls Claude API]",
        fontsize=7.5, color=SLATE, fontstyle="italic", zorder=3)

metrics_l2 = [
    ("Groundedness", "Fraction of cited values\ntraceable to FM input"),
    ("Faithfulness", "LLM-as-Judge 1-5:\nreasoning follows evidence?"),
    ("Consistency", "Jaccard across 3 runs\nat temperature=0"),
]
for i, (mname, mdesc) in enumerate(metrics_l2):
    bx = bx_start + i * (2.85 + 0.15)
    by = l2_y - 0.3
    box = FancyBboxPatch((bx, by - 0.35), 2.85, 0.7,
                          boxstyle="round,pad=0.06", facecolor=CREAM,
                          edgecolor=STEEL, linewidth=0.8, zorder=3, alpha=0.95)
    ax.add_patch(box)
    ax.text(bx + 2.85/2, by + 0.15, mname, fontsize=7.5, fontweight="bold",
            ha="center", color=NAVY, zorder=4)
    ax.text(bx + 2.85/2, by - 0.12, mdesc, fontsize=6, ha="center",
            va="center", color=SLATE, zorder=4, linespacing=1.2)

ax.text(layer_x + layer_w - 0.3, l2_y + l2_h/2 - 0.2,
        "m6_fm_llm  m2_snapshot_llm", fontsize=7, ha="right", color=SLATE,
        fontweight="bold", zorder=3)
ax.text(layer_x + layer_w - 0.3, l2_y + l2_h/2 - 0.45,
        "Key: m6_fm_llm.Ground > m2_snapshot_llm.Ground -- FM reduces hallucination",
        fontsize=6.5, ha="right", color=NAVY, fontstyle="italic", zorder=3)

# ==============================================================
#  LAYER 3: End-to-End Decision Quality
# ==============================================================
l3_y = 2.4
l3_h = 1.6

l3_bg = FancyBboxPatch((layer_x, l3_y - l3_h/2), layer_w, l3_h,
                         boxstyle="round,pad=0.12", facecolor=GREEN_L,
                         edgecolor=GREEN_D, linewidth=1.5, zorder=2)
ax.add_patch(l3_bg)

ax.text(layer_x + 0.2, l3_y + l3_h/2 - 0.2, "Layer 3: End-to-End Decision Quality",
        fontsize=10, fontweight="bold", color=GREEN_D, zorder=3)
ax.text(layer_x + 0.2, l3_y + l3_h/2 - 0.45,
        "\"Which method produces the best risk management decisions?\"    [Core contribution]",
        fontsize=7.5, color=SLATE, fontstyle="italic", zorder=3)

metrics_l3 = [
    ("b5_decision: Decision Quality", "Precision, recall, severity rho,\nstability, FIR, cost-adjusted"),
    ("b4_stress: Stress Test", "Contagion loss MAE,\noverlap@10"),
    ("Explanation Quality", "LLM-as-Judge 1-5\n(rules = N/A by design)"),
]
for i, (mname, mdesc) in enumerate(metrics_l3):
    bx = bx_start + i * (2.85 + 0.15)
    by = l3_y - 0.3
    box = FancyBboxPatch((bx, by - 0.35), 2.85, 0.7,
                          boxstyle="round,pad=0.06", facecolor=CREAM,
                          edgecolor=GREEN_D, linewidth=0.8, zorder=3, alpha=0.95)
    ax.add_patch(box)
    ax.text(bx + 2.85/2, by + 0.15, mname, fontsize=7.5, fontweight="bold",
            ha="center", color=GREEN_D, zorder=4)
    ax.text(bx + 2.85/2, by - 0.12, mdesc, fontsize=6, ha="center",
            va="center", color=SLATE, zorder=4, linespacing=1.2)

ax.text(layer_x + layer_w - 0.3, l3_y + l3_h/2 - 0.2,
        "m6_fm_llm  m5_fm_rules  m1_persistence_rules  m2_snapshot_llm", fontsize=7, ha="right", color=SLATE,
        fontweight="bold", zorder=3)

# ==============================================================
#  FLOW ARROWS
# ==============================================================
arrow_x = layer_x + layer_w + 0.3
ax.annotate("", xy=(arrow_x, l2_y + l2_h/2 + 0.05),
            xytext=(arrow_x, l1_y - l1_h/2 - 0.05),
            arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.5))
ax.text(arrow_x + 0.15, (l1_y - l1_h/2 + l2_y + l2_h/2)/2,
        "Necessary\ncondition", fontsize=7, ha="left", va="center",
        color=SLATE, fontstyle="italic")

ax.annotate("", xy=(arrow_x, l3_y + l3_h/2 + 0.05),
            xytext=(arrow_x, l2_y - l2_h/2 - 0.05),
            arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.5))
ax.text(arrow_x + 0.15, (l2_y - l2_h/2 + l3_y + l3_h/2)/2,
        "Necessary\ncondition", fontsize=7, ha="left", va="center",
        color=SLATE, fontstyle="italic")

# ==============================================================
#  THREE CLAIMS
# ==============================================================
claims_y = 0.55
claims_bg = FancyBboxPatch((1.5, -0.15), 10.0, 1.15,
                            boxstyle="round,pad=0.1", facecolor=WARM_GRAY,
                            edgecolor="#D0CCC4", linewidth=0.8, zorder=1)
ax.add_patch(claims_bg)

ax.text(6.5, claims_y + 0.25, "Three claims to prove:", fontsize=9,
        fontweight="bold", ha="center", color=NAVY)

claims = [
    "(1)  m6_fm_llm.Precision > m5_fm_rules.Precision    (LLM reasoning > rule engine)",
    "(2)  m6_fm_llm.Groundedness > m2_snapshot_llm.Groundedness    (FM data reduces hallucination)",
    "(3)  m6_fm_llm.Explanation > 0,  Rules = N/A    (LLM provides interpretability)",
]
for i, claim in enumerate(claims):
    ax.text(6.5, claims_y - 0.15 - i * 0.28, claim, fontsize=7.5,
            ha="center", color=CHARCOAL, zorder=3)

plt.savefig("/home/figurich/CodeProjects/graph-dexposure/DeXposure_Agent/figures/fig3_evaluation_framework.pdf",
            format="pdf")
plt.savefig("/home/figurich/CodeProjects/graph-dexposure/DeXposure_Agent/figures/fig3_evaluation_framework.png",
            format="png", dpi=300)
print("Fig 3 saved.")
