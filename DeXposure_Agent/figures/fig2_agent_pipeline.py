#!/usr/bin/env python3
"""
Figure 2: Agent Pipeline -- Algorithm 1 Visualization
Financial color palette matching Fig 1
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

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
MUTED_RED = "#C0706B"
LIGHT_RED = "#F2E0DE"
TEAL      = "#3D7A80"
TEAL_LIGHT= "#DFF0F0"
GREEN_D   = "#3A6B4E"
GREEN_L   = "#E2EDE6"
AMBER     = "#B8860B"
AMBER_L   = "#FDF5E6"
WARM_GRAY = "#E8E4DE"
LOOP_BG   = "#F7F5F0"

fig, ax = plt.subplots(1, 1, figsize=(11, 14))
ax.set_xlim(0, 11)
ax.set_ylim(0, 14)
ax.axis("off")
fig.patch.set_facecolor(CREAM)
ax.set_facecolor(CREAM)

def draw_box(ax, cx, cy, w, h, label, sublabel, fc, ec, fontsize=9):
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle="round,pad=0.1", facecolor=fc,
                          edgecolor=ec, linewidth=1.2, zorder=3)
    ax.add_patch(box)
    ax.text(cx, cy + 0.12, label, fontsize=fontsize, fontweight="bold",
            ha="center", va="center", color=NAVY, zorder=4)
    if sublabel:
        ax.text(cx, cy - 0.18, sublabel, fontsize=7, ha="center", va="center",
                color=SLATE, zorder=4)

def draw_diamond(ax, cx, cy, w, h, label, sublabel, fc, ec):
    diamond = plt.Polygon([(cx, cy+h/2), (cx+w/2, cy), (cx, cy-h/2), (cx-w/2, cy)],
                           closed=True, facecolor=fc, edgecolor=ec,
                           linewidth=1.2, zorder=3)
    ax.add_patch(diamond)
    ax.text(cx, cy + 0.05, label, fontsize=8, fontweight="bold",
            ha="center", va="center", color=NAVY, zorder=4)
    if sublabel:
        ax.text(cx, cy - 0.2, sublabel, fontsize=6.5, ha="center", va="center",
                color=SLATE, zorder=4)

def arrow_down(ax, x, y_from, y_to, label=None):
    ax.annotate("", xy=(x, y_to), xytext=(x, y_from),
                arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.2))
    if label:
        ax.text(x + 0.15, (y_from + y_to)/2, label, fontsize=7, ha="left",
                va="center", color=SLATE)

def arrow_right(ax, x_from, x_to, y, label=None):
    ax.annotate("", xy=(x_to, y), xytext=(x_from, y),
                arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.2))

CX = 5.5

# Title
ax.text(CX, 13.85, "Algorithm 1: DeXposure-Agent -- One Epoch",
        fontsize=12, fontweight="bold", ha="center", color=NAVY)

# ROW 1: Input
y = 13.2
draw_box(ax, CX, y, 3.5, 0.7, "Input: Snapshot G_t , X_t",
         "Credit-exposure graph at epoch t", ICE, STEEL)

# ROW 2: Data Health Gate
y_dh = 12.0
arrow_down(ax, CX, 13.2 - 0.35, y_dh + 0.45)
draw_diamond(ax, CX, y_dh, 3.2, 0.85,
             "Data-Health Gate", "DH_t >= tau_data (0.7)?", GOLD_LIGHT, GOLD)

# Safe mode branch
ax.annotate("", xy=(8.5, y_dh), xytext=(CX + 1.6, y_dh),
            arrowprops=dict(arrowstyle="-|>", color=DARK_RED, lw=1.0))
ax.text(CX + 2.3, y_dh + 0.2, "No", fontsize=7.5, color=DARK_RED, fontweight="bold")

safe_box = FancyBboxPatch((8.5, y_dh - 0.3), 2.0, 0.6,
                           boxstyle="round,pad=0.08", facecolor=LIGHT_RED,
                           edgecolor=DARK_RED, linewidth=1.0, zorder=3)
ax.add_patch(safe_box)
ax.text(9.5, y_dh + 0.05, "SAFE MODE", fontsize=8, fontweight="bold",
        ha="center", color=DARK_RED, zorder=4)
ax.text(9.5, y_dh - 0.17, "Suppress interventions", fontsize=6.5,
        ha="center", color=SLATE, zorder=4)

ax.text(CX - 0.3, y_dh - 0.5, "Yes", fontsize=7.5, color=GREEN_D, fontweight="bold")

# LOOP BOX
loop_top = 10.8
loop_bot = 6.0
loop_left = 1.5
loop_right = 9.5

loop_bg = FancyBboxPatch((loop_left, loop_bot), loop_right - loop_left, loop_top - loop_bot,
                          boxstyle="round,pad=0.15", facecolor=LOOP_BG,
                          edgecolor=SLATE, linewidth=1.0, linestyle="--", zorder=1)
ax.add_patch(loop_bg)
ax.text(loop_left + 0.2, loop_top - 0.15, "for each horizon  h in {1, 4, 8, 12}",
        fontsize=9, fontweight="bold", fontstyle="italic", color=SLATE, zorder=2)

arrow_down(ax, CX, y_dh - 0.45, loop_top)

# Stage 1: Forecast
y_fc = 10.1
draw_box(ax, 3.5, y_fc, 2.8, 0.65,
         "FM Forecast", "P_{t,h} <- DeXposure-FM.Forecast(G_t, h)",
         AMBER_L, AMBER)

ax.text(1.4, y_fc, "GPU\nServer", fontsize=7, ha="center", va="center",
        color=AMBER, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", facecolor=GOLD_LIGHT,
                  edgecolor=GOLD, alpha=0.9))
ax.annotate("", xy=(3.5 - 1.4, y_fc), xytext=(1.9, y_fc),
            arrowprops=dict(arrowstyle="<->", color=GOLD, lw=0.8))

# Stage 2: PredGraph
y_pg = 9.0
arrow_down(ax, 3.5, y_fc - 0.33, y_pg + 0.33)
draw_box(ax, 3.5, y_pg, 2.8, 0.65,
         "Build Predicted Graph", "G_hat (Eq. 2: threshold at pi_min)",
         AMBER_L, AMBER)

# Stage 3: MC Sampling
y_mc = 8.0
arrow_down(ax, 3.5, y_pg - 0.33, y_mc + 0.33)
mc_pts = np.array([
    [3.5 - 1.2, y_mc - 0.28],
    [3.5 + 1.2, y_mc - 0.28],
    [3.5 + 1.5, y_mc + 0.28],
    [3.5 - 0.9, y_mc + 0.28],
])
mc_poly = plt.Polygon(mc_pts, closed=True, facecolor=TEAL_LIGHT,
                       edgecolor=TEAL, linewidth=1.0, zorder=3)
ax.add_patch(mc_poly)
ax.text(3.5 + 0.15, y_mc + 0.05, "MC Sampling", fontsize=8.5, fontweight="bold",
        ha="center", va="center", color=NAVY, zorder=4)
ax.text(3.5 + 0.15, y_mc - 0.17, "M=50 samples from P_{t,h}",
        fontsize=6.5, ha="center", va="center", color=SLATE, zorder=4)

# Stage 4: Monitor
y_mon = 6.9
arrow_down(ax, 3.5, y_mc - 0.28, y_mon + 0.33)
draw_box(ax, 3.5, y_mon, 2.8, 0.65,
         "Risk Monitor", "M1-M7 metrics -> z-score alerts (Eq. 3)",
         ICE, STEEL)

# Alert output
arrow_right(ax, 4.9, 7.5, y_mon)
alert_box = FancyBboxPatch((7.5, y_mon - 0.25), 1.7, 0.5,
                            boxstyle="round,pad=0.06", facecolor=GOLD_LIGHT,
                            edgecolor=GOLD, linewidth=0.8, zorder=3)
ax.add_patch(alert_box)
ax.text(8.35, y_mon + 0.03, "Alerts A_t", fontsize=7.5, fontweight="bold",
        ha="center", color=AMBER, zorder=4)
ax.text(8.35, y_mon - 0.17, "with confidence C_t",
        fontsize=6, ha="center", color=SLATE, zorder=4)

# Stage 5: Scenario Engine
draw_box(ax, 7.5, 8.0, 2.2, 0.65,
         "Scenario Engine", "S1-S5 stress tests -> CVaR (Eq. 4)",
         TEAL_LIGHT, TEAL)

ax.annotate("", xy=(7.5 - 1.1, 8.0 + 0.15),
            xytext=(3.5 + 1.4, y_pg - 0.05),
            arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.0,
                            connectionstyle="arc3,rad=-0.2"))

# Scenario output
scenario_box = FancyBboxPatch((7.5, 6.9 - 0.25), 1.7, 0.5,
                               boxstyle="round,pad=0.06", facecolor=TEAL_LIGHT,
                               edgecolor=TEAL, linewidth=0.8, zorder=3)
ax.add_patch(scenario_box)
arrow_down(ax, 8.35, 8.0 - 0.33, 6.9 + 0.25)
ax.text(8.35, 6.93, "Scenario S", fontsize=7.5, fontweight="bold",
        ha="center", color=TEAL, zorder=4)
ax.text(8.35, 6.73, "ranked losses",
        fontsize=6, ha="center", color=SLATE, zorder=4)

# AGGREGATE
y_agg = 5.3
arrow_down(ax, CX, loop_bot, y_agg + 0.33)
draw_box(ax, CX, y_agg, 3.0, 0.65,
         "Aggregate", "Merge alerts & scenario losses across horizons",
         WARM_GRAY, SLATE)

# DECISION ENGINE
y_dec = 3.8
arrow_down(ax, CX, y_agg - 0.33, y_dec + 0.55)

dec_box = FancyBboxPatch((CX - 2.5, y_dec - 0.55), 5.0, 1.1,
                          boxstyle="round,pad=0.1", facecolor=GREEN_L,
                          edgecolor=GREEN_D, linewidth=1.5, zorder=3)
ax.add_patch(dec_box)
ax.text(CX, y_dec + 0.3, "Decision Engine (Playbook)", fontsize=9.5,
        fontweight="bold", ha="center", color=NAVY, zorder=4)

ax.text(CX - 1.3, y_dec - 0.05, "Safe-mode\ngate", fontsize=7, ha="center",
        color=DARK_RED, fontweight="bold", zorder=4)
ax.text(CX - 1.3, y_dec - 0.35, "blocks High/Critical\nif SAFE_MODE=1",
        fontsize=5.5, ha="center", color=SLATE, zorder=4)

ax.text(CX + 1.3, y_dec - 0.05, "Confidence\ngate", fontsize=7, ha="center",
        color=STEEL, fontweight="bold", zorder=4)
ax.text(CX + 1.3, y_dec - 0.35, "blocks if\nmean(C_t) < tau_conf",
        fontsize=5.5, ha="center", color=SLATE, zorder=4)

ax.plot([CX, CX], [y_dec - 0.45, y_dec + 0.15], color=WARM_GRAY, lw=0.5, ls=":", zorder=4)

ax.text(CX + 3.2, y_dec, "LLM Agent\n(Claude)", fontsize=7.5, ha="center",
        va="center", color=NAVY, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", facecolor=ICE,
                  edgecolor=STEEL, alpha=0.9))
ax.annotate("", xy=(CX + 2.5, y_dec), xytext=(CX + 2.8, y_dec),
            arrowprops=dict(arrowstyle="<-", color=STEEL, lw=0.8))

# OUTPUT
y_out = 2.3
arrow_down(ax, CX, y_dec - 0.55, y_out + 0.35)

out_items = [
    (CX - 2.5, "Alerts\n+ Evidence", GOLD_LIGHT, GOLD),
    (CX,       "Scenario\nSummary",   TEAL_LIGHT, TEAL),
    (CX + 2.5, "Decision\nTickets",   GREEN_L, GREEN_D),
]

for ox, label, fc, ec in out_items:
    box = FancyBboxPatch((ox - 1.0, y_out - 0.35), 2.0, 0.7,
                          boxstyle="round,pad=0.08", facecolor=fc,
                          edgecolor=ec, linewidth=1.0, zorder=3)
    ax.add_patch(box)
    ax.text(ox, y_out, label, fontsize=8, fontweight="bold",
            ha="center", va="center", color=NAVY, zorder=4)

for ox, _, _, _ in out_items:
    ax.plot([ox, ox], [y_out + 0.35, y_out + 0.55], color=NAVY, lw=0.8)
ax.plot([CX - 2.5, CX + 2.5], [y_out + 0.55, y_out + 0.55], color=NAVY, lw=0.8)
ax.plot([CX, CX], [y_out + 0.55, y_dec - 0.55], color=NAVY, lw=0.8)

ax.text(CX, 1.6, "AgentOutput: auditable, evidence-grounded risk intelligence",
        fontsize=9, ha="center", va="center", fontstyle="italic", color=SLATE,
        bbox=dict(boxstyle="round,pad=0.2", facecolor=LOOP_BG,
                  edgecolor=WARM_GRAY, alpha=0.8))

plt.savefig("/home/aijie/CodeProjects/graph-dexposure/DeXposure_Agent/figures/fig2_agent_pipeline.pdf",
            format="pdf")
plt.savefig("/home/aijie/CodeProjects/graph-dexposure/DeXposure_Agent/figures/fig2_agent_pipeline.png",
            format="png", dpi=300)
print("Fig 2 saved.")
