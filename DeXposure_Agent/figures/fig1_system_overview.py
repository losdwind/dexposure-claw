#!/usr/bin/env python3
"""
Figure 1: DeXposure-Framework -- Four-Layer Architecture (cover figure)

A single horizontal flow that reads left-to-right: G_t enters as a directed
weighted exposure graph; four layers transform it into a supervisory ticket;
the gate emits either the ticket or a safe-mode notice. Layer boundaries are
the unit of ablation (Tab.~Table_b5_decision_CrisisPeriod).

Style: The Economist -- cream paper, single red accent, serif title,
restrained colour beyond red. No 3-D, no shadow, no gradient.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.10,
})

# ── Economist palette ─────────────────────────────────────────────
PAPER     = "#FAF6EC"   # warm cream background
INK       = "#1A1A1A"
INK_2     = "#2A2A2A"
MUTE      = "#67635A"
DIM       = "#9A9588"
RULE      = "#D8D2C0"
RED       = "#C8102E"   # Economist red
RED_DK    = "#9A0C23"
BLUE      = "#2E5077"   # secondary cool tone
GREEN     = "#4A7C3E"
AMBER     = "#C5A55A"
HIGHLIGHT = "#FFF4D1"
SUBTLE    = "#F1ECDD"

# ── Canvas ────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(14, 6.4))
fig.patch.set_facecolor(PAPER)
ax.set_facecolor(PAPER)
ax.set_xlim(0, 14)
ax.set_ylim(0, 6.4)
ax.axis("off")

# ── Top masthead: serif title + red rule (Economist signature) ────
ax.plot([0.4, 13.6], [6.18, 6.18], color=RED, lw=2.4)
ax.text(0.4, 6.30, "DeXposure-Framework",
        fontsize=18, fontweight="bold", color=INK, ha="left", va="bottom",
        fontstyle="italic", family="serif")
ax.text(13.6, 6.30, "A four-layer query architecture for DeFi supervisory decisions",
        fontsize=10, color=MUTE, ha="right", va="bottom", fontstyle="italic")

# ── Input column ─────────────────────────────────────────────────
INPUT_X = 0.4
INPUT_W = 1.6
INPUT_Y0, INPUT_Y1 = 1.4, 5.4

# Mini graph visualisation
gcx, gcy = INPUT_X + INPUT_W/2, (INPUT_Y0 + INPUT_Y1)/2 + 0.1
sc = 0.34
protocols = {
    "Aave":     (gcx - 1.05*sc, gcy + 1.20*sc),
    "Compound": (gcx + 1.05*sc, gcy + 1.20*sc),
    "Lido":     (gcx - 1.45*sc, gcy - 0.10*sc),
    "Uniswap":  (gcx + 1.45*sc, gcy - 0.10*sc),
    "MakerDAO": (gcx,           gcy + 0.45*sc),
    "Curve":    (gcx - 0.80*sc, gcy - 1.20*sc),
    "Pendle":   (gcx + 0.80*sc, gcy - 1.20*sc),
}
edges = [
    ("Aave","MakerDAO",2.0),("Compound","MakerDAO",1.5),
    ("Lido","Aave",2.5),("Lido","Curve",1.0),
    ("Uniswap","Compound",1.2),("MakerDAO","Curve",1.8),
    ("Curve","Pendle",0.8),("Pendle","Uniswap",1.0),
    ("MakerDAO","Lido",1.3),("Aave","Uniswap",0.9),
]
for s, d, w in edges:
    x0, y0 = protocols[s]; x1, y1 = protocols[d]
    dx, dy = x1-x0, y1-y0; L = (dx*dx+dy*dy)**0.5
    sh = 0.14
    ax.annotate("", xy=(x1-dx/L*sh, y1-dy/L*sh),
                xytext=(x0+dx/L*sh, y0+dy/L*sh),
                arrowprops=dict(arrowstyle="-|>", color=DIM,
                                lw=0.4 + w*0.28, alpha=0.55,
                                connectionstyle="arc3,rad=0.10"))
for n, (x, y) in protocols.items():
    stressed = n in ("MakerDAO", "Curve")
    ax.add_patch(Circle((x, y), 0.12,
                        fc="#F0CACA" if stressed else "#D8E3F0",
                        ec=RED if stressed else BLUE,
                        lw=1.4 if stressed else 0.7, zorder=5))

# Input label cluster
ax.text(INPUT_X + INPUT_W/2, INPUT_Y1 - 0.10, "INPUT",
        fontsize=10.5, fontweight="bold", color=RED, ha="center")
ax.plot([INPUT_X + 0.35, INPUT_X + INPUT_W - 0.35],
        [INPUT_Y1 - 0.28, INPUT_Y1 - 0.28], color=RED, lw=1.2)
ax.text(INPUT_X + INPUT_W/2, INPUT_Y0 + 0.30,
        r"Weekly exposure graph",
        fontsize=8.5, color=INK, ha="center", fontstyle="italic")
ax.text(INPUT_X + INPUT_W/2, INPUT_Y0 + 0.08,
        r"$G_t = (V_t, E_t, w_t)$",
        fontsize=10, color=INK, ha="center", family="serif")

# Input → Layer1 arrow
ax.annotate("", xy=(2.30, 3.5), xytext=(INPUT_X + INPUT_W + 0.02, 3.5),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))

# ── Four layers (the contribution): equal columns ────────────────
LAYER_Y0, LAYER_Y1 = 1.20, 5.40
LAYER_W = 2.45
LAYER_GAP = 0.18
COL_X = [2.40, 2.40 + LAYER_W + LAYER_GAP,
         2.40 + 2*(LAYER_W + LAYER_GAP),
         2.40 + 3*(LAYER_W + LAYER_GAP)]

layers = [
    dict(
        roman="I", title="FM Prediction",
        accent=RED,
        comp_lines=[
            ("DeXposure-FM", True),
            ("hybrid predictor", False),
            (r"• keep edges of  $G_t$", False),
            ("• reweight by FM residual", False),
            (r"• add  $\pi \geqq \pi_{\min}$  edges", False),
        ],
        out_label="OUT",
        out_value=r"$\hat G_{t+h},\ \tilde G^{(1..S)}_{t+h}$",
        out_sub="predicted graph + MC samples",
        layer_tag="Layer 1",
    ),
    dict(
        roman="II", title="Monitor & Scenario",
        accent=BLUE,
        comp_lines=[
            ("monitor.py", True),
            ("• 7 network metrics", False),
            ("• rolling-baseline z-scores", False),
            ("scenario.py", True),
            ("• 5 shocks  $S_1$..$S_5$  + CVaR", False),
        ],
        out_label="OUT",
        out_value=r"alerts $\mathcal{A}_t$,  losses $\mathcal{L}_t$",
        out_sub="anomalies + contagion estimates",
        layer_tag="Layer 2",
    ),
    dict(
        roman="III", title="LLM Decision",
        accent=GREEN,
        comp_lines=[
            ("Claude Opus 4.7", True),
            ("• reads metrics + alerts", False),
            ("• reasons over evidence", False),
            ("• cites traceable values", False),
            ("• grounding $= 1.00$", False),
        ],
        out_label="OUT",
        out_value=r"ticket  $\tau$  (action, targets, rationale)",
        out_sub="auditable recommendation",
        layer_tag="Layer 3",
    ),
    dict(
        roman="IV", title="Safety Gate",
        accent=AMBER,
        comp_lines=[
            ("decision.py", True),
            ("• data-health  $\mathrm{DH}_t \geq \\tau_{\mathrm{data}}$", False),
            ("• confidence  $C_t \geq \\tau_{\mathrm{conf}}$", False),
            ("• action gate (Recommend / hold)", False),
            ("else $\\Rightarrow$ safe-mode notice", False),
        ],
        out_label="OUT",
        out_value=r"$\tau^\star$  or  SafeMode",
        out_sub="defensible supervisory output",
        layer_tag="Layer 4",
    ),
]

for i, L in enumerate(layers):
    x = COL_X[i]
    accent = L["accent"]

    # outer rounded card (very subtle)
    card = FancyBboxPatch((x, LAYER_Y0), LAYER_W, LAYER_Y1 - LAYER_Y0,
                          boxstyle="round,pad=0.04,rounding_size=0.05",
                          facecolor="#FFFFFF", edgecolor=RULE, linewidth=0.7,
                          zorder=2)
    ax.add_patch(card)

    # accent ribbon at top
    ax.add_patch(Rectangle((x, LAYER_Y1 - 0.18), LAYER_W, 0.18,
                           facecolor=accent, alpha=0.16, edgecolor="none", zorder=3))

    # layer tag (small, top-left)
    ax.text(x + 0.12, LAYER_Y1 - 0.10, L["layer_tag"].upper(),
            fontsize=8, fontweight="bold", color=accent,
            ha="left", va="center", zorder=5,
            family="sans-serif")
    # roman numeral large
    ax.text(x + LAYER_W - 0.12, LAYER_Y1 - 0.10, L["roman"],
            fontsize=14, fontweight="bold", color=accent,
            ha="right", va="center", zorder=5,
            family="serif", fontstyle="italic")

    # title
    ax.text(x + LAYER_W/2, LAYER_Y1 - 0.55, L["title"],
            fontsize=13, fontweight="bold", color=INK,
            ha="center", va="center", family="serif", zorder=5)
    ax.plot([x + 0.55, x + LAYER_W - 0.55],
            [LAYER_Y1 - 0.78, LAYER_Y1 - 0.78],
            color=accent, lw=1.2, zorder=5)

    # components
    cy = LAYER_Y1 - 1.10
    for j, (txt, bold) in enumerate(L["comp_lines"]):
        if bold:
            ax.text(x + 0.20, cy, txt, fontsize=9, fontweight="bold",
                    color=INK_2, ha="left", va="top", zorder=5, family="serif")
            cy -= 0.32
        else:
            ax.text(x + 0.34, cy, txt, fontsize=8.2, color=MUTE,
                    ha="left", va="top", zorder=5, family="serif")
            cy -= 0.28

    # OUT section (bottom)
    out_y = LAYER_Y0 + 0.45
    ax.plot([x + 0.18, x + LAYER_W - 0.18],
            [out_y + 0.42, out_y + 0.42],
            color=RULE, lw=0.6, zorder=5)
    ax.text(x + 0.20, out_y + 0.28, L["out_label"], fontsize=8,
            fontweight="bold", color=accent, ha="left", va="center",
            family="sans-serif", zorder=5)
    ax.text(x + LAYER_W/2, out_y + 0.05, L["out_value"], fontsize=9.2,
            color=INK, ha="center", va="center", family="serif", zorder=5)
    ax.text(x + LAYER_W/2, out_y - 0.20, L["out_sub"], fontsize=7.5,
            color=DIM, ha="center", va="center", fontstyle="italic",
            family="serif", zorder=5)

    # inter-layer arrow (between this column and next)
    if i < len(layers) - 1:
        x_arr = COL_X[i+1] - LAYER_GAP/2
        ax.annotate("", xy=(x_arr + 0.08, 3.30), xytext=(x_arr - 0.08, 3.30),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0))

# ── Final arrow to output token ──────────────────────────────────
final_x = COL_X[-1] + LAYER_W + 0.05
ax.annotate("", xy=(final_x + 0.50, 3.50), xytext=(final_x, 3.50),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))

# Output token cluster (a small card with a ticket-style mock)
TICKET_X = final_x + 0.55
TICKET_Y0, TICKET_Y1 = 1.4, 5.4
TICKET_W = 13.6 - TICKET_X

ax.text(TICKET_X + TICKET_W/2, TICKET_Y1 - 0.10, "OUTPUT",
        fontsize=10.5, fontweight="bold", color=RED, ha="center")
ax.plot([TICKET_X + 0.35, TICKET_X + TICKET_W - 0.35],
        [TICKET_Y1 - 0.28, TICKET_Y1 - 0.28], color=RED, lw=1.2)

# Ticket mock
tk_x = TICKET_X + 0.10
tk_y0 = 1.50
tk_y1 = 4.85
tk_w = TICKET_W - 0.20
ax.add_patch(FancyBboxPatch((tk_x, tk_y0), tk_w, tk_y1 - tk_y0,
                            boxstyle="round,pad=0.04,rounding_size=0.05",
                            facecolor="#FFFFFF", edgecolor=RULE, linewidth=0.7))
ax.add_patch(Rectangle((tk_x, tk_y1 - 0.32), tk_w, 0.32,
                       facecolor=RED, alpha=0.12, edgecolor="none"))
ax.text(tk_x + tk_w/2, tk_y1 - 0.16, "SUPERVISORY TICKET",
        fontsize=8.5, fontweight="bold", color=RED_DK, ha="center", va="center",
        family="sans-serif")

# Ticket fields
fy = tk_y1 - 0.60
def field(label, value, col=INK):
    global fy
    ax.text(tk_x + 0.18, fy, label, fontsize=7.5, color=MUTE,
            fontstyle="italic", ha="left", va="top")
    ax.text(tk_x + 0.18, fy - 0.22, value, fontsize=9, color=col,
            fontweight="bold", ha="left", va="top", family="serif")
    fy -= 0.55

field("Action", "INVESTIGATE", RED_DK)
field("Severity", "Medium  ·  score 0.82")
field("Targets", "MakerDAO  ·  Curve", RED_DK)
field("Evidence", "Concentration z=2.41")
field("Confidence", "0.83  (gate: passed)")

# ── Bottom callout: dual-contribution framing ────────────────────
ax.add_patch(Rectangle((0.4, 0.36), 13.2, 0.78,
                       facecolor=HIGHLIGHT, edgecolor=RULE, lw=0.7))
ax.text(0.65, 0.95, "CONTRIBUTION I  ·  DeXposure-Framework",
        fontsize=9, fontweight="bold", color=RED_DK, ha="left", va="center",
        family="sans-serif")
ax.text(0.65, 0.65,
        "The architecture above is the unit of contribution: each layer is "
        "independently ablatable, with measurable F1 and explanation-quality lifts.",
        fontsize=8.5, color=INK_2, ha="left", va="center",
        family="serif", fontstyle="italic")

# Ablation tally on the right
ablation_x = 9.4
ax.text(ablation_x, 0.95, "Layer-wise ablation lifts (b5 decision quality)",
        fontsize=8, fontweight="bold", color=INK, ha="left", va="center",
        family="sans-serif")
ax.text(ablation_x, 0.65,
        r"$+\mathrm{FM}$  F1 +33\%   ·   $+\mathrm{LLM}$  F1 +27\%   ·   "
        r"$+\mathrm{Gate}$  Judge $+0.21$",
        fontsize=8.5, color=RED_DK, ha="left", va="center",
        family="serif")

# ── Save ─────────────────────────────────────────────────────────
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
_default = os.path.join(_here, "fig1_system_overview")
OUT = os.environ.get("FIG_OUT", _default)
plt.savefig(OUT + ".pdf", format="pdf", facecolor=PAPER)
plt.savefig(OUT + ".png", format="png", dpi=300, facecolor=PAPER)
print("Fig 1 (DeXposure-Framework architecture) saved.")
