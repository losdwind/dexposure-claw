#!/usr/bin/env python3
"""
Figure 1: DeXposure-Agent -- System Overview
Style: The Economist — red accent, clean serif, no box borders, thin rule dividers.
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
    "savefig.pad_inches": 0.12,
})

# -- The Economist palette --
RED       = "#E3120B"
DARK      = "#1A1A1A"
MID       = "#555555"
LIGHT     = "#999999"
RULE      = "#CCCCCC"
WHITE     = "#FFFFFF"
BLUE_NODE = "#B8D4E8"
RED_NODE  = "#F5C6C6"
BLUE_EDGE = "#5B8FA8"
RED_EDGE  = "#C0392B"
BAR_MAIN  = "#E3120B"
BAR_MUTED = "#D4C5A9"
BAR_STEEL = "#3D6B8E"

fig, ax = plt.subplots(1, 1, figsize=(14, 5.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5.5)
ax.axis("off")
fig.patch.set_facecolor(WHITE)

TOP = 5.15

# ══════════════════════════════════════════════════════
#  TITLES — Economist uses bold serif + thin red rule
# ══════════════════════════════════════════════════════
ax.text(1.5, TOP + 0.05, "Input", fontsize=12, fontweight="bold", ha="center", color=DARK)
ax.plot([0.2, 2.8], [TOP - 0.12, TOP - 0.12], color=RED, lw=2, zorder=3)

ax.text(5.0, TOP + 0.05, "DeXposure-Agent", fontsize=12, fontweight="bold", ha="center", color=DARK)
ax.plot([3.3, 6.7], [TOP - 0.12, TOP - 0.12], color=RED, lw=2, zorder=3)

ax.text(10.7, TOP + 0.05, "Output (weekly)", fontsize=12, fontweight="bold", ha="center", color=DARK)
ax.plot([8.15, 13.75], [TOP - 0.12, TOP - 0.12], color=RED, lw=2, zorder=3)

ax.text(1.5, TOP - 0.35, "Credit exposure network G_t",
        fontsize=7.5, ha="center", color=LIGHT, fontstyle="italic")

# ══════════════════════════════════════════════════════
#  LEFT: Network Graph
# ══════════════════════════════════════════════════════
ncx, ncy = 1.5, 3.0
sc = 0.4
protocols = {
    "Aave":     (ncx - 1.4*sc, ncy + 1.5*sc),
    "Compound": (ncx + 1.4*sc, ncy + 1.5*sc),
    "Lido":     (ncx - 1.8*sc, ncy + 0.0),
    "Uniswap":  (ncx + 1.8*sc, ncy + 0.0),
    "MakerDAO": (ncx,          ncy + 0.6*sc),
    "Curve":    (ncx - 1.0*sc, ncy - 1.4*sc),
    "Pendle":   (ncx + 1.0*sc, ncy - 1.4*sc),
}
edges = [
    ("Aave", "MakerDAO", 2.0), ("Compound", "MakerDAO", 1.5),
    ("Lido", "Aave", 2.5), ("Lido", "Curve", 1.0),
    ("Uniswap", "Compound", 1.2), ("MakerDAO", "Curve", 1.8),
    ("Curve", "Pendle", 0.8), ("Pendle", "Uniswap", 1.0),
    ("MakerDAO", "Lido", 1.3), ("Aave", "Uniswap", 0.9),
]

for src, dst, w in edges:
    x0, y0 = protocols[src]
    x1, y1 = protocols[dst]
    dx, dy = x1 - x0, y1 - y0
    d = np.sqrt(dx**2 + dy**2)
    sh = 0.18
    ax.annotate("", xy=(x1 - dx/d*sh, y1 - dy/d*sh),
                xytext=(x0 + dx/d*sh, y0 + dy/d*sh),
                arrowprops=dict(arrowstyle="-|>", color=LIGHT,
                                lw=0.3 + w*0.35, alpha=0.4,
                                connectionstyle="arc3,rad=0.1"))

for name, (x, y) in protocols.items():
    stressed = name in ("MakerDAO", "Curve")
    circle = plt.Circle((x, y), 0.15,
                         facecolor=RED_NODE if stressed else BLUE_NODE,
                         edgecolor=RED_EDGE if stressed else BLUE_EDGE,
                         linewidth=1.5 if stressed else 0.8, zorder=5)
    ax.add_patch(circle)
    ax.text(x, y - 0.23, name, fontsize=5, ha="center", va="top",
            color=RED_EDGE if stressed else DARK, fontstyle="italic",
            fontweight="bold" if stressed else "normal")

# Legend — minimal, no boxes
lx, ly = 0.5, 1.5
ax.add_patch(plt.Circle((lx, ly), 0.07, fc=BLUE_NODE, ec=BLUE_EDGE, lw=0.6))
ax.text(lx + 0.15, ly, "Protocol", fontsize=5, va="center", color=MID)
ax.add_patch(plt.Circle((lx + 1.0, ly), 0.07, fc=RED_NODE, ec=RED_EDGE, lw=1.0))
ax.text(lx + 1.15, ly, "Stressed", fontsize=5, va="center", color=RED_EDGE)
ax.annotate("", xy=(lx + 2.15, ly), xytext=(lx + 1.8, ly),
            arrowprops=dict(arrowstyle="-|>", color=LIGHT, lw=0.7))
ax.text(lx + 2.25, ly, "Exposure", fontsize=5, va="center", color=MID)

# ══════════════════════════════════════════════════════
#  ARROW: Input -> Agent
# ══════════════════════════════════════════════════════
ax.annotate("", xy=(3.5, 3.3), xytext=(2.8, 3.3),
            arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.5))
ax.text(3.1, 3.52, "G_t", fontsize=10, ha="center", fontstyle="italic",
        color=DARK, fontweight="bold")

# ══════════════════════════════════════════════════════
#  CENTER: Pipeline — no box borders, just text + thin rules
# ══════════════════════════════════════════════════════
px = 3.65
pw = 2.7
stages = [
    ("Data Health Gate",  4.5),
    ("FM Forecast",       3.9),
    ("Risk Monitor",      3.3),
    ("Stress Scenarios",  2.7),
    ("Decision Engine",   2.1),
]
bh = 0.35

for i, (label, y) in enumerate(stages):
    # Light background box, very subtle
    box = FancyBboxPatch((px, y - bh/2), pw, bh,
                          boxstyle="round,pad=0.06", facecolor="#F9F9F9",
                          edgecolor=RULE, linewidth=0.5, zorder=3)
    ax.add_patch(box)
    ax.text(px + pw/2, y, f"({i+1})  {label}", fontsize=8, fontweight="bold",
            ha="center", va="center", color=DARK, zorder=4)

for i in range(len(stages) - 1):
    ax.annotate("", xy=(px + pw/2, stages[i+1][1] + bh/2),
                xytext=(px + pw/2, stages[i][1] - bh/2),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.0))

# FM bracket
bk_x = px + pw + 0.08
bk_top = stages[1][1] + bh/2
bk_bot = stages[3][1] - bh/2
bk_w = 0.1
ax.plot([bk_x, bk_x + bk_w], [bk_top, bk_top], color=RED, lw=1.2)
ax.plot([bk_x + bk_w, bk_x + bk_w], [bk_top, bk_bot], color=RED, lw=1.2)
ax.plot([bk_x, bk_x + bk_w], [bk_bot, bk_bot], color=RED, lw=1.2)
ax.text(bk_x + bk_w + 0.08, (bk_top + bk_bot)/2 + 0.08, "DeXposure-FM",
        fontsize=5.5, ha="left", va="center", color=RED, fontweight="bold")
ax.text(bk_x + bk_w + 0.08, (bk_top + bk_bot)/2 - 0.08, "(GraphPFN)",
        fontsize=4.5, ha="left", va="center", color=MID)

# LLM bracket
lk_top = stages[4][1] + bh/2
lk_bot = stages[4][1] - bh/2
ax.plot([bk_x, bk_x + bk_w], [lk_top, lk_top], color=BAR_STEEL, lw=1.2)
ax.plot([bk_x + bk_w, bk_x + bk_w], [lk_top, lk_bot], color=BAR_STEEL, lw=1.2)
ax.plot([bk_x, bk_x + bk_w], [lk_bot, lk_bot], color=BAR_STEEL, lw=1.2)
ax.text(bk_x + bk_w + 0.08, stages[4][1] + 0.08, "LLM Agent",
        fontsize=5.5, ha="left", va="center", color=BAR_STEEL, fontweight="bold")
ax.text(bk_x + bk_w + 0.08, stages[4][1] - 0.08, "(Claude)",
        fontsize=4.5, ha="left", va="center", color=MID)

# ══════════════════════════════════════════════════════
#  ARROW: Agent -> Output
# ══════════════════════════════════════════════════════
ax.annotate("", xy=(8.0, 3.3), xytext=(7.35, 3.3),
            arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.5))

# ══════════════════════════════════════════════════════
#  RIGHT: Output — Economist style: thin rules, no box borders
# ══════════════════════════════════════════════════════
rx = 8.15
rw = 5.6
bar_left = rx + 0.12
bar_w = rw - 0.25
bar_h = 0.12
gap = 0.15

# ── Alerts ──
A_TOP = TOP - 0.35
A_H = 1.25
A_BOT = A_TOP - A_H

# Section divider (thin red rule at top, gray rule at bottom)
ax.plot([rx, rx + rw], [A_TOP + 0.02, A_TOP + 0.02], color=RED, lw=1.5)
ax.text(rx, A_TOP - 0.12, "Alerts (h = 4 weeks)", fontsize=8.5,
        fontweight="bold", color=DARK)
ax.text(rx + 2.5, A_TOP - 0.12, "How abnormal vs. history?",
        fontsize=5.5, color=LIGHT, fontstyle="italic")

metrics = ["Dominance", "Concentration", "Connectivity", "PR Inequality", "Deg Inequality"]
z_vals  = [1.2,         2.41,           0.8,            1.87,            1.1]

by0 = A_TOP - 0.35
bdy = 0.16
for i, (m, z) in enumerate(zip(metrics, z_vals)):
    by = by0 - i * bdy
    bw = min(z / 3.0, 1.0) * bar_w
    triggered = z >= 2.0
    ax.barh(by, bw, height=bar_h, left=bar_left,
            color=RED if triggered else BAR_MUTED,
            alpha=0.85 if triggered else 0.5, edgecolor="none", zorder=3)
    ax.text(bar_left + 0.05, by, m, fontsize=5, ha="left", va="center",
            color=WHITE if (triggered and bw > 1.5) else DARK,
            fontweight="bold", zorder=4)
    ax.text(bar_left + bw + 0.08, by, f"z={z:.1f}", fontsize=5, ha="left",
            va="center", color=RED if triggered else LIGHT,
            fontweight="bold" if triggered else "normal")

tx = bar_left + (2.0 / 3.0) * bar_w
ax.plot([tx, tx], [by0 + 0.1, by0 - 4*bdy - 0.08],
        color=RED, lw=0.7, ls="--", zorder=4)
ax.text(tx, by0 - 4*bdy - 0.16, "abnormal", fontsize=4.5,
        ha="center", color=RED)

# ── Stress Tests ──
B_H = 1.3
B_TOP = A_BOT - gap
B_BOT = B_TOP - B_H

ax.plot([rx, rx + rw], [B_TOP + 0.02, B_TOP + 0.02], color=RULE, lw=0.7)
ax.text(rx, B_TOP - 0.12, "Stress Tests", fontsize=8.5,
        fontweight="bold", color=DARK)
ax.text(rx + 1.5, B_TOP - 0.12, "What if ... happens? How much value lost?",
        fontsize=5.5, color=LIGHT, fontstyle="italic")

scenarios = ["Top-protocol fail", "Bridge cluster", "Stablecoin de-peg",
             "Lending shock", "Correlated top-10"]
losses    = [8.3, 3.2, 5.1, 2.8, 4.5]
sy0 = B_TOP - 0.35
sdy = 0.16
for i, (s, l) in enumerate(zip(scenarios, losses)):
    by = sy0 - i * sdy
    bw = (l / 15.0) * bar_w
    intensity = l / max(losses)
    ax.barh(by, bw, height=bar_h, left=bar_left, color=BAR_STEEL,
            alpha=0.3 + 0.5 * intensity, edgecolor="none", zorder=3)
    ax.text(bar_left + 0.05, by, s, fontsize=5, ha="left", va="center",
            color=DARK, fontweight="bold", zorder=4)
    ax.text(bar_left + bw + 0.08, by, f"{l}% loss", fontsize=5, ha="left",
            va="center", color=DARK)

cvx = bar_left + (12.7 / 15.0) * bar_w
ax.plot([cvx, cvx], [sy0 + 0.08, sy0 - 4*sdy - 0.08],
        color=RED, lw=0.7, ls="--", zorder=4)
ax.text(cvx + 0.08, sy0 - 4*sdy - 0.16, "Worst-case: 12.7%", fontsize=4.5,
        ha="center", color=RED, fontweight="bold")
ax.text(cvx + 0.08, sy0 - 4*sdy - 0.3, "(avg. of worst 5% simulations)", fontsize=4,
        ha="center", color=LIGHT)

# ── Recommendation Ticket ──
C_TOP = B_BOT - gap
C_BOT = 0.45

ax.plot([rx, rx + rw], [C_TOP + 0.02, C_TOP + 0.02], color=RULE, lw=0.7)
ax.text(rx, C_TOP - 0.12, "Recommendation Ticket", fontsize=8.5,
        fontweight="bold", color=DARK)

# Action
r1 = C_TOP - 0.35
ax.text(rx + 0.05, r1, "Action:", fontsize=6, color=LIGHT)
ax.text(rx + 0.7, r1, "INVESTIGATE", fontsize=7.5,
        fontweight="bold", color=RED, va="center")
ax.text(rx + 2.4, r1, "Severity: Medium", fontsize=6, color=MID, va="center")
ax.text(rx + 4.2, r1, "Score: 0.82", fontsize=6,
        color=DARK, fontweight="bold", va="center")

# Targets
r2 = C_TOP - 0.6
ax.text(rx + 0.05, r2, "Targets:", fontsize=6, color=LIGHT)
for j, tname in enumerate(["MakerDAO", "Curve"]):
    ttx = rx + 1.0 + j * 1.8
    c = plt.Circle((ttx, r2), 0.09, fc=RED_NODE, ec=RED_EDGE, lw=1.0, zorder=4)
    ax.add_patch(c)
    ax.text(ttx + 0.18, r2, tname, fontsize=5.5, va="center",
            color=RED_EDGE, fontweight="bold", zorder=4)

# Evidence
r3 = C_TOP - 0.82
ax.text(rx + 0.05, r3, "Evidence:", fontsize=6, color=LIGHT)
ax.text(rx + 1.0, r3, "Concentration alert (z=2.41) + Top-protocol loss (8.3%)",
        fontsize=5.5, va="center", color=DARK, fontstyle="italic")

# Rationale
r4 = C_TOP - 1.03
ax.text(rx + 0.05, r4, "Rationale:", fontsize=6, color=LIGHT)
ax.text(rx + 1.0, r4, "Concentration rising; if top protocol fails, contagion could spread.",
        fontsize=5.5, va="center", color=DARK)

plt.savefig("/home/aijie/CodeProjects/graph-dexposure/DeXposure_Agent/figures/fig1_system_overview.pdf",
            format="pdf")
plt.savefig("/home/aijie/CodeProjects/graph-dexposure/DeXposure_Agent/figures/fig1_system_overview.png",
            format="png", dpi=300)
print("Fig 1 saved.")
