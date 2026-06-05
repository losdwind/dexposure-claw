#!/usr/bin/env python3
"""Regenerate Figure 2 (fig4_method_comparison) with the 2026-06-03 cross-family
judge data.

The original `build_table3_and_fig4.py` consumed `comparison.json` from the
2026-05 llm_eval run, which used Claude Haiku 4.5 as the LLM-as-judge. Paper
v3 now reports judge scores under a stronger, cross-family panel (primary:
Claude Opus 4.8); the figure must show the same Opus 4.8 numbers as Tables 1
and 2 so a reader cross-checking the figure does not see a contradiction.

This regenerator:
  - keeps F1 / FIR for m1 / m5 / m2 / m6 / m7 from the original
    Opus 4.7 decision run (m5 F1 corrected to 0.0190 from b5_tkde_v4)
  - swaps the explanation-quality column to the Opus 4.8 judge scores from
    paper/results/rejudge_aggregated.json
  - re-derives the bar annotations dynamically so no number is hardcoded

Run from the repo root:
    python paper/scripts/rebuild_fig2_with_new_judges.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES_EMNLP = ROOT.parent / "paper-emnlp-industry" / "figures"
FIGURES_LOCAL = ROOT / "figures"


# ---------------------------------------------------------------------------
# Data: pulled from the same JSON files that Tables 1 / 2 / 3 use,
# so the figure cannot drift from the tables.
# ---------------------------------------------------------------------------

def load_data() -> dict:
    # Macro F1 for m1 / m5 comes from b5_tkde_v4 (the same source Table 1 uses)
    m1 = json.load(open(RESULTS / "run_20260515_pipeline_full" / "b5_tkde_v4"
                        / "b5_decision__m1_persistence_rules.json"))["results"][0]
    m5 = json.load(open(RESULTS / "run_20260515_pipeline_full" / "b5_tkde_v4"
                        / "b5_decision__m5_fm_rules.json"))["results"][0]

    # m2 / m6 / m7 from the original Opus 4.7 + Haiku 4.5 run summaries
    # (F1 / FIR are decision-model-only quantities, judge-independent).
    m2s = json.load(open(RESULTS / "run_20260515_pipeline_full" / "llm_eval"
                         / "summary_m2_snapshot_llm.json"))
    m6s = json.load(open(RESULTS / "run_20260515_pipeline_full" / "llm_eval"
                         / "summary_m6_fm_llm.json"))
    m7s = json.load(open(RESULTS / "run_20260515_pipeline_full" / "llm_eval"
                         / "summary_m7_fm_llm_gated.json"))

    def macro_f1(p, c):
        return 2 * p * c / (p + c) if (p + c) > 0 else 0.0

    # Opus 4.8 judge scores (the NEW judge column) from rejudge_aggregated.json
    agg = json.load(open(RESULTS / "rejudge_aggregated.json"))
    opus48 = {r["method"]: r["mean_quality"]
              for r in agg["rejudge_track_a"]
              if r["judge_model"] == "claude-opus-4.8"}

    return {
        "m1": {"f1": macro_f1(m1["ticket_precision"], m1["audit_completeness"]),
               "fir": m1["false_intervention_rate"]},
        "m5": {"f1": macro_f1(m5["ticket_precision"], m5["audit_completeness"]),
               "fir": m5["false_intervention_rate"]},
        "m2": {"f1": macro_f1(m2s["ticket_precision"], m2s["audit_completeness"]),
               "fir": m2s["false_intervention_rate"],
               "judge_opus48": opus48["m2_snapshot_llm"]},
        "m6": {"f1": macro_f1(m6s["ticket_precision"], m6s["audit_completeness"]),
               "fir": m6s["false_intervention_rate"],
               "judge_opus48": opus48["m6_fm_llm"]},
        "m7": {"f1": macro_f1(m7s["ticket_precision"], m7s["audit_completeness"]),
               "fir": m7s["false_intervention_rate"],
               "judge_opus48": opus48["m7_fm_llm_gated"]},
    }


def _pct_delta(new: float, old: float) -> int:
    return round(100.0 * (new - old) / old) if old > 0 else 0


def build_figure(data: dict, out_paths: list[Path]) -> None:
    """Render a column-width, three-panel figure for the EMNLP paper."""
    f1_order = ["m1", "m5", "m2", "m6", "m7"]
    judge_order = ["m2", "m6", "m7"]
    labels = {
        "m1": "m1 Persist",
        "m5": "m5 FM+R",
        "m2": "m2 LLM",
        "m6": "m6 FM+L",
        "m7": "m7 Gate",
    }
    colors = {
        "m1": "#9AA0A6",
        "m5": "#009E73",
        "m2": "#56B4E9",
        "m6": "#0072B2",
        "m7": "#D55E00",
    }
    ink = "#263238"
    muted = "#607D8B"
    grid = "#D7DEE2"

    f1_values = np.array([data[m]["f1"] for m in f1_order])
    judge_values = np.array([data[m]["judge_opus48"] for m in judge_order])
    fir_values = np.array([data[m]["fir"] for m in f1_order])

    fm_delta = _pct_delta(data["m6"]["f1"], data["m2"]["f1"])
    llm_delta = _pct_delta(data["m6"]["f1"], data["m5"]["f1"])
    gate_delta = _pct_delta(data["m7"]["f1"], data["m6"]["f1"])
    judge_rounded = [round(v, 2) for v in judge_values]
    fm_jdelta = judge_rounded[1] - judge_rounded[0]
    gate_jdelta = judge_rounded[2] - judge_rounded[1]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.0,
        "axes.titlesize": 8.4,
        "axes.labelsize": 7.8,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.3,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.035,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": ink,
        "axes.linewidth": 0.6,
    })

    fig, axes = plt.subplots(
        3, 1, figsize=(3.33, 4.55),
        gridspec_kw={"height_ratios": [1.22, 0.92, 1.08], "hspace": 0.78},
    )
    fig.patch.set_facecolor("white")

    def style_axis(ax, xlabel: str) -> None:
        ax.grid(axis="x", color=grid, linestyle="-", linewidth=0.45, alpha=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", colors=ink, length=2.5, width=0.55)
        ax.set_xlabel(xlabel, color=ink, labelpad=2)

    def add_note(ax, text: str, color: str) -> None:
        ax.text(
            1.0, 1.015, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.15, color=color, clip_on=False,
        )

    # Panel A: ticket F1.
    ax = axes[0]
    y = np.arange(len(f1_order))
    ax.barh(
        y, f1_values, height=0.58,
        color=[colors[m] for m in f1_order],
        edgecolor="white", linewidth=0.6,
    )
    ax.set_yticks(y, [labels[m] for m in f1_order])
    ax.invert_yaxis()
    ax.set_xlim(0.0, max(f1_values) * 1.38)
    ax.set_title("Ticket F1", loc="left", fontweight="bold", color=ink, pad=5)
    style_axis(ax, "F1 (higher better)")
    for yi, value in zip(y, f1_values):
        ax.text(
            value + ax.get_xlim()[1] * 0.025, yi, f"{value:.4f}",
            va="center", ha="left", fontsize=6.9, color=ink,
        )
    add_note(ax, f"+FM {fm_delta:+d}%; +LLM {llm_delta:+d}%", colors["m6"])
    ax.text(
        data["m7"]["f1"] + ax.get_xlim()[1] * 0.025, f1_order.index("m7") + 0.28,
        f"gate {gate_delta:+d}%", fontsize=6.3, color=colors["m7"],
        ha="left", va="center",
    )

    # Panel B: Opus 4.8 judge quality.
    ax = axes[1]
    y = np.arange(len(judge_order))
    ax.barh(
        y, judge_values, height=0.58,
        color=[colors[m] for m in judge_order],
        edgecolor="white", linewidth=0.6,
    )
    ax.set_yticks(y, [labels[m] for m in judge_order])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 5.0)
    ax.set_title("Explanation Quality", loc="left", fontweight="bold", color=ink, pad=5)
    style_axis(ax, "Opus 4.8 judge score")
    for yi, value in zip(y, judge_values):
        ax.text(value + 0.09, yi, f"{value:.2f}", va="center", ha="left",
                fontsize=6.9, color=ink)
    add_note(ax, f"+FM {fm_jdelta:+.2f}; +Gate {gate_jdelta:+.2f}", colors["m6"])

    # Panel C: false intervention rate.
    ax = axes[2]
    y = np.arange(len(f1_order))
    ax.barh(
        y, fir_values, height=0.58,
        color=[colors[m] for m in f1_order],
        edgecolor="white", linewidth=0.6,
    )
    ax.set_yticks(y, [labels[m] for m in f1_order])
    ax.invert_yaxis()
    ax.set_xlim(0.0, max(0.52, float(fir_values.max()) * 1.16))
    ax.set_title("Intervention Cost", loc="left", fontweight="bold", color=ink, pad=5)
    style_axis(ax, "FIR (lower better)")
    for yi, value in zip(y, fir_values):
        label = "0" if value == 0 else f"{value:.3f}"
        x_text = value + ax.get_xlim()[1] * 0.025
        ax.text(x_text, yi, label, va="center", ha="left",
                fontsize=6.9, color=muted if value == 0 else ink)

    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300)
    plt.close(fig)


def main():
    data = load_data()
    print("Data sources cross-check:")
    print(f"  m1 F1 = {data['m1']['f1']:.4f}  (Table 1 says 0.0076)")
    print(f"  m5 F1 = {data['m5']['f1']:.4f}  (Table 1 says 0.0190)")
    print(f"  m2 F1 = {data['m2']['f1']:.4f}  (Table 1 says 0.0183)")
    print(f"  m6 F1 = {data['m6']['f1']:.4f}  (Table 1 says 0.0241)")
    print(f"  m7 F1 = {data['m7']['f1']:.4f}  (Table 1 says 0.0233)")
    print(f"  m2 Opus 4.8 judge = {data['m2']['judge_opus48']:.3f}  (Table 1 says 2.24)")
    print(f"  m6 Opus 4.8 judge = {data['m6']['judge_opus48']:.3f}  (Table 1 says 2.41)")
    print(f"  m7 Opus 4.8 judge = {data['m7']['judge_opus48']:.3f}  (Table 1 says 2.45)")

    out_paths = [
        FIGURES_EMNLP / "fig4_method_comparison.pdf",
        FIGURES_EMNLP / "fig4_method_comparison.png",
        FIGURES_LOCAL / "fig4_method_comparison.pdf",
        FIGURES_LOCAL / "fig4_method_comparison.png",
    ]
    build_figure(data, out_paths)
    print("\nWrote:")
    for p in out_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
