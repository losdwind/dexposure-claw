#!/usr/bin/env python3
"""Build additional data-driven paper figures from paper/results.

Every value plotted here is loaded from JSON artifacts under paper/results.
The script intentionally avoids inline experimental constants except for
display labels and plotting style.

Run from the repository root:
    python3 paper/scripts/build_extra_result_figures.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RUN = RESULTS / "run_20260515_pipeline_full"
ABLATION_RUN = RESULTS / "run_20260517_161248_ablation_gpu"

METHOD_LABELS = {
    "m1_persistence_rules": "m1 Persist",
    "m3_evolvegcn": "m3 EvolveGCN",
    "m4_fm_only": "m4 FM-only",
    "m5_fm_rules": "m5 FM+Rules",
    "m2_snapshot_llm": "m2 LLM",
    "m6_fm_llm": "m6 FM+LLM",
    "m7_fm_llm_gated": "m7 FM+LLM+Gate",
}

SHORT_LABELS = {
    "m1_persistence_rules": "Persist",
    "m3_evolvegcn": "EvolveGCN",
    "m4_fm_only": "FM-only",
    "m5_fm_rules": "FM+Rules",
    "m2_snapshot_llm": "LLM",
    "m6_fm_llm": "FM+LLM",
    "m7_fm_llm_gated": "Gate",
}

COLORS = {
    "m1_persistence_rules": "#9AA0A6",
    "m3_evolvegcn": "#CC79A7",
    "m4_fm_only": "#56B4E9",
    "m5_fm_rules": "#0072B2",
    "m2_snapshot_llm": "#56B4E9",
    "m6_fm_llm": "#0072B2",
    "m7_fm_llm_gated": "#D55E00",
    "sonnet": "#009E73",
    "gemini": "#E69F00",
}

INK = "#263238"
MUTED = "#607D8B"
GRID = "#D7DEE2"


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(), parse_constant=lambda _: float("nan"))


def result_rows(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    rows = data.get("results")
    if rows is None:
        raise KeyError(f"{path} has no 'results' field")
    return rows


def write_figure(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES / f"{name}.pdf"
    png = FIGURES / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 7.7,
        "axes.titlesize": 8.4,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.7,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.045,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.8,
    })


def b1_paths() -> dict[str, Path]:
    return {
        "m1_persistence_rules": RUN / "main" / "b1_forecast__m1_persistence_rules.json",
        "m3_evolvegcn": RUN / "rerun" / "b1_forecast__m3_evolvegcn.json",
        "m4_fm_only": RUN / "rerun" / "b1_forecast__m4_fm_only.json",
        "m5_fm_rules": RUN / "rerun" / "b1_forecast__m5_fm_rules.json",
    }


def b4_paths() -> dict[str, Path]:
    return {
        "m1_persistence_rules": RUN / "main" / "b4_stress__m1_persistence_rules.json",
        "m3_evolvegcn": RUN / "rerun" / "b4_stress__m3_evolvegcn.json",
        "m4_fm_only": RUN / "rerun" / "b4_stress__m4_fm_only.json",
        "m5_fm_rules": RUN / "rerun" / "b4_stress__m5_fm_rules.json",
    }


def b6_paths() -> dict[str, Path]:
    return {
        "m1_persistence_rules": RUN / "main" / "b6_robustness__m1_persistence_rules.json",
        "m3_evolvegcn": RUN / "rerun" / "b6_robustness__m3_evolvegcn.json",
        "m4_fm_only": RUN / "rerun" / "b6_robustness__m4_fm_only.json",
        "m5_fm_rules": RUN / "rerun" / "b6_robustness__m5_fm_rules.json",
    }


def set_panel_title(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, loc="left", fontweight="bold", color=INK, pad=4)


def build_fig_b1_horizon_diagnostics() -> None:
    sources = b1_paths()
    metrics = [
        ("pagerank_mae", "PageRank MAE ($10^{-5}$)", lambda x: x * 1e5),
        ("rank_correlation", "Rank correlation", float),
        ("trend_consistency", "Trend consistency", float),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.15), sharex=True)
    for ax, (metric, ylabel, transform) in zip(axes, metrics):
        for method, path in sources.items():
            rows = sorted(result_rows(path), key=lambda r: int(r["horizon"]))
            horizons = [int(r["horizon"]) for r in rows]
            values = [transform(float(r[metric])) for r in rows]
            ax.plot(
                horizons, values, marker="o", linewidth=1.35, markersize=3.8,
                color=COLORS[method], label=SHORT_LABELS[method],
            )
        set_panel_title(ax, ylabel)
        ax.set_xlabel("Forecast horizon (weeks)")
        ax.set_xticks([1, 4, 8, 12])
        ax.tick_params(colors=INK)
        if metric == "pagerank_mae":
            ax.set_ylabel("lower better")
        else:
            ax.set_ylabel("higher better")
        if metric == "trend_consistency":
            ax.set_ylim(-0.03, 0.72)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.subplots_adjust(top=0.78, wspace=0.30)
    write_figure(fig, "fig6_b1_horizon_diagnostics")


def build_fig_b6_robustness_heatmap() -> None:
    methods = list(b6_paths())
    regimes = [
        "low_data_10pct",
        "low_data_25pct",
        "partial_graph_30",
        "noisy_features_01",
        "missing_features_20",
    ]
    matrix = np.zeros((len(methods), len(regimes)))
    for i, method in enumerate(methods):
        rows = {r["regime"]: r for r in result_rows(b6_paths()[method])}
        for j, regime in enumerate(regimes):
            matrix[i, j] = float(rows[regime]["relative_degradation"])

    fig, ax = plt.subplots(figsize=(3.45, 2.65))
    vmax = max(0.8, float(np.nanmax(np.abs(matrix))))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(regimes)))
    ax.set_xticklabels(["10% data", "25% data", "30% graph", "noise", "missing"], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([SHORT_LABELS[m] for m in methods])
    set_panel_title(ax, "b6 relative degradation")
    ax.grid(False)
    for i in range(len(methods)):
        for j in range(len(regimes)):
            value = matrix[i, j]
            color = "white" if abs(value) > 0.45 else INK
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=6.8, color=color)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    cbar.ax.set_ylabel("signed degradation", rotation=90, labelpad=6)
    write_figure(fig, "fig7_b6_robustness_heatmap")


def build_fig_b4_stress_fidelity() -> None:
    methods = list(b4_paths())
    scenarios = ["S1", "S2", "S3", "S4", "S5"]
    x = np.arange(len(scenarios))
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65), sharex=True)
    metric_specs = [
        ("loss_mae", "Loss MAE", "lower better"),
        ("target_overlap_at_k", "Target overlap@10", "higher better"),
    ]
    for ax, (metric, title, label) in zip(axes, metric_specs):
        for k, method in enumerate(methods):
            rows = {r["scenario"]: r for r in result_rows(b4_paths()[method])}
            values = [float(rows[s][metric]) for s in scenarios]
            offset = (k - (len(methods) - 1) / 2.0) * width
            ax.bar(
                x + offset, values, width=width * 0.92,
                color=COLORS[method], edgecolor="white", linewidth=0.45,
                label=SHORT_LABELS[method],
            )
        set_panel_title(ax, title)
        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.subplots_adjust(top=0.78, wspace=0.30)
    write_figure(fig, "fig8_b4_stress_fidelity")


def build_fig_b2_warning_budget() -> None:
    rows = result_rows(RUN / "main" / "b2_warning__h1_weighted_degree.json")
    events = ["terra_luna", "ftx", "svb_usdc"]
    labels = {"terra_luna": "Terra/Luna", "ftx": "FTX", "svb_usdc": "SVB/USDC"}
    colors = {"terra_luna": "#D55E00", "ftx": "#0072B2", "svb_usdc": "#009E73"}
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.2), sharex=True)
    for event in events:
        sub = sorted([r for r in rows if r["stress_event"] == event], key=lambda r: int(r["alert_budget"]))
        budgets = [int(r["alert_budget"]) for r in sub]
        precision = [float(r["precision"]) for r in sub]
        f1_scaled = [100.0 * float(r["f1_warning"]) for r in sub]
        axes[0].plot(budgets, precision, marker="o", color=colors[event], linewidth=1.25, markersize=3.4, label=labels[event])
        axes[1].plot(budgets, f1_scaled, marker="s", color=colors[event], linewidth=1.25, markersize=3.3, label=labels[event])
    set_panel_title(axes[0], "Precision")
    set_panel_title(axes[1], "F1-warning ($10^2$)")
    for ax in axes:
        ax.set_xlabel("Alert budget K")
        ax.set_xticks([5, 10, 20])
        ax.tick_params(colors=INK)
    axes[0].set_ylim(0.45, 1.05)
    axes[1].set_ylim(0.0, 1.75)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.06))
    fig.subplots_adjust(top=0.76, wspace=0.28)
    write_figure(fig, "fig9_b2_warning_budget")


def _macro_f1(precision: float, completeness: float) -> float:
    return 0.0 if precision + completeness == 0 else 2 * precision * completeness / (precision + completeness)


def build_fig_judge_decision_surface() -> None:
    agg = load_json(RESULTS / "rejudge_aggregated.json")
    llm_eval = RUN / "llm_eval"
    base_points = []
    for method in ["m6_fm_llm", "m7_fm_llm_gated"]:
        summary = load_json(llm_eval / f"summary_{method}.json")
        f1 = _macro_f1(float(summary["ticket_precision"]), float(summary["audit_completeness"]))
        fir = float(summary["false_intervention_rate"])
        quality = next(
            float(r["mean_quality"])
            for r in agg["rejudge_track_a"]
            if r["method"] == method and r["judge_model"] == "claude-opus-4.8"
        )
        base_points.append({
            "label": f"{SHORT_LABELS[method]}\nOpus 4.7",
            "method": method,
            "decision": "opus",
            "f1": f1,
            "fir": fir,
            "quality": quality,
        })
    redecide_points = []
    for row in agg["redecide_track_b"]:
        decision = row["decision_model"]
        method = row["method"]
        label = "Sonnet 4.6" if "sonnet" in decision else "Gemini 2.5"
        redecide_points.append({
            "label": f"{SHORT_LABELS[method]}\n{label}",
            "method": method,
            "decision": "sonnet" if "sonnet" in decision else "gemini",
            "f1": float(row["mean_f1"]),
            "fir": float(row["mean_fir"]),
            "quality": float(row["mean_quality"]),
        })

    fig, ax_scatter = plt.subplots(1, 1, figsize=(3.45, 2.55))
    offsets = {
        ("m6_fm_llm", "opus"): (-30, 10),
        ("m7_fm_llm_gated", "opus"): (-30, -12),
        ("m6_fm_llm", "sonnet"): (10, -11),
        ("m7_fm_llm_gated", "sonnet"): (10, 12),
        ("m6_fm_llm", "gemini"): (10, -12),
        ("m7_fm_llm_gated", "gemini"): (10, 11),
    }
    for point in base_points + redecide_points:
        marker = "o" if point["method"] == "m6_fm_llm" else "s"
        face = COLORS["sonnet"] if point["decision"] == "sonnet" else COLORS["gemini"] if point["decision"] == "gemini" else COLORS[point["method"]]
        ax_scatter.scatter(
            point["fir"], point["f1"], s=55 + 30 * point["quality"],
            marker=marker, color=face, edgecolor="white", linewidth=0.6, zorder=3,
        )
        dx, dy = offsets[(point["method"], point["decision"])]
        ax_scatter.annotate(
            point["label"],
            xy=(point["fir"], point["f1"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=5.9,
            ha="left" if dx > 0 else "right",
            va="center",
            color=INK,
            arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.45},
        )
    set_panel_title(ax_scatter, "Decision LLM operating point")
    ax_scatter.set_xlabel("FIR (lower better)")
    ax_scatter.set_ylabel("Ticket F1 (higher better)")
    ax_scatter.set_xlim(-0.02, 0.50)
    ax_scatter.set_ylim(0.010, 0.032)
    ax_scatter.grid(True)
    write_figure(fig, "fig10_judge_decision_surface")


def build_fig_ablation_reserve() -> None:
    a1_rows = load_json(ABLATION_RUN / "run_a1_isolated_results" / "a1_isolated.json")
    a6_rows = load_json(ABLATION_RUN / "run_supplementary_ablations_results" / "summary.json")["a6_crisis"]

    regimes = ["extreme", "extreme_topo", "extreme_feat", "severe"]
    labels = ["both", "topology", "features", "severe"]
    split_rows = [r for r in a1_rows if r["test_split"] == "2025-01~2025-08"]
    by_key = {(r["regime"], r["label"]): r for r in split_rows}

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65))
    x = np.arange(len(regimes))
    off = [float(by_key[(r, "gate_off")]["fir"]) for r in regimes]
    strict = [float(by_key[(r, "gate_on_strict")]["fir"]) for r in regimes]
    safe = [100.0 * float(by_key[(r, "gate_on_strict")]["safe_mode_pct"]) for r in regimes]
    axes[0].bar(x - 0.18, off, width=0.34, color="#D55E00", edgecolor="white", label="A1 off")
    axes[0].bar(x + 0.18, strict, width=0.34, color="#009E73", edgecolor="white", label="A1 strict")
    for xi, pct in zip(x, safe):
        axes[0].text(xi + 0.18, 0.025, f"{pct:.0f}% safe", rotation=90, ha="center", va="bottom", fontsize=6.3, color=MUTED)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("FIR")
    axes[0].set_ylim(0, max(off) * 1.18)
    set_panel_title(axes[0], "A1 data-health gate under degradation")
    axes[0].legend(loc="upper left")

    crisis_order = ["terra_luna", "ftx", "svb"]
    crisis_labels = ["Terra/Luna", "FTX", "SVB"]
    single = []
    multi = []
    precision = []
    for crisis in crisis_order:
        c_rows = [r for r in a6_rows if r["crisis_period"] == crisis]
        single_row = next(r for r in c_rows if r["horizons"] == [4])
        multi_row = next(r for r in c_rows if len(r["horizons"]) > 1)
        single.append(int(single_row["n_alerts_total"]))
        multi.append(int(multi_row["n_alerts_total"]))
        precision.append(float(multi_row["ticket_precision"]))
    x2 = np.arange(len(crisis_order))
    axes[1].bar(x2 - 0.18, single, width=0.34, color="#9AA0A6", edgecolor="white", label="h=4")
    axes[1].bar(x2 + 0.18, multi, width=0.34, color="#0072B2", edgecolor="white", label="multi-horizon")
    for xi, p in zip(x2, precision):
        axes[1].text(xi + 0.18, max(multi) * 0.06, f"prec {p:.2f}", rotation=90, ha="center", va="bottom", fontsize=6.3, color="white")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(crisis_labels)
    axes[1].set_ylabel("Total alerts")
    axes[1].set_ylim(0, max(multi) * 1.18)
    set_panel_title(axes[1], "A6 horizon reserve in crisis windows")
    axes[1].legend(loc="upper left")
    write_figure(fig, "fig11_ablation_reserve")


def main() -> None:
    configure_matplotlib()
    build_fig_b1_horizon_diagnostics()
    build_fig_b6_robustness_heatmap()
    build_fig_b4_stress_fidelity()
    build_fig_b2_warning_budget()
    build_fig_judge_decision_surface()
    build_fig_ablation_reserve()


if __name__ == "__main__":
    main()
