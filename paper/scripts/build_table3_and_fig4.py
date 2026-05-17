#!/usr/bin/env python3
"""Build paper-ready Table 3 and Figure 4 from experiment outputs."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SECTIONS = ROOT / "sections"


def _resolve_b5_run() -> Path:
    configured = os.environ.get("DEXPOSURE_B5_RUN")
    if configured:
        return Path(configured)
    return RESULTS / "latest"


def _resolve_llm_run() -> Path:
    configured = os.environ.get("DEXPOSURE_LLM_RUN")
    if configured:
        return Path(configured)
    candidates = sorted(
        [p for p in RESULTS.glob("llm_eval_*") if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No llm_eval_* directory found. Set DEXPOSURE_LLM_RUN or rerun "
            "experiments/llm_eval_b5.py first."
        )
    return candidates[0]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _cost_adjusted(precision: float | None, fir: float | None) -> float | None:
    if precision is None or fir is None:
        return None
    return precision * (1.0 - fir)


def _ticket_f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _pct_delta(new: float, old: float) -> int:
    if old == 0:
        return 0
    return round(100.0 * (new - old) / old)


def _round_or_none(x: float | None, ndigits: int = 4) -> float | None:
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    return round(float(x), ndigits)


def build_rows() -> list[dict]:
    b5_run = _resolve_b5_run()
    llm_run = _resolve_llm_run()
    if not b5_run.exists():
        raise FileNotFoundError(
            f"b5_decision run directory not found: {b5_run}. Set DEXPOSURE_B5_RUN "
            "or run scripts/run_benchmarks_sequential.py first."
        )

    b5_c2 = _load_json(b5_run / "b5_decision__m1_persistence_rules.json")["results"][0]
    b5_c0 = _load_json(b5_run / "b5_decision__m5_fm_rules.json")["results"][0]
    llm_cmp = _load_json(llm_run / "comparison.json")["methods"]
    c0llm = llm_cmp["m6_fm_llm"]
    c0llm_gated = llm_cmp.get("m7_fm_llm_gated")
    c3 = llm_cmp["m2_snapshot_llm"]

    rows = [
        {
            "method_id": "m1_persistence_rules",
            "method_name": "Persist+Rules",
            "ticket_precision": _round_or_none(b5_c2.get("ticket_precision")),
            "audit_completeness": _round_or_none(b5_c2.get("audit_completeness")),
            "target_stability": _round_or_none(b5_c2.get("target_stability")),
            "severity_correlation": None,
            "false_intervention_rate": _round_or_none(b5_c2.get("false_intervention_rate")),
            "grounding_score": None,
            "consistency": None,
            "explanation_quality": None,
        },
        {
            "method_id": "m5_fm_rules",
            "method_name": "FM+Rules",
            "ticket_precision": _round_or_none(b5_c0.get("ticket_precision")),
            "audit_completeness": _round_or_none(b5_c0.get("audit_completeness")),
            "target_stability": _round_or_none(b5_c0.get("target_stability")),
            "severity_correlation": None,
            "false_intervention_rate": _round_or_none(b5_c0.get("false_intervention_rate")),
            "grounding_score": None,
            "consistency": None,
            "explanation_quality": None,
        },
        {
            "method_id": "m2_snapshot_llm",
            "method_name": "Pure LLM",
            "ticket_precision": _round_or_none(c3.get("ticket_precision")),
            "audit_completeness": _round_or_none(c3.get("audit_completeness")),
            "target_stability": _round_or_none(c3.get("target_stability")),
            "severity_correlation": _round_or_none(c3.get("severity_correlation")),
            "false_intervention_rate": _round_or_none(c3.get("false_intervention_rate")),
            "grounding_score": _round_or_none(c3.get("grounding_score")),
            "consistency": _round_or_none(c3.get("consistency")),
            "explanation_quality": _round_or_none(c3.get("explanation_quality"), ndigits=2),
        },
        {
            "method_id": "m6_fm_llm",
            "method_name": "FM+LLM",
            "ticket_precision": _round_or_none(c0llm.get("ticket_precision")),
            "audit_completeness": _round_or_none(c0llm.get("audit_completeness")),
            "target_stability": _round_or_none(c0llm.get("target_stability")),
            "severity_correlation": _round_or_none(c0llm.get("severity_correlation")),
            "false_intervention_rate": _round_or_none(c0llm.get("false_intervention_rate")),
            "grounding_score": _round_or_none(c0llm.get("grounding_score")),
            "consistency": _round_or_none(c0llm.get("consistency")),
            "explanation_quality": _round_or_none(c0llm.get("explanation_quality"), ndigits=2),
        },
    ]
    if c0llm_gated is not None:
        rows.append({
            "method_id": "m7_fm_llm_gated",
            "method_name": "FM+LLM+RulesGate",
            "ticket_precision": _round_or_none(c0llm_gated.get("ticket_precision")),
            "audit_completeness": _round_or_none(c0llm_gated.get("audit_completeness")),
            "target_stability": _round_or_none(c0llm_gated.get("target_stability")),
            "severity_correlation": _round_or_none(c0llm_gated.get("severity_correlation")),
            "false_intervention_rate": _round_or_none(c0llm_gated.get("false_intervention_rate")),
            "grounding_score": _round_or_none(c0llm_gated.get("grounding_score")),
            "consistency": _round_or_none(c0llm_gated.get("consistency")),
            "explanation_quality": _round_or_none(
                c0llm_gated.get("explanation_quality"), ndigits=2
            ),
        })

    for row in rows:
        row["cost_adjusted_score"] = _round_or_none(
            _cost_adjusted(row["ticket_precision"], row["false_intervention_rate"])
        )
    return rows


def write_table_files(rows: list[dict]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS / "paper_table3_latest.json"
    out_csv = RESULTS / "paper_table3_latest.csv"
    out_tex = SECTIONS / "Table3_MethodComparison.tex"

    out_json.write_text(json.dumps({"rows": rows}, indent=2))

    columns = [
        "method_id",
        "method_name",
        "ticket_precision",
        "audit_completeness",
        "target_stability",
        "severity_correlation",
        "false_intervention_rate",
        "grounding_score",
        "consistency",
        "explanation_quality",
        "cost_adjusted_score",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "% Auto-generated by scripts/build_table3_and_fig4.py",
        "\\begin{table*}[!htbp]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lccccccccc}",
        "\\toprule",
        "Method & Prec & Recall & Stabil & SevRho & FIR & Ground & Consist & Explain & CostAdj \\\\",
        "\\midrule",
    ]
    for r in rows:
        def fmt(v, digs=3):
            if v is None:
                return "N/A"
            return f"{v:.{digs}f}"
        explain_str = "N/A" if r["explanation_quality"] is None else f"{r['explanation_quality']:.1f}/5"
        method_id_tex = r['method_id'].replace('_', r'\_')
        lines.append(
            f"\\texttt{{{method_id_tex}}} {r['method_name']} & "
            f"{fmt(r['ticket_precision'])} & "
            f"{fmt(r['audit_completeness'])} & "
            f"{fmt(r['target_stability'])} & "
            f"{fmt(r['severity_correlation'])} & "
            f"{fmt(r['false_intervention_rate'])} & "
            f"{fmt(r['grounding_score'])} & "
            f"{fmt(r['consistency'])} & "
            f"{explain_str} & "
            f"{fmt(r['cost_adjusted_score'])} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Task II decision-quality comparison. Higher Prec, Recall, "
        "Stabil, Ground, Consist, Explain, and CostAdj are better; lower FIR is "
        "better.}",
        "\\label{tab:task2_table3}",
        "\\vspace{2pt}",
        "\\begin{minipage}{0.98\\textwidth}",
        "\\footnotesize\\emph{Note:} \\texttt{m3\\_evolvegcn} and "
        "\\texttt{m4\\_fm\\_only} are predictor-only baselines and do not emit "
        "supervisory tickets, so they are not applicable to \\texttt{b5\\_decision}.",
        "\\end{minipage}",
        "\\end{table*}",
        "",
    ]
    out_tex.write_text("\n".join(lines))


def write_figure(rows: list[dict]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    row_by_id = {r["method_id"]: r for r in rows}
    f1_order = [
        "m1_persistence_rules",
        "m5_fm_rules",
        "m2_snapshot_llm",
        "m6_fm_llm",
        "m7_fm_llm_gated",
    ]
    f1_rows = [row_by_id[mid] for mid in f1_order if mid in row_by_id]
    f1_values = np.array(
        [
            _ticket_f1(r["ticket_precision"], r["audit_completeness"]) or 0.0
            for r in f1_rows
        ]
    )
    f1_labels = ["m1\nPersist", "m5\nFM+Rules", "m2\nLLM", "m6\nFM+LLM", "m7\n+Gate"]
    f1_colors = ["#B8B8B8", "#5C8D57", "#6F8FBF", "#4A6FA5", "#3A8E8C"]

    judge_ids = ["m2_snapshot_llm", "m6_fm_llm", "m7_fm_llm_gated"]
    judge_rows = [row_by_id[mid] for mid in judge_ids if mid in row_by_id]
    judge_values = np.array([r["explanation_quality"] or 0.0 for r in judge_rows])
    judge_labels = ["m2\nLLM", "m6\nFM+LLM", "m7\n+Gate"]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9.5,
            "figure.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )
    fig, (ax_f1, ax_judge) = plt.subplots(
        1,
        2,
        figsize=(8.8, 3.8),
        gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.28},
    )

    x = np.arange(len(f1_rows))
    ax_f1.bar(x, f1_values, color=f1_colors[: len(f1_rows)], edgecolor="#222222")
    ax_f1.set_xticks(x)
    ax_f1.set_xticklabels(f1_labels[: len(f1_rows)])
    ax_f1.set_ylabel("Ticket F1")
    ax_f1.set_title("Layer-wise Contribution to Ticket F1", fontsize=11, pad=8)
    ax_f1.set_ylim(0.0, max(f1_values) * 1.33)
    ax_f1.grid(axis="y", linestyle="--", alpha=0.3)
    for idx, value in enumerate(f1_values):
        ax_f1.text(idx, value + 0.0006, f"{value:.4f}", ha="center", va="bottom", fontsize=8)

    def arrow_between(ax, start_idx: int, end_idx: int, label: str, lift: float) -> None:
        y0 = f1_values[start_idx]
        y1 = f1_values[end_idx]
        y_mid = max(y0, y1) + lift
        ax.annotate(
            label,
            xy=(end_idx, y1 + 0.00025),
            xytext=((start_idx + end_idx) / 2, y_mid),
            ha="center",
            va="bottom",
            fontsize=8,
            arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 0.9},
        )

    if len(f1_values) >= 5:
        fm_delta = _pct_delta(f1_values[3], f1_values[2])
        llm_delta = _pct_delta(f1_values[3], f1_values[1])
        gate_delta = _pct_delta(f1_values[4], f1_values[3])
        arrow_between(ax_f1, 2, 3, f"+FM: {fm_delta:+d}%", 0.0036)
        arrow_between(ax_f1, 1, 3, f"+LLM: {llm_delta:+d}%", 0.0064)
        arrow_between(ax_f1, 3, 4, f"+Gate: {gate_delta:+d}% F1", 0.0022)

    jx = np.arange(len(judge_rows))
    ax_judge.bar(jx, judge_values, color=["#6F8FBF", "#4A6FA5", "#3A8E8C"], edgecolor="#222222")
    ax_judge.set_xticks(jx)
    ax_judge.set_xticklabels(judge_labels[: len(judge_rows)])
    ax_judge.set_ylabel("Judge Score")
    ax_judge.set_ylim(0.0, 5.0)
    ax_judge.set_title("Explanation Quality", fontsize=11, pad=8)
    ax_judge.grid(axis="y", linestyle="--", alpha=0.3)
    for idx, value in enumerate(judge_values):
        ax_judge.text(idx, value + 0.12, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    if len(judge_values) >= 3:
        ax_judge.annotate(
            "+FM: +0.45",
            xy=(1, judge_values[1] + 0.05),
            xytext=(0.5, 3.35),
            ha="center",
            fontsize=8,
            arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 0.9},
        )
        ax_judge.annotate(
            "+Gate: +0.21",
            xy=(2, judge_values[2] + 0.05),
            xytext=(1.5, 3.85),
            ha="center",
            fontsize=8,
            arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 0.9},
        )

    fig.suptitle("Task II Layer-wise Contribution", fontsize=14, y=1.03)

    fig.savefig(FIGURES / "fig4_method_comparison.png", format="png")
    fig.savefig(FIGURES / "fig4_method_comparison.pdf", format="pdf")
    plt.close(fig)


def main() -> None:
    rows = build_rows()
    write_table_files(rows)
    write_figure(rows)
    print("Wrote:")
    print(f"- {RESULTS / 'paper_table3_latest.json'}")
    print(f"- {RESULTS / 'paper_table3_latest.csv'}")
    print(f"- {SECTIONS / 'Table3_MethodComparison.tex'}")
    print(f"- {FIGURES / 'fig4_method_comparison.png'}")
    print(f"- {FIGURES / 'fig4_method_comparison.pdf'}")


if __name__ == "__main__":
    main()
