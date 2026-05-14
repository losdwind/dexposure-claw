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
        "\\begin{table}[t]",
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
        lines.append(
            f"{r['method_id']} {r['method_name']} & "
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
        "\\caption{Task II method comparison (Table 3).}",
        "\\label{tab:task2_table3}",
        "\\end{table}",
        "",
    ]
    out_tex.write_text("\n".join(lines))


def write_figure(rows: list[dict]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    methods = [r["method_id"] for r in rows]

    precision = [r["ticket_precision"] or 0.0 for r in rows]
    stability = [r["target_stability"] or 0.0 for r in rows]
    cost_adj = [r["cost_adjusted_score"] or 0.0 for r in rows]

    x = np.arange(len(methods))
    w = 0.24

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "figure.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )
    fig, ax = plt.subplots(figsize=(8.6, 4.4))

    ax.bar(x - w, precision, width=w, label="Ticket Precision", color="#4A6FA5")
    ax.bar(x, stability, width=w, label="Target Stability", color="#3A6B4E")
    ax.bar(x + w, cost_adj, width=w, label="Cost-Adjusted", color="#C5A55A")

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Task II Method Comparison (Key Metrics)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", frameon=False)

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
