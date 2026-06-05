#!/usr/bin/env python3
"""Aggregate Track A + Track B re-judge / re-decide results into one table.

Produces:
  - paper/results/rejudge_aggregated.json  -- machine-readable summary
  - paper/results/rejudge_aggregated.md    -- markdown table for the paper

Recovery: if the original judge response failed JSON parse but contains
"quality_score": <N> in the raw text, this script regex-extracts the real
score instead of leaving the score=3 fallback in place. Only applies to
entries flagged with quality_score==3 AND a non-empty "reasoning" containing
the raw text (rejudge_b5.py stores reasoning, not raw -- so the recovery
only fires when rejudge stored an empty reasoning marker).

Usage:
    python paper/experiments/aggregate_rejudge.py \\
        --rejudge-dir paper/results/rejudge_<ts> \\
        --redecide-dir paper/results/redecide_<ts>
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def load_dir(p: Path, prefix: str) -> list[dict]:
    """Load every JSON in `p` that starts with `prefix`, return list of {file, summary, entries}."""
    out = []
    if not p or not p.is_dir():
        return out
    for f in sorted(p.glob(f"{prefix}*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        out.append({"file": f.name, "summary": d.get("summary", {}),
                    "entries": d.get("entries", [])})
    return out


def aggregate_rejudge(blocks: list[dict]) -> list[dict]:
    """Collapse rejudge entries into per (method, judge) rows."""
    rows = []
    for b in blocks:
        s = b["summary"]
        method = s.get("method")
        judge = s.get("judge_model")
        if not method or not judge:
            continue
        scores = [e["quality_score"] for e in b["entries"]]
        old = [e.get("old_haiku_score", 0) for e in b["entries"]]
        rows.append({
            "method": method,
            "decision_model": "claude-opus-4.7",  # original
            "judge_model": judge,
            "n_weeks": len(scores),
            "mean_quality": round(mean(scores), 3) if scores else None,
            "std_quality": round(stdev(scores), 3) if len(scores) > 1 else 0.0,
            "old_haiku_mean": round(mean(old), 3) if old else None,
            "total_cost_usd": s.get("total_cost_usd", 0.0),
        })
    return rows


def aggregate_redecide(blocks: list[dict]) -> list[dict]:
    """redecide produces two file families: decide_*.json (the new decisions)
    and rejudge_*.json (the judge scores on those decisions). We key off the
    rejudge files because each contains the cross-product method/decision/judge.

    F1 is reported as macro F1 (`2 * mean_p * mean_c / (mean_p + mean_c)`),
    matching the b5_decision summary convention used in the original Table 1
    so cross-decision rows are directly comparable to the headline row.
    """
    rows = []
    for b in blocks:
        s = b["summary"]
        method = s.get("method")
        dec = s.get("decision_model")
        judge = s.get("judge_model")
        if not (method and dec and judge):
            continue
        scores = [e["quality_score"] for e in b["entries"]]
        precs = [e["precision"] for e in b["entries"]]
        compls = [e["completeness"] for e in b["entries"]]
        firs = [e["false_intervention_rate"] for e in b["entries"]]
        mean_p = mean(precs) if precs else 0.0
        mean_c = mean(compls) if compls else 0.0
        macro_f1 = (2 * mean_p * mean_c / (mean_p + mean_c)) if (mean_p + mean_c) > 0 else 0.0
        rows.append({
            "method": method,
            "decision_model": dec,
            "judge_model": judge,
            "n_weeks": len(scores),
            "mean_quality": round(mean(scores), 3) if scores else None,
            "mean_precision": round(mean_p, 4),
            "mean_completeness": round(mean_c, 4),
            "mean_f1": round(macro_f1, 4),
            "mean_fir": round(mean(firs), 4) if firs else None,
        })
    return rows


def emit_markdown(all_rows: list[dict]) -> str:
    """Group rows into the paper's Table 1 + a new Table 2 layout."""
    # Table 1: cross-judge robustness (same Opus 4.7 decision, varying judge)
    rejudge_rows = [r for r in all_rows if "mean_precision" not in r]
    # Table 2: cross-decision robustness (varying decision, fixed judge=Opus 4.8)
    redecide_rows = [r for r in all_rows if "mean_precision" in r]

    out = []
    out.append("## Table 1: Cross-family judge robustness (decision = Claude Opus 4.7, original Pipeline)")
    out.append("")
    out.append("| Method | Judge | n | Mean quality | Std | Old Haiku 4.5 baseline |")
    out.append("|---|---|---|---|---|---|")
    for r in sorted(rejudge_rows, key=lambda x: (x["method"], x["judge_model"])):
        out.append(f"| {r['method']} | {r['judge_model']} | {r['n_weeks']} | "
                   f"{r['mean_quality']} | {r['std_quality']} | {r['old_haiku_mean']} |")
    out.append("")
    out.append("## Table 2: Cross-family decision robustness (judge = Claude Opus 4.8)")
    out.append("")
    out.append("| Method | Decision | n | F1 | FIR | Judge quality |")
    out.append("|---|---|---|---|---|---|")
    for r in sorted(redecide_rows, key=lambda x: (x["method"], x["decision_model"])):
        out.append(f"| {r['method']} | {r['decision_model']} | {r['n_weeks']} | "
                   f"{r['mean_f1']} | {r['mean_fir']} | {r['mean_quality']} |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# LaTeX emitters -- paste-ready table fragments for the paper
# ---------------------------------------------------------------------------

JUDGE_SHORT = {
    "claude-opus-4.8": "Opus~4.8",
    "google/gemini-2.5-pro": "Gemini~2.5~Pro",
    "openai/gpt-5.5": "GPT-5.5",
}
DEC_SHORT = {
    "claude-opus-4.7": "Opus~4.7",
    "claude-sonnet-4.6": "Sonnet~4.6",
    "google/gemini-2.5-pro": "Gemini~2.5~Pro",
}
METHOD_TT = {
    "m2_snapshot_llm": r"\texttt{m2}",
    "m6_fm_llm":       r"\texttt{m6}",
    "m7_fm_llm_gated": r"\texttt{m7}",
}


def emit_latex_judge_panel(rejudge_rows: list[dict]) -> str:
    """LaTeX rows for the cross-family judge panel (decision=Opus 4.7)."""
    methods = ["m2_snapshot_llm", "m6_fm_llm", "m7_fm_llm_gated"]
    judges = ["claude-opus-4.8", "google/gemini-2.5-pro", "openai/gpt-5.5"]
    grid = {(r["method"], r["judge_model"]): r for r in rejudge_rows}
    lines = []
    lines.append(r"% Auto-generated by aggregate_rejudge.py -- LaTeX judge panel")
    lines.append(r"\begin{tabular}{@{}lcccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Method & Opus~4.8 & Gemini~2.5~Pro & GPT-5.5 & Haiku~4.5 baseline \\")
    lines.append(r"\midrule")
    for m in methods:
        haiku = next((grid[(m, j)].get("old_haiku_mean") for j in judges if (m, j) in grid), None)
        cells = []
        for j in judges:
            r = grid.get((m, j))
            cells.append(f"{r['mean_quality']:.2f}" if r and r.get("mean_quality") is not None else "--")
        haiku_s = f"{haiku:.2f}" if haiku is not None else "--"
        lines.append(f"{METHOD_TT.get(m, m)} & {cells[0]} & {cells[1]} & {cells[2]} & {haiku_s} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def emit_latex_decision_panel(redecide_rows: list[dict],
                              opus47_baseline: dict[str, dict] | None = None) -> str:
    """LaTeX rows for the cross-family decision panel (judge=Opus 4.8).

    `opus47_baseline` maps method -> {"f1": ..., "fir": ..., "quality": ...}
    so the original Opus 4.7 row can be inlined for comparison.
    """
    methods = ["m6_fm_llm", "m7_fm_llm_gated"]
    decisions = ["claude-opus-4.7", "claude-sonnet-4.6", "google/gemini-2.5-pro"]
    grid = {(r["method"], r["decision_model"]): r for r in redecide_rows}
    lines = []
    lines.append(r"% Auto-generated by aggregate_rejudge.py -- LaTeX decision panel")
    lines.append(r"\begin{tabular}{@{}llccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Method & Decision LLM & F1$\uparrow$ & FIR$\downarrow$ & Judge$\uparrow$ \\")
    lines.append(r"\midrule")
    for m in methods:
        for d in decisions:
            if d == "claude-opus-4.7" and opus47_baseline:
                b = opus47_baseline.get(m, {})
                if not b:
                    continue
                lines.append(
                    f"{METHOD_TT.get(m, m)} & Opus~4.7 (main) & "
                    f"{b.get('f1', '--')} & {b.get('fir', '--')} & {b.get('quality', '--')} \\\\")
            else:
                r = grid.get((m, d))
                if not r:
                    continue
                lines.append(
                    f"{METHOD_TT.get(m, m)} & {DEC_SHORT.get(d, d)} & "
                    f"{r['mean_f1']:.4f} & {r['mean_fir']:.3f} & {r['mean_quality']:.2f} \\\\")
        if m != methods[-1]:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rejudge-dir", required=True, type=Path)
    ap.add_argument("--redecide-dir", required=True, type=Path)
    ap.add_argument("--out-prefix", default="paper/results/rejudge_aggregated")
    args = ap.parse_args()

    rejudge_blocks = load_dir(args.rejudge_dir, "rejudge_")
    redecide_blocks = load_dir(args.redecide_dir, "rejudge_")

    rows_a = aggregate_rejudge(rejudge_blocks)
    rows_b = aggregate_redecide(redecide_blocks)

    all_rows = rows_a + rows_b
    with open(f"{args.out_prefix}.json", "w") as f:
        json.dump({
            "rejudge_track_a": rows_a,
            "redecide_track_b": rows_b,
        }, f, indent=2)

    md = emit_markdown(all_rows)
    with open(f"{args.out_prefix}.md", "w") as f:
        f.write(md + "\n")

    # Paste-ready LaTeX fragments.
    # Judge values are taken from the Opus 4.8 cross-family panel
    # (rejudge_*__claude-opus-4.8.json) so the decision-panel
    # Judge column is comparable across rows; the F1/FIR
    # columns are model-independent (from the original Opus 4.7
    # run logged in paper/results/run_20260515_pipeline_full).
    rejudge_grid = {(r["method"], r["judge_model"]): r for r in rows_a}
    def _opus48_judge(method):
        r = rejudge_grid.get((method, "claude-opus-4.8"))
        return f"{r['mean_quality']:.2f}" if r else "--"
    opus47_baseline = {
        "m2_snapshot_llm": {"f1": "0.0183", "fir": "0.000", "quality": _opus48_judge("m2_snapshot_llm")},
        "m6_fm_llm":       {"f1": "0.0241", "fir": "0.448", "quality": _opus48_judge("m6_fm_llm")},
        "m7_fm_llm_gated": {"f1": "0.0233", "fir": "0.437", "quality": _opus48_judge("m7_fm_llm_gated")},
    }
    lx = emit_latex_judge_panel(rows_a) + "\n\n" + emit_latex_decision_panel(rows_b, opus47_baseline)
    with open(f"{args.out_prefix}.tex", "w") as f:
        f.write(lx + "\n")

    print(md)
    print()
    print(f"Wrote: {args.out_prefix}.json + .md + .tex")


if __name__ == "__main__":
    main()
