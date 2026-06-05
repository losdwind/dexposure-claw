#!/usr/bin/env python3
"""Re-judge existing m6/m7/m2 LLM decision outputs with stronger/cross-family judges.

Background
----------
The original llm_eval_b5 run paired a strong decision LLM (Claude Opus 4.7)
with a much weaker, same-family judge (Claude Haiku 4.5). For EMNLP industry
review this triggers two methodology concerns:

  DA-C1 -- judge shares the same training family as the decision model
  Tier  -- judge is weaker than decision (judge >= decision is standard)

Both are addressed in one pass by re-judging the saved decision outputs with
a panel of stronger and cross-family judges. The decision outputs themselves
are unchanged; only the explanation-quality scores are recomputed.

This script is offline -- it only needs OPENROUTER_API_KEY and the saved
raw_<method>.json files. No FM API, no GPU.

Usage
-----
    export OPENROUTER_API_KEY=...
    python paper/experiments/rejudge_b5.py \
        --raw-dir paper/results/run_20260515_pipeline_full/llm_eval \
        --judge claude-opus-4-8 \
        --judge google/gemini-3.1-pro \
        --judge openai/gpt-5.5 \
        --method m6_fm_llm --method m7_fm_llm_gated --method m2_snapshot_llm

Output
------
    paper/results/rejudge_<timestamp>/
        rejudge_<method>__<judge>.json   per-week scores
        summary.json                      aggregated mean per (method, judge)
        rejudge.log
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from loguru import logger

# Reuse the prompt template + LLM caller from the main eval module so we
# stay byte-for-byte consistent with the original judging.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_eval_b5 as eval_mod  # noqa: E402


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------------------
# Parse user_prompt to recover the inputs the judge template needs
# ---------------------------------------------------------------------------

_NODES_EDGES_RE = re.compile(
    r"Nodes:\s*(\d+)\s+protocols?,\s*Edges:\s*(\d+)\s+weighted",
    re.IGNORECASE,
)
_METRIC_RE = re.compile(r"^\s*(M\d+_[a-z_]+):\s*([\-0-9.eE]+)\s*$", re.MULTILINE)


def parse_prompt_meta(user_prompt: str) -> dict:
    n_nodes = n_edges = 0
    m = _NODES_EDGES_RE.search(user_prompt)
    if m:
        n_nodes, n_edges = int(m.group(1)), int(m.group(2))
    metrics = {}
    for k, v in _METRIC_RE.findall(user_prompt):
        try:
            metrics[k] = float(v)
        except ValueError:
            pass
    return {"n_nodes": n_nodes, "n_edges": n_edges, "metrics": metrics}


def build_judge_prompt(date: str, horizon: int, meta: dict,
                       truly_stressed: list[str], llm_output: dict) -> tuple[str, str]:
    metrics_summary = ", ".join(f"{k}={v}" for k, v in meta["metrics"].items())
    user = eval_mod.JUDGE_USER_TEMPLATE.format(
        date=date,
        horizon=horizon,
        n_nodes=meta["n_nodes"],
        n_edges=meta["n_edges"],
        metrics_summary=metrics_summary or "n/a",
        n_stressed=len(truly_stressed),
        stressed_list=", ".join(sorted(truly_stressed)[:10]) or "none",
        risk_level=llm_output.get("risk_level", "?"),
        llm_targets=", ".join(p["protocol"] for p in llm_output.get("target_protocols", [])[:10]),
        rationale=llm_output.get("rationale", "N/A"),
    )
    return eval_mod.JUDGE_SYSTEM_PROMPT, user


# ---------------------------------------------------------------------------
# Per-judge run
# ---------------------------------------------------------------------------

def rejudge_method(raw_path: Path, method: str, judge_model: str,
                   horizon: int, out_dir: Path, resume: bool) -> dict:
    raw = json.load(open(raw_path))
    out_path = out_dir / f"rejudge_{method}__{judge_model.replace('/', '_')}.json"

    done: dict[str, dict] = {}
    if resume and out_path.exists():
        existing = json.load(open(out_path))
        for e in existing.get("entries", []):
            done[e["date"]] = e
        logger.info(f"  resume: {len(done)} weeks already scored for {method} x {judge_model}")

    entries: list[dict] = list(done.values())
    judge_costs = eval_mod.MODEL_COSTS.get(judge_model, {"input": 1.0, "output": 5.0})
    total_cost = 0.0
    n_calls = 0
    n_parse_fallback = 0

    # Default summary covers the all-weeks-already-done resume case so the
    # final return path always has a value.
    scores0 = [e["quality_score"] for e in entries]
    summary = {
        "method": method,
        "judge_model": judge_model,
        "n_weeks": len(entries),
        "mean_quality": round(sum(scores0) / len(scores0), 3) if scores0 else 0,
        "total_cost_usd": 0.0,
        "n_parse_fallback": 0,
        "resumed_complete": len(entries) == len(raw),
    }

    for wi, entry in enumerate(raw):
        date = entry["date"]
        if date in done:
            continue

        primary = entry["llm_outputs"][0]
        meta = parse_prompt_meta(entry["user_prompt"])
        system, user = build_judge_prompt(
            date=date, horizon=horizon, meta=meta,
            truly_stressed=entry["truly_stressed"],
            llm_output=primary,
        )

        # Force LLM_API_KEY refresh in case it was set after module import
        eval_mod.LLM_API_KEY = os.environ.get(
            "OPENROUTER_API_KEY", os.environ.get("ANTHROPIC_API_KEY", "")
        )
        if not eval_mod.LLM_API_KEY:
            logger.error("OPENROUTER_API_KEY not set; aborting")
            sys.exit(1)

        result = eval_mod.call_llm(system, user, model=judge_model)
        score = result.get("quality_score", 0)
        if not isinstance(score, (int, float)) or not (1 <= score <= 5):
            score = 3
            n_parse_fallback += 1

        cost = (result.get("input_tokens", 0) * judge_costs["input"]
                + result.get("output_tokens", 0) * judge_costs["output"]) / 1_000_000
        total_cost += cost
        n_calls += 1

        rec = {
            "date": date,
            "method": method,
            "judge_model": judge_model,
            "quality_score": int(score),
            "reasoning": result.get("reasoning", ""),
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            "old_haiku_score": entry.get("explanation_quality", 0),
            "decision_risk_level": primary.get("risk_level", "?"),
            "n_targets": len(primary.get("target_protocols", [])),
        }
        entries.append(rec)
        done[date] = rec
        logger.info(f"  [{wi+1}/{len(raw)}] {date}: score={int(score)} "
                    f"(old_haiku={entry.get('explanation_quality', 0)}) "
                    f"cost=${cost:.4f}")

        # Checkpoint after each week
        scores = [e["quality_score"] for e in entries]
        summary = {
            "method": method,
            "judge_model": judge_model,
            "n_weeks": len(entries),
            "mean_quality": round(sum(scores) / len(scores), 3) if scores else 0,
            "total_cost_usd": round(total_cost, 4),
            "n_parse_fallback": n_parse_fallback,
        }
        with open(out_path, "w") as f:
            json.dump({"summary": summary, "entries": entries}, f, indent=2)

    return summary if entries else {"method": method, "judge_model": judge_model, "n_weeks": 0}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Re-judge existing LLM decisions with new judges")
    ap.add_argument("--raw-dir", required=True,
                    help="Directory containing raw_<method>.json files")
    ap.add_argument("--method", action="append", default=[],
                    help="m6_fm_llm / m7_fm_llm_gated / m2_snapshot_llm (repeatable)")
    ap.add_argument("--judge", action="append", default=[], dest="judges",
                    help="Judge model name (repeatable)")
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--out-dir", default=None,
                    help="Override output directory (default: paper/results/rejudge_<ts>)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if not args.method:
        args.method = ["m6_fm_llm", "m7_fm_llm_gated", "m2_snapshot_llm"]
    if not args.judges:
        # OpenRouter slugs as of 2026-06. Anthropic uses dots, not dashes.
        # Cross-family judges are auto-set to reasoning effort "minimal" via
        # llm_eval_b5._is_reasoning_model so judge thinking budget matches
        # the Anthropic baseline (which has no extended thinking by default).
        # Gemini 3.1 Pro Preview was dropped: with effort=minimal it returns
        # empty content; default reasoning would confound the comparison.
        args.judges = [
            "claude-opus-4.8",        # same-family upgrade (tier > Opus 4.7)
            "google/gemini-2.5-pro",  # cross-family, reasoning forced minimal
            "openai/gpt-5.5",         # cross-family, reasoning forced minimal
        ]

    raw_dir = Path(args.raw_dir).resolve()
    if not raw_dir.is_dir():
        logger.error(f"raw-dir does not exist: {raw_dir}")
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else (
        RESULTS_DIR / f"rejudge_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / "rejudge.log"
    logger.add(str(log_file), level="DEBUG",
               format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}")

    logger.info(f"raw_dir: {raw_dir}")
    logger.info(f"out_dir: {out_dir}")
    logger.info(f"methods: {args.method}")
    logger.info(f"judges:  {args.judges}")

    grid: dict[tuple[str, str], dict] = {}
    for method in args.method:
        raw_path = raw_dir / f"raw_{method}.json"
        if not raw_path.exists():
            logger.warning(f"missing {raw_path}, skip")
            continue
        for judge in args.judges:
            logger.info(f"--- {method} x {judge} ---")
            try:
                summary = rejudge_method(
                    raw_path=raw_path,
                    method=method,
                    judge_model=judge,
                    horizon=args.horizon,
                    out_dir=out_dir,
                    resume=args.resume,
                )
            except SystemExit:
                raise
            except Exception as e:
                logger.exception(f"failed: {method} x {judge}: {e}")
                summary = {"method": method, "judge_model": judge, "error": str(e)}
            grid[(method, judge)] = summary

    # Aggregated comparison table
    table = []
    methods_seen = sorted({k[0] for k in grid})
    judges_seen = sorted({k[1] for k in grid})
    for m in methods_seen:
        row = {"method": m}
        for j in judges_seen:
            s = grid.get((m, j), {})
            row[j] = s.get("mean_quality", None)
        table.append(row)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"grid": [
            {"method": m, "judge": j, **grid[(m, j)]} for (m, j) in grid
        ], "table": table, "judges": judges_seen, "methods": methods_seen},
            f, indent=2)

    # Pretty-print
    print("\n=== Re-judge mean explanation quality (1-5) ===")
    hdr = f"{'method':<22} " + " ".join(f"{j[:18]:>20}" for j in judges_seen)
    print(hdr)
    print("-" * len(hdr))
    for row in table:
        cells = " ".join(
            f"{(row.get(j) if row.get(j) is not None else float('nan')):>20.2f}"
            for j in judges_seen
        )
        print(f"{row['method']:<22} {cells}")

    logger.info(f"All done. Output: {out_dir}")


if __name__ == "__main__":
    main()
