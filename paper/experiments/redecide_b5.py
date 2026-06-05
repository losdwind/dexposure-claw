#!/usr/bin/env python3
"""Re-decide existing FM-conditioned LLM prompts with new decision LLMs.

Background
----------
Track A (rejudge_b5.py) swaps only the judge model while keeping the original
Claude Opus 4.7 decisions. Track B answers a different question: would a
weaker but tier-matched-to-judge decision model (Claude Sonnet 4.6) produce
materially different supervisory tickets given the SAME FM forecast inputs?
And do non-Anthropic decision models (GPT-5, Gemini 2.5 Pro) reproduce the
same Pareto operating point?

This script reuses the saved system_prompt + user_prompt from the original
run, so the FM forecast is held identical across decision models. Only the
decision LLM changes. After re-deciding, the script also re-judges with the
configured judge so each new (decision, judge) cell gets a complete row.

Like rejudge_b5.py, this runs purely against the OpenRouter API -- no FM
service, no GPU server.

Usage
-----
    export OPENROUTER_API_KEY=...
    python paper/experiments/redecide_b5.py \\
        --raw-dir paper/results/run_20260515_pipeline_full/llm_eval \\
        --decision claude-sonnet-4-6 \\
        --decision google/gemini-2.5-pro \\
        --judge claude-opus-4-8 \\
        --method m6_fm_llm --method m7_fm_llm_gated

Output
------
    paper/results/redecide_<timestamp>/
        decide_<method>__<decision>.json     per-week decisions
        rejudge_<method>__<decision>__<judge>.json   per-week judge scores
        summary.json                           aggregated mean per cell
        redecide.log
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_eval_b5 as eval_mod  # noqa: E402
import rejudge_b5 as judge_mod  # noqa: E402


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
GATED_LLM_METHODS = {"m7_fm_llm_gated"}


def _refresh_key():
    eval_mod.LLM_API_KEY = os.environ.get(
        "OPENROUTER_API_KEY", os.environ.get("ANTHROPIC_API_KEY", "")
    )
    if not eval_mod.LLM_API_KEY:
        logger.error("OPENROUTER_API_KEY not set")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Decide-then-judge per method
# ---------------------------------------------------------------------------

def redecide_and_judge(
    raw_path: Path,
    method: str,
    decision_model: str,
    judge_model: str,
    consistency_runs: int,
    horizon: int,
    out_dir: Path,
    resume: bool,
) -> dict:
    raw = json.load(open(raw_path))
    safe_dec = decision_model.replace("/", "_")
    safe_judge = judge_model.replace("/", "_")
    decide_path = out_dir / f"decide_{method}__{safe_dec}.json"
    judge_path = out_dir / f"rejudge_{method}__{safe_dec}__{safe_judge}.json"

    decide_done: dict[str, dict] = {}
    judge_done: dict[str, dict] = {}
    if resume:
        if decide_path.exists():
            for e in json.load(open(decide_path)).get("entries", []):
                decide_done[e["date"]] = e
        if judge_path.exists():
            for e in json.load(open(judge_path)).get("entries", []):
                judge_done[e["date"]] = e
        if decide_done or judge_done:
            logger.info(f"  resume: decide={len(decide_done)} judge={len(judge_done)}")

    decide_entries: list[dict] = list(decide_done.values())
    judge_entries: list[dict] = list(judge_done.values())
    dec_costs = eval_mod.MODEL_COSTS.get(decision_model, {"input": 3.0, "output": 15.0})
    jud_costs = eval_mod.MODEL_COSTS.get(judge_model, {"input": 1.0, "output": 5.0})
    total_dec_cost = 0.0
    total_jud_cost = 0.0

    for wi, entry in enumerate(raw):
        date = entry["date"]
        system_prompt = entry["system_prompt"]
        user_prompt = entry["user_prompt"]
        truly_stressed = entry["truly_stressed"]

        # --------- Decision ---------
        if date not in decide_done:
            _refresh_key()
            run_outputs = []
            for ri in range(consistency_runs):
                out = eval_mod.call_llm(system_prompt, user_prompt, model=decision_model)
                if method in GATED_LLM_METHODS:
                    out = eval_mod.apply_action_gate(out)
                run_outputs.append(out)
                total_dec_cost += (
                    out.get("input_tokens", 0) * dec_costs["input"]
                    + out.get("output_tokens", 0) * dec_costs["output"]
                ) / 1_000_000

            primary = run_outputs[0]
            actual_losses = {}  # unavailable at this stage; FIR uses targets vs truly_stressed set
            ev = eval_mod.assess_week(primary, set(truly_stressed), actual_losses, user_prompt)
            consistency = eval_mod.compute_consistency(run_outputs)

            decide_rec = {
                "date": date,
                "method": method,
                "decision_model": decision_model,
                "llm_outputs": run_outputs,
                "assessment": ev,
                "consistency": consistency,
                "truly_stressed_count": len(truly_stressed),
            }
            decide_entries.append(decide_rec)
            decide_done[date] = decide_rec
            self_assessment = ev
            self_primary = primary
        else:
            decide_rec = decide_done[date]
            self_primary = decide_rec["llm_outputs"][0]
            self_assessment = decide_rec["assessment"]

        # --------- Judge ---------
        if date not in judge_done:
            meta = judge_mod.parse_prompt_meta(user_prompt)
            j_sys, j_user = judge_mod.build_judge_prompt(
                date=date, horizon=horizon, meta=meta,
                truly_stressed=truly_stressed,
                llm_output=self_primary,
            )
            _refresh_key()
            j_result = eval_mod.call_llm(j_sys, j_user, model=judge_model)
            score = j_result.get("quality_score", 0)
            if not isinstance(score, (int, float)) or not (1 <= score <= 5):
                score = 3
            total_jud_cost += (
                j_result.get("input_tokens", 0) * jud_costs["input"]
                + j_result.get("output_tokens", 0) * jud_costs["output"]
            ) / 1_000_000

            judge_rec = {
                "date": date,
                "method": method,
                "decision_model": decision_model,
                "judge_model": judge_model,
                "quality_score": int(score),
                "reasoning": j_result.get("reasoning", ""),
                "precision": self_assessment.get("precision", 0.0),
                "completeness": self_assessment.get("completeness", 0.0),
                "false_intervention_rate": self_assessment.get("false_intervention_rate", 0.0),
                "n_targets": len(self_primary.get("target_protocols", [])),
            }
            judge_entries.append(judge_rec)
            judge_done[date] = judge_rec

        # Checkpoint after each week
        with open(decide_path, "w") as f:
            json.dump({
                "summary": {
                    "method": method,
                    "decision_model": decision_model,
                    "n_weeks": len(decide_entries),
                    "total_decision_cost_usd": round(total_dec_cost, 4),
                },
                "entries": decide_entries,
            }, f, indent=2)
        scores = [e["quality_score"] for e in judge_entries]
        with open(judge_path, "w") as f:
            json.dump({
                "summary": {
                    "method": method,
                    "decision_model": decision_model,
                    "judge_model": judge_model,
                    "n_weeks": len(judge_entries),
                    "mean_quality": round(sum(scores) / len(scores), 3) if scores else 0,
                    "total_judge_cost_usd": round(total_jud_cost, 4),
                },
                "entries": judge_entries,
            }, f, indent=2)

        if (wi + 1) % 5 == 0 or wi == len(raw) - 1:
            logger.info(
                f"  [{wi+1}/{len(raw)}] {date}: dec={decision_model[:20]} jud={judge_model[:20]} "
                f"score={judge_done[date]['quality_score']} "
                f"prec={self_assessment.get('precision', 0):.3f} "
                f"FIR={self_assessment.get('false_intervention_rate', 0):.3f}"
            )

    return {
        "method": method,
        "decision_model": decision_model,
        "judge_model": judge_model,
        "n_weeks": len(judge_entries),
        "mean_quality": round(sum(e["quality_score"] for e in judge_entries) / max(1, len(judge_entries)), 3),
        "mean_precision": round(sum(e["precision"] for e in judge_entries) / max(1, len(judge_entries)), 4),
        "mean_completeness": round(sum(e["completeness"] for e in judge_entries) / max(1, len(judge_entries)), 4),
        "mean_fir": round(sum(e["false_intervention_rate"] for e in judge_entries) / max(1, len(judge_entries)), 4),
        "decision_cost_usd": round(total_dec_cost, 4),
        "judge_cost_usd": round(total_jud_cost, 4),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Re-decide saved FM prompts with new decision LLMs")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--method", action="append", default=[],
                    help="m6_fm_llm / m7_fm_llm_gated / m2_snapshot_llm (repeatable)")
    ap.add_argument("--decision", action="append", default=[], dest="decisions",
                    help="Decision model (repeatable). Default: Sonnet 4.6 + Gemini 2.5 Pro")
    ap.add_argument("--judge", default="claude-opus-4.8",
                    help="Judge model used for ALL decision models")
    ap.add_argument("--consistency-runs", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if not args.method:
        args.method = ["m6_fm_llm", "m7_fm_llm_gated"]
    if not args.decisions:
        # OpenRouter slugs (Anthropic uses dots, not dashes)
        args.decisions = [
            "claude-sonnet-4.6",      # tier-matched-to-judge primary
            "google/gemini-2.5-pro",  # cross-family decision
        ]

    raw_dir = Path(args.raw_dir).resolve()
    if not raw_dir.is_dir():
        logger.error(f"raw-dir does not exist: {raw_dir}")
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else (
        RESULTS_DIR / f"redecide_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / "redecide.log"
    logger.add(str(log_file), level="DEBUG",
               format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}")

    logger.info(f"raw_dir:   {raw_dir}")
    logger.info(f"out_dir:   {out_dir}")
    logger.info(f"methods:   {args.method}")
    logger.info(f"decisions: {args.decisions}")
    logger.info(f"judge:     {args.judge}")

    grid: dict[tuple[str, str, str], dict] = {}
    for method in args.method:
        raw_path = raw_dir / f"raw_{method}.json"
        if not raw_path.exists():
            logger.warning(f"missing {raw_path}, skip")
            continue
        for dec in args.decisions:
            logger.info(f"--- {method} | decision={dec} | judge={args.judge} ---")
            try:
                summary = redecide_and_judge(
                    raw_path=raw_path,
                    method=method,
                    decision_model=dec,
                    judge_model=args.judge,
                    consistency_runs=args.consistency_runs,
                    horizon=args.horizon,
                    out_dir=out_dir,
                    resume=args.resume,
                )
            except SystemExit:
                raise
            except Exception as e:
                logger.exception(f"failed: {method} x {dec}: {e}")
                summary = {"method": method, "decision_model": dec,
                           "judge_model": args.judge, "error": str(e)}
            grid[(method, dec, args.judge)] = summary

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"cells": [{
            "method": k[0], "decision": k[1], "judge": k[2], **v
        } for k, v in grid.items()]}, f, indent=2)

    # Pretty-print
    print("\n=== Re-decide summary ===")
    hdr = f"{'method':<22} {'decision':<26} {'judge':<22} {'F1?':>8} {'FIR':>7} {'judge':>7}"
    print(hdr); print("-" * len(hdr))
    for (method, dec, jud), s in grid.items():
        f1 = (2 * s.get("mean_precision", 0) * s.get("mean_completeness", 0)
              / max(1e-9, s.get("mean_precision", 0) + s.get("mean_completeness", 0)))
        print(f"{method:<22} {dec[:24]:<26} {jud[:20]:<22} {f1:>8.4f} "
              f"{s.get('mean_fir', 0):>7.3f} {s.get('mean_quality', 0):>7.2f}")
    logger.info(f"All done. Output: {out_dir}")


if __name__ == "__main__":
    main()
