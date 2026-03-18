#!/usr/bin/env python3
"""Expanding-window walk-forward cross-validation (15 folds).

Protocol (from EXPERIMENT_PLAN Section 1.2):
  - Start with minimal training window: 2020-03 ~ 2022-06
  - Expand training window by ~4 weeks per fold
  - Evaluate on next 4 weeks
  - 15 folds total, covering 2022-07 ~ 2025-08
  - Report mean +/- std across folds for B1 key metrics
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger


@dataclass
class FoldResult:
    fold: int
    train_start: str
    train_end: str
    eval_start: str
    eval_end: str
    metrics: dict[str, float] = field(default_factory=dict)


NUM_FOLDS = 15
TRAIN_START = "2020-03-30"
FIRST_EVAL_START = "2022-07-04"
FOLD_STEP_WEEKS = 4
EVAL_WINDOW_WEEKS = 4


def generate_folds() -> list[FoldResult]:
    """Generate the 15 expanding-window folds."""
    folds = []
    train_start = datetime.strptime(TRAIN_START, "%Y-%m-%d")
    eval_start = datetime.strptime(FIRST_EVAL_START, "%Y-%m-%d")

    for i in range(NUM_FOLDS):
        train_end = eval_start - timedelta(days=1)
        eval_end = eval_start + timedelta(weeks=EVAL_WINDOW_WEEKS) - timedelta(days=1)
        folds.append(FoldResult(
            fold=i + 1,
            train_start=train_start.strftime("%Y-%m-%d"),
            train_end=train_end.strftime("%Y-%m-%d"),
            eval_start=eval_start.strftime("%Y-%m-%d"),
            eval_end=eval_end.strftime("%Y-%m-%d"),
        ))
        eval_start += timedelta(weeks=FOLD_STEP_WEEKS)
    return folds


def run_fold(fold: FoldResult, method_id: str = "C0") -> FoldResult:
    """Run a single fold of walk-forward evaluation."""
    logger.info(f"Fold {fold.fold}: train={fold.train_start}~{fold.train_end}, eval={fold.eval_start}~{fold.eval_end}")
    # TODO: load data for fold, train/eval, compute B1 metrics
    raise NotImplementedError("Walk-forward fold evaluation not yet connected")


def main():
    parser = argparse.ArgumentParser(description="Walk-forward cross-validation")
    parser.add_argument("--method", default="C0")
    parser.add_argument("--output", default="results/walk_forward.json")
    parser.add_argument("--folds", type=int, default=NUM_FOLDS)
    args = parser.parse_args()

    folds = generate_folds()[:args.folds]
    logger.info(f"Running {len(folds)} folds for method {args.method}")

    for f in folds:
        logger.info(f"  Fold {f.fold}: train {f.train_start} ~ {f.train_end} | eval {f.eval_start} ~ {f.eval_end}")

    results = []
    for f in folds:
        try:
            result = run_fold(f, args.method)
            results.append(asdict(result))
        except NotImplementedError:
            logger.warning(f"Fold {f.fold}: not yet implemented, skipping")
            results.append(asdict(f))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    logger.info(f"Results saved to {out}")


if __name__ == "__main__":
    main()
