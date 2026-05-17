#!/usr/bin/env python3
"""Hyperparameter grid search on validation split.

Search grid (from EXPERIMENT_PLAN Section 2.5):
  pi_min: {0.1, 0.2, 0.3}
  z_threshold: {1.5, 2.0, 2.5}
  rolling_window: {13, 26, 52}
  tau_data: {0.5, 0.6, 0.7}
  tau_conf: {0.4, 0.5, 0.6}
  lambda_tail: {0.0, 0.25, 0.5}
  mc_samples: {20, 50, 100}
  top_k: {5, 10, 20}

Optimizes a composite objective from b1_forecast (Rank Correlation) + b2_warning (F1-warning) + b5_decision (Ticket Precision).
Saves best config to results/best_agent_config.json.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from loguru import logger

from dexposure_agent.config import AgentConfig


SEARCH_GRID = {
    "pi_min": [0.1, 0.2, 0.3],
    "z_threshold": [1.5, 2.0, 2.5],
    "rolling_window": [13, 26, 52],
    "tau_data": [0.5, 0.6, 0.7],
    "tau_conf": [0.4, 0.5, 0.6],
    "lambda_tail": [0.0, 0.25, 0.5],
    "mc_samples": [20, 50, 100],
    "top_k": [5, 10, 20],
}

VAL_SPLIT = ("2024-07", "2024-12")


def objective(trial) -> float:
    """Optuna objective function. Returns composite score (to maximize)."""
    config = AgentConfig(
        pi_min=trial.suggest_categorical("pi_min", SEARCH_GRID["pi_min"]),
        z_threshold=trial.suggest_categorical("z_threshold", SEARCH_GRID["z_threshold"]),
        rolling_window=trial.suggest_categorical("rolling_window", SEARCH_GRID["rolling_window"]),
        tau_data=trial.suggest_categorical("tau_data", SEARCH_GRID["tau_data"]),
        tau_conf=trial.suggest_categorical("tau_conf", SEARCH_GRID["tau_conf"]),
        lambda_tail=trial.suggest_categorical("lambda_tail", SEARCH_GRID["lambda_tail"]),
        mc_samples=trial.suggest_categorical("mc_samples", SEARCH_GRID["mc_samples"]),
        top_k=trial.suggest_categorical("top_k", SEARCH_GRID["top_k"]),
    )
    logger.info(f"Trial {trial.number}: {config.model_dump()}")

    # TODO: Run b1_forecast, b2_warning, b5_decision on validation split with this config
    # rank_corr = run_b1_on_val(config)
    # f1_warning = run_b2_on_val(config)
    # ticket_prec = run_b5_on_val(config)
    # composite = (rank_corr + f1_warning + ticket_prec) / 3

    raise NotImplementedError("Validation-split evaluation not yet connected")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for DeXposure-Agent")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--output", default="results/best_agent_config.json")
    args = parser.parse_args()

    try:
        import optuna
    except ImportError:
        logger.error("optuna not installed. Run: pip install optuna")
        return

    study = optuna.create_study(direction="maximize", study_name="dexposure-agent-tuning")
    study.optimize(objective, n_trials=args.n_trials)

    best = study.best_params
    logger.info(f"Best params: {best}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(best, indent=2))
    logger.info(f"Saved to {out}")


if __name__ == "__main__":
    main()
