#!/usr/bin/env python3
"""C3: LLM-Agent competitor wrapper.

Calls a large language model (Claude or GPT-4) with tabular risk metrics
serialised as a structured prompt. The LLM is asked to produce:
  - Protocol-level risk scores
  - Recommended supervisory actions

Applicable to: B2, B5, B6 (not B1 -- too many numeric predictions;
not B3 -- no calibrated uncertainty; not B4 -- no contagion simulation).

Prompt design:
  System: You are a DeFi systemic risk analyst ...
  User:   Here is the current DeFi network state as of {date}: {tabular_metrics}
          Predict which protocols are at elevated risk over the next {h} weeks ...
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional
from loguru import logger


@dataclass
class LLMAgentConfig:
    model: str = "claude-opus-4-5"          # or "gpt-4o"
    horizon: int = 4
    max_tokens: int = 2048
    temperature: float = 0.0                # greedy for reproducibility
    api_key_env: str = "ANTHROPIC_API_KEY"  # env var name
    prompt_template: Optional[str] = None   # override default prompt


@dataclass
class LLMAgentPrediction:
    """Output format shared across all agent-level competitors."""
    method_id: str = "C3"
    horizon: int = 4
    pagerank_pred: dict[str, float] = field(default_factory=dict)
    hhi_pred: float = float("nan")
    density_pred: float = float("nan")
    gini_pred: float = float("nan")
    risk_scores: dict[str, float] = field(default_factory=dict)
    uncertainty: dict[str, float] = field(default_factory=dict)
    recommended_actions: list[dict[str, Any]] = field(default_factory=list)
    raw_response: Optional[str] = None      # LLM completion for audit

    def __str__(self) -> str:
        return (
            f"LLMAgentPrediction(model={self.method_id}, h={self.horizon}, "
            f"n_protocols={len(self.risk_scores)}, "
            f"n_actions={len(self.recommended_actions)})"
        )


DEFAULT_SYSTEM_PROMPT = """You are a DeFi systemic risk analyst.
You will be given a summary of the current DeFi protocol network state as tabular metrics.
Your task is to identify protocols at elevated risk of distress over the specified horizon,
and recommend appropriate supervisory actions.
Respond in structured JSON only."""

DEFAULT_USER_TEMPLATE = """Current DeFi network state ({date}), horizon={horizon} weeks:

{tabular_metrics}

Identify the top protocols at risk and recommended actions.
Return JSON with keys: "risk_scores" (dict protocol->float[0,1]),
"recommended_actions" (list of dicts with keys: protocol, action, reason)."""


def _build_prompt(
    tabular_metrics: dict[str, Any],
    date: str,
    horizon: int,
    template: Optional[str] = None,
) -> str:
    """Serialise tabular metrics into an LLM prompt string."""
    import json
    metrics_str = json.dumps(tabular_metrics, indent=2)
    tmpl = template or DEFAULT_USER_TEMPLATE
    return tmpl.format(
        date=date,
        horizon=horizon,
        tabular_metrics=metrics_str,
    )


def run_llm_agent(
    graph: Any,
    config: LLMAgentConfig | None = None,
    **kwargs,
) -> LLMAgentPrediction:
    """Run the LLM-Agent: convert graph metrics to text and call an LLM.

    Args:
        graph: Temporal graph object. The latest snapshot is converted to
               tabular metrics (PageRank, TVL, degree, etc.) and passed as
               a structured prompt to the LLM.
        config: LLMAgentConfig. Uses defaults if None.
        **kwargs: Extra config key-value pairs that override config fields.

    Returns:
        LLMAgentPrediction parsed from the LLM JSON response.
    """
    if config is None:
        config = LLMAgentConfig()
    for k, v in kwargs.items():
        if hasattr(config, k):
            setattr(config, k, v)

    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        logger.warning(
            f"LLM-Agent: environment variable {config.api_key_env!r} not set; "
            "API calls will fail at runtime."
        )

    logger.info(
        f"LLM-Agent (C3) | model={config.model} | horizon={config.horizon} | "
        f"temperature={config.temperature}"
    )
    # TODO: extract latest snapshot from graph as tabular metrics dict
    # TODO: call _build_prompt(metrics, date, config.horizon, config.prompt_template)
    # TODO: send system+user prompt to config.model via anthropic or openai SDK
    # TODO: parse JSON response into LLMAgentPrediction
    raise NotImplementedError("LLM-Agent (C3) API call not yet implemented")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="C3: LLM-Agent competitor")
    parser.add_argument("--data-dir", default="data/", help="Data directory")
    parser.add_argument("--test-split", default="2025-01~2025-08")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--model", default="claude-opus-4-5",
                        help="LLM model identifier")
    args = parser.parse_args()

    cfg = LLMAgentConfig(horizon=args.horizon, model=args.model)
    # TODO: load graph from args.data_dir
    pred = run_llm_agent(graph=None, config=cfg)
    print(pred)
