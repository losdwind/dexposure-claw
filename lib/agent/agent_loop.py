"""End-to-end agent loop implementing Algorithm 1.

Orchestrates the full DeXposure-Agent pipeline for one epoch:
    DataHealth -> (for each horizon: Forecast -> PredGraph -> MC Sample -> Monitor -> Scenario)
               -> Aggregate -> Decision -> AgentOutput
"""
from __future__ import annotations

import logging
from typing import Any

from lib.agent.config import AgentConfig
from lib.agent.types import GraphSnapshot, AgentOutput, ScenarioSummary, ScenarioLoss
from lib.agent.data_health import compute_data_health
from lib.agent.pred_graph import build_pred_graph, mc_sample
from lib.agent.monitor import run_monitor
from lib.agent.scenario import run_scenarios
from lib.agent.decision import generate_tickets

logger = logging.getLogger(__name__)


async def call_forecast_api(
    graph: GraphSnapshot,
    horizon: int,
    config: AgentConfig,
) -> dict[str, Any]:
    """Call the GPU server forecast API. Replaced by mock in tests.

    Args:
        graph:   The current GraphSnapshot (date used as request context).
        horizon: Forecast horizon in weeks.
        config:  AgentConfig with api_base_url.

    Returns:
        Dict with keys edge_probs, edge_weights, weight_stds, node_ids.
        Returns empty prediction if the API is unreachable.
    """
    try:
        import httpx
        async with httpx.AsyncClient(base_url=config.api_base_url, timeout=300) as client:
            resp = await client.post("/forecast", json={
                "date": graph.date,
                "horizon": horizon,
            })
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.warning(
            "call_forecast_api failed (date=%s horizon=%d): %s — returning empty prediction",
            graph.date, horizon, exc,
        )
        return {"edge_probs": {}, "edge_weights": {}, "weight_stds": {}, "node_ids": []}


def _aggregate_scenarios(all_scenario_losses: list[ScenarioLoss]) -> ScenarioSummary:
    """Merge scenario losses across horizons, keeping worst expected_loss per scenario_id.

    Args:
        all_scenario_losses: Flat list of ScenarioLoss objects from all horizons.

    Returns:
        ScenarioSummary with losses ranked by expected_loss descending.
    """
    best_per_scenario: dict[str, ScenarioLoss] = {}
    for sl in all_scenario_losses:
        key = sl.scenario_id
        if key not in best_per_scenario or sl.expected_loss > best_per_scenario[key].expected_loss:
            best_per_scenario[key] = sl

    ranked = sorted(best_per_scenario.values(), key=lambda s: s.expected_loss, reverse=True)

    worst = ranked[0] if ranked else ScenarioLoss(
        scenario_id="",
        scenario_name="",
        horizon=0,
        expected_loss=0.0,
        cvar_loss=0.0,
        distressed_count=0,
        propagation_depth=0,
    )

    return ScenarioSummary(
        ranked_losses=ranked,
        worst_scenario=worst.scenario_id,
        worst_horizon=worst.horizon,
    )


async def run_epoch(
    graph: GraphSnapshot,
    baseline_history: list[dict[str, float]],
    config: AgentConfig,
) -> AgentOutput:
    """Algorithm 1: full agent loop for one epoch.

    Steps:
        1. DataHealth gate — compute DH_t and safe_mode flag.
        2. For each horizon h in config.horizons:
            a. Forecast via GPU server API (or mock).
            b. Build predicted graph G_hat.
            c. Draw MC samples for uncertainty quantification.
            d. Monitor — compare G_hat metrics to rolling baseline.
            e. Scenario engine — stress-test G_hat.
        3. Aggregate scenario losses across horizons.
        4. Decision — generate ranked recommendation tickets.

    Args:
        graph:            Current DeFi credit-exposure GraphSnapshot.
        baseline_history: Past metric dicts for rolling baseline comparison.
        config:           AgentConfig with all hyperparameters.

    Returns:
        AgentOutput containing data health, alerts, scenario summary, and tickets.
    """
    logger.info("run_epoch | date=%s horizons=%s", graph.date, config.horizons)

    # Step 1: DataHealth gate
    dh = compute_data_health(graph, config)
    logger.info("DataHealth | score=%.4f safe_mode=%s", dh.score, dh.safe_mode)

    all_alerts = []
    all_scenario_losses: list[ScenarioLoss] = []

    for h in config.horizons:
        logger.debug("Horizon h=%d: calling forecast API", h)

        # Step 2a: Forecast
        prediction = await call_forecast_api(graph, h, config)

        # Step 2b: Build predicted graph G_hat
        g_hat = build_pred_graph(prediction, config, date=graph.date)
        logger.debug(
            "Horizon h=%d: G_hat nodes=%d edges=%d",
            h, len(g_hat.nodes), len(g_hat.edges),
        )

        # Step 2c: MC samples for uncertainty quantification
        samples = mc_sample(prediction, config, date=graph.date)

        # Step 2d: Monitor — z-score alerts on predicted metrics vs baseline
        monitor_result = run_monitor(
            predicted_graph=g_hat,
            mc_samples=samples,
            baseline_history=baseline_history,
            horizon=h,
            config=config,
            dh_score=dh.score,
        )
        all_alerts.extend(monitor_result.alerts)
        logger.debug("Horizon h=%d: %d alert(s) generated", h, len(monitor_result.alerts))

        # Step 2e: Scenario engine — stress tests S1-S5
        scenario_result = run_scenarios(g_hat, samples, config, horizon=h)
        all_scenario_losses.extend(scenario_result.ranked_losses)

    # Step 3: Aggregate scenario losses across all horizons
    scenario_summary = _aggregate_scenarios(all_scenario_losses)
    logger.info(
        "Scenarios | worst=%s worst_horizon=%d",
        scenario_summary.worst_scenario, scenario_summary.worst_horizon,
    )

    # Step 4: Decision — generate recommendation tickets
    decision = generate_tickets(all_alerts, scenario_summary, dh, config)
    logger.info(
        "Decision | tickets=%d suppressed=%s",
        len(decision.tickets), decision.suppressed,
    )

    return AgentOutput(
        epoch_date=graph.date,
        data_health=dh,
        alerts=all_alerts,
        scenario_summary=scenario_summary,
        tickets=decision.tickets,
        suppressed=decision.suppressed,
    )
