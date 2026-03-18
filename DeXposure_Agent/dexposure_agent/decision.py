"""Decision Module for DeXposure-Agent.

Implements Decision Agent v1 (Section 2.7): playbook constraints,
ticket generation and scoring (Eq. 7).

Algorithm 1 final step: consumes Monitor alerts and Scenario Engine results,
produces ranked recommendation Tickets.
"""
from __future__ import annotations

import logging
from statistics import mean

from dexposure_agent.config import AgentConfig
from dexposure_agent.types import (
    Alert,
    DataHealthResult,
    DecisionResult,
    ScenarioSummary,
    Ticket,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Playbook constants
# ---------------------------------------------------------------------------

PLAYBOOK: dict[str, dict] = {
    "Monitor": {
        "severity": "Low",
        "requires_safe_off": False,
        "min_alerts": 1,
    },
    "Investigate": {
        "severity": "Medium",
        "requires_safe_off": False,
        "min_alerts": 1,
    },
    "Recommend-Reduce": {
        "severity": "High",
        "requires_safe_off": True,
        "min_alerts": 2,
    },
    "Contingency": {
        "severity": "Critical",
        "requires_safe_off": True,
        "min_alerts": 3,
    },
}

SEVERITY_WEIGHTS: dict[str, float] = {
    "Low": 0.25,
    "Medium": 0.5,
    "High": 0.75,
    "Critical": 1.0,
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_tickets(
    alerts: list[Alert],
    scenario_summary: ScenarioSummary,
    data_health: DataHealthResult,
    config: AgentConfig,
) -> DecisionResult:
    """Generate ranked recommendation tickets from alerts and scenario results.

    Implements the decision gate (Eq. 7):
      - safe_mode blocks intervention actions (requires_safe_off)
      - min_alerts enforces minimum evidence count
      - confidence gate tau_conf blocks intervention when mean confidence is low

    Args:
        alerts: Triggered alerts from the Monitor module.
        scenario_summary: Ranked loss scenarios from the Scenario Engine.
        data_health: Data quality assessment including safe_mode flag.
        config: Agent hyperparameters (tau_conf, etc.).

    Returns:
        DecisionResult with sorted tickets and suppressed flag.
    """
    if not alerts:
        logger.info("No alerts — returning empty DecisionResult")
        return DecisionResult(tickets=[], suppressed=data_health.safe_mode)

    num_alerts = len(alerts)
    mean_confidence = mean(a.confidence for a in alerts)
    scenario_impact = _compute_scenario_impact(scenario_summary)

    logger.debug(
        "Decision: num_alerts=%d mean_confidence=%.3f scenario_impact=%.3f safe_mode=%s",
        num_alerts,
        mean_confidence,
        scenario_impact,
        data_health.safe_mode,
    )

    tickets: list[Ticket] = []

    for action, rules in PLAYBOOK.items():
        severity: str = rules["severity"]
        requires_safe_off: bool = rules["requires_safe_off"]
        min_alerts: int = rules["min_alerts"]

        # Eq. 7 constraint (i): safe_mode blocks intervention actions
        if requires_safe_off and data_health.safe_mode:
            logger.debug("Action %s blocked: safe_mode=True", action)
            continue

        # Eq. 7 constraint (ii): minimum alert count
        if num_alerts < min_alerts:
            logger.debug(
                "Action %s blocked: num_alerts=%d < min_alerts=%d",
                action,
                num_alerts,
                min_alerts,
            )
            continue

        # Eq. 7 constraint (iii): confidence gate for intervention actions
        if requires_safe_off and mean_confidence < config.tau_conf:
            logger.debug(
                "Action %s blocked: mean_confidence=%.3f < tau_conf=%.3f",
                action,
                mean_confidence,
                config.tau_conf,
            )
            continue

        # Score = severity_weight * mean_confidence * scenario_impact
        severity_weight = SEVERITY_WEIGHTS[severity]
        score = severity_weight * mean_confidence * scenario_impact

        targets = _extract_targets(alerts, scenario_summary, config.top_k)
        triggering_alert_ids = [a.metric_id for a in alerts]
        scenario_impact_map = (
            {scenario_summary.worst_scenario: scenario_impact}
            if scenario_summary.worst_scenario
            else {}
        )

        rationale = (
            f"{action} triggered by {num_alerts} alert(s); "
            f"mean_conf={mean_confidence:.2f}; scenario_impact={scenario_impact:.2f}"
        )

        ticket = Ticket(
            action=action,
            severity=severity,
            score=score,
            targets=targets,
            triggering_alerts=triggering_alert_ids,
            scenario_impact=scenario_impact_map,
            rationale=rationale,
        )
        tickets.append(ticket)
        logger.debug("Ticket created: action=%s score=%.4f", action, score)

    # Sort by score descending
    tickets.sort(key=lambda t: t.score, reverse=True)

    logger.info(
        "Decision complete: %d ticket(s) generated suppressed=%s",
        len(tickets),
        data_health.safe_mode,
    )

    return DecisionResult(tickets=tickets, suppressed=data_health.safe_mode)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _compute_scenario_impact(scenario_summary: ScenarioSummary) -> float:
    """Aggregate scenario impact as mean CVaR loss across ranked scenarios.

    Falls back to 1.0 (neutral multiplier) when no scenarios exist so that
    alert-only tickets are still scored non-zero.
    """
    if not scenario_summary.ranked_losses:
        return 1.0
    return mean(sl.cvar_loss for sl in scenario_summary.ranked_losses)


def _extract_targets(
    alerts: list[Alert],
    scenario_summary: ScenarioSummary,
    top_k: int,
) -> list[str]:
    """Collect unique top targets from scenario losses and alert attributions."""
    targets: list[str] = []

    # Targets from scenario top_targets
    for sl in scenario_summary.ranked_losses:
        for t in sl.top_targets:
            if t not in targets:
                targets.append(t)

    # Targets from alert attribution maps
    for alert in alerts:
        for node in alert.attribution:
            if node not in targets:
                targets.append(node)

    return targets[:top_k]
