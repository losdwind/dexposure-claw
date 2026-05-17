import sys
import types
import unittest
from types import SimpleNamespace

sys.modules.setdefault("loguru", types.SimpleNamespace(logger=types.SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None)))
from experiments.ablations import _aggregate_b4_results
from experiments.b5_decision_quality import _build_scenario_summary
from dexposure_agent.types import ScenarioSummary


class AblationAggregationTest(unittest.TestCase):
    def test_aggregate_b4_results_uses_current_stress_overlap_field(self):
        results = [
            SimpleNamespace(loss_mae=0.2, target_overlap_at_k=0.4),
            SimpleNamespace(loss_mae=0.4, target_overlap_at_k=0.8),
        ]

        loss_mae, overlap_at_10 = _aggregate_b4_results(results)

        self.assertAlmostEqual(loss_mae, 0.3)
        self.assertAlmostEqual(overlap_at_10, 0.6)

    def test_b5_skip_scenario_returns_empty_scenario_summary(self):
        config = SimpleNamespace(skip_scenario=True)

        summary = _build_scenario_summary(
            pred_graph=SimpleNamespace(),
            mc_samples=[SimpleNamespace()],
            config=config,
            horizon=4,
        )

        self.assertIsInstance(summary, ScenarioSummary)
        self.assertEqual(summary.ranked_losses, [])
        self.assertEqual(summary.worst_scenario, "")
        self.assertEqual(summary.worst_horizon, 0)
