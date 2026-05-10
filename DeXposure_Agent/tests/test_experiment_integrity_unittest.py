"""Source-level integrity checks for the DeXposure-Agent experiments.

These tests intentionally avoid project dependencies such as pydantic, torch,
and scipy so they can run in a bare Python environment before the full
experiment stack is installed.
"""
from __future__ import annotations

import importlib
import pathlib
import json
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "DeXposure_Agent"


def read(relative_path: str) -> str:
    return (AGENT_ROOT / relative_path).read_text()


def import_run_all():
    if str(AGENT_ROOT) not in sys.path:
        sys.path.insert(0, str(AGENT_ROOT))
    return importlib.import_module("experiments.run_all")


class ExperimentIntegrityTests(unittest.TestCase):
    def test_prediction_helper_fails_closed_instead_of_proxying_to_persistence(self):
        source = read("experiments/predict_helper.py")

        self.assertNotIn("Others: Persistence proxy", source)
        self.assertNotIn("using persistence", source)
        self.assertIn("PredictionUnavailable", source)

    def test_evolvegcn_predictor_does_not_fallback_to_persistence(self):
        source = read("experiments/competitors/evolvegcn.py")

        self.assertNotIn("using persistence", source)
        self.assertNotIn("return current_snapshot", source)

    def test_b1_uses_shared_prediction_helper(self):
        source = read("experiments/b1_risk_forecasting.py")

        self.assertNotIn("_PROXY_METHODS", source)
        self.assertNotIn("def _predict_graph", source)
        self.assertIn("from experiments.predict_helper import predict_graph", source)

    def test_llm_prompt_declares_conservative_action_constraints(self):
        source = read("experiments/llm_eval_b5.py")

        self.assertIn("Recommend-Reduce requires", source)
        self.assertIn("Contingency requires", source)
        self.assertIn("apply_action_gate", source)

    def test_llm_judge_does_not_compare_against_empty_rule_context(self):
        source = read("experiments/llm_eval_b5.py")

        self.assertNotIn("REPORT A (baseline rule-engine output", source)
        self.assertNotIn("truly_stressed, set(), primary", source)

    def test_b2_is_marked_as_shared_heuristic_not_method_comparison(self):
        b2_source = read("experiments/b2_early_warning.py")
        run_all_source = read("experiments/run_all.py")
        runner_source = read("scripts/run_benchmarks_sequential.py")
        methods_source = read("experiments/methods.py")

        self.assertIn('B2_APPLICABLE_METHODS = {"H0"}', b2_source)
        self.assertIn('"H0": MethodSpec', methods_source)
        self.assertIn('label="WeightedDegreeHeuristic"', methods_source)
        self.assertIn('"B2": {"H0"}', run_all_source)
        self.assertIn('"run_b2", ["H0"]', runner_source)

    def test_master_runner_uses_canonical_methods_and_structured_results(self):
        source = read("experiments/run_all.py")

        self.assertIn("from experiments.methods import", source)
        self.assertNotIn("METHOD_NAMES = {", source)
        self.assertNotIn("str(r) for r in results", source)
        self.assertNotIn('("ablations", "C0")', source)
        self.assertIn("traceback.format_exc", source)
        self.assertIn("wall_seconds", source)
        self.assertIn("git_commit", source)
        self.assertIn("DGLBACKEND", source)
        self.assertIn("ALL DONE", source)

    def test_master_runner_uses_prediction_unavailable_type_not_name_matching(self):
        source = read("experiments/run_all.py")

        self.assertIn("from experiments.exceptions import PredictionUnavailable", source)
        self.assertIn("isinstance(exc, PredictionUnavailable)", source)
        self.assertNotIn("UNAVAILABLE_EXCEPTION_NAMES", source)
        self.assertIn("Unavailable predictor exceptions are registered", source)
        self.assertIn("experiments.predict_helper", source)

    def test_master_runner_serializes_non_finite_floats_explicitly(self):
        run_all = import_run_all()

        payload = run_all._jsonify(
            {"nan": float("nan"), "pos": float("inf"), "neg": float("-inf")}
        )

        self.assertEqual(payload, {"nan": "nan", "pos": "+inf", "neg": "-inf"})

    def test_master_runner_does_not_expand_unknown_object_dicts(self):
        run_all = import_run_all()

        class UnknownObject:
            def __init__(self):
                self.secret_internal_state = "do-not-serialize"

            def __repr__(self):
                return "<UnknownObject>"

        self.assertEqual(run_all._jsonify(UnknownObject()), "<UnknownObject>")

    def test_master_runner_classifies_prediction_unavailable_separately(self):
        run_all = import_run_all()

        def fake_run(**kwargs):
            raise run_all.PredictionUnavailable("missing checkpoint")

        fake_module = types.SimpleNamespace(run_fake=fake_run)

        with mock.patch.object(run_all.importlib, "import_module", return_value=fake_module):
            with mock.patch.dict(run_all.BENCHMARK_MODULES, {"BX": "fake.module"}):
                with mock.patch.dict(run_all.BENCHMARK_FUNCS, {"BX": "run_fake"}):
                    result = run_all._run_benchmark(
                        "BX",
                        "C7",
                        data_dir="data/",
                        test_split="2025-01~2025-08",
                        results_dir=AGENT_ROOT / "results",
                    )

        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["exception_type"], "PredictionUnavailable")

    def test_master_runner_continues_after_unavailable_or_not_implemented(self):
        run_all = import_run_all()

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(run_all, "_setup_logger", return_value=pathlib.Path(tmp) / "run.log"):
                with mock.patch.object(run_all, "_collect_metadata", return_value={}):
                    with mock.patch.object(run_all, "_write_state"):
                        with mock.patch.object(
                            run_all,
                            "_run_benchmark",
                            side_effect=[
                                {"status": "not_implemented"},
                                {"status": "unavailable"},
                                {"status": "ok"},
                            ],
                        ) as run_benchmark:
                            code = run_all.main(
                                [
                                    "--benchmarks",
                                    "B1",
                                    "--methods",
                                    "C7,C4,C0",
                                    "--output",
                                    tmp,
                                ]
                            )

        self.assertEqual(code, 1)
        self.assertEqual(run_benchmark.call_count, 3)

    def test_artifact_builders_do_not_reference_deleted_legacy_runs(self):
        for relative_path in (
            "scripts/build_table3_and_fig4.py",
            "scripts/build_task1_artifacts.py",
        ):
            source = read(relative_path)
            self.assertNotIn("202603", source)
            self.assertNotIn("20260409", source)
            self.assertIn("latest", source)

    def test_result_audit_passes_after_old_artifacts_are_removed(self):
        audit_script = AGENT_ROOT / "experiments" / "audit_results.py"
        result = subprocess.run(
            [
                sys.executable,
                str(audit_script),
                "--results-dir",
                str(AGENT_ROOT / "results"),
                "--format",
                "json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)

        self.assertEqual(payload["n_issues"], 0)
        self.assertEqual(payload["issues"], [])


if __name__ == "__main__":
    unittest.main()
