#!/usr/bin/env python3
"""Unit tests for ResultExporter output artifacts."""

# ruff: noqa: E402

import json
import sys
import tempfile
import unittest
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from optimization.events.event_definitions import EventDispatcher, OptimizationCompleted
from optimization.result_exporter import ResultExporter
from optimization.evaluator import EvaluationStatus


class TestResultExporter(unittest.TestCase):
    def setUp(self) -> None:
        self.exporter = ResultExporter()
        self.results = [
            {
                "value": {"k1": 0.5},
                "plan_fingerprint": "plan_a",
                "execute_time": 1000.0,
                "improvement": 0.0,
                "is_real_execution": True,
                "phase": "baseline",
                "sample_index": 0,
                "candidate_source": "default_configuration",
                "observation_source": "executed_plan",
                "evaluation_status": EvaluationStatus.EXECUTED.value,
                "sql_file": "q1.sql",
            },
            {
                "value": {"k1": 0.2},
                "plan_fingerprint": "plan_b",
                "execute_time": 700.0,
                "improvement": 30.0,
                "is_real_execution": True,
                "phase": "warm_start",
                "sample_index": 0,
                "candidate_source": "sobol_warm_start",
                "hinted_sql": "SELECT /*+ set_var(k1='0.2') */ 1",
                "observation_source": "executed_plan",
                "evaluation_status": EvaluationStatus.EXECUTED.value,
            },
            {
                "value": {"k1": 0.9},
                "plan_fingerprint": "plan_c",
                "execute_time": 1300.0,
                "improvement": -30.0,
                "is_real_execution": False,
                "is_admission_estimate": True,
                "phase": "optimization",
                "iteration": 0,
                "sample_index": 0,
                "candidate_source": "TCBO+Reasoner",
                "is_timeout": False,
                "observation_source": "subplan_admission_estimate",
                "evaluation_status": EvaluationStatus.SUBPLAN_REJECTED.value,
            },
            {
                "value": {"k1": 0.1},
                "plan_fingerprint": "plan_b",
                "execute_time": 700.0,
                "improvement": 30.0,
                "is_real_execution": False,
                "is_repository_reuse": True,
                "is_admission_estimate": False,
                "phase": "optimization",
                "iteration": 0,
                "sample_index": 1,
                "candidate_source": "TCBO+Reasoner",
                "observation_source": "plan_repository_duplicate",
                "evaluation_status": EvaluationStatus.DUPLICATE_PLAN.value,
            },
            {
                "value": {"k1": 0.8},
                "plan_fingerprint": "plan_d",
                "execute_time": 2000.0,
                "improvement": -100.0,
                "is_real_execution": True,
                "is_admission_estimate": False,
                "phase": "optimization",
                "iteration": 1,
                "sample_index": 0,
                "candidate_source": "TCBO+Reasoner Prompt Optimization",
                "is_timeout": True,
                "observation_source": "timeout_censored_execution",
                "evaluation_status": EvaluationStatus.EXECUTED.value,
            },
        ]

    def test_save_summary_and_data_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "result_q1_obelisk.json"

            self.exporter._save_results(self.results, str(result_path))
            self.exporter._save_summary(
                best_result=self.results[1],
                baseline_exec_time=1000.0,
                result_path=str(result_path),
                optimizer_settings={"tcbo_num_trust_regions": 2},
                total_rounds=3,
                warm_start_rounds=1,
            )
            self.exporter._save_data_summary(
                self.results,
                result_path=str(result_path),
                baseline_time_ms=1000.0,
                warm_start_rounds=1,
                optimizer_settings={"tcbo_num_trust_regions": 2},
                total_rounds=3,
            )

            summary_path = Path(str(result_path).replace(".json", "_summary.json"))
            data_summary_path = Path(str(result_path).replace(".json", "_data_summary.json"))

            self.assertTrue(result_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(data_summary_path.exists())

            with open(result_path, "r", encoding="utf-8") as file_handle:
                raw_results = json.load(file_handle)
            self.assertTrue(all("plan_fingerprint" in result for result in raw_results))
            self.assertTrue(all("plan_id" not in result for result in raw_results))

            with open(summary_path, "r", encoding="utf-8") as file_handle:
                summary = json.load(file_handle)
            self.assertNotIn("best_plan_id", summary)
            self.assertEqual(summary["best_plan_fingerprint"], "plan_b")
            self.assertAlmostEqual(summary["improvement_percent"], 30.0)
            self.assertEqual(summary["best_hinted_sql"], "SELECT /*+ set_var(k1='0.2') */ 1")
            self.assertEqual(summary["best_observation_source"], "executed_plan")
            self.assertEqual(summary["optimizer_settings"]["tcbo_num_trust_regions"], 2)
            self.assertEqual(summary["total_rounds"], 3)
            self.assertEqual(summary["warm_start_rounds"], 1)

            with open(data_summary_path, "r", encoding="utf-8") as file_handle:
                data_summary = json.load(file_handle)
            self.assertEqual(data_summary["total_samples"], 5)
            self.assertEqual(data_summary["real_executions"], 3)
            self.assertEqual(data_summary["true_observations"], 3)
            self.assertEqual(data_summary["estimated_executions"], 1)
            self.assertEqual(data_summary["admission_estimates"], 1)
            self.assertEqual(data_summary["repository_plan_reuses"], 1)
            self.assertNotIn("cached_plan_reuses", data_summary)
            self.assertEqual(data_summary["timeout_observations"], 1)
            self.assertEqual(data_summary["distinct_plans"], 4)
            self.assertEqual(data_summary["best_observation_index"], 1)
            self.assertEqual(data_summary["optimizer_settings"]["tcbo_num_trust_regions"], 2)
            self.assertEqual(data_summary["total_rounds"], 3)
            self.assertEqual(data_summary["warm_start_rounds"], 1)
            self.assertNotIn("warm_start_times", data_summary)
            self.assertEqual(
                data_summary["evaluation_status_counts"][EvaluationStatus.EXECUTED.value],
                3,
            )
            self.assertEqual(
                data_summary["observation_source_counts"]["subplan_admission_estimate"],
                1,
            )
            self.assertEqual(data_summary["candidate_source_counts"]["TCBO+Reasoner"], 2)
            self.assertEqual(data_summary["candidate_source_counts"]["TCBO+Reasoner Prompt Optimization"], 1)
            self.assertIn("phases", data_summary)
            self.assertEqual(data_summary["phases"]["baseline"]["num_samples"], 1)
            self.assertEqual(data_summary["phases"]["warm_start"]["num_samples"], 1)
            self.assertEqual(data_summary["phases"]["optimization"]["num_samples"], 3)
            self.assertEqual(data_summary["phases"]["optimization"]["repository_reuse_count"], 1)
            self.assertNotIn("cached_count", data_summary["phases"]["optimization"])
            self.assertEqual(data_summary["phases"]["optimization"]["timeout_count"], 1)
            self.assertEqual(data_summary["phases"]["optimization"]["true_observation_count"], 1)
            self.assertIn("time_series", data_summary)
            self.assertEqual(data_summary["time_series"][2]["phase"], "optimization")
            self.assertEqual(data_summary["time_series"][1]["plan_fingerprint"], "plan_b")
            self.assertNotIn("plan_id", data_summary["time_series"][1])
            self.assertTrue(data_summary["time_series"][3]["is_true_observation"])
            self.assertTrue(data_summary["time_series"][3]["is_repository_reuse"])
            self.assertNotIn("is_cached_observation", data_summary["time_series"][3])
            self.assertEqual(
                data_summary["time_series"][3]["observation_source"],
                "plan_repository_duplicate",
            )

    def test_exporter_raises_when_result_path_cannot_be_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_dir_path = Path(tmp_dir) / "missing" / "result_q1_obelisk.json"
            event = OptimizationCompleted(
                timestamp=0.0,
                event_type="optimization_completed",
                context={
                    "result_path": str(missing_dir_path),
                    "baseline_exec_time": 1000.0,
                    "warm_start_rounds": 1,
                    "total_rounds": 3,
                },
                results=self.results,
                best_result=self.results[1],
                total_duration=1.0,
            )

            with self.assertRaises(FileNotFoundError):
                self.exporter.on_optimization_completed(event)

    def test_exporter_requires_result_path(self) -> None:
        event = OptimizationCompleted(
            timestamp=0.0,
            event_type="optimization_completed",
            context={},
            results=self.results,
            best_result=self.results[1],
            total_duration=1.0,
        )

        with self.assertRaisesRegex(ValueError, "result_path is required"):
            self.exporter.on_optimization_completed(event)

    def test_repository_reuse_accepts_legacy_cached_observation_input(self) -> None:
        legacy_result = {
            "is_cached_observation": True,
            "evaluation_status": EvaluationStatus.EXECUTED.value,
        }

        self.assertTrue(self.exporter._is_repository_reuse(legacy_result))

    def test_plan_fingerprint_accepts_legacy_plan_id_input(self) -> None:
        self.assertEqual(
            self.exporter._plan_fingerprint({"plan_id": "legacy_plan"}),
            "legacy_plan",
        )

    def test_event_dispatcher_propagates_subscriber_failures(self) -> None:
        dispatcher = EventDispatcher()

        def fail(_event):
            raise RuntimeError("subscriber failed")

        dispatcher.subscribe(OptimizationCompleted.__name__, fail)
        event = OptimizationCompleted(
            timestamp=0.0,
            event_type="optimization_completed",
            context={},
            results=self.results,
            best_result=self.results[1],
            total_duration=1.0,
        )

        with self.assertRaisesRegex(RuntimeError, "subscriber failed"):
            dispatcher.publish(event)


if __name__ == "__main__":
    unittest.main(verbosity=2)
