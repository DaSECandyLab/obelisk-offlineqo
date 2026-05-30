#!/usr/bin/env python3
"""Unit tests for OBELISK optimization-pipeline observation records."""

# ruff: noqa: E402

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from optimization.evaluator import EvaluationStatus
from optimization.events.event_definitions import (
    EventDispatcher,
    OptimizationCompleted,
    OptimizationStarted,
)
from optimization.optimization_pipeline import OptimizationPipeline
from optimization.result_exporter import ResultExporter
from util.config import AppConfig, LLMRuntimeConfig, OptimizationConfig
from util.knob_space import KnobSpace


class TestOptimizationPipelineRecords(unittest.TestCase):
    def test_candidate_phase_name_uses_reasoner_when_enabled(self) -> None:
        self.assertEqual(
            OptimizationPipeline._candidate_phase_name(
                "TCBO",
                try_number=0,
                reasoner_enabled=False,
            ),
            "TCBO Sampling",
        )
        self.assertEqual(
            OptimizationPipeline._candidate_phase_name(
                "TCBO",
                try_number=0,
                reasoner_enabled=True,
            ),
            "TCBO+Reasoner",
        )
        self.assertEqual(
            OptimizationPipeline._candidate_phase_name(
                "TCBO",
                try_number=1,
                reasoner_enabled=True,
            ),
            "TCBO+Reasoner Prompt Optimization",
        )

    def test_optimize_rejects_inconsistent_round_budget(self) -> None:
        pipeline = object.__new__(OptimizationPipeline)

        with self.assertRaisesRegex(ValueError, "warm_start_rounds cannot exceed total_rounds"):
            pipeline.optimize(
                sql_filepath="/tmp/q1.sql",
                result_filepath="/tmp/result.json",
                total_rounds=1,
                warm_start_rounds=2,
            )

        with self.assertRaisesRegex(ValueError, "total_rounds must be non-negative"):
            pipeline.optimize(
                sql_filepath="/tmp/q1.sql",
                result_filepath="/tmp/result.json",
                total_rounds=-1,
                warm_start_rounds=0,
            )

    def test_make_result_record_captures_obelisk_observation_source(self) -> None:
        pipeline = object.__new__(OptimizationPipeline)
        knob_space = KnobSpace(["k1"], {"k1": (0.1, 10.0)})

        duplicate_record = pipeline._make_result_record(
            phase="optimization",
            knob_space=knob_space,
            vector=[0.5],
            plan_fingerprint="plan_a",
            execute_time_ms=500.0,
            baseline_exec_time_ms=1000.0,
            evaluation_status=EvaluationStatus.DUPLICATE_PLAN,
            sql_filepath="/tmp/q1.sql",
            timeout_threshold_ms=2000.0,
            iteration=1,
            sample_index=2,
            candidate_source="TCBO+Reasoner",
        )

        self.assertEqual(duplicate_record["phase"], "optimization")
        self.assertEqual(duplicate_record["plan_fingerprint"], "plan_a")
        self.assertNotIn("plan_id", duplicate_record)
        self.assertEqual(duplicate_record["observation_source"], "plan_repository_duplicate")
        self.assertNotIn("is_cached_observation", duplicate_record)
        self.assertFalse(duplicate_record["is_admission_estimate"])
        self.assertFalse(duplicate_record["is_real_execution"])
        self.assertTrue(duplicate_record["is_true_observation"])
        self.assertTrue(duplicate_record["is_repository_reuse"])
        self.assertFalse(duplicate_record["is_admission_rejected"])
        self.assertEqual(duplicate_record["normalized_vector"], [0.5])
        self.assertAlmostEqual(duplicate_record["value"]["k1"], 1.0)

        timeout_record = pipeline._make_result_record(
            phase="warm_start",
            knob_space=knob_space,
            vector=[0.5],
            plan_fingerprint="plan_b",
            execute_time_ms=2000.0,
            baseline_exec_time_ms=1000.0,
            evaluation_status=EvaluationStatus.TIMEOUT,
            sql_filepath="/tmp/q1.sql",
            timeout_threshold_ms=2000.0,
            sample_index=0,
            candidate_source="sobol_warm_start",
        )

        self.assertEqual(timeout_record["observation_source"], "timeout_censored_execution")
        self.assertTrue(timeout_record["is_timeout"])
        self.assertTrue(timeout_record["is_real_execution"])
        self.assertFalse(timeout_record["is_true_observation"])

        executed_at_tau_record = pipeline._make_result_record(
            phase="warm_start",
            knob_space=knob_space,
            vector=[0.5],
            plan_fingerprint="plan_c",
            execute_time_ms=2000.0,
            baseline_exec_time_ms=1000.0,
            evaluation_status=EvaluationStatus.EXECUTED,
            sql_filepath="/tmp/q1.sql",
            timeout_threshold_ms=2000.0,
            sample_index=1,
            candidate_source="sobol_warm_start",
        )

        self.assertEqual(executed_at_tau_record["observation_source"], "executed_plan")
        self.assertFalse(executed_at_tau_record["is_timeout"])
        self.assertTrue(executed_at_tau_record["is_true_observation"])

    def test_baseline_uses_physical_default_configuration(self) -> None:
        class FakeExecutor:
            timeout_threshold_ms = None

        class FakeDatabaseService:
            def __init__(self):
                self.executor = FakeExecutor()
                self.executed_vectors = []

            def get_relevant_knobs(self, _sql_filepath):
                return [{"var": "k1", "min": 0.1, "max": 100.0, "default": 1.0}]

            def update_knob_space(self, _knob_space):
                return None

            def execute_with_knobs(self, _sql_filepath, knob_vector, _timeout_ms, is_warm_start=False):
                self.executed_vectors.append(list(knob_vector))
                return "plan_default", 1000.0, False, EvaluationStatus.EXECUTED

            def set_timeout_threshold(self, timeout_threshold):
                self.executor.timeout_threshold_ms = timeout_threshold
                return timeout_threshold

        pipeline = object.__new__(OptimizationPipeline)
        pipeline.app_config = AppConfig(
            optimization=OptimizationConfig(
                baseline_timeout_ms=10_000,
                batch=1,
                retry_attempts=0,
                max_no_improvement=1,
            )
        )
        pipeline.db_service = FakeDatabaseService()
        pipeline.dispatcher = EventDispatcher()

        completed_events = []
        pipeline.dispatcher.subscribe(
            OptimizationCompleted.__name__,
            lambda event: completed_events.append(event),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = str(Path(tmp_dir) / "q1.json")
            pipeline.optimize(
                sql_filepath="/tmp/q1.sql",
                result_filepath=result_path,
                total_rounds=0,
                warm_start_rounds=0,
                strategy="tcbo",
            )

        self.assertAlmostEqual(pipeline.db_service.executed_vectors[0][0], 1.0 / 3.0)
        self.assertAlmostEqual(completed_events[0].results[0]["value"]["k1"], 1.0)

    def test_optimize_loop_can_improve_with_fake_db_and_guider(self) -> None:
        class FakeExecutor:
            timeout_threshold_ms = None

            def build_hinted_sql(self, sql_filepath, knobs):
                self.last_hinted_sql_args = (sql_filepath, knobs)
                return "SELECT /*+ set_var(k1='0.2') */ 1"

        class FakeDatabaseService:
            def __init__(self):
                self.executor = FakeExecutor()
                self.knob_space = None

            def get_relevant_knobs(self, _sql_filepath):
                return [{"var": "k1", "min": 0.1, "max": 10.0, "default": 1.0}]

            def update_knob_space(self, knob_space):
                self.knob_space = knob_space

            def execute_with_knobs(self, _sql_filepath, knob_vector, _timeout_ms, is_warm_start=False):
                x_value = knob_vector[0]
                latency_ms = 100.0 + ((x_value - 0.2) ** 2) * 1000.0
                plan_id = f"plan_{round(x_value, 3)}"
                return plan_id, latency_ms, False, EvaluationStatus.EXECUTED

            def set_timeout_threshold(self, timeout_threshold):
                self.executor.timeout_threshold_ms = timeout_threshold
                return timeout_threshold

        class FakeGuider:
            def __init__(self, knob_space, *_args, **_kwargs):
                self.knob_space = knob_space
                self.strategy = object()
                self.observations = []

            def warm_start_sampling(self):
                return [[0.8]]

            def record_observation(self, vector, perf, plan_id=None, plan_fingerprint=None, is_timeout=None):
                self.observations.append((vector, perf, plan_fingerprint or plan_id))

            def get_next_points(self, **_kwargs):
                return [[0.2]]

        pipeline = object.__new__(OptimizationPipeline)
        pipeline.app_config = AppConfig(
            optimization=OptimizationConfig(
                baseline_timeout_ms=10_000,
                batch=1,
                retry_attempts=0,
                max_no_improvement=2,
            )
        )
        pipeline.db_service = FakeDatabaseService()
        pipeline.dispatcher = EventDispatcher()
        pipeline.result_exporter = ResultExporter()
        pipeline.result_exporter.subscribe_to_dispatcher(pipeline.dispatcher)

        started_events = []
        completed_events = []
        pipeline.dispatcher.subscribe(
            OptimizationStarted.__name__,
            lambda event: started_events.append(event),
        )
        pipeline.dispatcher.subscribe(
            OptimizationCompleted.__name__,
            lambda event: completed_events.append(event),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = str(Path(tmp_dir) / "q1.json")
            with patch("optimization.optimization_pipeline.Guider", FakeGuider):
                pipeline.optimize(
                    sql_filepath="/tmp/q1.sql",
                    result_filepath=result_path,
                    total_rounds=2,
                    warm_start_rounds=1,
                    strategy="tcbo",
                )
            summary_path = Path(str(result_path).replace(".json", "_summary.json"))
            with open(summary_path, "r", encoding="utf-8") as file_handle:
                summary = json.load(file_handle)

            self.assertEqual(summary["best_hinted_sql"], "SELECT /*+ set_var(k1='0.2') */ 1")

        self.assertEqual(len(completed_events), 1)
        event = completed_events[0]
        self.assertLess(event.best_result["execute_time"], event.context["baseline_exec_time"])
        self.assertEqual(event.best_result["normalized_vector"], [0.2])
        self.assertEqual(event.best_result["hinted_sql"], "SELECT /*+ set_var(k1='0.2') */ 1")
        self.assertEqual(event.context["optimizer_settings"]["tcbo_num_trust_regions"], 4)
        self.assertEqual(event.context["total_rounds"], 2)
        self.assertEqual(event.context["warm_start_rounds"], 1)
        self.assertEqual(len(started_events), 1)
        self.assertEqual(started_events[0].total_rounds, 2)
        self.assertEqual(started_events[0].warm_start_rounds, 1)
        self.assertEqual(started_events[0].total_trials, 2)
        self.assertEqual(started_events[0].warm_start_times, 1)

    def test_duplicate_plan_reuse_does_not_trigger_rejection_retry(self) -> None:
        class FakeExecutor:
            timeout_threshold_ms = None

        class FakeDatabaseService:
            def __init__(self):
                self.executor = FakeExecutor()

            def get_relevant_knobs(self, _sql_filepath):
                return [{"var": "k1", "min": 0.1, "max": 10.0, "default": 1.0}]

            def update_knob_space(self, _knob_space):
                return None

            def execute_with_knobs(self, _sql_filepath, knob_vector, _timeout_ms, is_warm_start=False):
                if is_warm_start:
                    return "plan_warm", 900.0, False, EvaluationStatus.EXECUTED
                if knob_vector == [0.3]:
                    return "plan_other", 950.0, False, EvaluationStatus.EXECUTED
                return "plan_warm", 900.0, False, EvaluationStatus.DUPLICATE_PLAN

            def set_timeout_threshold(self, timeout_threshold):
                self.executor.timeout_threshold_ms = timeout_threshold
                return timeout_threshold

        class FakeGuider:
            try_numbers = []

            def __init__(self, knob_space, *_args, **_kwargs):
                self.knob_space = knob_space
                self.strategy = object()

            def warm_start_sampling(self):
                return [[0.3]]

            def record_observation(self, *_args, **_kwargs):
                return None

            def get_next_points(self, **kwargs):
                self.try_numbers.append(kwargs["try_number"])
                return [[0.4]]

        pipeline = object.__new__(OptimizationPipeline)
        pipeline.app_config = AppConfig(
            optimization=OptimizationConfig(
                baseline_timeout_ms=10_000,
                batch=1,
                retry_attempts=2,
                max_no_improvement=2,
            )
        )
        pipeline.db_service = FakeDatabaseService()
        pipeline.dispatcher = EventDispatcher()

        completed_events = []
        pipeline.dispatcher.subscribe(
            OptimizationCompleted.__name__,
            lambda event: completed_events.append(event),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = str(Path(tmp_dir) / "q1.json")
            with patch("optimization.optimization_pipeline.Guider", FakeGuider):
                pipeline.optimize(
                    sql_filepath="/tmp/q1.sql",
                    result_filepath=result_path,
                    total_rounds=2,
                    warm_start_rounds=1,
                    strategy="tcbo",
                )

        self.assertEqual(FakeGuider.try_numbers, [0])
        duplicate_results = [
            result for result in completed_events[0].results
            if result["evaluation_status"] == EvaluationStatus.DUPLICATE_PLAN.value
        ]
        self.assertEqual(len(duplicate_results), 1)
        self.assertTrue(duplicate_results[0]["is_true_observation"])
        self.assertFalse(duplicate_results[0]["is_admission_rejected"])

    def test_warm_start_uses_evaluator_admission_gate(self) -> None:
        class FakeExecutor:
            timeout_threshold_ms = None

        class FakeDatabaseService:
            def __init__(self):
                self.executor = FakeExecutor()
                self.is_warm_start_flags = []

            def get_relevant_knobs(self, _sql_filepath):
                return [{"var": "k1", "min": 0.1, "max": 10.0, "default": 1.0}]

            def update_knob_space(self, _knob_space):
                return None

            def execute_with_knobs(self, _sql_filepath, knob_vector, _timeout_ms, is_warm_start=False):
                self.is_warm_start_flags.append(is_warm_start)
                if is_warm_start:
                    return "plan_default", 1_000.0, False, EvaluationStatus.EXECUTED
                return "plan_default", 1_000.0, False, EvaluationStatus.DUPLICATE_PLAN

            def set_timeout_threshold(self, timeout_threshold):
                self.executor.timeout_threshold_ms = timeout_threshold
                return timeout_threshold

        class FakeGuider:
            observations = []
            safety_feedback = []

            def __init__(self, knob_space, *_args, **_kwargs):
                self.knob_space = knob_space
                self.strategy = object()

            def warm_start_sampling(self):
                return [[0.4]]

            def record_observation(self, vector, perf, plan_id=None, plan_fingerprint=None, is_timeout=None):
                self.observations.append((vector, perf, plan_fingerprint or plan_id))

            def record_admission_rejection(self, vector, estimated_latency=None, plan_fingerprint=None):
                self.safety_feedback.append((vector, estimated_latency, plan_fingerprint))

        FakeGuider.observations = []
        FakeGuider.safety_feedback = []

        pipeline = object.__new__(OptimizationPipeline)
        pipeline.app_config = AppConfig(
            optimization=OptimizationConfig(
                baseline_timeout_ms=10_000,
                batch=1,
                retry_attempts=0,
                max_no_improvement=2,
            )
        )
        pipeline.db_service = FakeDatabaseService()
        pipeline.dispatcher = EventDispatcher()

        completed_events = []
        pipeline.dispatcher.subscribe(
            OptimizationCompleted.__name__,
            lambda event: completed_events.append(event),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = str(Path(tmp_dir) / "q1.json")
            with patch("optimization.optimization_pipeline.Guider", FakeGuider):
                pipeline.optimize(
                    sql_filepath="/tmp/q1.sql",
                    result_filepath=result_path,
                    total_rounds=1,
                    warm_start_rounds=1,
                    strategy="tcbo",
                )

        self.assertEqual(pipeline.db_service.is_warm_start_flags, [True, False])
        self.assertEqual(FakeGuider.safety_feedback, [])
        self.assertIn(([0.4], 1_000.0, "plan_default"), FakeGuider.observations)

        warm_results = [
            result for result in completed_events[0].results
            if result["phase"] == "warm_start"
        ]
        self.assertEqual(len(warm_results), 1)
        self.assertEqual(warm_results[0]["evaluation_status"], EvaluationStatus.DUPLICATE_PLAN.value)
        self.assertTrue(warm_results[0]["is_repository_reuse"])
        self.assertTrue(warm_results[0]["is_true_observation"])
        self.assertFalse(warm_results[0]["is_admission_rejected"])

    def test_optimize_propagates_unexpected_sample_error(self) -> None:
        class FakeExecutor:
            timeout_threshold_ms = None

        class FakeDatabaseService:
            def __init__(self):
                self.executor = FakeExecutor()

            def get_relevant_knobs(self, _sql_filepath):
                return [{"var": "k1", "min": 0.1, "max": 10.0, "default": 1.0}]

            def update_knob_space(self, _knob_space):
                return None

            def execute_with_knobs(self, _sql_filepath, knob_vector, _timeout_ms, is_warm_start=False):
                if knob_vector == [0.5]:
                    return "plan_default", 1000.0, False, EvaluationStatus.EXECUTED
                raise RuntimeError("db disconnected")

            def set_timeout_threshold(self, timeout_threshold):
                self.executor.timeout_threshold_ms = timeout_threshold
                return timeout_threshold

        class FakeGuider:
            def __init__(self, knob_space, *_args, **_kwargs):
                self.knob_space = knob_space
                self.strategy = object()

            def warm_start_sampling(self):
                return []

            def record_observation(self, *_args, **_kwargs):
                return None

            def get_next_points(self, **_kwargs):
                return [[0.2]]

        pipeline = object.__new__(OptimizationPipeline)
        pipeline.app_config = AppConfig(
            optimization=OptimizationConfig(
                baseline_timeout_ms=10_000,
                batch=1,
                retry_attempts=2,
                max_no_improvement=2,
            )
        )
        pipeline.db_service = FakeDatabaseService()
        pipeline.dispatcher = EventDispatcher()

        completed_events = []
        pipeline.dispatcher.subscribe(
            OptimizationCompleted.__name__,
            lambda event: completed_events.append(event),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = str(Path(tmp_dir) / "q1.json")
            with patch("optimization.optimization_pipeline.Guider", FakeGuider):
                with self.assertRaisesRegex(RuntimeError, "db disconnected"):
                    pipeline.optimize(
                        sql_filepath="/tmp/q1.sql",
                        result_filepath=result_path,
                        total_rounds=1,
                        warm_start_rounds=0,
                        strategy="tcbo",
                    )

        self.assertEqual(completed_events, [])

    def test_optimize_records_actual_candidate_source_from_guider(self) -> None:
        class FakeExecutor:
            timeout_threshold_ms = None

        class FakeDatabaseService:
            def __init__(self):
                self.executor = FakeExecutor()

            def get_relevant_knobs(self, _sql_filepath):
                return [{"var": "k1", "min": 0.1, "max": 10.0, "default": 1.0}]

            def update_knob_space(self, _knob_space):
                return None

            def execute_with_knobs(self, _sql_filepath, knob_vector, _timeout_ms, is_warm_start=False):
                if knob_vector == [0.5]:
                    return "plan_default", 1000.0, False, EvaluationStatus.EXECUTED
                return "plan_sample", 900.0, False, EvaluationStatus.EXECUTED

            def set_timeout_threshold(self, timeout_threshold):
                self.executor.timeout_threshold_ms = timeout_threshold
                return timeout_threshold

        class FakeTCBOStrategy:
            pass

        class FakeGuider:
            def __init__(self, knob_space, *_args, **_kwargs):
                self.knob_space = knob_space
                self.strategy = FakeTCBOStrategy()
                self.last_candidate_source = ""

            def warm_start_sampling(self):
                return []

            def record_observation(self, *_args, **_kwargs):
                return None

            def get_next_points(self, **_kwargs):
                self.last_candidate_source = "TCBO Sampling"
                return [[0.3]]

        pipeline = object.__new__(OptimizationPipeline)
        pipeline.app_config = AppConfig(
            optimization=OptimizationConfig(
                baseline_timeout_ms=10_000,
                batch=1,
                retry_attempts=0,
                max_no_improvement=2,
            ),
            llm=LLMRuntimeConfig(api_key="test-key"),
        )
        pipeline.db_service = FakeDatabaseService()
        pipeline.dispatcher = EventDispatcher()

        completed_events = []
        pipeline.dispatcher.subscribe(
            OptimizationCompleted.__name__,
            lambda event: completed_events.append(event),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = str(Path(tmp_dir) / "q1.json")
            with patch("optimization.optimization_pipeline.Guider", FakeGuider):
                pipeline.optimize(
                    sql_filepath="/tmp/q1.sql",
                    result_filepath=result_path,
                    total_rounds=1,
                    warm_start_rounds=0,
                    strategy="tcbo",
                )

        optimization_results = [
            result for result in completed_events[0].results
            if result["phase"] == "optimization"
        ]
        self.assertEqual(len(optimization_results), 1)
        self.assertEqual(optimization_results[0]["candidate_source"], "TCBO Sampling")

    def test_subplan_reject_updates_safety_feedback_without_objective_observation(self) -> None:
        class FakeExecutor:
            timeout_threshold_ms = None

        class FakeDatabaseService:
            def __init__(self):
                self.executor = FakeExecutor()

            def get_relevant_knobs(self, _sql_filepath):
                return [{"var": "k1", "min": 0.1, "max": 10.0, "default": 1.0}]

            def update_knob_space(self, _knob_space):
                return None

            def execute_with_knobs(self, _sql_filepath, knob_vector, _timeout_ms, is_warm_start=False):
                if is_warm_start:
                    return "plan_warm", 900.0, False, EvaluationStatus.EXECUTED
                if knob_vector == [0.9]:
                    return "plan_bad", 3_000.0, True, EvaluationStatus.SUBPLAN_REJECTED
                return "plan_good", 700.0, False, EvaluationStatus.EXECUTED

            def set_timeout_threshold(self, timeout_threshold):
                self.executor.timeout_threshold_ms = timeout_threshold
                return timeout_threshold

        class FakeGuider:
            observations = []
            safety_feedback = []
            try_numbers = []

            def __init__(self, knob_space, *_args, **_kwargs):
                self.knob_space = knob_space
                self.strategy = object()

            def warm_start_sampling(self):
                return [[0.4]]

            def record_observation(self, vector, perf, plan_id=None, plan_fingerprint=None, is_timeout=None):
                self.observations.append((vector, perf, plan_fingerprint or plan_id))

            def record_admission_rejection(self, vector, estimated_latency=None, plan_fingerprint=None):
                self.safety_feedback.append((vector, estimated_latency, plan_fingerprint))

            def get_next_points(self, **kwargs):
                self.try_numbers.append(kwargs["try_number"])
                return [[0.9]] if kwargs["try_number"] == 0 else [[0.2]]

        FakeGuider.observations = []
        FakeGuider.safety_feedback = []
        FakeGuider.try_numbers = []

        pipeline = object.__new__(OptimizationPipeline)
        pipeline.app_config = AppConfig(
            optimization=OptimizationConfig(
                baseline_timeout_ms=10_000,
                batch=1,
                retry_attempts=2,
                max_no_improvement=2,
            )
        )
        pipeline.db_service = FakeDatabaseService()
        pipeline.dispatcher = EventDispatcher()

        completed_events = []
        pipeline.dispatcher.subscribe(
            OptimizationCompleted.__name__,
            lambda event: completed_events.append(event),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = str(Path(tmp_dir) / "q1.json")
            with patch("optimization.optimization_pipeline.Guider", FakeGuider):
                pipeline.optimize(
                    sql_filepath="/tmp/q1.sql",
                    result_filepath=result_path,
                    total_rounds=2,
                    warm_start_rounds=1,
                    strategy="tcbo",
                )

        self.assertEqual(FakeGuider.try_numbers, [0, 1])
        self.assertEqual(FakeGuider.safety_feedback, [([0.9], 3_000.0, "plan_bad")])
        self.assertNotIn(([0.9], 3_000.0, "plan_bad"), FakeGuider.observations)
        subplan_results = [
            result for result in completed_events[0].results
            if result["evaluation_status"] == EvaluationStatus.SUBPLAN_REJECTED.value
        ]
        self.assertEqual(len(subplan_results), 1)
        self.assertFalse(subplan_results[0]["is_true_observation"])
        self.assertTrue(subplan_results[0]["is_admission_rejected"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
