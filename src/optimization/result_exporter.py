"""
Result exporter that handles saving and exporting optimization results
"""
import json
import os
from statistics import fmean, median
from typing import Dict, List

from optimization.events.event_definitions import (
    OptimizationCompleted
)
from optimization.evaluator import EvaluationStatus
from util.logger import logger


class ResultExporter:
    """Handles saving and exporting optimization results"""

    def __init__(self):
        self.event_handlers = {
            OptimizationCompleted.__name__: self.on_optimization_completed,
        }

    def subscribe_to_dispatcher(self, dispatcher):
        """Subscribe all handlers to the event dispatcher"""
        for event_type, handler in self.event_handlers.items():
            dispatcher.subscribe(event_type, handler)

    def on_optimization_completed(self, event: OptimizationCompleted):
        """Handle optimization completed event and save results"""
        results = event.results
        best_result = event.best_result
        result_path = event.context.get("result_path", "")
        baseline_exec_time = event.context.get("baseline_exec_time", 0.0)
        warm_start_rounds = event.context.get(
            "warm_start_rounds",
            event.context.get("warm_start_times", 0),
        )
        total_rounds = event.context.get(
            "total_rounds",
            event.context.get("total_trials"),
        )
        optimizer_settings = event.context.get("optimizer_settings", {})

        if not result_path:
            raise ValueError("result_path is required for saving optimization results")

        self._save_results(results, result_path)
        self._save_summary(
            best_result,
            baseline_exec_time,
            result_path,
            optimizer_settings=optimizer_settings,
            total_rounds=total_rounds,
            warm_start_rounds=warm_start_rounds,
        )
        self._save_data_summary(
            results,
            result_path,
            baseline_exec_time,
            warm_start_rounds,
            optimizer_settings=optimizer_settings,
            total_rounds=total_rounds,
        )

    def _save_results(self, results: List[Dict], result_filepath: str):
        """Save optimization results to JSON file"""
        try:
            with open(result_filepath, 'w') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
            raise

    def _save_summary(
        self,
        best_result: Dict,
        baseline_exec_time: float,
        result_path: str,
        optimizer_settings: Dict | None = None,
        total_rounds: int | None = None,
        warm_start_rounds: int | None = None,
    ) -> None:
        """Save optimization summary"""
        if not best_result:
            return

        improvement = self._calculate_improvement(best_result["execute_time"], baseline_exec_time)

        summary = {
            "sql_file": best_result.get("sql_file", ""),
            "baseline_time": baseline_exec_time,
            "best_time": best_result["execute_time"],
            "improvement_percent": improvement,
            "best_plan_fingerprint": self._plan_fingerprint(best_result),
            "best_config": best_result["value"],
            "best_hinted_sql": best_result.get("hinted_sql", ""),
            "best_observation_source": best_result.get("observation_source", ""),
            "best_evaluation_status": best_result.get("evaluation_status", ""),
            "optimizer_settings": optimizer_settings or {},
        }
        if total_rounds is not None:
            summary["total_rounds"] = total_rounds
        if warm_start_rounds is not None:
            summary["warm_start_rounds"] = warm_start_rounds

        summary_path = result_path.replace('.json', '_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    def _save_data_summary(
        self,
        results: List[Dict],
        result_path: str,
        baseline_time_ms: float,
        warm_start_rounds: int,
        optimizer_settings: Dict | None = None,
        total_rounds: int | None = None,
    ) -> None:
        """Save detailed data summary"""
        if not results:
            return

        out_dir = os.path.dirname(os.path.abspath(result_path))
        os.makedirs(out_dir, exist_ok=True)

        total = len(results)
        real_count = sum(1 for r in results if r.get("is_real_execution", True))
        admission_estimate_count = sum(1 for r in results if r.get("is_admission_estimate", False))
        true_observation_count = sum(
            1 for r in results
            if self._is_true_objective_observation(r)
        )
        est_count = admission_estimate_count
        repository_reuse_count = sum(
            1 for r in results
            if r.get("evaluation_status") == EvaluationStatus.DUPLICATE_PLAN.value
        )
        timeout_count = sum(1 for r in results if r.get("is_timeout", False))
        distinct_plan_count = len({
            self._plan_fingerprint(r)
            for r in results
            if self._plan_fingerprint(r)
        })
        status_counts = self._count_by_key(results, "evaluation_status")
        source_counts = self._count_by_key(results, "observation_source")
        candidate_source_counts = self._count_by_key(results, "candidate_source")

        phase_slices = self._phase_slices(results, warm_start_rounds)

        def _stats(slice_list: List[Dict]) -> Dict:
            if not slice_list:
                return {
                    "num_samples": 0,
                    "real_count": 0,
                    "true_observation_count": 0,
                    "estimated_count": 0,
                    "admission_estimate_count": 0,
                    "repository_reuse_count": 0,
                    "best_time_ms": None,
                    "avg_time_ms": None,
                    "median_time_ms": None,
                }
            times = [r.get("execute_time", 0.0) for r in slice_list]
            measured_times = [
                r.get("execute_time", 0.0)
                for r in slice_list
                if not r.get("is_admission_estimate", False)
            ]
            times_sorted = sorted(times)
            best = min(times) if times else None
            best_measured = min(measured_times) if measured_times else None
            avg = float(fmean(times)) if times else None
            med = float(median(times_sorted)) if times else None
            real_c = sum(1 for r in slice_list if r.get("is_real_execution", True))
            estimate_c = sum(1 for r in slice_list if r.get("is_admission_estimate", False))
            true_observation_c = sum(
                1 for r in slice_list
                if self._is_true_objective_observation(r)
            )
            repository_reuse_c = sum(
                1 for r in slice_list
                if r.get("evaluation_status") == EvaluationStatus.DUPLICATE_PLAN.value
            )
            timeout_c = sum(1 for r in slice_list if r.get("is_timeout", False))
            return {
                "num_samples": len(slice_list),
                "real_count": real_c,
                "true_observation_count": true_observation_c,
                "estimated_count": estimate_c,
                "admission_estimate_count": estimate_c,
                "repository_reuse_count": repository_reuse_c,
                "timeout_count": timeout_c,
                "best_time_ms": best,
                "best_measured_time_ms": best_measured,
                "avg_time_ms": avg,
                "median_time_ms": med,
            }

        baseline_stats = _stats(phase_slices["baseline"])
        warm_stats = _stats(phase_slices["warm_start"])
        opt_stats = _stats(phase_slices["optimization"])

        # Global best
        measured = [r for r in results if not r.get("is_admission_estimate", False)]
        if measured:
            best_time_ms = min(r.get("execute_time", float("inf")) for r in measured)
            best_index = min(
                (
                    (i, r)
                    for i, r in enumerate(results)
                    if not r.get("is_admission_estimate", False)
                ),
                key=lambda item: item[1].get("execute_time", float("inf")),
            )[0]
        else:
            best_time_ms = min(r.get("execute_time", float("inf")) for r in results)
            best_index = min(
                range(len(results)),
                key=lambda i: results[i].get("execute_time", float("inf")),
            )

        improvement_pct = self._calculate_improvement(best_time_ms, baseline_time_ms)

        # Time series
        series = [
            {
                "index": i,
                "phase": r.get("phase", ""),
                "iteration": r.get("iteration"),
                "sample_index": r.get("sample_index"),
                "execute_time_ms": r.get("execute_time", 0.0),
                "is_real_execution": r.get("is_real_execution", True),
                "is_true_observation": r.get(
                    "is_true_observation",
                    self._default_true_objective_observation(r),
                ),
                "is_repository_reuse": self._is_repository_reuse(r),
                "is_admission_estimate": r.get("is_admission_estimate", False),
                "is_admission_rejected": r.get("is_admission_rejected", False),
                "is_timeout": r.get("is_timeout", False),
                "observation_source": r.get("observation_source", ""),
                "evaluation_status": r.get("evaluation_status", ""),
                "plan_fingerprint": self._plan_fingerprint(r),
            }
            for i, r in enumerate(results)
        ]

        summary = {
            "total_samples": total,
            "real_executions": real_count,
            "true_observations": true_observation_count,
            "estimated_executions": est_count,
            "admission_estimates": admission_estimate_count,
            "repository_plan_reuses": repository_reuse_count,
            "timeout_observations": timeout_count,
            "distinct_plans": distinct_plan_count,
            "baseline_time_ms": baseline_time_ms,
            "best_time_ms": best_time_ms,
            "best_observation_index": best_index,
            "improvement_percent": improvement_pct,
            "evaluation_status_counts": status_counts,
            "observation_source_counts": source_counts,
            "candidate_source_counts": candidate_source_counts,
            "optimizer_settings": optimizer_settings or {},
            "total_rounds": total_rounds,
            "warm_start_rounds": warm_start_rounds,
            "phases": {
                "baseline": baseline_stats,
                "warm_start": warm_stats,
                "optimization": opt_stats,
            },
            "time_series": series,
        }

        out_path = os.path.join(out_dir, os.path.basename(result_path).replace('.json', '_data_summary.json'))
        with open(out_path, 'w') as f:
            json.dump(summary, f, indent=2)

    @staticmethod
    def _calculate_improvement(current_exec_time: float, baseline_exec_time: float) -> float:
        """Calculate percentage improvement in execution time"""
        if baseline_exec_time == 0:
            return 0.0
        return ((baseline_exec_time - current_exec_time) / baseline_exec_time) * 100

    @staticmethod
    def _default_true_objective_observation(result: Dict) -> bool:
        return (
            not result.get("is_admission_estimate", False)
            and not result.get("is_timeout", False)
        )

    @classmethod
    def _is_true_objective_observation(cls, result: Dict) -> bool:
        return result.get(
            "is_true_observation",
            cls._default_true_objective_observation(result),
        )

    @staticmethod
    def _is_repository_reuse(result: Dict) -> bool:
        return bool(
            result.get("is_repository_reuse")
            or result.get("is_cached_observation")
            or result.get("evaluation_status") == EvaluationStatus.DUPLICATE_PLAN.value
        )

    @staticmethod
    def _plan_fingerprint(result: Dict) -> str:
        return str(result.get("plan_fingerprint") or result.get("plan_id") or "")

    @staticmethod
    def _count_by_key(results: List[Dict], key: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for result in results:
            value = str(result.get(key, "") or "unknown")
            counts[value] = counts.get(value, 0) + 1
        return counts

    @staticmethod
    def _phase_slices(results: List[Dict], warm_start_rounds: int) -> Dict[str, List[Dict]]:
        if any(result.get("phase") for result in results):
            return {
                "baseline": [r for r in results if r.get("phase") == "baseline"],
                "warm_start": [r for r in results if r.get("phase") == "warm_start"],
                "optimization": [r for r in results if r.get("phase") == "optimization"],
            }

        return {
            "baseline": results[0:1],
            "warm_start": results[1:1 + max(0, warm_start_rounds)] if len(results) > 1 else [],
            "optimization": results[1 + max(0, warm_start_rounds):],
        }
