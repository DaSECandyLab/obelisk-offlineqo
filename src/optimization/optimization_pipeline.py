"""
Core optimization pipeline with event-driven architecture
"""
import os
import time
from typing import Dict, List

from optimization.database_service import DatabaseService
from optimization.events.event_definitions import (
    EventDispatcher, OptimizationCompleted, OptimizationStarted
)
from optimization.evaluator import (
    EvaluationStatus,
    is_admission_estimate,
    is_true_observation,
)
from optimization.guider import Guider
from optimization.result_exporter import ResultExporter
from llm.llm_config import LLMConfig
from util.config import AppConfig, load_app_config
from util.knob_space import KnobSpace
from util.logger import logger


class OptimizationPipeline:
    """Core optimization pipeline with event-driven architecture"""

    def __init__(
        self,
        repository_name: str = "",
        autocommit: bool = True,
        app_config: AppConfig | None = None,
        cache_name: str | None = None,
    ):
        self.app_config = app_config or load_app_config()
        selected_repository = cache_name if cache_name is not None else repository_name
        self.db_service = DatabaseService(
            repository_name=selected_repository,
            autocommit=autocommit,
            app_config=self.app_config,
        )
        self.dispatcher = EventDispatcher()
        self.result_exporter = ResultExporter()
        self.result_exporter.subscribe_to_dispatcher(self.dispatcher)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db_service.close()

    @staticmethod
    def _get_plan_fingerprint_preview(plan_fingerprint: str) -> str:
        """Get a short display form of the plan fingerprint F(P)."""
        if plan_fingerprint and len(plan_fingerprint) >= 8:
            return f"[{plan_fingerprint[:4]}...{plan_fingerprint[-4:]}]"
        return "[unknown]"

    @staticmethod
    def _calculate_improvement(current_exec_time: float, baseline_exec_time: float) -> float:
        """Calculate percentage improvement in execution time"""
        if baseline_exec_time == 0:
            return 0.0
        return ((baseline_exec_time - current_exec_time) / baseline_exec_time) * 100

    def _log_phase(self, phase: str, iteration: int = None) -> None:
        if iteration is not None:
            logger.info(f"Phase: {phase} iteration={iteration + 1}")
        else:
            logger.info(f"Phase: {phase}")

    def _print_sample_result(self, sample_type: str, exec_time: float, improvement: float,
                            plan_fingerprint: str, is_best: bool = False) -> None:
        """Print execution result with metrics"""
        status = "[NEW BEST]" if is_best else "[OK]"
        logger.info(f"{status} {sample_type}: {exec_time/1000:.3f}s ({improvement:+.1f}%) [{plan_fingerprint}]")

    def _observation_source(
        self,
        evaluation_status: EvaluationStatus,
        execute_time_ms: float,
        timeout_threshold_ms: float | None,
    ) -> str:
        if evaluation_status == EvaluationStatus.DUPLICATE_PLAN:
            return "plan_repository_duplicate"
        if evaluation_status == EvaluationStatus.SUBPLAN_REJECTED:
            return "subplan_admission_estimate"
        if evaluation_status == EvaluationStatus.TIMEOUT:
            return "timeout_censored_execution"
        return "executed_plan"

    def _make_result_record(
        self,
        *,
        phase: str,
        knob_space: KnobSpace,
        vector: List[float],
        plan_fingerprint: str,
        execute_time_ms: float,
        baseline_exec_time_ms: float,
        evaluation_status: EvaluationStatus,
        sql_filepath: str,
        timeout_threshold_ms: float | None,
        iteration: int | None = None,
        sample_index: int | None = None,
        candidate_source: str = "",
    ) -> Dict:
        admission_estimate = is_admission_estimate(evaluation_status)
        source = self._observation_source(
            evaluation_status,
            execute_time_ms,
            timeout_threshold_ms,
        )
        is_timeout = source == "timeout_censored_execution"
        true_objective_observation = (
            is_true_observation(evaluation_status)
            and not is_timeout
        )
        return {
            "phase": phase,
            "iteration": iteration,
            "sample_index": sample_index,
            "candidate_source": candidate_source,
            "value": knob_space.vector_to_config(vector),
            "normalized_vector": vector,
            "plan_fingerprint": plan_fingerprint,
            "execute_time": execute_time_ms,
            "improvement": self._calculate_improvement(execute_time_ms, baseline_exec_time_ms),
            "is_real_execution": evaluation_status in {
                EvaluationStatus.EXECUTED,
                EvaluationStatus.TIMEOUT,
            },
            "is_true_observation": true_objective_observation,
            "is_repository_reuse": evaluation_status == EvaluationStatus.DUPLICATE_PLAN,
            "is_admission_estimate": admission_estimate,
            "is_admission_rejected": admission_estimate,
            "is_timeout": is_timeout,
            "observation_source": source,
            "evaluation_status": evaluation_status.value,
            "sql_file": os.path.basename(sql_filepath),
        }

    @staticmethod
    def _optimizer_settings(optimization_config, timeout_threshold_ms: float) -> Dict:
        return {
            "timeout_multiplier": optimization_config.timeout_multiplier,
            "timeout_threshold_ms": timeout_threshold_ms,
            "topk": optimization_config.topk,
            "batch": optimization_config.batch,
            "retry_attempts": optimization_config.retry_attempts,
            "max_no_improvement": optimization_config.max_no_improvement,
            "tcbo_num_trust_regions": optimization_config.tcbo_num_trust_regions,
            "tcbo_risk_threshold": optimization_config.tcbo_risk_threshold,
            "tcbo_candidate_count": optimization_config.tcbo_candidate_count,
        }

    @staticmethod
    def _candidate_phase_name(
        strategy_name: str,
        try_number: int,
        reasoner_enabled: bool,
    ) -> str:
        if not reasoner_enabled:
            return f"{strategy_name} Sampling"
        if try_number == 0:
            return f"{strategy_name}+Reasoner"
        return f"{strategy_name}+Reasoner Prompt Optimization"

    def optimize(
        self,
        sql_filepath: str,
        result_filepath: str,
        total_rounds: int | None = None,
        warm_start_rounds: int | None = None,
        strategy: str = "tcbo",
        *,
        total_trials: int | None = None,
        warm_start_times: int | None = None,
    ) -> None:
        """Main optimization pipeline"""
        if total_rounds is None:
            total_rounds = total_trials
        if warm_start_rounds is None:
            warm_start_rounds = warm_start_times
        if total_rounds is None or warm_start_rounds is None:
            raise ValueError("total_rounds and warm_start_rounds are required")
        total_rounds = int(total_rounds)
        warm_start_rounds = int(warm_start_rounds)
        if total_rounds < 0:
            raise ValueError("total_rounds must be non-negative")
        if warm_start_rounds < 0:
            raise ValueError("warm_start_rounds must be non-negative")
        if warm_start_rounds > total_rounds:
            raise ValueError("warm_start_rounds cannot exceed total_rounds")

        try:
            exp_start_ts = time.time()
            self.dispatcher.publish(OptimizationStarted(
                timestamp=exp_start_ts,
                event_type="optimization_started",
                context={},
                sql_filepath=sql_filepath,
                result_filepath=result_filepath,
                total_rounds=total_rounds,
                warm_start_rounds=warm_start_rounds,
                strategy=strategy,
            ))

            logger.info(
                "SQL optimization started: file=%s total_rounds=%d warm_start_rounds=%d strategy=%s",
                os.path.basename(sql_filepath),
                total_rounds,
                warm_start_rounds,
                strategy,
            )

            optimization_config = self.app_config.optimization
            relevant_knobs = self.db_service.get_relevant_knobs(sql_filepath)
            results: List[Dict] = []
            baseline_timeout_ms = optimization_config.baseline_timeout_ms

            self._log_phase("BASELINE")
            knob_space = KnobSpace.from_search_space(relevant_knobs)
            self.db_service.update_knob_space(knob_space)
            default_vector = knob_space.get_default_vector()
            plan_fingerprint, baseline_exec_time, _, evaluation_status = self.db_service.execute_with_knobs(
                sql_filepath,
                default_vector,
                baseline_timeout_ms,
                is_warm_start=True,
            )
            results.append(self._make_result_record(
                phase="baseline",
                knob_space=knob_space,
                vector=default_vector,
                plan_fingerprint=plan_fingerprint,
                execute_time_ms=baseline_exec_time,
                baseline_exec_time_ms=baseline_exec_time,
                evaluation_status=evaluation_status,
                sql_filepath=sql_filepath,
                timeout_threshold_ms=baseline_timeout_ms,
                sample_index=0,
                candidate_source="default_configuration",
            ))

            fingerprint_preview = self._get_plan_fingerprint_preview(plan_fingerprint)
            logger.info(f"Baseline execution time: {baseline_exec_time/1000:.3f} seconds {fingerprint_preview:>15}")

            timeout_threshold = optimization_config.timeout_multiplier * baseline_exec_time
            timeout_threshold = self.db_service.set_timeout_threshold(timeout_threshold)
            logger.info(
                "Evaluator timeout threshold tau set to %.3f seconds (%.2fx baseline)",
                timeout_threshold / 1000,
                optimization_config.timeout_multiplier,
            )
            optimizer_settings = self._optimizer_settings(
                optimization_config,
                timeout_threshold,
            )

            llm_config = LLMConfig.from_app_config(self.app_config)
            guider = Guider(
                knob_space=knob_space,
                warm_start_rounds=warm_start_rounds,
                strategy=strategy,
                timeout_threshold=timeout_threshold,
                llm_config=llm_config,
                tcbo_num_trust_regions=optimization_config.tcbo_num_trust_regions,
                tcbo_risk_threshold=optimization_config.tcbo_risk_threshold,
                tcbo_candidate_count=optimization_config.tcbo_candidate_count,
            )
            guider.record_observation(
                default_vector,
                baseline_exec_time,
                plan_fingerprint=plan_fingerprint,
                is_timeout=evaluation_status == EvaluationStatus.TIMEOUT,
            )
            self.db_service.update_knob_space(guider.knob_space)
            no_improvement_count = 0
            max_no_improvement = optimization_config.max_no_improvement
            previous_top5 = []
            best_exec_time = baseline_exec_time

            self._log_phase("WARM_START")
            timeout_threshold_ms = self.db_service.executor.timeout_threshold_ms
            logger.info(
                "Warm-start Evaluator timeout tau: %.3fs",
                timeout_threshold_ms / 1000,
            )
            warm_vectors = guider.warm_start_sampling()

            for i, vector in enumerate(warm_vectors, 1):
                plan_fingerprint, current_exec_time, is_rejected, evaluation_status = self.db_service.execute_with_knobs(
                    sql_filepath,
                    vector,
                    timeout_threshold_ms,
                    is_warm_start=False,
                )
                if is_rejected:
                    guider.record_admission_rejection(
                        vector,
                        estimated_latency=current_exec_time,
                        plan_fingerprint=plan_fingerprint,
                    )
                else:
                    guider.record_observation(
                        vector,
                        current_exec_time,
                        plan_fingerprint=plan_fingerprint,
                        is_timeout=evaluation_status == EvaluationStatus.TIMEOUT,
                    )
                    best_exec_time = min(current_exec_time, best_exec_time)
                improvement = self._calculate_improvement(current_exec_time, baseline_exec_time)
                results.append(self._make_result_record(
                    phase="warm_start",
                    knob_space=guider.knob_space,
                    vector=vector,
                    plan_fingerprint=plan_fingerprint,
                    execute_time_ms=current_exec_time,
                    baseline_exec_time_ms=baseline_exec_time,
                    evaluation_status=evaluation_status,
                    sql_filepath=sql_filepath,
                    timeout_threshold_ms=timeout_threshold_ms,
                    sample_index=i - 1,
                    candidate_source="sobol_warm_start",
                ))

                if not is_rejected:
                    is_best = current_exec_time == best_exec_time
                    if evaluation_status == EvaluationStatus.DUPLICATE_PLAN:
                        logger.info(
                            "Reused warm-start duplicate plan %d from repository: time=%.3fs plan=%s",
                            i,
                            current_exec_time / 1000,
                            plan_fingerprint,
                        )
                    else:
                        self._print_sample_result(f"Warm sample {i}", current_exec_time, improvement, plan_fingerprint, is_best)
                else:
                    logger.info(
                        "Admission-rejected warm sample %d: status=%s estimate=%.3fs plan=%s",
                        i,
                        evaluation_status.value,
                        current_exec_time / 1000,
                        plan_fingerprint,
                    )

            for iteration in range(max(0, total_rounds - warm_start_rounds)):
                self._log_phase("OPTIMIZATION", iteration)

                try_number = 0
                failed_vectors = []
                while try_number <= optimization_config.retry_attempts:
                    strategy_name = type(guider.strategy).__name__.replace("Strategy", "")
                    phase_name = self._candidate_phase_name(
                        strategy_name,
                        try_number,
                        llm_config.can_call_remote(),
                    )
                    logger.info(f"Phase: {phase_name} (try {try_number + 1})")

                    sample_vectors = guider.get_next_points(
                        sql_path=sql_filepath,
                        topk=optimization_config.topk,
                        try_number=try_number,
                        last_failed_vectors=failed_vectors[-5:],
                        batch=optimization_config.batch,
                    )
                    candidate_source = getattr(
                        guider,
                        "last_candidate_source",
                        "",
                    ) or phase_name
                    if candidate_source != phase_name:
                        logger.info("Candidate source resolved to: %s", candidate_source)

                    reject_list, accept_list = [], []

                    for i, vector in enumerate(sample_vectors):
                        try:
                            plan_fingerprint, current_exec_time, admission_rejected, evaluation_status = self.db_service.execute_with_knobs(
                                sql_filepath,
                                vector,
                                timeout_threshold_ms,
                                is_warm_start=False,
                            )
                            if admission_rejected:
                                guider.record_admission_rejection(
                                    vector,
                                    estimated_latency=current_exec_time,
                                    plan_fingerprint=plan_fingerprint,
                                )
                            else:
                                guider.record_observation(
                                    vector,
                                    current_exec_time,
                                    plan_fingerprint=plan_fingerprint,
                                    is_timeout=evaluation_status == EvaluationStatus.TIMEOUT,
                                )
                                best_exec_time = min(current_exec_time, best_exec_time)
                            improvement = self._calculate_improvement(current_exec_time, baseline_exec_time)
                            results.append(self._make_result_record(
                                phase="optimization",
                                knob_space=guider.knob_space,
                                vector=vector,
                                plan_fingerprint=plan_fingerprint,
                                execute_time_ms=current_exec_time,
                                baseline_exec_time_ms=baseline_exec_time,
                                evaluation_status=evaluation_status,
                                sql_filepath=sql_filepath,
                                timeout_threshold_ms=timeout_threshold_ms,
                                iteration=iteration,
                                sample_index=i,
                                candidate_source=candidate_source,
                            ))

                            if not admission_rejected:
                                is_best = current_exec_time == best_exec_time
                                if evaluation_status == EvaluationStatus.DUPLICATE_PLAN:
                                    logger.info(
                                        "Reused duplicate plan %d from repository: time=%.3fs plan=%s",
                                        i,
                                        current_exec_time / 1000,
                                        plan_fingerprint,
                                    )
                                else:
                                    self._print_sample_result(f"Sample {i}", current_exec_time, improvement, plan_fingerprint, is_best)
                                accept_list.append(vector)
                            else:
                                logger.info(
                                    "Admission-rejected sample %d: status=%s estimate=%.3fs plan=%s",
                                    i,
                                    evaluation_status.value,
                                    current_exec_time / 1000,
                                    plan_fingerprint,
                                )
                                reject_list.append(vector)

                        except Exception:
                            logger.exception("Error processing sample %d", i)
                            raise

                    if accept_list:
                        logger.info(
                            "Executed/reused %d samples, admission-rejected %d samples",
                            len(accept_list),
                            len(reject_list),
                        )
                        break
                    else:
                        logger.warning(
                            "All %d samples were admission-rejected, trying next strategy",
                            len(sample_vectors),
                        )
                        failed_vectors.extend(reject_list)
                        try_number += 1

                    if try_number > optimization_config.retry_attempts:
                        logger.error("All retry attempts failed, using fallback")
                        break

                measured_results = [
                    r for r in results
                    if not r.get("is_admission_estimate", False)
                ]
                sorted_results = sorted(measured_results, key=lambda x: x["execute_time"])
                current_top5 = [r["execute_time"] for r in sorted_results[:5]] if sorted_results else []

                has_improvement = False
                if not previous_top5:
                    has_improvement = True
                elif len(current_top5) != len(previous_top5):
                    has_improvement = True
                else:
                    has_improvement = any(curr < prev for curr, prev in zip(current_top5, previous_top5))

                if has_improvement:
                    no_improvement_count = 0
                    previous_top5 = current_top5
                else:
                    no_improvement_count += 1
                    logger.info(f"No improvement for {no_improvement_count} iterations")

                if no_improvement_count >= max_no_improvement:
                    logger.info(f"Early stopping: No improvement for {max_no_improvement} consecutive iterations")
                    break

            measured_results = [
                r for r in results
                if not r.get("is_admission_estimate", False)
            ]
            if measured_results:
                best_result = min(measured_results, key=lambda x: x["execute_time"])
                logger.info(
                    "Best result from %d measured/repository-reused observations out of %d total results",
                    len(measured_results),
                    len(results),
                )
            else:
                best_result = min(results, key=lambda x: x["execute_time"])
                logger.warning("No measured observations found, using all results for best result")

            self._attach_hinted_sql(sql_filepath, best_result)
            self._log_final_summary(
                sql_filepath,
                baseline_exec_time,
                best_result["execute_time"],
                best_result,
                result_filepath,
            )
            self.dispatcher.publish(OptimizationCompleted(
                timestamp=time.time(),
                event_type="optimization_completed",
                context={
                    "result_path": result_filepath,
                    "baseline_exec_time": baseline_exec_time,
                    "warm_start_rounds": warm_start_rounds,
                    "total_rounds": total_rounds,
                    "strategy": strategy,
                    "timeout_threshold_ms": timeout_threshold_ms,
                    "optimizer_settings": optimizer_settings,
                },
                results=results,
                best_result=best_result,
                total_duration=time.time() - exp_start_ts
            ))

        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            raise

    def _attach_hinted_sql(self, sql_filepath: str, best_result: Dict) -> None:
        """Attach the hinted SQL that reproduces the best observed configuration."""
        build_hinted_sql = getattr(self.db_service.executor, "build_hinted_sql", None)
        if not callable(build_hinted_sql):
            return
        try:
            best_result["hinted_sql"] = build_hinted_sql(
                sql_filepath,
                best_result.get("value", {}),
            )
        except Exception as error:
            logger.warning("Failed to build hinted SQL for best result: %s", error)

    def _log_final_summary(self, sql_filepath: str, baseline_exec_time: float, best_exec_time: float,
                           best_result: Dict, result_path: str) -> None:
        improvement = self._calculate_improvement(best_exec_time, baseline_exec_time)
        logger.info(
            "Optimization summary: file=%s baseline=%.3fs best=%.3fs improvement=%.2f%% plan=%s result=%s",
            os.path.basename(sql_filepath),
            baseline_exec_time / 1000,
            best_exec_time / 1000,
            improvement,
            best_result.get("plan_fingerprint", best_result.get("plan_id", "")),
            result_path,
        )


class SQLService(OptimizationPipeline):
    def __init__(
        self,
        repository_name: str = "",
        autocommit: bool = True,
        app_config: AppConfig | None = None,
        cache_name: str | None = None,
    ) -> None:
        super().__init__(
            repository_name=repository_name,
            autocommit=autocommit,
            app_config=app_config,
            cache_name=cache_name,
        )
