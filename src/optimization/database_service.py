"""
Database service for SQL execution and connection management
"""
import re
from typing import List, Tuple

from db.db_connection import DBConnectionFactory
from db.sql_executor import SQLExecutor
from optimization.evaluator import EvaluationStatus, is_admission_estimate
from util.config import AppConfig
from util.logger import logger


class DatabaseService:
    """Handles database connections and SQL execution"""

    def __init__(
        self,
        repository_name: str = "",
        autocommit: bool = True,
        app_config: AppConfig | None = None,
        cache_name: str | None = None,
    ) -> None:
        self.connection = None
        self.cursor = None
        self.executor = None
        self.app_config = app_config
        selected_repository = cache_name if cache_name is not None else repository_name
        self.repository_name = selected_repository or self._repository_name_for_schema()
        self.cache_name = self.repository_name
        self._initialize(autocommit)

    def _repository_name_for_schema(self) -> str:
        if self.app_config is None:
            return "default"
        raw_name = self.app_config.database.name or "default"
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name).strip("._")
        return safe_name or "default"

    def _initialize(self, autocommit: bool) -> None:
        """Initialize database connection"""
        try:
            self.connection = DBConnectionFactory.create_connection(
                autocommit,
                app_config=self.app_config,
            )
            self.cursor = self.connection.cursor()
            self.executor = SQLExecutor(self.cursor, repository_name=self.repository_name)
        except Exception as e:
            logger.error(f"Failed to initialize SQLService: {str(e)}")
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_relevant_knobs(self, sql_filepath: str):
        """Get relevant knobs for a given SQL file"""
        return self.executor.get_relevant_knobs(sql_filepath)

    def execute_with_knobs(
        self,
        sql_filepath: str,
        knob_vector: List[float],
        timeout_ms: int,
        is_warm_start: bool = False,
    ) -> Tuple[str, float, bool, EvaluationStatus]:
        """Execute SQL with given knob vector"""
        try:
            knob_config = self.executor.knob_space.vector_to_config(knob_vector)

            if is_warm_start:
                plan_fingerprint, execution_time, timed_out = self._execute_with_timeout_result(
                    sql_filepath,
                    knob_config,
                    timeout_ms,
                )
                status = EvaluationStatus.TIMEOUT if timed_out else EvaluationStatus.EXECUTED
                return plan_fingerprint, execution_time, False, status

            repository_plan_fingerprint, repository_execution_time, admission_status = self.executor.admission_check(
                sql_filepath,
                knob_config,
                timeout_ms,
            )

            if repository_execution_time is None:
                plan_fingerprint, execution_time, timed_out = self._execute_with_timeout_result(
                    sql_filepath,
                    knob_config,
                    timeout_ms,
                )
                status = EvaluationStatus.TIMEOUT if timed_out else EvaluationStatus.EXECUTED
                return plan_fingerprint, execution_time, False, status

            return (
                repository_plan_fingerprint,
                repository_execution_time,
                is_admission_estimate(admission_status),
                admission_status,
            )

        except Exception as e:
            logger.error(f"Error executing SQL with knobs: {str(e)}")
            raise

    def _execute_with_timeout_result(
        self,
        sql_filepath: str,
        knob_config: dict,
        timeout_ms: int,
    ) -> Tuple[str, float, bool]:
        execute_with_status = getattr(self.executor, "execute_with_timeout_result", None)
        if callable(execute_with_status):
            return execute_with_status(sql_filepath, knob_config, timeout_ms)

        plan_fingerprint, execution_time = self.executor.execute_with_timeout(
            sql_filepath,
            knob_config,
            timeout_ms,
        )
        return plan_fingerprint, execution_time, execution_time >= timeout_ms

    def set_timeout_threshold(self, timeout_threshold_ms: float) -> float:
        """Set the Evaluator timeout threshold tau."""
        return self.executor.set_timeout_threshold(timeout_threshold_ms)

    def set_fail_time(self, timeout_threshold: float) -> float:
        """Backward-compatible alias for older scripts."""
        return self.set_timeout_threshold(timeout_threshold)

    def update_knob_space(self, knob_space):
        """Update the knob space used by the executor"""
        self.executor.knob_space = knob_space

    def close(self) -> None:
        """Close database connections"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
