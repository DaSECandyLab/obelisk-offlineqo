#!/usr/bin/env python3
"""Unit tests for TiDB connection session validation."""

# ruff: noqa: E402

import sys
import unittest
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from db.db_connection import DBConnectionFactory
from optimization.evaluator import EvaluationStatus
from optimization.database_service import DatabaseService
from util.config import AppConfig, DatabaseConfig
from util.knob_space import KnobSpace


class FakeCursor:
    def __init__(self, fetches):
        self._fetches = list(fetches)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def execute(self, _query):
        return 1

    def fetchone(self):
        return self._fetches.pop(0)


class FakeConnection:
    def __init__(self, fetches):
        self._fetches = fetches
        self.closed = False

    def cursor(self):
        return FakeCursor(self._fetches)

    def close(self):
        self.closed = True


class TestDBConnectionFactory(unittest.TestCase):
    def test_rejects_enabled_copr_cache(self) -> None:
        connection = FakeConnection([
            {"@@tidb_mem_quota_query": 1},
            {"value": "128"},
        ])

        with self.assertRaises(RuntimeError):
            DBConnectionFactory._initialize_session_settings(
                connection,
                mem_quota_bytes=1,
                validate_copr_cache=True,
            )
        self.assertTrue(connection.closed)

    def test_accepts_disabled_copr_cache(self) -> None:
        connection = FakeConnection([
            {"@@tidb_mem_quota_query": 1},
            {"Value": "0"},
        ])

        DBConnectionFactory._initialize_session_settings(
            connection,
            mem_quota_bytes=1,
            validate_copr_cache=True,
        )
        self.assertFalse(connection.closed)

    def test_database_service_defaults_repository_to_schema_name(self) -> None:
        service = object.__new__(DatabaseService)
        service.app_config = AppConfig(database=DatabaseConfig(name="tenant/schema"))

        self.assertEqual(service._repository_name_for_schema(), "tenant_schema")

    def test_duplicate_plan_reuse_is_true_observation_not_admission_reject(self) -> None:
        class FakeExecutor:
            knob_space = KnobSpace(["k1"])

            def admission_check(self, *_args):
                return "plan_a", 123.0, EvaluationStatus.DUPLICATE_PLAN

            def execute_with_timeout(self, *_args):
                raise AssertionError("duplicate plans must not be re-executed")

        service = object.__new__(DatabaseService)
        service.executor = FakeExecutor()

        plan_id, latency, admission_rejected, status = service.execute_with_knobs(
            "q.sql",
            [0.5],
            1000,
            is_warm_start=False,
        )

        self.assertEqual(plan_id, "plan_a")
        self.assertEqual(latency, 123.0)
        self.assertFalse(admission_rejected)
        self.assertEqual(status, EvaluationStatus.DUPLICATE_PLAN)

    def test_database_service_returns_explicit_timeout_status(self) -> None:
        class FakeExecutor:
            knob_space = KnobSpace(["k1"])

            def admission_check(self, *_args):
                return "plan_a", None, EvaluationStatus.ADMITTED

            def execute_with_timeout_result(self, *_args):
                return "plan_timeout", 1000.0, True

        service = object.__new__(DatabaseService)
        service.executor = FakeExecutor()

        plan_id, latency, admission_rejected, status = service.execute_with_knobs(
            "q.sql",
            [0.5],
            1000,
            is_warm_start=False,
        )

        self.assertEqual(plan_id, "plan_timeout")
        self.assertEqual(latency, 1000.0)
        self.assertFalse(admission_rejected)
        self.assertEqual(status, EvaluationStatus.TIMEOUT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
