#!/usr/bin/env python3
"""Unit tests for the OBELISK plan repository."""

# ruff: noqa: E402

import sys
import uuid
import sqlite3
import unittest
from pathlib import Path
from unittest.mock import patch

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from db.cache_manager import CacheManager
from db.plan_repository import PlanRepository


class TestPlanRepository(unittest.TestCase):
    def test_plan_repository_is_primary_cache_type(self) -> None:
        self.assertIs(PlanRepository, CacheManager)

    def test_plan_repository_persists_across_instances(self) -> None:
        repository_name = f"test-repository-{uuid.uuid4().hex}"
        manager = PlanRepository(repository_name=repository_name)
        try:
            manager.record_executed_plan(
                "plan_a",
                "select * from t where a = ?",
                123.0,
                {"operatorInfo": "demo, plan_a"},
            )

            reloaded = PlanRepository(repository_name=repository_name)
            self.assertEqual(
                reloaded.get_execution_time("plan_a", "select * from t where a = ?"),
                123.0,
            )
            self.assertEqual(
                reloaded.get_verbose_plan(
                    "plan_a",
                    "select * from t where a = ?",
                ),
                {"operatorInfo": "demo, plan_a"},
            )
        finally:
            PlanRepository(repository_name=repository_name).clear()

    def test_plan_repository_keeps_multiple_executed_records_for_same_plan(self) -> None:
        repository_name = f"test-repository-{uuid.uuid4().hex}"
        manager = PlanRepository(repository_name=repository_name)
        query_template = "select * from t where a = ?"
        try:
            manager.record_executed_plan(
                "plan_a",
                query_template,
                123.0,
                {"operatorInfo": "demo old, plan_a"},
            )
            manager.record_executed_plan(
                "plan_a",
                query_template,
                456.0,
                {"operatorInfo": "demo new, plan_a"},
            )

            reloaded = PlanRepository(repository_name=repository_name)
            records = reloaded.get_matching_plan_records("plan_a", query_template)

            self.assertEqual([latency for latency, _ in records], [456.0, 123.0])
            self.assertEqual(
                [plan["operatorInfo"] for _, plan in records],
                ["demo new, plan_a", "demo old, plan_a"],
            )
            self.assertEqual(reloaded.get_execution_time("plan_a", query_template), 456.0)
        finally:
            PlanRepository(repository_name=repository_name).clear()

    def test_subplan_records_can_be_retrieved_across_query_templates(self) -> None:
        repository_name = f"test-repository-{uuid.uuid4().hex}"
        manager = PlanRepository(repository_name=repository_name)
        try:
            manager.record_executed_plan(
                "subplan_a",
                "select * from t where a = ?",
                123.0,
                {"operatorInfo": "old query, subplan_a"},
            )
            manager.record_executed_plan(
                "subplan_a",
                "select * from u where b = ?",
                456.0,
                {"operatorInfo": "new query, subplan_a"},
            )

            reloaded = PlanRepository(repository_name=repository_name)
            all_records = reloaded.get_matching_plan_records("subplan_a")
            query_records = reloaded.get_matching_plan_records(
                "subplan_a",
                "select * from t where a = ?",
            )

            self.assertEqual([latency for latency, _ in all_records], [456.0, 123.0])
            self.assertEqual([latency for latency, _ in query_records], [123.0])
        finally:
            PlanRepository(repository_name=repository_name).clear()

    def test_plan_repository_write_failure_is_not_silenced(self) -> None:
        repository_name = f"test-repository-{uuid.uuid4().hex}"
        manager = PlanRepository(repository_name=repository_name)
        try:
            with patch.object(manager, "_get_connection", side_effect=RuntimeError("sqlite locked")):
                with self.assertRaisesRegex(RuntimeError, "sqlite locked"):
                    manager.record_executed_plan(
                        "plan_a",
                        "select ?",
                        10.0,
                        {"operatorInfo": "demo, plan_a"},
                    )
        finally:
            PlanRepository(repository_name=repository_name).clear()

    def test_plan_repository_exact_match_read_failure_is_not_silenced(self) -> None:
        repository_name = f"test-repository-{uuid.uuid4().hex}"
        manager = PlanRepository(repository_name=repository_name)
        try:
            with patch.object(manager, "_get_connection", side_effect=RuntimeError("sqlite locked")):
                with self.assertRaisesRegex(RuntimeError, "sqlite locked"):
                    manager.get_execution_time("plan_a", "select ?")
        finally:
            PlanRepository(repository_name=repository_name).clear()

    def test_plan_repository_subplan_read_failure_is_not_silenced(self) -> None:
        repository_name = f"test-repository-{uuid.uuid4().hex}"
        manager = PlanRepository(repository_name=repository_name)
        try:
            with patch.object(manager, "_get_connection", side_effect=RuntimeError("sqlite locked")):
                with self.assertRaisesRegex(RuntimeError, "sqlite locked"):
                    manager.get_matching_plan_records("plan_a")
        finally:
            PlanRepository(repository_name=repository_name).clear()

    def test_plan_repository_migrates_legacy_unique_schema(self) -> None:
        repository_name = f"test-repository-{uuid.uuid4().hex}"
        db_path = Path("cache") / f"{repository_name}.db"
        db_path.parent.mkdir(exist_ok=True)
        try:
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE plan_repository (
                        plan_fingerprint TEXT NOT NULL,
                        query_template TEXT NOT NULL,
                        execution_time_ms REAL NOT NULL,
                        verbose_plan TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (plan_fingerprint, query_template)
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO plan_repository
                        (plan_fingerprint, query_template, execution_time_ms, verbose_plan)
                    VALUES ('plan_a', 'select ?', 111.0, '{"operatorInfo":"old"}')
                    """
                )
                conn.commit()

            repository = PlanRepository(repository_name=repository_name)
            repository.record_executed_plan(
                "plan_a",
                "select ?",
                222.0,
                {"operatorInfo": "new"},
            )

            records = repository.get_matching_plan_records("plan_a", "select ?")
            self.assertEqual([latency for latency, _ in records], [222.0, 111.0])
        finally:
            PlanRepository(repository_name=repository_name).clear()


if __name__ == "__main__":
    unittest.main(verbosity=2)
