"""OBELISK plan repository R backed by SQLite."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from util.logger import logger


class PlanRepository:
    """Persistent repository R for executed plan evidence.

    Each record follows the paper's tuple <F(P), Q, e_P>: plan fingerprint,
    SQL statement key, and verbose plan with runtime annotations.
    """

    def __init__(
        self,
        repository_name: str = "default",
        *,
        cache_name: str | None = None,
        clear_on_init: bool = False,
    ) -> None:
        # cache_name is accepted for backward compatibility with older scripts.
        name = cache_name if cache_name is not None else repository_name
        self.repository_name = name or "default"
        self.db_path = str(Path("cache") / f"{self.repository_name}.db")
        self._ensure_db_directory()
        if clear_on_init:
            self.clear()
        self._init_database()

    def _ensure_db_directory(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            self._ensure_plan_repository_schema(cursor)
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_plan_repository_fingerprint_query
                ON plan_repository(plan_fingerprint, query_template)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_plan_repository_query
                ON plan_repository(query_template)
                """
            )
            self._migrate_legacy_plan_cache(cursor)
            conn.commit()
            logger.info("Initialized plan repository R at %s", self.db_path)

    def _ensure_plan_repository_schema(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name = 'plan_repository'
            """
        )
        if cursor.fetchone() is None:
            self._create_plan_repository_table(cursor)
            return

        cursor.execute("PRAGMA table_info(plan_repository)")
        columns = {row["name"] for row in cursor.fetchall()}
        if "record_id" in columns:
            return

        cursor.execute("ALTER TABLE plan_repository RENAME TO plan_repository_legacy")
        self._create_plan_repository_table(cursor)
        cursor.execute(
            """
            INSERT INTO plan_repository
                (plan_fingerprint, query_template, execution_time_ms,
                 verbose_plan, created_at, updated_at)
            SELECT plan_fingerprint, query_template, execution_time_ms,
                   verbose_plan, created_at, updated_at
            FROM plan_repository_legacy
            """
        )
        cursor.execute("DROP TABLE plan_repository_legacy")

    @staticmethod
    def _create_plan_repository_table(cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS plan_repository (
                record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_fingerprint TEXT NOT NULL,
                query_template TEXT NOT NULL,
                execution_time_ms REAL NOT NULL,
                verbose_plan TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    @staticmethod
    def _migrate_legacy_plan_cache(cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name = 'plan_cache'
            """
        )
        if cursor.fetchone() is None:
            return

        cursor.execute(
            """
            INSERT INTO plan_repository
                (plan_fingerprint, query_template, execution_time_ms, verbose_plan)
            SELECT plan_digest, sql_filepath, execution_time_ms, plan_details
            FROM plan_cache
            WHERE NOT EXISTS (
                SELECT 1
                FROM plan_repository
                WHERE plan_fingerprint = plan_cache.plan_digest
                  AND query_template = plan_cache.sql_filepath
                  AND execution_time_ms = plan_cache.execution_time_ms
                  AND verbose_plan = plan_cache.plan_details
            )
            """
        )

    def record_executed_plan(
        self,
        plan_fingerprint: str,
        query_template: str,
        execution_time_ms: float,
        verbose_plan: dict[str, Any],
    ) -> None:
        """Store an actually executed plan record <F(P), Q, e_P>."""
        if not plan_fingerprint:
            raise ValueError("plan_fingerprint must be non-empty")
        if not query_template:
            raise ValueError("query_template must be non-empty")
        if execution_time_ms <= 0:
            raise ValueError("execution_time_ms must be positive")

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO plan_repository
                        (plan_fingerprint, query_template, execution_time_ms,
                         verbose_plan, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        plan_fingerprint,
                        query_template,
                        float(execution_time_ms),
                        json.dumps(verbose_plan),
                    ),
                )
                conn.commit()
        except Exception as error:
            logger.error(
                "Failed to record plan fingerprint %s for query template %s: %s",
                plan_fingerprint,
                query_template,
                error,
            )
            raise

    def get_execution_time(
        self,
        plan_fingerprint: str,
        query_template: str,
    ) -> float | None:
        """Return recorded execution time for an exact full-plan match."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT execution_time_ms
                    FROM plan_repository
                    WHERE plan_fingerprint = ? AND query_template = ?
                    ORDER BY record_id DESC
                    LIMIT 1
                    """,
                    (plan_fingerprint, query_template),
                )
                row = cursor.fetchone()
                return row["execution_time_ms"] if row else None
        except Exception as error:
            logger.error(
                "Failed to fetch execution time for %s under query template %s: %s",
                plan_fingerprint,
                query_template,
                error,
            )
            raise

    def get_execution_time_and_verbose_plan(
        self,
        plan_fingerprint: str,
        query_template: str,
    ) -> tuple[float | None, dict[str, Any] | None]:
        """Return recorded latency and verbose plan e_P for F(P), Q."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT execution_time_ms, verbose_plan
                    FROM plan_repository
                    WHERE plan_fingerprint = ? AND query_template = ?
                    ORDER BY record_id DESC
                    LIMIT 1
                    """,
                    (plan_fingerprint, query_template),
                )
                row = cursor.fetchone()
                if row is None:
                    return None, None
                return row["execution_time_ms"], json.loads(row["verbose_plan"])
        except Exception as error:
            logger.error(
                "Failed to fetch verbose plan for %s under query template %s: %s",
                plan_fingerprint,
                query_template,
                error,
            )
            raise

    def get_matching_plan_records(
        self,
        plan_fingerprint: str,
        query_template: str | None = None,
    ) -> list[tuple[float, dict[str, Any]]]:
        """Return historical records in S(p) for F(p).

        Full-plan duplicate reuse is scoped by Q through get_execution_time().
        Subplan admission follows the paper's schema-level repository design and
        can reuse matching fingerprints across query templates in the same
        schema, so query_template is optional here.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if query_template is None:
                    cursor.execute(
                        """
                        SELECT execution_time_ms, verbose_plan
                        FROM plan_repository
                        WHERE plan_fingerprint = ?
                        ORDER BY record_id DESC
                        """,
                        (plan_fingerprint,),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT execution_time_ms, verbose_plan
                        FROM plan_repository
                        WHERE plan_fingerprint = ? AND query_template = ?
                        ORDER BY record_id DESC
                        """,
                        (plan_fingerprint, query_template),
                    )
                return [
                    (row["execution_time_ms"], json.loads(row["verbose_plan"]))
                    for row in cursor.fetchall()
                ]
        except Exception as error:
            logger.error(
                "Failed to fetch matching plan records for %s under query template %s: %s",
                plan_fingerprint,
                query_template,
                error,
            )
            raise

    def get_verbose_plan(
        self,
        plan_fingerprint: str,
        query_template: str,
    ) -> dict[str, Any] | None:
        """Return the stored verbose plan e_P for a repository entry."""
        _, verbose_plan = self.get_execution_time_and_verbose_plan(
            plan_fingerprint,
            query_template,
        )
        return verbose_plan

    def list_plan_fingerprints(
        self,
        query_template: str | None = None,
    ) -> list[tuple[str, str]]:
        """List stored (F(P), Q) keys, optionally filtered by Q."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if query_template is None:
                    cursor.execute(
                        """
                        SELECT DISTINCT plan_fingerprint, query_template
                        FROM plan_repository
                        """
                    )
                else:
                    cursor.execute(
                        """
                        SELECT DISTINCT plan_fingerprint, query_template
                        FROM plan_repository
                        WHERE query_template = ?
                        """,
                        (query_template,),
                    )
                return [
                    (row["plan_fingerprint"], row["query_template"])
                    for row in cursor.fetchall()
                ]
        except Exception as error:
            logger.error("Failed to list plan fingerprints: %s", error)
            return []

    def clear(self, query_template: str | None = None) -> None:
        """Clear repository entries, optionally only for one query template."""
        try:
            if query_template is None:
                if os.path.exists(self.db_path):
                    os.remove(self.db_path)
                    logger.info("Removed plan repository: %s", self.db_path)
                return

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM plan_repository WHERE query_template = ?",
                    (query_template,),
                )
                conn.commit()
                logger.info("Cleared repository entries for query template %s", query_template)
        except Exception as error:
            logger.error("Failed to clear plan repository: %s", error)

    def get_repository_stats(self) -> dict[str, Any]:
        """Return plan repository statistics."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) as total FROM plan_repository")
                total_count = cursor.fetchone()["total"]
                cursor.execute(
                    """
                    SELECT query_template, COUNT(*) as count
                    FROM plan_repository
                    GROUP BY query_template
                    """
                )
                query_stats = {
                    row["query_template"]: row["count"]
                    for row in cursor.fetchall()
                }
                return {
                    "total_plan_count": total_count,
                    "query_stats": query_stats,
                    "sql_stats": query_stats,
                }
        except Exception as error:
            logger.error("Failed to get plan repository stats: %s", error)
            return {"total_plan_count": 0, "query_stats": {}, "sql_stats": {}}

    # Backward-compatible CacheManager API.
    def cache_plan(
        self,
        plan_digest: str,
        sql_filepath: str,
        execution_time_ms: float,
        plan_details: dict[str, Any],
    ) -> None:
        self.record_executed_plan(
            plan_digest,
            sql_filepath,
            execution_time_ms,
            plan_details,
        )

    def get_plan_execution_time(
        self,
        plan_digest: str,
        sql_filepath: str,
    ) -> float | None:
        return self.get_execution_time(plan_digest, sql_filepath)

    def get_plan_execution_time_and_details(
        self,
        plan_digest: str,
        sql_filepath: str,
    ) -> tuple[float | None, dict[str, Any] | None]:
        return self.get_execution_time_and_verbose_plan(plan_digest, sql_filepath)

    def get_plan_details(
        self,
        plan_digest: str,
        sql_filepath: str,
    ) -> dict[str, Any] | None:
        return self.get_verbose_plan(plan_digest, sql_filepath)

    def get_all_plan_digests(
        self,
        sql_filepath: str | None = None,
    ) -> list[tuple[str, str]]:
        return self.list_plan_fingerprints(sql_filepath)

    def clear_cache(self, sql_filepath: str | None = None) -> None:
        self.clear(sql_filepath)

    def get_cache_stats(self) -> dict[str, Any]:
        return self.get_repository_stats()


__all__ = ["PlanRepository"]
