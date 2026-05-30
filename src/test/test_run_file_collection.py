#!/usr/bin/env python3
"""Unit tests for SQL file discovery helpers in run.py."""

# ruff: noqa: E402

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Avoid loading full optimization stack (torch/botorch) for pure file-discovery tests.
stub_pipeline = types.ModuleType("optimization.optimization_pipeline")
stub_pipeline.SQLService = object
sys.modules.setdefault("optimization.optimization_pipeline", stub_pipeline)

import run as run_module
from util.config import AppConfig, DatabaseConfig, RunConfig


class TestRunFileCollection(unittest.TestCase):
    def test_derive_repository_name_uses_schema_safe_name(self) -> None:
        self.assertEqual(run_module.derive_repository_name("imdb"), "imdb")
        self.assertEqual(run_module.derive_repository_name("tenant/schema"), "tenant_schema")
        self.assertEqual(run_module.derive_repository_name(""), "default")

    def test_parse_args_accepts_repository_name_and_legacy_cache_name(self) -> None:
        old_argv = sys.argv
        try:
            sys.argv = ["run.py", "--repository-name", "repo_a"]
            self.assertEqual(run_module.parse_args().repository_name, "repo_a")

            sys.argv = ["run.py", "--cache-name", "repo_b"]
            self.assertEqual(run_module.parse_args().repository_name, "repo_b")
        finally:
            sys.argv = old_argv

    def test_parse_args_accepts_paper_round_names_and_legacy_trial_names(self) -> None:
        old_argv = sys.argv
        try:
            sys.argv = ["run.py", "--total-rounds", "21", "--warm-start-rounds", "6"]
            args = run_module.parse_args()
            self.assertEqual(args.total_rounds, 21)
            self.assertEqual(args.warm_start_rounds, 6)

            sys.argv = ["run.py", "--trials", "9", "--warm-times", "4"]
            args = run_module.parse_args()
            self.assertEqual(args.total_rounds, 9)
            self.assertEqual(args.warm_start_rounds, 4)
        finally:
            sys.argv = old_argv

    def test_collect_sql_files_recursive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            sql_dir = root / "job"
            (sql_dir / "nested").mkdir(parents=True)
            (sql_dir / "nested" / "a.sql").write_text("select 1", encoding="utf-8")
            (sql_dir / "b.sql").write_text("select 2", encoding="utf-8")

            files = run_module.collect_sql_files(sql_dir)
            self.assertEqual([p.as_posix() for p in files], sorted(p.as_posix() for p in files))
            self.assertEqual({p.name for p in files}, {"a.sql", "b.sql"})

    def test_build_result_path_uses_obelisk_artifact_name(self) -> None:
        self.assertEqual(
            run_module.build_result_path(Path("results/job"), Path("sql/job/1a.sql")),
            Path("results/job/result_1a_obelisk.json"),
        )

    def test_load_ceb_task_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            sql_dir = Path(tmp_dir) / "ceb-3k"
            (sql_dir / "1a").mkdir(parents=True)
            (sql_dir / "1a" / "1a2.sql").write_text("select 1", encoding="utf-8")
            (sql_dir / "x.sql").write_text("select 2", encoding="utf-8")

            (sql_dir / "tasks.txt").write_text("CEB_1A2\n", encoding="utf-8")

            files = run_module.load_ceb_task_filter(sql_dir)
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].name, "1a2.sql")

    def test_main_reports_sql_failures_after_batch_attempt(self) -> None:
        class FakeSQLService:
            processed = []

            def __init__(self, *_args, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

            def optimize(self, sql_filepath, *_args):
                sql_name = Path(sql_filepath).name
                self.processed.append(sql_name)
                if sql_name == "a.sql":
                    raise RuntimeError("db failed")

        old_argv = sys.argv
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            sql_dir = root / "sql"
            results_dir = root / "results"
            sql_dir.mkdir()
            (sql_dir / "a.sql").write_text("select 1", encoding="utf-8")
            (sql_dir / "b.sql").write_text("select 2", encoding="utf-8")
            app_config = AppConfig(
                database=DatabaseConfig(name="demo"),
                run=RunConfig(
                    sql_dir=str(sql_dir),
                    results_dir=str(results_dir),
                    total_rounds=1,
                    warm_start_rounds=0,
                    strategy="tcbo",
                ),
            )

            try:
                sys.argv = ["run.py"]
                FakeSQLService.processed = []
                with (
                    patch.object(run_module, "load_app_config", return_value=app_config),
                    patch.object(run_module, "SQLService", FakeSQLService),
                    patch.object(run_module, "tqdm", side_effect=lambda items, **_kwargs: items),
                ):
                    with self.assertRaisesRegex(RuntimeError, "Failed to optimize 1 SQL file"):
                        run_module.main()
            finally:
                sys.argv = old_argv

        self.assertEqual(FakeSQLService.processed, ["a.sql", "b.sql"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
