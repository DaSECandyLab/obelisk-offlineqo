#!/usr/bin/env python3
"""Unit tests for TOML configuration loading."""

# ruff: noqa: E402

import os
import sys
import tempfile
import unittest
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from util.config import Config, load_app_config


class TestConfig(unittest.TestCase):
    def test_defaults_use_paper_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "obelisk.toml"
            config_path.write_text("", encoding="utf-8")

            config = load_app_config(config_path)

        self.assertEqual(config.run.warm_start_rounds, 6)
        self.assertEqual(config.run.total_rounds, 21)
        self.assertEqual(config.run.warm_times, 6)
        self.assertEqual(config.run.trials, 21)
        self.assertAlmostEqual(config.llm.temperature, 0.7)

    def test_loads_toml_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "obelisk.toml"
            config_path.write_text(
                """
[database]
host = "10.0.0.1"
port = 4010
name = "demo_db"

[llm]
enabled = false
model_name = "test-model"
base_url = "https://llm.example/v1"

[run]
sql_dir = "sql/job"
total_rounds = 3
warm_start_rounds = 1
repository_name = "demo_repository"

[optimization]
batch = 2
timeout_multiplier = 3.0
tcbo_num_trust_regions = 3
tcbo_risk_threshold = 0.1
tcbo_candidate_count = 128
""",
                encoding="utf-8",
            )

            config = load_app_config(config_path)
            self.assertEqual(config.database.host, "10.0.0.1")
            self.assertEqual(config.database.port, 4010)
            self.assertEqual(config.database.name, "demo_db")
            self.assertFalse(config.llm.enabled)
            self.assertEqual(config.llm.model_name, "test-model")
            self.assertEqual(config.llm.base_url, "https://llm.example/v1")
            self.assertEqual(config.run.total_rounds, 3)
            self.assertEqual(config.run.warm_start_rounds, 1)
            self.assertEqual(config.run.repository_name, "demo_repository")
            self.assertEqual(config.run.cache_name, "demo_repository")
            self.assertEqual(config.optimization.batch, 2)
            self.assertAlmostEqual(config.optimization.timeout_multiplier, 3.0)
            self.assertEqual(config.optimization.tcbo_num_trust_regions, 3)
            self.assertAlmostEqual(config.optimization.tcbo_risk_threshold, 0.1)
            self.assertEqual(config.optimization.tcbo_candidate_count, 128)

    def test_env_overrides_preserve_legacy_db_scripts(self) -> None:
        old_value = os.environ.get("TIDB_DB_NAME")
        os.environ["TIDB_DB_NAME"] = "env_db"
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "obelisk.toml"
            config_path.write_text("[database]\nname = \"file_db\"\n", encoding="utf-8")
            try:
                db_config = Config(config_path)
                self.assertEqual(db_config.tidb_db_name, "env_db")
            finally:
                if old_value is None:
                    os.environ.pop("TIDB_DB_NAME", None)
                else:
                    os.environ["TIDB_DB_NAME"] = old_value

    def test_legacy_cache_name_maps_to_repository_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "obelisk.toml"
            config_path.write_text(
                """
[run]
cache_name = "legacy_cache"
""",
                encoding="utf-8",
            )

            config = load_app_config(config_path)
            self.assertEqual(config.run.repository_name, "legacy_cache")

    def test_legacy_trial_names_map_to_round_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "obelisk.toml"
            config_path.write_text(
                """
[run]
trials = 9
warm_times = 4
""",
                encoding="utf-8",
            )

            config = load_app_config(config_path)
            self.assertEqual(config.run.total_rounds, 9)
            self.assertEqual(config.run.warm_start_rounds, 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
