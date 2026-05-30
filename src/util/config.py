# Copyright 2023 PingCAP, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "etc" / "obelisk.toml"


@dataclass(slots=True)
class DatabaseConfig:
    host: str = "127.0.0.1"
    port: int = 4000
    user: str = "root"
    password: str = ""
    name: str = "imdb"
    ca_path: str = ""
    autocommit: bool = True
    mem_quota_bytes: int = 16 * 1024 * 1024 * 1024
    validate_copr_cache: bool = True


@dataclass(slots=True)
class RunConfig:
    sql_dir: str = "sql/job"
    results_dir: str = "results/job"
    total_rounds: int = 21
    warm_start_rounds: int = 6
    strategy: str = "tcbo"
    repository_name: str = ""

    @property
    def trials(self) -> int:
        """Backward-compatible name for older local configs/scripts."""
        return self.total_rounds

    @trials.setter
    def trials(self, value: int) -> None:
        self.total_rounds = value

    @property
    def warm_times(self) -> int:
        """Backward-compatible name for older local configs/scripts."""
        return self.warm_start_rounds

    @warm_times.setter
    def warm_times(self, value: int) -> None:
        self.warm_start_rounds = value

    @property
    def cache_name(self) -> str:
        """Backward-compatible name for older local configs/scripts."""
        return self.repository_name

    @cache_name.setter
    def cache_name(self, value: str) -> None:
        self.repository_name = value


@dataclass(slots=True)
class OptimizationConfig:
    baseline_timeout_ms: int = 3_600_000
    timeout_multiplier: float = 2.0
    topk: int = 5
    batch: int = 5
    retry_attempts: int = 8
    max_no_improvement: int = 3
    tcbo_num_trust_regions: int = 4
    tcbo_risk_threshold: float = 0.05
    tcbo_candidate_count: int = 2_000


@dataclass(slots=True)
class LLMRuntimeConfig:
    enabled: bool = True
    model_name: str = "gpt-4.1-mini"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.7
    max_retries: int = 5
    retry_delay: int = 5
    max_new_tokens: int = 2048
    top_p: float = 0.7
    prompt_optimizer_enabled: bool = False
    prompt_optimizer_iterations: int = 1
    prompt_optimizer_top_n: int = 3


@dataclass(slots=True)
class AppConfig:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    run: RunConfig = field(default_factory=RunConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    llm: LLMRuntimeConfig = field(default_factory=LLMRuntimeConfig)
    source_path: Path = DEFAULT_CONFIG_PATH


class Config:
    """Backward-compatible DB config facade.

    New code should prefer ``load_app_config().database``. This class keeps
    older helper scripts working while configuration moves to TOML.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        database = load_app_config(config_path).database
        self.tidb_host = database.host
        self.tidb_port = database.port
        self.tidb_user = database.user
        self.tidb_password = database.password
        self.tidb_db_name = database.name
        self.ca_path = database.ca_path


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    raw_path = config_path or os.getenv("OBELISK_CONFIG")
    if raw_path:
        path = Path(raw_path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    return DEFAULT_CONFIG_PATH


def load_app_config(config_path: str | Path | None = None) -> AppConfig:
    path = resolve_config_path(config_path)
    data = _load_toml(path)

    app_config = AppConfig(
        database=_database_config(data.get("database", {})),
        run=_run_config(data.get("run", {})),
        optimization=_optimization_config(data.get("optimization", {})),
        llm=_llm_config(data.get("llm", {})),
        source_path=path,
    )
    _apply_env_overrides(app_config)
    return app_config


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. Copy etc/obelisk.toml.tpl "
            "to etc/obelisk.toml, keep secrets only in the local .toml file, "
            "or pass --config."
        )

    with path.open("rb") as file:
        content = tomllib.load(file)

    if not isinstance(content, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return content


def _database_config(data: dict[str, Any]) -> DatabaseConfig:
    return DatabaseConfig(
        host=str(data.get("host", "127.0.0.1")),
        port=int(data.get("port", 4000)),
        user=str(data.get("user", "root")),
        password=str(data.get("password", "")),
        name=str(data.get("name", "imdb")),
        ca_path=str(data.get("ca_path", "")),
        autocommit=bool(data.get("autocommit", True)),
        mem_quota_bytes=int(data.get("mem_quota_bytes", 16 * 1024 * 1024 * 1024)),
        validate_copr_cache=bool(data.get("validate_copr_cache", True)),
    )


def _run_config(data: dict[str, Any]) -> RunConfig:
    return RunConfig(
        sql_dir=str(data.get("sql_dir", "sql/job")),
        results_dir=str(data.get("results_dir", "results/job")),
        total_rounds=int(data.get("total_rounds", data.get("trials", 21))),
        warm_start_rounds=int(data.get("warm_start_rounds", data.get("warm_times", 6))),
        strategy=str(data.get("strategy", "tcbo")),
        repository_name=str(data.get("repository_name", data.get("cache_name", ""))),
    )


def _optimization_config(data: dict[str, Any]) -> OptimizationConfig:
    return OptimizationConfig(
        baseline_timeout_ms=int(data.get("baseline_timeout_ms", 3_600_000)),
        timeout_multiplier=float(data.get("timeout_multiplier", 2.0)),
        topk=int(data.get("topk", 5)),
        batch=int(data.get("batch", 5)),
        retry_attempts=int(data.get("retry_attempts", 8)),
        max_no_improvement=int(data.get("max_no_improvement", 3)),
        tcbo_num_trust_regions=int(data.get("tcbo_num_trust_regions", 4)),
        tcbo_risk_threshold=float(data.get("tcbo_risk_threshold", 0.05)),
        tcbo_candidate_count=int(data.get("tcbo_candidate_count", 2_000)),
    )


def _bool_value(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _llm_config(data: dict[str, Any]) -> LLMRuntimeConfig:
    return LLMRuntimeConfig(
        enabled=_bool_value(data.get("enabled"), True),
        model_name=str(data.get("model_name", "gpt-4.1-mini")),
        api_key=str(data.get("api_key", "")),
        base_url=str(
            data.get("base_url")
            or data.get("api_base")
            or data.get("openai_api_base")
            or ""
        ),
        temperature=float(data.get("temperature", 0.7)),
        max_retries=int(data.get("max_retries", 5)),
        retry_delay=int(data.get("retry_delay", 5)),
        max_new_tokens=int(data.get("max_new_tokens", 2048)),
        top_p=float(data.get("top_p", 0.7)),
        prompt_optimizer_enabled=_bool_value(
            data.get("prompt_optimizer_enabled"),
            False,
        ),
        prompt_optimizer_iterations=int(data.get("prompt_optimizer_iterations", 1)),
        prompt_optimizer_top_n=int(data.get("prompt_optimizer_top_n", 3)),
    )


def _apply_env_overrides(app_config: AppConfig) -> None:
    """Keep old scripts usable while TOML remains the source of truth."""

    database = app_config.database
    database.host = os.getenv("TIDB_HOST", database.host)
    database.port = int(os.getenv("TIDB_PORT", str(database.port)))
    database.user = os.getenv("TIDB_USER", database.user)
    database.password = os.getenv("TIDB_PASSWORD", database.password)
    database.name = os.getenv("TIDB_DB_NAME", database.name)
    database.ca_path = os.getenv("CA_PATH", database.ca_path)

    app_config.llm.api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("LLM_API_KEY")
        or app_config.llm.api_key
    )
    app_config.llm.base_url = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or os.getenv("LLM_BASE_URL")
        or os.getenv("LLM_API_BASE")
        or app_config.llm.base_url
    )

    llm_enabled = os.getenv("OBELISK_LLM_ENABLED") or os.getenv("LLM_ENABLED")
    if llm_enabled is not None:
        app_config.llm.enabled = _bool_value(llm_enabled, app_config.llm.enabled)
