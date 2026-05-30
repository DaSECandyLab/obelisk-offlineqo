from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from util.config import AppConfig, load_app_config


@dataclass(slots=True)
class LLMConfig:
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

    @classmethod
    def from_app_config(cls, app_config: AppConfig | None = None) -> "LLMConfig":
        settings = (app_config or load_app_config()).llm
        return cls(
            enabled=settings.enabled,
            model_name=settings.model_name,
            api_key=settings.api_key,
            base_url=settings.base_url,
            temperature=settings.temperature,
            max_retries=settings.max_retries,
            retry_delay=settings.retry_delay,
            max_new_tokens=settings.max_new_tokens,
            top_p=settings.top_p,
            prompt_optimizer_enabled=settings.prompt_optimizer_enabled,
            prompt_optimizer_iterations=settings.prompt_optimizer_iterations,
            prompt_optimizer_top_n=settings.prompt_optimizer_top_n,
        )

    @classmethod
    def from_toml(cls, config_path: str | Path | None = None) -> "LLMConfig":
        return cls.from_app_config(load_app_config(config_path))

    def can_call_remote(self) -> bool:
        """Return whether Reasoner is allowed to make a remote LLM request."""
        if not self.enabled:
            return False
        if self.api_key:
            return True
        if not self.base_url:
            return False

        host = (urlparse(self.base_url).hostname or "").lower()
        return host in {"localhost", "127.0.0.1", "::1"}
