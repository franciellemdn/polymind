"""
Settings
--------
Centralised configuration using pydantic-settings.
All values read from environment variables or .env file.
"""

from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Keys
    openrouter_api_key: str = ""
    langsmith_api_key: str = ""

    @field_validator("openrouter_api_key", "langsmith_api_key", mode="before")
    @classmethod
    def strip_quotes(cls, v: str) -> str:
        if isinstance(v, str):
            return v.strip("'\"")
        return v

    # Endpoints
    ollama_base_url: str = "http://localhost:11434"

    # Graph defaults
    default_max_iterations: int = 2
    complexity_threshold: float = 0.55
    budget_usd_per_run: float = 1.0

    # Tracing
    langchain_tracing_v2: bool = False
    langchain_project: str = "polymind"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    # Eval
    eval_output_dir: str = "data/traces"
    eval_n_runs: int = 2


settings = Settings()
