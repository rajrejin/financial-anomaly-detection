"""API settings (environment overrides with the ANOMALY_ prefix)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ANOMALY_", env_file=None)

    model_dir: str = "models_export"
    api_key: str | None = None
    stale_days: int = 5
    max_range_days: int = 3700
    request_timeout_s: float = 30.0
    json_logs: bool = True
    cors_origins: str = ""  # CORS is off unless origins are listed
