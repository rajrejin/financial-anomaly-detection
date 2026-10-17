"""Request and response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Contribution(BaseModel):
    feature: str
    contribution: float


class ScoreResponse(BaseModel):
    date: str | None = None
    score: float
    threshold: float
    alarm: bool
    contributions: list[Contribution]
    contribution_kind: str
    stale: bool | None = None


class ScoreRequest(BaseModel):
    features: dict[str, float] = Field(
        description="Trailing z-scored feature vector; keys must equal the model feature names."
    )


class Episode(BaseModel):
    start: str
    end: str
    days: int
    peak_date: str
    peak_score: float


class AnomaliesResponse(BaseModel):
    start: str
    end: str
    threshold: float
    episodes: list[Episode]


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool
    model_loaded: bool
    last_data_date: str | None = None
    data_age_days: int | None = None
    stale: bool | None = None
    quality_ok: bool | None = None


class ModelResponse(BaseModel):
    name: str
    version: int
    alias: str
    run_id: str
    family: str
    threshold: float
    features: list[str]
    manifest: dict[str, Any]


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
