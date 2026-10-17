"""FastAPI service for the promoted anomaly model."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from api.schemas import (
    AnomaliesResponse,
    HealthResponse,
    ModelResponse,
    ReadyResponse,
    ScoreRequest,
    ScoreResponse,
)
from api.settings import Settings
from finanomaly.logging_utils import setup_logging
from finanomaly.serving.model_loader import ModelLoadError, load_bundle
from finanomaly.serving.scoring import ScoringError, ScoringService

log = logging.getLogger("api")

REQS = Counter("anomaly_requests_total", "Requests", ["path", "method", "status"])
LAT = Histogram("anomaly_request_latency_seconds", "Request latency", ["path"])
ALARMS = Counter("anomaly_alarms_returned_total", "Alarm episodes or flags returned")
DATA_AGE = Gauge("anomaly_data_age_days", "Age of the last scored day in days")
MODEL_LOADED = Gauge("anomaly_model_loaded", "1 if a model is loaded")


class State:
    service: ScoringService | None = None
    load_error: str | None = None
    settings: Settings


def load_service(settings: Settings, state: State) -> None:
    try:
        bundle = load_bundle(Path(settings.model_dir))
        state.service = ScoringService(bundle, stale_days=settings.stale_days)
        state.load_error = None
        MODEL_LOADED.set(1)
        DATA_AGE.set(state.service.data_age_days())
    except (ModelLoadError, FileNotFoundError, OSError) as exc:
        state.service = None
        state.load_error = str(exc)
        MODEL_LOADED.set(0)
        log.error("model load failed: %s", exc)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    state = State()
    state.settings = settings
    setup_logging(json_logs=settings.json_logs)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        load_service(settings, state)
        yield

    app = FastAPI(title="Financial anomaly detection API", version="0.1.0", lifespan=lifespan)
    app.state.app_state = state
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=settings.cors_origins.split(","), allow_methods=["GET"]
        )

    def err(status: int, code: str, message: str, request: Request) -> JSONResponse:
        rid = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=status,
            content={"error": {"code": code, "message": message, "request_id": rid}},
        )

    @app.middleware("http")
    async def middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.request_id = rid
        t0 = time.time()
        try:
            response = await asyncio.wait_for(call_next(request), settings.request_timeout_s)
        except TimeoutError:
            response = err(504, "timeout", "request timed out", request)
        except Exception:  # never leak stack traces
            log.exception("unhandled error", extra={"request_id": rid})
            response = err(500, "internal_error", "internal server error", request)
        dt = time.time() - t0
        route = request.scope.get("route")
        path = route.path if route is not None else "unmatched"
        REQS.labels(path, request.method, str(response.status_code)).inc()
        LAT.labels(path).observe(dt)
        response.headers["x-request-id"] = rid
        log.info(
            "request",
            extra={
                "request_id": rid,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round(dt * 1000, 1),
            },
        )
        return response

    @app.exception_handler(HTTPException)
    async def http_exc(request: Request, exc: HTTPException) -> JSONResponse:
        return err(exc.status_code, f"http_{exc.status_code}", str(exc.detail), request)

    @app.exception_handler(RequestValidationError)
    async def validation_exc(request: Request, exc: RequestValidationError) -> JSONResponse:
        msgs = "; ".join(
            f"{'.'.join(str(x) for x in e['loc'][1:])}: {e['msg']}" for e in exc.errors()
        )
        return err(422, "validation_error", msgs, request)

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if settings.api_key and x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    def service() -> ScoringService:
        if state.service is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return state.service

    def parse_date(value: str, name: str) -> pd.Timestamp:
        try:
            return pd.Timestamp(value)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=f"{name} is not a valid date") from exc

    @app.get("/v1/health", response_model=HealthResponse)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/ready", response_model=ReadyResponse)
    def ready(response: Response) -> dict[str, Any]:
        if state.service is None:
            response.status_code = 503
            return {"ready": False, "model_loaded": False}
        svc = state.service
        DATA_AGE.set(svc.data_age_days())
        return {
            "ready": True,
            "model_loaded": True,
            "last_data_date": str(svc.last_date.date()),
            "data_age_days": svc.data_age_days(),
            "stale": svc.is_stale(),
            "quality_ok": svc.b.status.get("quality_ok", True),
        }

    @app.get("/v1/model", response_model=ModelResponse, dependencies=[Depends(require_key)])
    def model_info() -> dict[str, Any]:
        svc = service()
        m = svc.b.manifest
        return {
            "name": m["model_name"],
            "version": m["model_version"],
            "alias": m.get("alias", "production"),
            "run_id": m["run_id"],
            "family": m["family"],
            "threshold": m["threshold"],
            "features": svc.b.features,
            "manifest": {k: v for k, v in m.items() if k != "files"},
        }

    @app.get("/v1/score", response_model=ScoreResponse, dependencies=[Depends(require_key)])
    def score_get(
        date: str | None = Query(default=None, description="YYYY-MM-DD; latest if omitted"),
    ) -> dict[str, Any]:
        svc = service()
        try:
            res = svc.score_date(parse_date(date, "date") if date else None)
        except ScoringError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        ALARMS.inc(int(res["alarm"]))
        return res

    @app.post("/v1/score", response_model=ScoreResponse, dependencies=[Depends(require_key)])
    def score_post(req: ScoreRequest) -> dict[str, Any]:
        try:
            res = service().score_vector(req.features)
        except ScoringError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        ALARMS.inc(int(res["alarm"]))
        return res

    @app.get("/v1/anomalies", response_model=AnomaliesResponse, dependencies=[Depends(require_key)])
    def anomalies(start: str, end: str) -> dict[str, Any]:
        svc = service()
        s, e = parse_date(start, "start"), parse_date(end, "end")
        if e < s:
            raise HTTPException(status_code=422, detail="end must not precede start")
        if (e - s).days > settings.max_range_days:
            raise HTTPException(
                status_code=422, detail=f"range longer than {settings.max_range_days} days"
            )
        eps = svc.anomalies(s, e)
        ALARMS.inc(len(eps))
        return {
            "start": str(s.date()),
            "end": str(e.date()),
            "threshold": svc.b.threshold,
            "episodes": eps,
        }

    @app.get("/metrics")
    def metrics() -> Response:
        if state.service is not None:
            DATA_AGE.set(state.service.data_age_days())
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
