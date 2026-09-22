"""FastAPI service for battery capacity estimation.

Run locally:
    uvicorn src.api:app --reload
    open http://localhost:8000/docs

The checkpoint to serve is chosen with the MODEL_PATH environment variable,
defaulting to models/lstm/fold0.pt.
"""

import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from src.config import MODELS_DIR, get_device
from src.predict import NOMINAL_CAPACITY_AH, load_model, predict, state_of_health

MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "lstm" / "fold0.pt"))
MAX_BATCH = 1000

# ---------------------------------------------------------------------------
# Counters for /metrics. A lock keeps them consistent under concurrent requests.
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_stats = {"requests": 0, "predictions": 0, "errors": 0, "latency_sum": 0.0}


def _record(n_predictions: int, seconds: float, error: bool = False) -> None:
    with _lock:
        _stats["requests"] += 1
        _stats["predictions"] += n_predictions
        _stats["latency_sum"] += seconds
        if error:
            _stats["errors"] += 1


# ---------------------------------------------------------------------------
# Request and response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    snippet: list[list[float]] = Field(
        ...,
        description="One charging snippet: 128 timesteps x 9 channels "
                    "(avg_V, current, SOC, max_V, min_V, max_T, min_T, "
                    "V_spread, T_spread), raw unscaled values.",
    )

    @field_validator("snippet")
    @classmethod
    def check_rectangular(cls, v):
        # Shape against the loaded model is checked in the endpoint; here we
        # only reject input that isn't a rectangular 2-D array, which would
        # otherwise fail deep inside numpy with an unhelpful message.
        if not v:
            raise ValueError("snippet is empty")
        widths = {len(row) for row in v}
        if len(widths) != 1:
            raise ValueError("all timesteps must have the same number of channels")
        return v


class BatchPredictRequest(BaseModel):
    snippets: list[list[list[float]]] = Field(
        ..., description=f"Up to {MAX_BATCH} snippets, each 128 x 9."
    )


class PredictResponse(BaseModel):
    capacity_ah: float
    soh_percent: float
    model_name: str
    fold: int


class BatchPredictResponse(BaseModel):
    capacity_ah: list[float]
    soh_percent: list[float]
    count: int
    model_name: str
    fold: int


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup rather than per request."""
    app.state.bundle = load_model(MODEL_PATH, device=get_device())
    app.state.started = time.time()
    yield


app = FastAPI(
    title="EV Battery Capacity Prediction",
    description=(
        "Estimates lithium-ion battery capacity from a 128-timestep charging "
        "window. Mileage is deliberately not an input: the model reads charging "
        "behaviour only."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _as_array(rows, bundle) -> np.ndarray:
    """Validate shape against the loaded model and return a float32 array."""
    x = np.asarray(rows, dtype=np.float32)
    expected = (bundle["seq_len"], bundle["n_features"])
    if x.shape[-2:] != expected:
        raise HTTPException(
            status_code=422,
            detail=f"expected each snippet to be {expected[0]} timesteps x "
                   f"{expected[1]} channels, got {tuple(x.shape[-2:])}",
        )
    return x


@app.get("/health", tags=["ops"])
def health():
    """Liveness probe. Orchestrators call this to decide if the container is serving."""
    ready = getattr(app.state, "bundle", None) is not None
    return {"status": "ok" if ready else "loading",
            "uptime_seconds": round(time.time() - app.state.started, 1)}


@app.get("/model/info", tags=["ops"])
def model_info():
    """Which model is being served, and what input it expects."""
    b = app.state.bundle
    return {
        "model_name": b["model_name"],
        "fold": b["fold"],
        "checkpoint": b["path"],
        "input_shape": [b["seq_len"], b["n_features"]],
        "channels": ["avg_cell_voltage", "charging_current", "soc",
                     "max_cell_voltage", "min_cell_voltage",
                     "max_cell_temp", "min_cell_temp",
                     "voltage_spread", "temp_spread"][: b["n_features"]],
        "nominal_capacity_ah": NOMINAL_CAPACITY_AH,
        "test_metrics": b["metrics"],
        "hyperparameters": b["params"],
    }


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict_one(req: PredictRequest):
    """Estimate capacity for a single charging snippet."""
    t0 = time.perf_counter()
    b = app.state.bundle
    try:
        x = _as_array(req.snippet, b)
        capacity = float(predict(b, x)[0])
    except HTTPException:
        _record(0, time.perf_counter() - t0, error=True)
        raise
    except Exception as exc:
        _record(0, time.perf_counter() - t0, error=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    _record(1, time.perf_counter() - t0)
    return PredictResponse(
        capacity_ah=round(capacity, 3),
        soh_percent=round(float(state_of_health(capacity)), 2),
        model_name=b["model_name"],
        fold=b["fold"],
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
def predict_batch(req: BatchPredictRequest):
    """Estimate capacity for many snippets at once - e.g. scoring a fleet."""
    t0 = time.perf_counter()
    b = app.state.bundle

    if not req.snippets:
        raise HTTPException(status_code=422, detail="snippets is empty")
    if len(req.snippets) > MAX_BATCH:
        raise HTTPException(
            status_code=422,
            detail=f"batch too large: {len(req.snippets)} > {MAX_BATCH}",
        )

    try:
        x = _as_array(req.snippets, b)
        capacities = predict(b, x)
    except HTTPException:
        _record(0, time.perf_counter() - t0, error=True)
        raise
    except Exception as exc:
        _record(0, time.perf_counter() - t0, error=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    _record(len(capacities), time.perf_counter() - t0)
    return BatchPredictResponse(
        capacity_ah=[round(float(c), 3) for c in capacities],
        soh_percent=[round(float(s), 2) for s in state_of_health(capacities)],
        count=len(capacities),
        model_name=b["model_name"],
        fold=b["fold"],
    )


@app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
def metrics():
    """Counters in Prometheus exposition format.

    Nothing scrapes this yet. It exists so monitoring can be added later as
    configuration rather than a code change.
    """
    with _lock:
        s = dict(_stats)

    return "\n".join([
        "# HELP ev_api_requests_total Requests handled.",
        "# TYPE ev_api_requests_total counter",
        f"ev_api_requests_total {s['requests']}",
        "# HELP ev_api_predictions_total Individual snippets scored.",
        "# TYPE ev_api_predictions_total counter",
        f"ev_api_predictions_total {s['predictions']}",
        "# HELP ev_api_errors_total Requests that failed.",
        "# TYPE ev_api_errors_total counter",
        f"ev_api_errors_total {s['errors']}",
        "# HELP ev_api_request_duration_seconds_sum Total time in request handlers.",
        "# TYPE ev_api_request_duration_seconds_sum counter",
        f"ev_api_request_duration_seconds_sum {s['latency_sum']:.6f}",
        "",
    ])
