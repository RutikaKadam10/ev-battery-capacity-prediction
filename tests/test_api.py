"""API tests. A temporary checkpoint is served, so no trained model is needed."""

import importlib
import os

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from src.config import LSTM
from src.models import build_model
from src.train import save_checkpoint

SEQ, FEAT = 128, 9


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Serve a freshly initialised model from a temporary checkpoint.

    MODEL_PATH is read when src.api is imported, so the env var is set first
    and the module reloaded. TestClient's context manager triggers the lifespan
    handler that loads the model.
    """
    path = tmp_path_factory.mktemp("models") / "fold0.pt"
    torch.manual_seed(0)
    scaler = {
        "mean": np.full((1, 1, FEAT), 20.0, dtype=np.float32),
        "std": np.full((1, 1, FEAT), 2.0, dtype=np.float32),
        "y_mean": 40.49,
        "y_std": 1.86,
    }
    save_checkpoint("lstm", 0, build_model("lstm"), scaler, LSTM,
                    {"rmse": 1.69}, path)

    os.environ["MODEL_PATH"] = str(path)
    import src.api
    importlib.reload(src.api)          # re-read MODEL_PATH from the environment

    with TestClient(src.api.app) as c:
        yield c

    del os.environ["MODEL_PATH"]


def _snippet():
    rng = np.random.default_rng(0)
    return rng.normal(20, 2, size=(SEQ, FEAT)).round(4).tolist()


# ---------------------------------------------------------------------------
# Ops endpoints
# ---------------------------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_name"] == "lstm"
    assert body["input_shape"] == [SEQ, FEAT]
    assert len(body["channels"]) == FEAT


def test_metrics_is_prometheus_text(client):
    client.post("/predict", json={"snippet": _snippet()})
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "ev_api_predictions_total" in r.text
    assert "# TYPE" in r.text


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def test_predict_returns_plausible_capacity(client):
    r = client.post("/predict", json={"snippet": _snippet()})
    assert r.status_code == 200
    body = r.json()

    # An untrained model still outputs near the target mean, because the
    # prediction is inverted through the scaler. Anything far outside the
    # dataset's range would mean the inversion is wrong.
    assert 20 < body["capacity_ah"] < 60
    assert 40 < body["soh_percent"] < 130
    assert body["model_name"] == "lstm"


def test_predict_is_deterministic(client):
    payload = {"snippet": _snippet()}
    first = client.post("/predict", json=payload).json()["capacity_ah"]
    second = client.post("/predict", json=payload).json()["capacity_ah"]
    assert first == second          # dropout must be off at inference


def test_batch_matches_single(client):
    s = _snippet()
    single = client.post("/predict", json={"snippet": s}).json()["capacity_ah"]
    batch = client.post("/predict/batch", json={"snippets": [s, s]}).json()

    assert batch["count"] == 2
    assert batch["capacity_ah"] == [single, single]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_wrong_timestep_count_is_rejected(client):
    r = client.post("/predict", json={"snippet": _snippet()[:64]})
    assert r.status_code == 422
    assert "128" in r.json()["detail"]


def test_wrong_channel_count_is_rejected(client):
    bad = [row[:7] for row in _snippet()]
    r = client.post("/predict", json={"snippet": bad})
    assert r.status_code == 422


def test_ragged_snippet_is_rejected(client):
    bad = _snippet()
    bad[5] = bad[5][:4]
    r = client.post("/predict", json={"snippet": bad})
    assert r.status_code == 422


def test_empty_batch_is_rejected(client):
    r = client.post("/predict/batch", json={"snippets": []})
    assert r.status_code == 422
