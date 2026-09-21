"""Checkpoint tests: what the serving API will depend on."""

import numpy as np
import pytest
import torch

from src.config import LSTM, TRANSFORMER
from src.models import build_model
from src.train import save_checkpoint


def _scaler(n_features=9):
    rng = np.random.default_rng(0)
    return {
        "mean": rng.normal(20, 5, size=(1, 1, n_features)).astype(np.float32),
        "std": rng.uniform(0.5, 3, size=(1, 1, n_features)).astype(np.float32),
        "y_mean": 40.49,
        "y_std": 1.86,
    }


@pytest.mark.parametrize("name,params", [("lstm", LSTM),
                                         ("transformer", TRANSFORMER)])
def test_checkpoint_round_trip(tmp_path, name, params):
    """Save, reload with weights_only=True, rebuild, and get identical output."""
    torch.manual_seed(0)
    model = build_model(name).eval()
    scaler = _scaler()
    path = tmp_path / "ckpt.pt"

    save_checkpoint(name, 0, model, scaler, params, {"rmse": 1.7}, path)

    # weights_only=True refuses arbitrary pickled objects - the safe way to
    # load a file the API did not create itself
    ckpt = torch.load(path, weights_only=True)

    rebuilt = build_model(ckpt["model_name"], n_features=ckpt["n_features"],
                          params=ckpt["params"])
    rebuilt.load_state_dict(ckpt["state_dict"])
    rebuilt.eval()

    x = torch.randn(3, 128, 9)
    with torch.no_grad():
        torch.testing.assert_close(model(x), rebuilt(x))


def test_checkpoint_scaler_reproduces_training_normalisation(tmp_path):
    """The API rebuilds the scaler from stored lists; it must match exactly."""
    scaler = _scaler()
    path = tmp_path / "ckpt.pt"
    save_checkpoint("lstm", 0, build_model("lstm"), scaler, LSTM, {}, path)

    stored = torch.load(path, weights_only=True)["scaler"]
    mean = np.asarray(stored["mean"], dtype=np.float32).reshape(1, 1, -1)
    std = np.asarray(stored["std"], dtype=np.float32).reshape(1, 1, -1)

    raw = np.random.default_rng(1).normal(20, 5, size=(2, 128, 9)).astype(np.float32)
    np.testing.assert_allclose((raw - mean) / std,
                               (raw - scaler["mean"]) / scaler["std"], rtol=1e-6)
    assert stored["y_mean"] == scaler["y_mean"]
    assert stored["y_std"] == scaler["y_std"]


def test_checkpoint_records_what_serving_needs(tmp_path):
    path = tmp_path / "ckpt.pt"
    save_checkpoint("transformer", 2, build_model("transformer"), _scaler(),
                    TRANSFORMER, {"rmse": 1.71}, path)
    ckpt = torch.load(path, weights_only=True)

    for key in ("model_name", "fold", "state_dict", "params",
                "n_features", "seq_len", "scaler", "metrics"):
        assert key in ckpt
    assert ckpt["n_features"] == 9
    assert ckpt["seq_len"] == 128
