"""Inference: load a checkpoint and turn raw snippets into capacity estimates.

Kept separate from the API so it can be used from scripts and tested without
starting a server.
"""

from pathlib import Path

import numpy as np
import torch

from src.models import build_model

# Nominal (healthy) pack capacity, used only to report State of Health as a
# percentage. 46.17 Ah is the maximum observed across the full brand-1 fleet.
#
# Deliberately not in params.yaml: the `data` section there is a DVC dependency
# of both training stages, so adding a key would mark them out of date and
# trigger a retrain for a display-only constant.
NOMINAL_CAPACITY_AH = 46.17


def load_model(path: Path | str, device: torch.device | None = None) -> dict:
    """Load a checkpoint written by src.train.save_checkpoint.

    Returns a bundle holding the model, the scaler used in training, and the
    metadata the API reports.

    weights_only=True refuses arbitrary pickled objects - the safe way to load
    a file that may not have been produced locally.
    """
    device = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=device, weights_only=True)

    model = build_model(ckpt["model_name"], n_features=ckpt["n_features"],
                        params=ckpt["params"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()            # eval() disables dropout

    s = ckpt["scaler"]
    return {
        "model": model,
        "device": device,
        # (1, 1, C) so it broadcasts over (batch, timesteps, channels)
        "mean": np.asarray(s["mean"], dtype=np.float32).reshape(1, 1, -1),
        "std": np.asarray(s["std"], dtype=np.float32).reshape(1, 1, -1),
        "y_mean": float(s["y_mean"]),
        "y_std": float(s["y_std"]),
        "model_name": ckpt["model_name"],
        "fold": int(ckpt["fold"]),
        "n_features": int(ckpt["n_features"]),
        "seq_len": int(ckpt["seq_len"]),
        "params": ckpt["params"],
        "metrics": ckpt.get("metrics", {}),
        "path": str(path),
    }


def predict(bundle: dict, snippets: np.ndarray) -> np.ndarray:
    """Capacity in Ah for a batch of raw (unscaled) snippets.

    snippets: (n, seq_len, n_features), raw sensor values as recorded.
    returns:  (n,) capacity in Ah.

    The scaler saved with the checkpoint is applied here. Normalising with any
    other statistics produces confidently wrong numbers and no error.
    """
    x = np.asarray(snippets, dtype=np.float32)
    if x.ndim == 2:                                  # single snippet
        x = x[None, ...]

    expected = (bundle["seq_len"], bundle["n_features"])
    if x.shape[1:] != expected:
        raise ValueError(f"expected snippets of shape {expected}, got {x.shape[1:]}")

    x = (x - bundle["mean"]) / bundle["std"]

    with torch.no_grad():                            # no gradient tracking
        out = bundle["model"](torch.from_numpy(x).to(bundle["device"]))

    # invert the target standardisation, back to Ah
    return out.cpu().numpy() * bundle["y_std"] + bundle["y_mean"]


def state_of_health(capacity_ah: np.ndarray | float,
                    nominal: float = NOMINAL_CAPACITY_AH) -> np.ndarray | float:
    """Capacity as a percentage of a healthy pack."""
    return capacity_ah / nominal * 100.0
