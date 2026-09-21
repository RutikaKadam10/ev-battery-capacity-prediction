"""Evaluation: metrics, prediction in Ah, per-mileage-band breakdown."""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import BANDS


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """All metrics in Ah.

    corr and std_ratio distinguish a collapsed model (constant output,
    std_ratio ~0) from a compressed one (tracks reality but hedges toward the
    mean, std_ratio ~0.3-0.5 with corr > 0.3).
    """
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "r2": float(r2_score(y_true, y_pred)),
        "corr": float(np.corrcoef(y_pred, y_true)[0, 1]),
        "std_ratio": float(y_pred.std() / y_true.std()),
    }


def predict(model, loader, scaler: dict, device) -> tuple[np.ndarray, np.ndarray]:
    """Run a loader through the model. Returns (predictions, targets) in Ah."""
    model.eval()                                       # disable dropout
    preds, actuals = [], []

    with torch.no_grad():                              # no gradient tracking
        for xb, yb in loader:
            preds.append(model(xb.to(device)).cpu().numpy())
            actuals.append(yb.numpy())

    # invert target standardisation
    p = np.concatenate(preds) * scaler["y_std"] + scaler["y_mean"]
    a = np.concatenate(actuals) * scaler["y_std"] + scaler["y_mean"]
    return p, a


def band_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                 mileage_km: np.ndarray, min_n: int = 50) -> list[dict]:
    """Metrics per mileage band.

    Aggregate RMSE is dominated by the 100-150k band (41% of snippets), so a
    breakdown is needed to see where a model actually helps. Bands with fewer
    than min_n test snippets are skipped as too noisy to report.
    """
    cut = pd.cut(mileage_km / 1000, bins=BANDS)
    rows = []
    for band in cut.categories:
        m = np.asarray(cut == band)
        if m.sum() < min_n:
            continue
        rows.append({"band": str(band), "n": int(m.sum()),
                     **metrics(y_true[m], y_pred[m])})
    return rows
