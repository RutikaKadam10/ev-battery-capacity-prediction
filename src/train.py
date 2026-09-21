"""Training entrypoint.

Usage:
    python -m src.train --model lstm --fold 0
    python -m src.train --model lstm --all-folds
    python -m src.train --model transformer --all-folds --experiment transformer-refactored
    python -m src.train --model lstm --fold 0 --no-mlflow

Checkpoints go to models/<model>/fold<k>.pt and contain the scaler, so the
serving API normalises inputs with exactly the statistics used in training.
"""

import argparse
import json
import os
import subprocess
import time

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

from src.config import (DATA, LSTM, MLFLOW, MODELS_DIR, REPORTS_DIR,
                        TRANSFORMER, ensure_dirs, get_device, set_seed)
from src.data import build_cache, make_loaders
from src.evaluate import band_metrics, metrics, predict
from src.models import build_model, count_params

MODEL_PARAMS = {"lstm": LSTM, "transformer": TRANSFORMER}

# Reference points, out-of-sample on the same folds (notebooks 03-05)
BASELINE_RMSE_MEAN = 1.877
BASELINE_RMSE_LINEAR = 1.095


# ---------------------------------------------------------------------------
# One fold
# ---------------------------------------------------------------------------

def train_fold(name: str, data: dict, fold: int, p: dict, device,
               verbose: bool = False):
    """Train one fold. Returns (model, scaler, history, test_metrics, extras)."""
    # Seed per fold, so any fold can be rerun alone and reproduce itself.
    set_seed(DATA["seed"] + fold)

    loaders, scaler, (tr, va, te) = make_loaders(data, fold, p["batch_size"])
    model = build_model(name, n_features=DATA["n_channels"], params=p).to(device)

    criterion = nn.MSELoss()
    if name == "transformer":
        optimizer = torch.optim.AdamW(model.parameters(), lr=p["lr"],
                                      weight_decay=p["weight_decay"])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=p["lr"],
                                     weight_decay=p["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4)

    history = {"train_loss": [], "val_rmse": [], "lr": []}
    best_val, best_state, stalled = float("inf"), None, 0

    for epoch in range(p["epochs"]):
        model.train()                                  # enable dropout
        running = 0.0

        for xb, yb in loaders["train"]:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()                      # clear old gradients
            loss = criterion(model(xb), yb)            # forward + loss
            loss.backward()                            # gradients

            if "clip" in p:                            # transformer only
                nn.utils.clip_grad_norm_(model.parameters(), p["clip"])

            optimizer.step()                           # update parameters
            running += loss.item() * len(yb)

        train_loss = running / len(loaders["train"].dataset)

        vp, va_true = predict(model, loaders["val"], scaler, device)
        val_rmse = float(np.sqrt(mean_squared_error(va_true, vp)))

        scheduler.step(val_rmse)
        history["train_loss"].append(train_loss)
        history["val_rmse"].append(val_rmse)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # keep the best checkpoint, not the last
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            stalled = 0
        else:
            stalled += 1

        if verbose:
            print(f"  epoch {epoch+1:>3}  train {train_loss:.4f}  "
                  f"val_rmse {val_rmse:.4f}{'  *' if stalled == 0 else ''}")

        if stalled >= p["patience"]:
            break

    model.load_state_dict(best_state)
    tp, ta = predict(model, loaders["test"], scaler, device)

    extras = {
        "test_idx": te, "test_pred": tp, "test_actual": ta,
        "epochs_run": len(history["train_loss"]),
        "best_val_rmse": best_val,
        "n_params": count_params(model),
    }
    return model, scaler, history, metrics(ta, tp), extras


def save_checkpoint(name, fold, model, scaler, p, met, path):
    """Save weights plus everything the API needs to reproduce predictions.

    Scaler arrays are stored as plain lists so the checkpoint loads with
    torch.load(weights_only=True), which rejects numpy objects.
    """
    torch.save({
        "model_name": name,
        "fold": fold,
        "state_dict": model.state_dict(),
        "params": p,
        "n_features": DATA["n_channels"],
        "seq_len": DATA["seq_len"],
        "scaler": {
            "mean": scaler["mean"].reshape(-1).tolist(),
            "std": scaler["std"].reshape(-1).tolist(),
            "y_mean": scaler["y_mean"],
            "y_std": scaler["y_std"],
        },
        "metrics": met,
    }, path)


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

def setup_mlflow(experiment: str) -> None:
    """Environment variables if set (CI/Docker), browser OAuth otherwise."""
    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    else:
        import dagshub
        dagshub.init(repo_owner=MLFLOW["repo_owner"],
                     repo_name=MLFLOW["repo_name"], mlflow=True)
    mlflow.set_experiment(experiment)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def log_fold(name, fold, p, met, extras, history, bands, ckpt):
    mlflow.log_params({**p, "model": name, "fold": fold,
                       "n_features": DATA["n_channels"], "seed": DATA["seed"] + fold})

    for ep, (tl, vr, lr_) in enumerate(zip(history["train_loss"],
                                           history["val_rmse"], history["lr"])):
        mlflow.log_metric("train_loss", tl, step=ep)
        mlflow.log_metric("val_rmse", vr, step=ep)
        mlflow.log_metric("learning_rate", lr_, step=ep)

    mlflow.log_metrics(met)
    mlflow.log_metrics({
        "epochs_run": extras["epochs_run"],
        "best_val_rmse": extras["best_val_rmse"],
        "n_params": extras["n_params"],
        "gap_vs_mean_baseline": BASELINE_RMSE_MEAN - met["rmse"],
        "gap_vs_linear_baseline": BASELINE_RMSE_LINEAR - met["rmse"],
    })
    for b in bands:
        key = b["band"].strip("(]").replace(", ", "_")
        mlflow.log_metric(f"rmse_band_{key}", b["rmse"])
    mlflow.log_artifact(str(ckpt))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run(name: str, folds: list[int], experiment: str | None,
        verbose: bool = False) -> pd.DataFrame:
    ensure_dirs()
    device = get_device()
    p = MODEL_PARAMS[name]
    data = build_cache(DATA["n_channels"])

    out_dir = MODELS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    use_mlflow = experiment is not None
    if use_mlflow:
        setup_mlflow(experiment)
        parent = mlflow.start_run(run_name=f"{name}-{len(folds)}fold")
        mlflow.set_tags({"git_commit": git_commit(), "source": "src/train.py"})

    rows, band_rows = [], []
    try:
        for fold in folds:
            print(f"\n{'=' * 60}\n{name.upper()}  fold {fold}\n{'=' * 60}")
            t0 = time.time()
            model, scaler, hist, met, extras = train_fold(
                name, data, fold, p, device, verbose)
            elapsed = time.time() - t0

            ckpt = out_dir / f"fold{fold}.pt"
            save_checkpoint(name, fold, model, scaler, p, met, ckpt)

            bands = band_metrics(extras["test_actual"], extras["test_pred"],
                                 data["mil"][extras["test_idx"]])
            band_rows += [{"fold": fold, **b} for b in bands]

            if use_mlflow:
                with mlflow.start_run(run_name=f"fold-{fold}", nested=True):
                    log_fold(name, fold, p, met, extras, hist, bands, ckpt)
                    mlflow.log_metric("train_seconds", elapsed)

            rows.append({"fold": fold, **met, "epochs": extras["epochs_run"],
                         "seconds": round(elapsed, 1)})
            print(f"  RMSE {met['rmse']:.4f}  R2 {met['r2']:>7.4f}  "
                  f"corr {met['corr']:.3f}  ({elapsed:.0f}s, "
                  f"{extras['epochs_run']} epochs)")

        res = pd.DataFrame(rows)

        # DVC reads metrics from JSON
        summary = {f"{m}_mean": float(res[m].mean())
                   for m in ("rmse", "mae", "mape", "r2", "corr", "std_ratio")}
        if len(res) > 1:
            summary.update({f"{m}_std": float(res[m].std())
                            for m in ("rmse", "r2")})
        with open(REPORTS_DIR / f"{name}_metrics.json", "w") as f:
            json.dump(summary, f, indent=2)
        res.to_csv(REPORTS_DIR / f"{name}_per_fold.csv", index=False)
        pd.DataFrame(band_rows).to_csv(REPORTS_DIR / f"{name}_per_band.csv",
                                       index=False)

        if use_mlflow:
            mlflow.log_metrics(summary)
            mlflow.log_artifact(str(REPORTS_DIR / f"{name}_metrics.json"))
    finally:
        if use_mlflow:
            mlflow.end_run()

    return res


def main():
    ap = argparse.ArgumentParser(description="Train LSTM or Transformer.")
    ap.add_argument("--model", required=True, choices=["lstm", "transformer"])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fold", type=int)
    g.add_argument("--all-folds", action="store_true")
    ap.add_argument("--experiment", default=None,
                    help="MLflow experiment name (default: <model>-refactored)")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    folds = list(range(DATA["n_folds"])) if args.all_folds else [args.fold]
    experiment = None if args.no_mlflow else \
        (args.experiment or f"{args.model}-refactored")

    res = run(args.model, folds, experiment, args.verbose)

    print(f"\n{res.round(4).to_string(index=False)}")
    if len(res) > 1:
        print(f"\n{args.model}: RMSE {res['rmse'].mean():.3f} "
              f"± {res['rmse'].std():.3f}")


if __name__ == "__main__":
    main()
