"""Data loading: cache, fold splits, scaling, Dataset, DataLoaders.

Logic is lifted unchanged from notebooks 04/05 so that results reproduce.

Usage:
    python -m src.data              # build (or verify) the cache
"""

import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import CACHE_DIR, DATA, MANIFEST_PATH, ensure_dirs


# ---------------------------------------------------------------------------
# Fold manifest
# ---------------------------------------------------------------------------

def load_manifest() -> dict:
    """Read the fixed fold assignment produced in notebook 03."""
    with open(MANIFEST_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def build_cache(n_channels: int = DATA["n_channels"]) -> dict:
    """Load all snippets into memory, caching to .npy.

    Returns a dict with X (n, 128, C), y (n,), cars (n,), mil (n,), folds (n,).

    n_channels = 7 : raw sensors, timestamp channel dropped
    n_channels = 9 : plus cell voltage spread and temperature spread
    """
    ensure_dirs()
    manifest = load_manifest()
    car_to_fold = {int(k): v for k, v in manifest["car_to_fold"].items()}

    files = {k: CACHE_DIR / f"{k}_{n_channels}ch.npy"
             for k in ("X", "y", "cars", "mil")}

    if all(p.exists() for p in files.values()):
        X, y, cars, mil = (np.load(files[k]) for k in ("X", "y", "cars", "mil"))
    else:
        paths, path_cars = [], []
        for car, car_files in manifest["files"].items():
            paths.extend(car_files)
            path_cars.extend([int(car)] * len(car_files))

        n = len(paths)
        X = np.empty((n, DATA["seq_len"], n_channels), dtype=np.float32)
        y = np.empty(n, dtype=np.float32)
        mil = np.empty(n, dtype=np.float32)
        cars = np.asarray(path_cars, dtype=np.int32)

        for i, p in enumerate(tqdm(paths, desc=f"building {n_channels}ch cache")):
            arr, meta = torch.load(p, weights_only=False)
            raw = arr[:, :7]                                   # drop timestamp

            if n_channels == 9:
                v_spread = (raw[:, 3] - raw[:, 4])[:, None]    # cell voltage imbalance
                t_spread = (raw[:, 5] - raw[:, 6])[:, None]    # temperature spread
                X[i] = np.hstack([raw, v_spread, t_spread])
            else:
                X[i] = raw

            y[i] = meta["capacity"]
            mil[i] = meta["mileage"]                           # evaluation only

        for k, p in files.items():
            np.save(p, {"X": X, "y": y, "cars": cars, "mil": mil}[k])

    folds = np.array([car_to_fold[int(c)] for c in cars], dtype=np.int8)
    return {"X": X, "y": y, "cars": cars, "mil": mil, "folds": folds}


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def split_indices(folds: np.ndarray, test_fold: int,
                  n_folds: int = DATA["n_folds"]):
    """Three-way split by vehicle: train / validation / test.

    Validation rotates with the test fold. The network needs held-out data to
    decide when to stop and which checkpoint to keep; using the test fold for
    that would leak test information into training.
    """
    val_fold = (test_fold + 1) % n_folds
    train_idx = np.where((folds != test_fold) & (folds != val_fold))[0]
    val_idx = np.where(folds == val_fold)[0]
    test_idx = np.where(folds == test_fold)[0]
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def fit_scaler(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Per-channel input stats and target stats, from the training split only.

    Returned as a dict so it can be saved inside each checkpoint. The serving
    API must normalise with exactly these values.
    """
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std[std < 1e-8] = 1.0                                     # guard constants
    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "y_mean": float(y_train.mean()),
        "y_std": float(y_train.std()),
    }


# ---------------------------------------------------------------------------
# Dataset and loaders
# ---------------------------------------------------------------------------

class SnippetDataset(Dataset):
    """Snippets held in memory, inputs and target standardised once."""

    def __init__(self, X, y, idx, scaler):
        self.X = ((X[idx] - scaler["mean"]) / scaler["std"]).astype(np.float32)
        self.y = ((y[idx] - scaler["y_mean"]) / scaler["y_std"]).astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])


def make_loaders(data: dict, test_fold: int, batch_size: int):
    """Build train/val/test loaders for one fold.

    Returns (loaders, scaler, (train_idx, val_idx, test_idx)).
    """
    tr, va, te = split_indices(data["folds"], test_fold)
    scaler = fit_scaler(data["X"][tr], data["y"][tr])

    X, y = data["X"], data["y"]
    loaders = {
        "train": DataLoader(SnippetDataset(X, y, tr, scaler),
                            batch_size=batch_size, shuffle=True),
        "val": DataLoader(SnippetDataset(X, y, va, scaler),
                          batch_size=batch_size, shuffle=False),
        "test": DataLoader(SnippetDataset(X, y, te, scaler),
                           batch_size=batch_size, shuffle=False),
    }
    return loaders, scaler, (tr, va, te)


if __name__ == "__main__":
    data = build_cache()
    print(f"X      {data['X'].shape}  {data['X'].nbytes / 1024**2:.0f} MB")
    print(f"y      {data['y'].min():.2f} - {data['y'].max():.2f} Ah "
          f"(mean {data['y'].mean():.2f}, std {data['y'].std():.2f})")
    print(f"cars   {len(np.unique(data['cars']))}")
    print(f"folds  {np.bincount(data['folds'])}")

    for f in range(DATA["n_folds"]):
        tr, va, te = split_indices(data["folds"], f)
        shared = set(data["cars"][tr]) & set(data["cars"][te])
        print(f"fold {f}: train {len(tr):>6,}  val {len(va):>6,}  "
              f"test {len(te):>6,}  shared vehicles train/test: {len(shared)}")
