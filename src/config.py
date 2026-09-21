"""Project configuration.

Loads params.yaml once and exposes it, plus absolute paths resolved from the
repo root. Every other module imports from here, so a hyperparameter or path
change is a one-line edit to params.yaml rather than a hunt through source.
"""

import random
from pathlib import Path

import numpy as np
import torch
import yaml

# Repo root = parent of src/. Resolving from this file means scripts run
# correctly regardless of the current working directory.
ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "params.yaml"


def load_params(path: Path = PARAMS_PATH) -> dict:
    """Read params.yaml into a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


PARAMS = load_params()

DATA = PARAMS["data"]
LSTM = PARAMS["lstm"]
TRANSFORMER = PARAMS["transformer"]
MLFLOW = PARAMS["mlflow"]

# Absolute paths built from the relative ones in params.yaml
_p = {k: ROOT / v for k, v in PARAMS["paths"].items()}
RAW_DIR = _p["raw"]
RAW_SNIPPETS_DIR = RAW_DIR / "data"
PROCESSED_DIR = _p["processed"]
CACHE_DIR = _p["cache"]
MODELS_DIR = _p["models"]
REPORTS_DIR = _p["reports"]
FIG_DIR = _p["figures"]

# Fold assignment fixed in notebook 03. Read-only: regenerating it would give a
# different split and void every comparison against the baselines.
MANIFEST_PATH = PROCESSED_DIR / "fold_manifest.json"

# Mileage bands in thousand km, for per-band evaluation
BANDS = [0, 50, 100, 150, 200, 300]


def get_device() -> torch.device:
    """Best available device: Apple Silicon GPU, then NVIDIA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = DATA["seed"]) -> None:
    """Seed every source of randomness used in training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dirs() -> None:
    """Create output directories if absent."""
    for d in (CACHE_DIR, MODELS_DIR, REPORTS_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"root      {ROOT}")
    for name, path in [("raw", RAW_DIR), ("processed", PROCESSED_DIR),
                       ("cache", CACHE_DIR), ("manifest", MANIFEST_PATH),
                       ("models", MODELS_DIR), ("reports", REPORTS_DIR)]:
        print(f"{name:<10}{path}  {'ok' if path.exists() else 'MISSING'}")
    print(f"device    {get_device()}")
    print(f"data      {DATA}")
