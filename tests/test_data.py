"""Data pipeline tests. Synthetic arrays only, except the manifest check."""

import json

import numpy as np
import pytest

from src.config import MANIFEST_PATH
from src.data import SnippetDataset, fit_scaler, split_indices


@pytest.fixture
def synthetic():
    """30 vehicles, 5 folds, 20 snippets each."""
    rng = np.random.default_rng(0)
    cars = np.repeat(np.arange(30), 20)
    car_to_fold = {c: c % 5 for c in range(30)}
    folds = np.array([car_to_fold[c] for c in cars], dtype=np.int8)
    X = rng.normal(40, 3, size=(len(cars), 128, 9)).astype(np.float32)
    y = rng.normal(40, 2, size=len(cars)).astype(np.float32)
    return {"X": X, "y": y, "cars": cars, "folds": folds}


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("test_fold", range(5))
def test_splits_are_disjoint_and_complete(synthetic, test_fold):
    tr, va, te = split_indices(synthetic["folds"], test_fold, n_folds=5)

    assert not set(tr) & set(va)
    assert not set(tr) & set(te)
    assert not set(va) & set(te)
    assert len(tr) + len(va) + len(te) == len(synthetic["folds"])


@pytest.mark.parametrize("test_fold", range(5))
def test_no_vehicle_crosses_split_boundary(synthetic, test_fold):
    """The leakage test. Overlapping snippets from one vehicle on both sides
    of a boundary would let the model memorise vehicles and inflate scores."""
    tr, va, te = split_indices(synthetic["folds"], test_fold, n_folds=5)
    cars = synthetic["cars"]

    assert not set(cars[tr]) & set(cars[te])
    assert not set(cars[tr]) & set(cars[va])
    assert not set(cars[va]) & set(cars[te])


def test_validation_rotates_with_test_fold(synthetic):
    for test_fold in range(5):
        _, va, _ = split_indices(synthetic["folds"], test_fold, n_folds=5)
        assert set(synthetic["folds"][va]) == {(test_fold + 1) % 5}


def test_real_manifest_assigns_each_vehicle_to_one_fold():
    """Checks the committed fold assignment itself, not just the split code."""
    if not MANIFEST_PATH.exists():
        pytest.skip("fold manifest not present (e.g. CI without DVC pull)")

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    seen = {}
    for fold, cars in manifest["folds"].items():
        for car in cars:
            assert car not in seen, f"vehicle {car} in folds {seen[car]} and {fold}"
            seen[car] = fold

    assert len(seen) == manifest["config"]["n_cars"]


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def test_scaler_uses_training_data_only(synthetic):
    tr, _, te = split_indices(synthetic["folds"], 0, n_folds=5)
    X, y = synthetic["X"].copy(), synthetic["y"].copy()

    before = fit_scaler(X[tr], y[tr])
    X[te] += 1000.0                                 # corrupt only the test rows
    y[te] += 1000.0
    after = fit_scaler(X[tr], y[tr])

    np.testing.assert_allclose(before["mean"], after["mean"])
    assert before["y_mean"] == after["y_mean"]


def test_training_split_is_standardised(synthetic):
    tr, _, _ = split_indices(synthetic["folds"], 0, n_folds=5)
    scaler = fit_scaler(synthetic["X"][tr], synthetic["y"][tr])
    ds = SnippetDataset(synthetic["X"], synthetic["y"], tr, scaler)

    np.testing.assert_allclose(ds.X.mean(axis=(0, 1)), 0, atol=1e-3)
    np.testing.assert_allclose(ds.X.std(axis=(0, 1)), 1, atol=1e-3)
    assert abs(ds.y.mean()) < 1e-3
    assert abs(ds.y.std() - 1) < 1e-3


def test_target_scaling_round_trips(synthetic):
    """Predictions are inverted back to Ah; the inversion must be exact."""
    tr, _, _ = split_indices(synthetic["folds"], 0, n_folds=5)
    scaler = fit_scaler(synthetic["X"][tr], synthetic["y"][tr])
    ds = SnippetDataset(synthetic["X"], synthetic["y"], tr, scaler)

    restored = ds.y * scaler["y_std"] + scaler["y_mean"]
    np.testing.assert_allclose(restored, synthetic["y"][tr], rtol=1e-5)


def test_constant_channel_does_not_divide_by_zero(synthetic):
    X = synthetic["X"].copy()
    X[:, :, 5] = 14.0                               # e.g. a constant temperature
    tr, _, _ = split_indices(synthetic["folds"], 0, n_folds=5)
    scaler = fit_scaler(X[tr], synthetic["y"][tr])
    ds = SnippetDataset(X, synthetic["y"], tr, scaler)

    assert np.isfinite(ds.X).all()
