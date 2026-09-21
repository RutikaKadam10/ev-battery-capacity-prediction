"""Model tests: shapes, parameter counts, and the order-invariance property."""

import pytest
import torch

from src.models import LSTMNet, TransformerNet, build_model, count_params

BATCH, SEQ, FEAT = 4, 128, 9


@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.randn(BATCH, SEQ, FEAT)


@pytest.mark.parametrize("name", ["lstm", "transformer"])
def test_output_shape(name, x):
    model = build_model(name).eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (BATCH,)
    assert torch.isfinite(out).all()


def test_selected_configs_match_logged_parameter_counts():
    """params.yaml must describe the configurations that were actually selected.
    The Transformer '05-small' run logged 9,409 parameters."""
    assert count_params(build_model("lstm")) == 7_681
    assert count_params(build_model("transformer")) == 9_409


@pytest.mark.parametrize("pooling", ["last", "mean"])
def test_lstm_pooling_variants(pooling, x):
    model = LSTMNet(n_features=FEAT, hidden=32, pooling=pooling).eval()
    with torch.no_grad():
        assert model(x).shape == (BATCH,)


@pytest.mark.parametrize("readout", ["last", "mean", "cls"])
def test_transformer_readout_variants(readout, x):
    model = TransformerNet(n_features=FEAT, d_model=32, n_heads=2, n_layers=1,
                           dim_ff=64, readout=readout).eval()
    with torch.no_grad():
        assert model(x).shape == (BATCH,)


def test_transformer_without_posenc_is_order_invariant(x):
    """Without positional encoding, self-attention cannot see order: shuffling
    the timesteps must leave a mean-pooled output unchanged. This is the
    property the '04-no-posenc' ablation relied on."""
    model = TransformerNet(n_features=FEAT, d_model=32, n_heads=2, n_layers=1,
                           dim_ff=64, readout="mean", use_posenc=False).eval()
    perm = torch.randperm(SEQ)
    with torch.no_grad():
        torch.testing.assert_close(model(x), model(x[:, perm, :]),
                                   rtol=1e-4, atol=1e-5)


def test_transformer_with_posenc_is_order_sensitive(x):
    model = TransformerNet(n_features=FEAT, d_model=32, n_heads=2, n_layers=1,
                           dim_ff=64, readout="mean", use_posenc=True).eval()
    perm = torch.randperm(SEQ)
    with torch.no_grad():
        assert not torch.allclose(model(x), model(x[:, perm, :]), atol=1e-4)


def test_eval_mode_is_deterministic(x):
    """Dropout must be off at inference, or repeated calls would differ."""
    model = build_model("lstm").eval()
    with torch.no_grad():
        torch.testing.assert_close(model(x), model(x))
