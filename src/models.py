"""Model definitions: LSTM and Transformer.

Lifted unchanged from notebooks 04 and 05 so results reproduce. The ablation
flags (pooling, dropout_before_output, readout, use_posenc) are kept so every
logged configuration stays runnable, even though params.yaml selects only the
final ones.

Usage:
    python -m src.models            # shape and parameter-count check
"""

import math

import torch
import torch.nn as nn

from src.config import DATA, LSTM, TRANSFORMER


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

class LSTMNet(nn.Module):
    """1 LSTM layer -> 2 FC layers.

    pooling='last'           final hidden state as the sequence summary
    pooling='mean'           average of all hidden states (tested, rejected)
    dropout_before_output    reproduces the original, broken head
    """

    def __init__(self, n_features, hidden=128, fc_hidden=64, dropout=0.3,
                 pooling="last", dropout_before_output=False):
        super().__init__()
        self.pooling = pooling

        # batch_first=True -> (batch, seq, feature)
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden,
                            num_layers=1, batch_first=True)

        if dropout_before_output:
            layers = [nn.Dropout(dropout), nn.Linear(hidden, fc_hidden),
                      nn.LeakyReLU(), nn.Dropout(dropout),
                      nn.Linear(fc_hidden, 1)]
        else:
            layers = [nn.Linear(hidden, fc_hidden), nn.LeakyReLU(),
                      nn.Dropout(dropout), nn.Linear(fc_hidden, 1)]

        self.head = nn.Sequential(*layers)

        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out, _ = self.lstm(x)                         # (B, seq, hidden)
        summary = out.mean(dim=1) if self.pooling == "mean" else out[:, -1, :]
        return self.head(summary).squeeze(-1)         # (B,)


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding, added to the embedded input.

    Self-attention is permutation-invariant; this gives each position a
    distinct signature. Registered as a buffer, not a learned parameter.
    """

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerNet(nn.Module):
    """Transformer encoder over the charging window.

    readout: 'last' | 'mean' | 'cls'
    use_posenc=False makes the model order-invariant.
    """

    def __init__(self, n_features=9, d_model=64, n_heads=4, n_layers=2,
                 dim_ff=128, dropout=0.2, readout="last", use_posenc=True):
        super().__init__()
        self.readout = readout

        self.input_proj = nn.Linear(n_features, d_model)
        self.posenc = PositionalEncoding(d_model) if use_posenc else None

        if readout == "cls":
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
            norm_first=True,               # pre-norm, more stable on small data
        )
        # enable_nested_tensor is incompatible with norm_first; disabling it
        # silences the warning without changing behaviour
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.input_proj(x)                        # (B, 128, d_model)

        if self.readout == "cls":
            h = torch.cat([self.cls.expand(h.size(0), -1, -1), h], dim=1)

        if self.posenc is not None:
            h = self.posenc(h)

        h = self.encoder(h)

        if self.readout == "mean":
            summary = h.mean(dim=1)
        elif self.readout == "cls":
            summary = h[:, 0, :]
        else:
            summary = h[:, -1, :]

        return self.head(summary).squeeze(-1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(name: str, n_features: int = DATA["n_channels"],
                params: dict | None = None) -> nn.Module:
    """Construct a model from params.yaml (or an override dict)."""
    if name == "lstm":
        p = params or LSTM
        return LSTMNet(n_features=n_features, hidden=p["hidden"],
                       fc_hidden=p["fc_hidden"], dropout=p["dropout"],
                       pooling=p.get("pooling", "last"))
    if name == "transformer":
        p = params or TRANSFORMER
        return TransformerNet(n_features=n_features, d_model=p["d_model"],
                              n_heads=p["n_heads"], n_layers=p["n_layers"],
                              dim_ff=p["dim_ff"], dropout=p["dropout"],
                              readout=p["readout"],
                              use_posenc=p["use_posenc"])
    raise ValueError(f"unknown model: {name}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    x = torch.randn(4, DATA["seq_len"], DATA["n_channels"])

    # Transformer '05-small' logged 9,409 parameters in notebook 05; matching
    # that confirms params.yaml describes the selected configuration.
    expected = {"lstm": None, "transformer": 9409}

    for name in ("lstm", "transformer"):
        m = build_model(name).eval()
        with torch.no_grad():
            out = m(x)
        n = count_params(m)
        check = ""
        if expected[name] is not None:
            check = "  matches notebook" if n == expected[name] else \
                    f"  MISMATCH (notebook: {expected[name]:,})"
        print(f"{name:<12} in {tuple(x.shape)} -> out {tuple(out.shape)}  "
              f"params {n:,}{check}")
