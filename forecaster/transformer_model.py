"""
Workload Transformer Forecaster.

Architecture:
  Input  (batch, seq=20, 4)
    → Linear(4, d_model=64)
    → + sinusoidal positional encoding
    → TransformerEncoderLayer × n_layers  (d_model=64, nhead=4, d_ff=256, dropout=0.2)
    → take last token  (batch, 64)
    → n_horizons × Linear(64, 3)          # 3 quantiles (q10, q50, q90) per horizon

MC Dropout: keep model.train() at inference — dropout fires in both attention
and FFN layers, providing richer epistemic uncertainty than LSTM recurrent dropout.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Tuple


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)   # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class WorkloadTransformer(nn.Module):
    """
    Transformer encoder for multi-horizon workload forecasting.

    Outputs: (batch, n_horizons, 3)  — 3 quantiles per horizon.
    """

    def __init__(
        self,
        input_dim: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 256,
        n_layers: int = 2,
        dropout: float = 0.2,
        seq_len: int = 20,
        n_horizons: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_horizons = n_horizons

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=seq_len + 10, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # One output head per horizon, each outputting 3 quantiles
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 3),   # (q10, q50, q90)
            )
            for _ in range(n_horizons)
        ])

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            out: (batch, n_horizons, 3)  — quantile predictions
        """
        # Project input
        h = self.input_proj(x)          # (batch, seq, d_model)
        h = self.pos_enc(h)             # + positional encoding

        # Transformer encoder
        h = self.encoder(h)             # (batch, seq, d_model)

        # Use the last token as the summary representation
        h_last = h[:, -1, :]            # (batch, d_model)

        # Compute quantile outputs for each horizon
        outs = [head(h_last) for head in self.output_heads]   # n_horizons × (batch, 3)
        out = torch.stack(outs, dim=1)                         # (batch, n_horizons, 3)
        return out

    def predict_q50(self, x: torch.Tensor) -> torch.Tensor:
        """Returns only the median (q50) predictions: (batch, n_horizons)."""
        out = self.forward(x)   # (batch, n_horizons, 3)
        return out[:, :, 1]     # index 1 = q50
