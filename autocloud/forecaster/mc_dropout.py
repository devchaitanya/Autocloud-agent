"""
MC Dropout inference wrapper.

Performs K stochastic forward passes with model.train() (dropout active)
to estimate both point predictions and epistemic uncertainty.

Returns:
  means:     (n_horizons,)  — mean of q50 across K passes
  variances: (n_horizons,)  — variance of q50 across K passes (epistemic uncertainty)
"""
from __future__ import annotations

import torch
import numpy as np
from typing import Tuple

from .transformer_model import WorkloadTransformer


class MCDropoutForecaster:
    def __init__(
        self,
        model: WorkloadTransformer,
        k_samples: int = 30,
        device: str = "cpu",
    ):
        self.model = model
        self.k_samples = k_samples
        self.device = device
        self.model.to(device)

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            x: (seq_len, input_dim) or (1, seq_len, input_dim) numpy array

        Returns:
            means:     (n_horizons,) float32 — mean q50 predictions
            variances: (n_horizons,) float32 — variance across MC samples
        """
        if x.ndim == 2:
            x = x[np.newaxis, ...]         # add batch dim → (1, seq, feat)

        x_t = torch.from_numpy(x).float().to(self.device)

        # Keep model in train mode → dropout active
        self.model.train()

        preds = []
        for _ in range(self.k_samples):
            out = self.model(x_t)           # (1, n_horizons, 3)
            q50 = out[0, :, 1]             # (n_horizons,) — median predictions
            preds.append(q50.cpu().numpy())

        preds = np.stack(preds, axis=0)    # (K, n_horizons)
        means = preds.mean(axis=0)         # (n_horizons,)
        variances = preds.var(axis=0)      # (n_horizons,)

        return means.astype(np.float32), variances.astype(np.float32)

    def predict_batch(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch version for evaluation.

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            means:     (batch, n_horizons)
            variances: (batch, n_horizons)
        """
        x_t = torch.from_numpy(x).float().to(self.device)
        self.model.train()

        preds = []
        with torch.no_grad():
            for _ in range(self.k_samples):
                out = self.model(x_t)      # (batch, n_horizons, 3)
                q50 = out[:, :, 1]        # (batch, n_horizons)
                preds.append(q50.cpu().numpy())

        preds = np.stack(preds, axis=0)   # (K, batch, n_horizons)
        means = preds.mean(axis=0)        # (batch, n_horizons)
        variances = preds.var(axis=0)     # (batch, n_horizons)

        return means.astype(np.float32), variances.astype(np.float32)
