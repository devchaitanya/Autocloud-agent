"""
Transformer forecaster trainer.

Loss: Pinball (quantile) loss summed over q ∈ {0.1, 0.5, 0.9} and all 4 horizons.
Optimizer: AdamW with cosine annealing LR schedule.
Early stopping on validation quantile loss (patience=10 epochs).
"""
from __future__ import annotations

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple

# Allow running from any working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from forecaster.transformer_model import WorkloadTransformer
from environment.workload import generate_forecast_dataset, split_day1, make_sequences


# ------------------------------------------------------------------ #
# Quantile (Pinball) loss
# ------------------------------------------------------------------ #

class QuantileLoss(nn.Module):
    def __init__(self, quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (batch, n_horizons, 3)  — quantile predictions
            target: (batch, n_horizons)     — true demand values
        """
        target_exp = target.unsqueeze(-1).expand_as(pred)   # (batch, n_horizons, 3)
        q = torch.tensor(self.quantiles, device=pred.device, dtype=pred.dtype)
        errors = target_exp - pred                           # (batch, n_horizons, 3)
        loss = torch.max(q * errors, (q - 1) * errors)      # pinball loss per quantile
        return loss.mean()


# ------------------------------------------------------------------ #
# Training loop
# ------------------------------------------------------------------ #

def train_forecaster(
    model: Optional[WorkloadTransformer] = None,
    seed: int = 42,
    epochs: int = 50,
    patience: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    seq_len: int = 20,
    horizons: Tuple[int, ...] = (1, 5, 10, 15),
    device: Optional[str] = None,
    verbose: bool = True,
    save_path: Optional[str] = None,
    # Optional: pass pre-loaded Alibaba trace data directly
    train_data: Optional[np.ndarray] = None,
    val_data: Optional[np.ndarray] = None,
) -> WorkloadTransformer:
    """
    Train the Transformer forecaster.

    If train_data/val_data are provided (from AlibabaTraceLoader), uses those.
    Otherwise falls back to synthetic data (for unit tests / offline use).

    Returns the best model (lowest validation loss).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if train_data is None or val_data is None:
        # Fallback: synthetic data
        rng = np.random.default_rng(seed)
        day1_data, _ = generate_forecast_dataset(rng)
        train_data, val_data = split_day1(day1_data, train_fraction=0.6)

    X_train, y_train = make_sequences(train_data, seq_len=seq_len, horizons=horizons)
    X_val, y_val = make_sequences(val_data, seq_len=seq_len, horizons=horizons)

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Insufficient data to create sequences. Check seq_len and horizons.")

    # DataLoaders
    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).float(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    if model is None:
        model = WorkloadTransformer(
            input_dim=4,
            d_model=64,
            n_heads=4,
            d_ff=256,
            n_layers=2,
            dropout=0.2,
            seq_len=seq_len,
            n_horizons=len(horizons),
        )
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = QuantileLoss(quantiles=(0.1, 0.5, 0.9))

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)              # (batch, n_horizons, 3)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        # ── Validate ───────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_mape = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * len(xb)
                # MAPE on q50 (index 1) for first horizon
                q50 = pred[:, 0, 1]
                target = yb[:, 0]
                mask = target > 0.01   # avoid division by near-zero
                if mask.sum() > 0:
                    val_mape += (((q50[mask] - target[mask]).abs() / target[mask].abs()).mean().item() * 100) * len(xb)
                n_val += len(xb)
        val_loss /= len(val_ds)
        val_mape /= n_val

        scheduler.step()

        if verbose and epoch % 5 == 0:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val MAPE: {val_mape:.1f}%"
            )

        # ── Early stopping ─────────────────────────────────────
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        if verbose:
            print(f"Model saved to {save_path}")

    if verbose:
        print(f"Best val loss: {best_val_loss:.4f}")

    return model


def compute_mape(model: WorkloadTransformer, X: np.ndarray, y: np.ndarray, device: str = "cpu") -> float:
    """Compute MAPE on q50 predictions for first horizon."""
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(X).float().to(device)
        pred = model(x_t)       # (N, n_horizons, 3)
        q50 = pred[:, 0, 1].cpu().numpy()   # first horizon, median
    target = y[:, 0]
    mask = target > 0.01
    if mask.sum() == 0:
        return 0.0
    return float(np.abs((q50[mask] - target[mask]) / target[mask]).mean() * 100)


if __name__ == "__main__":
    import sys
    print("Training Transformer forecaster on synthetic data...")
    model = train_forecaster(verbose=True, epochs=50, save_path="forecaster_weights.pt")
    print("Done.")

    # Quick MAPE check
    rng = np.random.default_rng(42)
    from environment.workload import generate_forecast_dataset, split_day1, make_sequences
    day1_data, _ = generate_forecast_dataset(rng)
    _, val_data = split_day1(day1_data)
    X_val, y_val = make_sequences(val_data)
    mape = compute_mape(model, X_val, y_val)
    print(f"Validation MAPE (q50, horizon t+1): {mape:.1f}%")
