"""
Centralized artifact path resolution.

Resolves paths to trained model checkpoints, workload traces, and forecaster
weights without hardcoding in every script.

Resolution order (first match wins):
  1. Explicit path passed by the caller
  2. outputs/ directory  (Kaggle training artifacts)
  3. checkpoints/        (local synced copy)
"""
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class ArtifactPaths:
    """Resolve and validate paths to all trained-model artifacts."""

    project_root: Path = field(default_factory=lambda: _PROJECT_ROOT)
    checkpoint_dir: Optional[str] = None
    workload_file: Optional[str] = None
    forecaster_path: Optional[str] = None

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root)
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self._resolve_checkpoint_dir()
        if self.workload_file is None:
            self.workload_file = self._find_workload()
        if self.forecaster_path is None:
            self.forecaster_path = self._find_forecaster()

    # Resolution helpers

    def _resolve_checkpoint_dir(self) -> str:
        candidates = [
            self.project_root / "outputs" / "rl_agents",
            self.project_root.parent / "outputs" / "rl_agents",
            self.project_root / "checkpoints",
        ]
        for c in candidates:
            if (c / "so_actor_final.pt").exists():
                return str(c)
        return str(self.project_root / "checkpoints")

    def _find_workload(self) -> Optional[str]:
        candidates = [
            self.project_root / "outputs" / "train_Forecaster" / "day2_processed.npy",
            self.project_root / "outputs" / "train_Forecaster" / "day1_processed.npy",
            self.project_root.parent / "outputs" / "train_Forecaster" / "day2_processed.npy",
            self.project_root.parent / "outputs" / "train_Forecaster" / "day1_processed.npy",
            self.project_root / "data" / "day2_processed.npy",
            self.project_root / "data" / "day1_processed.npy",
            self.project_root / "checkpoints" / "day2_processed.npy",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return None

    def _find_forecaster(self) -> Optional[str]:
        candidates = [
            self.project_root / "outputs" / "train_Forecaster" / "forecaster_weights.pt",
            self.project_root.parent / "outputs" / "train_Forecaster" / "forecaster_weights.pt",
            self.project_root / "checkpoints" / "forecaster_weights.pt",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return None

    # Validation

    def validate_checkpoints(self, tag: str = "final") -> None:
        """Raise FileNotFoundError if any required .pt files are missing."""
        required = [
            f"so_actor_{tag}.pt", f"so_critic_{tag}.pt",
            f"con_actor_{tag}.pt", f"con_critic_{tag}.pt",
            f"sch_actor_{tag}.pt", f"sch_critic_{tag}.pt",
        ]
        missing = [
            f for f in required
            if not os.path.isfile(os.path.join(self.checkpoint_dir, f))
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing RL checkpoints in '{self.checkpoint_dir}': {missing}"
            )

    # Workload loader

    def make_workload_fn(self):
        """Build a CloudEnv-compatible workload_fn from the resolved .npy path."""
        if self.workload_file is None:
            return None
        data = np.load(self.workload_file)
        rates = np.clip(data[:, 0] if data.ndim > 1 else data, 0.1, 1.0)
        n = len(rates)

        def fn(sim_time: float) -> float:
            return float(rates[int(sim_time / 30.0) % n])

        return fn

    # Forecaster loader

    def load_forecaster(self, device: str = "cpu"):
        """Load MCDropoutForecaster if weights are available, else None."""
        if self.forecaster_path is None:
            return None
        import torch
        from autocloud.forecaster.transformer_model import WorkloadTransformer
        from autocloud.forecaster.mc_dropout import MCDropoutForecaster

        model = WorkloadTransformer(
            input_dim=4, d_model=64, n_heads=4, d_ff=256, n_layers=2,
            dropout=0.2, seq_len=20, n_horizons=4,
        )
        model.load_state_dict(torch.load(self.forecaster_path, map_location=device))
        model.to(device)
        return MCDropoutForecaster(model, k_samples=30, device=device)
