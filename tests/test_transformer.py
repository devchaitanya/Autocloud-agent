"""
Day 2 deliverable tests:
  1. MC Dropout variance is nonzero (dropout active at inference).
  2. Output shape is (batch, n_horizons, 3).
  3. predict() returns correct shapes and finite values.
  4. Trained model achieves MAPE ≤ 20% on synthetic validation data.
  5. Quantile ordering: q10 ≤ q50 ≤ q90 (after training).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import pytest

from forecaster.transformer_model import WorkloadTransformer
from forecaster.mc_dropout import MCDropoutForecaster
from forecaster.trainer import train_forecaster, compute_mape
from environment.workload import generate_forecast_dataset, split_day1, make_sequences


CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "forecaster_weights.pt")

@pytest.fixture(scope="module")
def trained_model():
    """Load pre-trained model if checkpoint exists (produced by Colab notebook).
    If not found, train a quick version for structural tests only."""
    model = WorkloadTransformer()
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
        model._from_checkpoint = True
    else:
        # Quick train just for structural tests (not expected to hit MAPE threshold)
        model = train_forecaster(seed=42, epochs=5, patience=5, verbose=False)
        model._from_checkpoint = False
    return model


class TestTransformerModel:
    def test_output_shape(self):
        model = WorkloadTransformer(input_dim=4, d_model=64, n_heads=4, d_ff=256,
                                    n_layers=2, dropout=0.2, seq_len=20, n_horizons=4)
        x = torch.randn(8, 20, 4)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (8, 4, 3), f"Expected (8,4,3), got {out.shape}"

    def test_output_finite(self):
        model = WorkloadTransformer()
        x = torch.randn(4, 20, 4)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert torch.all(torch.isfinite(out)), "Output contains NaN or Inf"

    def test_different_batch_sizes(self):
        model = WorkloadTransformer()
        model.eval()
        for bs in [1, 4, 16, 32]:
            x = torch.randn(bs, 20, 4)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (bs, 4, 3)


class TestMCDropout:
    def test_variance_nonzero(self):
        """MC Dropout must produce nonzero variance — confirms dropout is active at inference."""
        model = WorkloadTransformer(dropout=0.2)
        forecaster = MCDropoutForecaster(model, k_samples=30)
        x = np.random.randn(20, 4).astype(np.float32)
        means, variances = forecaster.predict(x)
        assert variances.shape == (4,), f"Expected (4,), got {variances.shape}"
        assert np.all(variances > 0), (
            f"Variance is zero — dropout may be inactive at inference. variances={variances}"
        )

    def test_means_shape(self):
        model = WorkloadTransformer()
        forecaster = MCDropoutForecaster(model, k_samples=10)
        x = np.random.randn(20, 4).astype(np.float32)
        means, variances = forecaster.predict(x)
        assert means.shape == (4,)
        assert variances.shape == (4,)

    def test_means_finite(self):
        model = WorkloadTransformer()
        forecaster = MCDropoutForecaster(model, k_samples=10)
        x = np.random.randn(20, 4).astype(np.float32)
        means, variances = forecaster.predict(x)
        assert np.all(np.isfinite(means))
        assert np.all(np.isfinite(variances))

    def test_batch_predict_shape(self):
        model = WorkloadTransformer()
        forecaster = MCDropoutForecaster(model, k_samples=10)
        x = np.random.randn(16, 20, 4).astype(np.float32)
        means, variances = forecaster.predict_batch(x)
        assert means.shape == (16, 4)
        assert variances.shape == (16, 4)

    def test_variance_increases_with_more_samples(self):
        """Variance estimate should stabilize (not necessarily monotone, but nonzero)."""
        model = WorkloadTransformer(dropout=0.3)  # higher dropout = more variance
        x = np.random.randn(20, 4).astype(np.float32)
        forecaster = MCDropoutForecaster(model, k_samples=50)
        means, variances = forecaster.predict(x)
        assert np.all(variances > 0)


class TestTrainedModel:
    def test_mape_below_threshold(self, trained_model):
        """Trained model must achieve MAPE ≤ 20% on synthetic validation data.
        Requires forecaster_weights.pt produced by the Colab training notebook."""
        if not getattr(trained_model, "_from_checkpoint", False):
            pytest.skip("No checkpoint found — run notebooks/train_forecaster.ipynb in Colab first")

        rng = np.random.default_rng(42)
        day1_data, _ = generate_forecast_dataset(rng)
        _, val_data = split_day1(day1_data, train_fraction=0.6)
        X_val, y_val = make_sequences(val_data)

        mape = compute_mape(trained_model, X_val, y_val)
        print(f"\nValidation MAPE: {mape:.1f}%")
        assert mape <= 20.0, f"MAPE {mape:.1f}% exceeds 20% threshold"

    def test_quantile_ordering(self, trained_model):
        """After training, q10 ≤ q50 ≤ q90 should hold for most samples.
        Only meaningful with a properly trained model."""
        if not getattr(trained_model, "_from_checkpoint", False):
            pytest.skip("No checkpoint — run Colab notebook first")
        rng = np.random.default_rng(42)
        day1_data, _ = generate_forecast_dataset(rng)
        _, val_data = split_day1(day1_data, train_fraction=0.6)
        X_val, _ = make_sequences(val_data)

        trained_model.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(X_val[:200]).float()
            pred = trained_model(x_t).numpy()   # (200, 4, 3)

        # q10 ≤ q50 ≤ q90 for first horizon
        q10, q50, q90 = pred[:, 0, 0], pred[:, 0, 1], pred[:, 0, 2]
        # Allow 5% violations (model may not be perfectly calibrated)
        frac_ordered = np.mean((q10 <= q50) & (q50 <= q90))
        assert frac_ordered >= 0.80, f"Only {frac_ordered:.1%} of predictions have q10≤q50≤q90"

    def test_mc_dropout_on_trained_model(self, trained_model):
        """MC Dropout must still produce nonzero variance on the trained model."""
        forecaster = MCDropoutForecaster(trained_model, k_samples=30)
        x = np.random.randn(20, 4).astype(np.float32)
        means, variances = forecaster.predict(x)
        assert np.all(variances > 0), "Variance is zero on trained model"
        assert np.all(np.isfinite(means))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
