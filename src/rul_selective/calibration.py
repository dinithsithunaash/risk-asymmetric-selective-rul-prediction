"""Asymmetric split conformalized quantile regression (CQR)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """One-sided split-conformal order statistic with finite-sample correction."""
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("scores must be a non-empty one-dimensional array.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one.")
    rank = min(len(scores), int(np.ceil((len(scores) + 1) * (1 - alpha))))
    return float(np.partition(scores, rank - 1)[rank - 1])


@dataclass
class AsymmetricCQR:
    """CQR corrections with a tighter lower-tail miscoverage budget for safety."""

    alpha: float = 0.10
    unsafe_miscoverage_fraction: float = 0.20
    lower_correction: float | None = None
    upper_correction: float | None = None

    @property
    def lower_alpha(self) -> float:
        return self.alpha * self.unsafe_miscoverage_fraction

    @property
    def upper_alpha(self) -> float:
        return self.alpha - self.lower_alpha

    def fit(self, predicted_quantiles: np.ndarray, targets: np.ndarray) -> "AsymmetricCQR":
        predictions = np.asarray(predicted_quantiles, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64).reshape(-1)
        if predictions.ndim != 2 or predictions.shape[1] != 3:
            raise ValueError("predicted_quantiles must have shape [n_samples, 3].")
        if len(predictions) != len(targets):
            raise ValueError("predictions and targets must have matching sample counts.")
        lower_scores = predictions[:, 0] - targets
        upper_scores = targets - predictions[:, 2]
        # Never contract the learned interval: this keeps calibration conservative.
        self.lower_correction = max(0.0, _finite_sample_quantile(lower_scores, self.lower_alpha))
        self.upper_correction = max(0.0, _finite_sample_quantile(upper_scores, self.upper_alpha))
        return self

    def predict(self, predicted_quantiles: np.ndarray) -> np.ndarray:
        """Return [lower, median, upper] calibrated intervals."""
        if self.lower_correction is None or self.upper_correction is None:
            raise RuntimeError("Call fit before predict.")
        predictions = np.asarray(predicted_quantiles, dtype=np.float64)
        if predictions.ndim != 2 or predictions.shape[1] != 3:
            raise ValueError("predicted_quantiles must have shape [n_samples, 3].")
        calibrated = predictions.copy()
        calibrated[:, 0] -= self.lower_correction
        calibrated[:, 2] += self.upper_correction
        return calibrated.astype(np.float32)
