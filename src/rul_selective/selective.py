"""Programmatic abstain-and-escalate routing based on calibrated interval width."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RoutingResult:
    """Predictions and decisions suitable for a maintenance-system integration."""

    lower: np.ndarray
    median: np.ndarray
    upper: np.ndarray
    interval_width: np.ndarray
    accepted: np.ndarray
    action: np.ndarray


@dataclass(frozen=True)
class SelectiveRouter:
    """Accept narrow calibrated intervals and escalate uncertain engine assets."""

    max_interval_width: float

    def __post_init__(self) -> None:
        if self.max_interval_width <= 0:
            raise ValueError("max_interval_width must be positive.")

    def route(self, calibrated_quantiles: np.ndarray) -> RoutingResult:
        values = np.asarray(calibrated_quantiles, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("calibrated_quantiles must have shape [n_samples, 3].")
        width = values[:, 2] - values[:, 0]
        accepted = width <= self.max_interval_width
        action = np.where(
            accepted, "AUTOMATED_MAINTENANCE_SCHEDULE", "ESCALATE_HUMAN_REVIEW"
        )
        return RoutingResult(
            lower=values[:, 0],
            median=values[:, 1],
            upper=values[:, 2],
            interval_width=width,
            accepted=accepted,
            action=action,
        )
