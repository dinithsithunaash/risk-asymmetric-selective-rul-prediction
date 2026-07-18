"""Metrics that expose interval validity, unsafe failures, and selective risk."""

from __future__ import annotations

import numpy as np

from .selective import RoutingResult


def evaluate(
    calibrated_quantiles: np.ndarray, targets: np.ndarray, routing: RoutingResult
) -> dict[str, float]:
    """Compute interval coverage and performance conditional on automated acceptance."""
    intervals = np.asarray(calibrated_quantiles)
    actual = np.asarray(targets).reshape(-1)
    if len(intervals) != len(actual):
        raise ValueError("intervals and targets must have equal lengths.")
    lower_miss = actual < intervals[:, 0]
    upper_miss = actual > intervals[:, 2]
    absolute_error = np.abs(intervals[:, 1] - actual)
    accepted = routing.accepted
    return {
        "coverage": float((~(lower_miss | upper_miss)).mean()),
        "unsafe_lower_miss_rate": float(lower_miss.mean()),
        "upper_miss_rate": float(upper_miss.mean()),
        "mean_interval_width": float(routing.interval_width.mean()),
        "abstention_rate": float((~accepted).mean()),
        "median_mae": float(absolute_error.mean()),
        "accepted_median_mae": (
            float(absolute_error[accepted].mean()) if accepted.any() else float("nan")
        ),
    }
