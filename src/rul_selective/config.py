"""Typed configuration for the RUL prediction pipeline."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ModelConfig:
    """TCN and window hyperparameters."""

    window_size: int = 30
    channels: Tuple[int, ...] = (64, 64, 64)
    kernel_size: int = 3
    dropout: float = 0.15
    quantiles: Tuple[float, float, float] = (0.1, 0.5, 0.9)

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2.")
        if self.kernel_size < 2:
            raise ValueError("kernel_size must be at least 2.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if tuple(sorted(self.quantiles)) != self.quantiles:
            raise ValueError("quantiles must be in strictly increasing order.")
        if not (0 < self.quantiles[0] < self.quantiles[1] < self.quantiles[2] < 1):
            raise ValueError("quantiles must be strictly between zero and one.")


@dataclass(frozen=True)
class RiskConfig:
    """Risk, calibration, and selective-routing policy."""

    alpha: float = 0.10
    unsafe_miscoverage_fraction: float = 0.20
    overestimate_penalty: float = 3.0
    max_interval_width: float = 35.0

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between zero and one.")
        if not 0 < self.unsafe_miscoverage_fraction < 1:
            raise ValueError("unsafe_miscoverage_fraction must be between zero and one.")
        if self.overestimate_penalty < 1:
            raise ValueError("overestimate_penalty must be at least one.")
        if self.max_interval_width <= 0:
            raise ValueError("max_interval_width must be positive.")

    @property
    def lower_alpha(self) -> float:
        """Small unsafe error budget for cases where true RUL falls below the interval."""
        return self.alpha * self.unsafe_miscoverage_fraction

    @property
    def upper_alpha(self) -> float:
        return self.alpha - self.lower_alpha
