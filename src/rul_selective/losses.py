"""Risk-asymmetric quantile-regression objective."""

from __future__ import annotations

import torch


def risk_asymmetric_pinball_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    overestimate_penalty: float = 3.0,
) -> torch.Tensor:
    """Pinball loss that costs optimistic (predicted RUL > actual) errors more heavily."""
    if predictions.shape[-1] != len(quantiles):
        raise ValueError("Number of outputs must equal number of quantiles.")
    if overestimate_penalty < 1:
        raise ValueError("overestimate_penalty must be at least one.")
    errors = targets.unsqueeze(-1) - predictions
    quantile_tensor = predictions.new_tensor(quantiles)
    pinball = torch.maximum(quantile_tensor * errors, (quantile_tensor - 1) * errors)
    safety_weight = torch.where(
        predictions > targets.unsqueeze(-1), overestimate_penalty, 1.0
    )
    return (pinball * safety_weight).mean()
