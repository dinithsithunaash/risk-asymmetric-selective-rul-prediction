"""Shared batched inference helpers."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@torch.no_grad()
def predict_quantiles(
    model: torch.nn.Module, dataset: Dataset, device: torch.device, batch_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Run a quantile model and return predictions and target RUL values."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for windows, batch_targets, _ in loader:
        predictions.append(model(windows.to(device)).cpu())
        targets.append(batch_targets.cpu())
    return torch.cat(predictions).numpy(), torch.cat(targets).numpy()
