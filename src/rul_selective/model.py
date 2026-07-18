"""Causal temporal convolutional quantile regressor."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


class Chomp1d(nn.Module):
    """Remove right padding so each TCN output is strictly causal."""

    def __init__(self, amount: int) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values[:, :, :-self.amount] if self.amount else values


class TemporalBlock(nn.Module):
    """Two residual dilated causal convolutions."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return functional.relu(self.net(values) + self.residual(values))


class TCNQuantileRegressor(nn.Module):
    """Predict ordered low, median, and high RUL quantiles from [batch, time, feature]."""

    def __init__(
        self,
        input_features: int,
        channels: tuple[int, ...] = (64, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.15,
        output_quantiles: int = 3,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        current_channels = input_features
        for level, out_channels in enumerate(channels):
            blocks.append(
                TemporalBlock(
                    current_channels, out_channels, kernel_size, dilation=2**level, dropout=dropout
                )
            )
            current_channels = out_channels
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(current_channels, output_quantiles)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 3:
            raise ValueError("Expected windows with shape [batch, time, features].")
        encoding = self.backbone(windows.transpose(1, 2))[:, :, -1]
        raw_quantiles = self.head(encoding)
        lower = raw_quantiles[:, :1]
        increments = functional.softplus(raw_quantiles[:, 1:])
        return torch.cat((lower, lower + torch.cumsum(increments, dim=1)), dim=1)
