"""NASA C-MAPSS loading, RUL labeling, scaling, and sliding-window datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

SETTINGS = ("setting_1", "setting_2", "setting_3")
SENSORS = tuple(f"sensor_{index}" for index in range(1, 22))
COLUMNS = ("unit_id", "cycle", *SETTINGS, *SENSORS)
FEATURE_COLUMNS = (*SETTINGS, *SENSORS)


@dataclass
class Standardizer:
    """Feature-wise standardizer fit only on non-calibration training engines."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "Standardizer":
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale < 1e-8] = 1.0
        return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.mean) / self.scale).astype(np.float32)


@dataclass
class EngineFrame:
    """Array representation of C-MAPSS cycles, features, and optionally RUL labels."""

    unit_ids: np.ndarray
    cycles: np.ndarray
    features: np.ndarray
    rul: np.ndarray | None = None

    @property
    def units(self) -> np.ndarray:
        return np.unique(self.unit_ids)

    def subset_units(self, units: Iterable[int]) -> "EngineFrame":
        selected = np.isin(self.unit_ids, np.asarray(list(units)))
        return EngineFrame(
            unit_ids=self.unit_ids[selected],
            cycles=self.cycles[selected],
            features=self.features[selected],
            rul=None if self.rul is None else self.rul[selected],
        )

    def with_features(self, features: np.ndarray) -> "EngineFrame":
        return EngineFrame(self.unit_ids, self.cycles, features, self.rul)


def load_cmapss(path: str | Path) -> EngineFrame:
    """Load a whitespace-delimited C-MAPSS trajectory file with 26 columns."""
    values = np.loadtxt(path, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[1] != len(COLUMNS):
        raise ValueError(
            f"Expected {len(COLUMNS)} columns in {path}, found {values.shape[1]}. "
            "Export N-MAPSS data to the C-MAPSS-compatible layout first."
        )
    return EngineFrame(
        unit_ids=values[:, 0].astype(np.int64),
        cycles=values[:, 1].astype(np.int64),
        features=values[:, 2:].astype(np.float32),
    )


def label_train_rul(frame: EngineFrame, cap_rul: float | None = 125.0) -> EngineFrame:
    """Label run-to-failure train trajectories as final_cycle - current_cycle."""
    max_cycles = {
        unit: frame.cycles[frame.unit_ids == unit].max() for unit in frame.units
    }
    rul = np.asarray(
        [max_cycles[unit] - cycle for unit, cycle in zip(frame.unit_ids, frame.cycles)],
        dtype=np.float32,
    )
    if cap_rul is not None:
        rul = np.minimum(rul, cap_rul)
    return EngineFrame(frame.unit_ids, frame.cycles, frame.features, rul)


def label_test_rul(
    frame: EngineFrame, rul_path: str | Path, cap_rul: float | None = 125.0
) -> EngineFrame:
    """Attach RUL targets to test trajectories from the ordered NASA RUL file."""
    final_ruls = np.loadtxt(rul_path, dtype=np.float32).reshape(-1)
    units = frame.units
    if len(final_ruls) != len(units):
        raise ValueError(
            f"RUL file has {len(final_ruls)} labels for {len(units)} test units."
        )
    target_at_final = dict(zip(units, final_ruls))
    final_cycles = {
        unit: frame.cycles[frame.unit_ids == unit].max() for unit in units
    }
    rul = np.asarray(
        [
            target_at_final[unit] + final_cycles[unit] - cycle
            for unit, cycle in zip(frame.unit_ids, frame.cycles)
        ],
        dtype=np.float32,
    )
    if cap_rul is not None:
        rul = np.minimum(rul, cap_rul)
    return EngineFrame(frame.unit_ids, frame.cycles, frame.features, rul)


def split_units(
    units: Sequence[int], calibration_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return reproducible, unit-disjoint training and calibration IDs."""
    if len(units) < 3:
        raise ValueError("At least three engines are required for train/calibration split.")
    generator = np.random.default_rng(seed)
    shuffled = generator.permutation(np.asarray(units))
    calibration_count = max(1, round(len(shuffled) * calibration_fraction))
    calibration_count = min(calibration_count, len(shuffled) - 1)
    return shuffled[calibration_count:], shuffled[:calibration_count]


class WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Fixed-length causal windows, optionally retaining just each engine's last window."""

    def __init__(
        self,
        frame: EngineFrame,
        window_size: int,
        terminal_only: bool = False,
        stride: int = 1,
    ) -> None:
        if frame.rul is None:
            raise ValueError("RUL labels are required to build a supervised dataset.")
        if stride < 1:
            raise ValueError("stride must be positive.")
        windows: list[np.ndarray] = []
        targets: list[float] = []
        unit_ids: list[int] = []
        for unit in frame.units:
            mask = frame.unit_ids == unit
            features = frame.features[mask]
            targets_for_unit = frame.rul[mask]
            if len(features) < window_size:
                continue
            end_indices = [len(features) - 1] if terminal_only else range(
                window_size - 1, len(features), stride
            )
            for end in end_indices:
                windows.append(features[end - window_size + 1 : end + 1])
                targets.append(float(targets_for_unit[end]))
                unit_ids.append(int(unit))
        if not windows:
            raise ValueError("No windows available; reduce window_size or provide longer runs.")
        self.windows = torch.from_numpy(np.stack(windows).astype(np.float32))
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.unit_ids = torch.tensor(unit_ids, dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.windows[index], self.targets[index], self.unit_ids[index]
