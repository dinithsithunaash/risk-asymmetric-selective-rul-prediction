"""Meaningful CPU/CUDA-independent end-to-end synthetic smoke test."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .calibration import AsymmetricCQR
from .data import EngineFrame, Standardizer, WindowDataset, split_units
from .evaluation import evaluate
from .inference import predict_quantiles
from .losses import risk_asymmetric_pinball_loss
from .model import TCNQuantileRegressor
from .selective import SelectiveRouter


def synthetic_engines(
    engines: int = 12, cycles: int = 42, features: int = 24, seed: int = 17
) -> EngineFrame:
    """Create degradational trajectories with known RUL labels."""
    generator = np.random.default_rng(seed)
    unit_ids = np.repeat(np.arange(1, engines + 1), cycles)
    cycle = np.tile(np.arange(1, cycles + 1), engines)
    progress = np.tile(np.linspace(0.0, 1.0, cycles), engines)
    sensor_noise = generator.normal(0.0, 0.12, size=(len(unit_ids), features))
    trends = np.linspace(0.1, 1.0, features)[None, :] * progress[:, None]
    values = (sensor_noise + trends).astype(np.float32)
    rul = np.tile(np.arange(cycles - 1, -1, -1), engines).astype(np.float32)
    return EngineFrame(unit_ids, cycle, values, rul)


def main() -> None:
    torch.manual_seed(17)
    frame = synthetic_engines()
    train_units, calibration_units = split_units(frame.units, calibration_fraction=0.25, seed=17)
    scaler = Standardizer.fit(frame.subset_units(train_units).features)
    frame = frame.with_features(scaler.transform(frame.features))
    train_set = WindowDataset(frame.subset_units(train_units), window_size=12, stride=3)
    calibration_set = WindowDataset(
        frame.subset_units(calibration_units), window_size=12, terminal_only=True
    )
    model = TCNQuantileRegressor(
        input_features=24, channels=(16, 16), kernel_size=3, dropout=0.0
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for windows, targets, _ in DataLoader(train_set, batch_size=16, shuffle=True):
        optimizer.zero_grad(set_to_none=True)
        output = model(windows)
        if output.shape != (len(windows), 3):
            raise AssertionError(f"Unexpected output shape: {tuple(output.shape)}")
        if not bool(torch.all(output[:, 0] <= output[:, 1])) or not bool(
            torch.all(output[:, 1] <= output[:, 2])
        ):
            raise AssertionError("Model did not produce ordered quantiles.")
        loss = risk_asymmetric_pinball_loss(output, targets, overestimate_penalty=3.0)
        loss.backward()
        optimizer.step()
        break
    predictions, targets = predict_quantiles(model, calibration_set, torch.device("cpu"))
    calibrator = AsymmetricCQR(alpha=0.2, unsafe_miscoverage_fraction=0.25).fit(
        predictions, targets
    )
    calibrated = calibrator.predict(predictions)
    if not bool(np.all(calibrated[:, 0] <= calibrated[:, 1])) or not bool(
        np.all(calibrated[:, 1] <= calibrated[:, 2])
    ):
        raise AssertionError("Calibration broke quantile ordering.")
    router = SelectiveRouter(max_interval_width=float(np.median(calibrated[:, 2] - calibrated[:, 0])))
    routing = router.route(calibrated)
    if routing.accepted.all() or (~routing.accepted).all():
        raise AssertionError("Router did not exercise both accept and abstain paths.")
    metrics = evaluate(calibrated, targets, routing)
    print(
        "SANITY_CHECK_PASS "
        f"train_windows={len(train_set)} calibration_engines={len(calibration_set)} "
        f"loss={loss.item():.5f} accepted={routing.accepted.sum()}/{len(routing.accepted)} "
        f"coverage={metrics['coverage']:.3f}"
    )


if __name__ == "__main__":
    main()
