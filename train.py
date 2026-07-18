"""Train, calibrate, and optionally evaluate risk-asymmetric selective RUL prediction."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rul_selective.calibration import AsymmetricCQR
from rul_selective.config import ModelConfig, RiskConfig
from rul_selective.data import (
    FEATURE_COLUMNS,
    Standardizer,
    WindowDataset,
    label_test_rul,
    label_train_rul,
    load_cmapss,
    split_units,
)
from rul_selective.evaluation import evaluate
from rul_selective.inference import predict_quantiles
from rul_selective.losses import risk_asymmetric_pinball_loss
from rul_selective.model import TCNQuantileRegressor
from rul_selective.selective import SelectiveRouter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/default"))
    parser.add_argument("--test-file", type=Path)
    parser.add_argument("--rul-file", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--unsafe-miscoverage-fraction", type=float, default=0.20)
    parser.add_argument("--overestimate-penalty", type=float, default=3.0)
    parser.add_argument("--max-interval-width", type=float, default=35.0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def configure_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    if bool(args.test_file) != bool(args.rul_file):
        raise ValueError("--test-file and --rul-file must be supplied together.")
    model_config = ModelConfig(window_size=args.window_size)
    risk_config = RiskConfig(
        alpha=args.alpha,
        unsafe_miscoverage_fraction=args.unsafe_miscoverage_fraction,
        overestimate_penalty=args.overestimate_penalty,
        max_interval_width=args.max_interval_width,
    )
    configure_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_train = label_train_rul(load_cmapss(args.train_file))
    train_units, calibration_units = split_units(
        full_train.units, args.calibration_fraction, args.seed
    )
    unscaled_train = full_train.subset_units(train_units)
    standardizer = Standardizer.fit(unscaled_train.features)
    full_train = full_train.with_features(standardizer.transform(full_train.features))
    train_dataset = WindowDataset(
        full_train.subset_units(train_units), model_config.window_size, stride=1
    )
    calibration_dataset = WindowDataset(
        full_train.subset_units(calibration_units), model_config.window_size, terminal_only=True
    )
    model = TCNQuantileRegressor(
        input_features=len(FEATURE_COLUMNS),
        channels=model_config.channels,
        kernel_size=model_config.kernel_size,
        dropout=model_config.dropout,
    ).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for windows, targets, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(windows.to(device))
            loss = risk_asymmetric_pinball_loss(
                prediction,
                targets.to(device),
                model_config.quantiles,
                risk_config.overestimate_penalty,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(loss.item())
        print(f"epoch={epoch:03d} train_loss={np.mean(losses):.5f}")

    calibration_predictions, calibration_targets = predict_quantiles(
        model, calibration_dataset, device, args.batch_size
    )
    calibrator = AsymmetricCQR(
        alpha=risk_config.alpha,
        unsafe_miscoverage_fraction=risk_config.unsafe_miscoverage_fraction,
    ).fit(calibration_predictions, calibration_targets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": (
            model.module.state_dict()
            if isinstance(model, torch.nn.DataParallel)
            else model.state_dict()
        ),
        "model_config": asdict(model_config),
        "risk_config": asdict(risk_config),
        "standardizer_mean": standardizer.mean,
        "standardizer_scale": standardizer.scale,
        "lower_correction": calibrator.lower_correction,
        "upper_correction": calibrator.upper_correction,
    }
    torch.save(checkpoint, args.output_dir / "model.pt")
    metadata = {
        "device": str(device),
        "gpu_count": torch.cuda.device_count() if device.type == "cuda" else 0,
        "train_engines": int(len(train_units)),
        "calibration_engines": int(len(calibration_units)),
        "calibration_lower_correction": calibrator.lower_correction,
        "calibration_upper_correction": calibrator.upper_correction,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))

    if args.test_file:
        test_frame = label_test_rul(load_cmapss(args.test_file), args.rul_file)
        test_frame = test_frame.with_features(standardizer.transform(test_frame.features))
        test_dataset = WindowDataset(test_frame, model_config.window_size, terminal_only=True)
        predictions, targets = predict_quantiles(model, test_dataset, device, args.batch_size)
        calibrated = calibrator.predict(predictions)
        routing = SelectiveRouter(risk_config.max_interval_width).route(calibrated)
        metrics = evaluate(calibrated, targets, routing)
        (args.output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
