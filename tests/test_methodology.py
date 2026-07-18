"""Focused tests for the risk-asymmetric selective-prediction mechanics."""

import numpy as np
import torch

from rul_selective.calibration import AsymmetricCQR
from rul_selective.losses import risk_asymmetric_pinball_loss
from rul_selective.model import TCNQuantileRegressor
from rul_selective.selective import SelectiveRouter


def test_tcn_returns_ordered_quantiles() -> None:
    model = TCNQuantileRegressor(24, channels=(8, 8), dropout=0.0)
    output = model(torch.randn(3, 12, 24))
    assert output.shape == (3, 3)
    assert torch.all(output[:, 0] <= output[:, 1])
    assert torch.all(output[:, 1] <= output[:, 2])


def test_optimistic_error_costs_more_than_equally_sized_pessimistic_error() -> None:
    targets = torch.tensor([10.0])
    optimistic = torch.tensor([[12.0, 12.0, 12.0]])
    pessimistic = torch.tensor([[8.0, 8.0, 8.0]])
    assert risk_asymmetric_pinball_loss(optimistic, targets, overestimate_penalty=3.0) > (
        risk_asymmetric_pinball_loss(pessimistic, targets, overestimate_penalty=3.0)
    )


def test_asymmetric_cqr_widens_and_router_abstains() -> None:
    predictions = np.array([[10.0, 15.0, 20.0], [10.0, 15.0, 20.0], [10.0, 15.0, 20.0]])
    targets = np.array([5.0, 16.0, 25.0])
    calibrated = AsymmetricCQR(alpha=0.4, unsafe_miscoverage_fraction=0.25).fit(
        predictions, targets
    ).predict(predictions)
    assert np.all(calibrated[:, 0] <= predictions[:, 0])
    assert np.all(calibrated[:, 2] >= predictions[:, 2])
    routes = SelectiveRouter(max_interval_width=5.0).route(calibrated)
    assert not routes.accepted.any()
    assert set(routes.action) == {"ESCALATE_HUMAN_REVIEW"}
