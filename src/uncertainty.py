"""Rolling prediction errors following Costa and Iyengar (2022), Section 2.1.

Source: https://github.com/Iyengar-Lab/E2E-DRO (e2e_net.forward).
Residuals are recomputed with the current predictor, not archived forecasts.
The caller must fit that predictor without using future validation/test outcomes.
"""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ErrorEstimate:
    indices: np.ndarray
    residuals: torch.Tensor       # [history, assets]
    mean_error: torch.Tensor      # [assets]
    covariance: torch.Tensor      # [assets, assets]

    @property
    def variance(self):
        return self.covariance.diagonal()


def historical_errors(panel, predictions, t, lookback):
    """Use the latest `lookback` observable targets at signal date t.

    predictions[j] must be the CURRENT model's prediction for panel.features[j].
    Predictions can cover just a prefix of the panel. Gradients are preserved.
    This is nominal residual dispersion, not DRO or a covariance of mean estimates.
    """
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    if not 0 <= t < len(panel.dates):
        raise ValueError("t is outside the panel")
    if predictions.ndim != 2 or predictions.shape[1] != len(panel.assets):
        raise ValueError("predictions must have shape [dates, assets]")

    # Signals are available at the close: a label ending today is observable.
    eligible = (panel.dates < panel.dates[t]) & (panel.label_end <= panel.dates[t])
    indices = np.flatnonzero(eligible)[-lookback:]
    if len(indices) < lookback:
        raise ValueError(f"Need {lookback} observed errors; found {len(indices)} at {panel.dates[t]}")
    if indices[-1] >= len(predictions):
        raise ValueError("predictions do not cover the required history")

    index = torch.as_tensor(indices, device=predictions.device)
    predicted = predictions[index]
    observed = torch.as_tensor(panel.targets[indices], dtype=predictions.dtype,
                               device=predictions.device)
    residuals = observed - predicted
    if not torch.isfinite(residuals).all():
        raise ValueError("Historical errors contain nonfinite values")
    mean_error = residuals.mean(dim=0)
    centered = residuals - mean_error
    # Costa–Iyengar's nominal empirical distribution assigns probability 1/T.
    covariance = centered.T @ centered / lookback
    return ErrorEstimate(indices, residuals, mean_error, covariance)


def predict_history(model, panel, stop):
    """Predict dates [0, stop) deterministically, retaining parameter gradients.

    Evaluate one cross-section at a time, as required by ReturnPredictor.
    Eval mode prevents dropout noise and repeated BatchNorm buffer updates.
    """
    if not 1 <= stop <= len(panel.dates):
        raise ValueError("stop must specify a nonempty panel prefix")
    parameter = next(model.parameters())
    features = torch.as_tensor(panel.features[:stop], dtype=parameter.dtype,
                               device=parameter.device)
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        return torch.stack([model(features_at_date) for features_at_date in features])
    finally:
        for module, training in modes:
            module.training = training
