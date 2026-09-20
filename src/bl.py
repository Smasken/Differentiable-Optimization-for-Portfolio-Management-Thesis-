"""Differentiable Black–Litterman mean update with one view per asset (P = I)."""
import math

import numpy as np
import pandas as pd
import torch
from torch import nn


class BLUpdate(nn.Module):
    def __init__(self, variance, tau=0.05):
        super().__init__()
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be finite and positive")
        variance = torch.as_tensor(variance, dtype=torch.float64)
        if variance.ndim != 1 or not torch.isfinite(variance).all() or (variance <= 0).any():
            raise ValueError("variance must be a positive finite vector")
        self.tau = tau
        self.register_buffer("variance", variance.clone().detach())

    def forward(self, views, prior, covariance, variance=None):
        """Optional variance allows a future learned head to retain its gradients."""
        omega = self.variance if variance is None else variance
        if views.ndim != 1 or prior.shape != views.shape or omega.shape != views.shape:
            raise ValueError("views, prior and variance must be matching vectors")
        if covariance.shape != (len(views), len(views)):
            raise ValueError("covariance shape mismatch")
        if not all(torch.isfinite(x).all() for x in (views, prior, covariance, omega)):
            raise ValueError("BL inputs must be finite")
        if (omega <= 0).any():
            raise ValueError("view variances must be strictly positive")
        # Equivalent to the precision-form posterior; no explicit inverses needed.
        prior_covariance = self.tau * covariance
        correction = torch.linalg.solve(prior_covariance + torch.diag(omega), views - prior)
        return prior + prior_covariance @ correction


def equilibrium_prior(covariance, market_weights, delta):
    """Reverse-optimization prior in risk-free-excess units: delta * Sigma * w."""
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive")
    if market_weights.ndim != 1 or covariance.shape != (len(market_weights), len(market_weights)):
        raise ValueError("market weights and covariance shapes do not match")
    if not torch.isfinite(market_weights).all() or (market_weights < 0).any():
        raise ValueError("market weights must be finite and nonnegative")
    if not torch.isclose(market_weights.sum(), market_weights.new_tensor(1.0)):
        raise ValueError("market weights must sum to one")
    return delta * (covariance @ market_weights)


def load_market_weights(path, assets):
    """CSV: date (availability date), then one market-weight column per asset."""
    weights = pd.read_csv(path, index_col="date")
    weights.index = pd.to_datetime(weights.index)
    if weights.empty or weights.index.hasnans or not weights.index.is_unique:
        raise ValueError("Market weights require unique, nonmissing availability dates")
    if not weights.index.is_monotonic_increasing:
        raise ValueError("Market-weight dates must be increasing")
    if set(weights.columns) != set(assets) or len(weights.columns) != len(assets):
        raise ValueError("Market-weight columns must match panel assets exactly")
    weights = weights.loc[:, list(assets)].astype(float)
    if not np.isfinite(weights.to_numpy()).all() or (weights.to_numpy() < 0).any():
        raise ValueError("Market weights must be finite and nonnegative")
    if not np.allclose(weights.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("Each row of market weights must sum to one")
    return weights


def historical_bl_inputs(panel, market_weights, factors, delta=2.5, lookback=252):
    """Reverse CAPM priors and full excess-return covariance, using known data.

    The one-year window and delta default are configurable pilot settings.
    Subtract the historical market-premium estimate to match market-relative targets.
    Weight dates must be availability dates, not dates of later-revised observations.
    """
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    aligned = factors.reindex(pd.to_datetime(panel.label_end))
    market = aligned["Mkt-RF"].to_numpy()
    risk_free = aligned["RF"].to_numpy()
    panel_market = panel.raw_returns - panel.targets
    if not np.isfinite(market).all() or not np.isfinite(risk_free).all():
        raise ValueError("Daily factors do not cover all panel target dates")
    if not np.allclose(panel_market, (market + risk_free)[:, None], atol=1e-8, rtol=1e-6):
        raise ValueError("Panel targets must be relative to the supplied market")

    priors, covariances = {}, {}
    for t, signal_date in enumerate(panel.dates):
        observed = np.flatnonzero(panel.label_end <= signal_date)
        weight_row = market_weights.index.searchsorted(pd.Timestamp(signal_date), side="right") - 1
        if len(observed) < lookback or weight_row < 0:
            continue
        ids = observed[-lookback:]
        excess = panel.raw_returns[ids] - risk_free[ids, None]
        covariance = torch.tensor(np.cov(excess, rowvar=False, ddof=1), dtype=torch.float64)
        weights = torch.tensor(market_weights.iloc[weight_row].to_numpy(), dtype=torch.float64)
        prior_excess = equilibrium_prior(covariance, weights, delta)
        priors[t] = prior_excess - market[ids].mean()
        covariances[t] = covariance
    if not priors:
        raise ValueError("No dates have both market weights and sufficient return history")
    return priors, covariances
