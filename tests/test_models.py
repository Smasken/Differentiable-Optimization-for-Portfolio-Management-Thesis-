import numpy as np
import torch
from data import synthetic_panel, rolling_splits, covariance_factors
from mean_variance import PortfolioLayer
from wang_e2e import run


def test_paper_example_and_gradient():
    x = torch.tensor([[.3,.1],[.35,.2],[.1,.9],[.2,1.]], dtype=torch.float64)
    realized = torch.tensor([.32,.39,.92,1.04], dtype=torch.float64)
    covariance = torch.tensor([[.0064,.00288,0,0],[.00288,.0144,0,.0084],
                               [0,0,.36,.294],[0,.0084,.294,.49]], dtype=torch.float64)
    factor = torch.linalg.cholesky(covariance).T
    previous = torch.full((4,), .25, dtype=torch.float64)
    # Wang Section 3.1, Table 1: independently reported utilities.
    for eta, expected in [(10, [.3197,.3201,.3190]), (1,[.6284,.6253,.6290])]:
        layer = PortfolioLayer(4, eta, cap=1.0)
        utilities = []
        for tilt in ([1,1],[1.05,.95],[.95,1.05]):
            mu = x @ (torch.tensor([.652,.927],dtype=torch.float64)*torch.tensor(tilt))
            weights = layer(mu, factor, previous)
            utilities.append(-layer.loss(weights, realized, factor, previous).item())
        np.testing.assert_allclose(utilities, expected, atol=1e-4)
    for risk in ("variance", "sd"):
        layer = PortfolioLayer(3, cap=.8, risk=risk)
        mu = torch.tensor([.02,.03,.015], dtype=torch.float64, requires_grad=True)
        factor = torch.eye(3, dtype=torch.float64)*.3
        previous = torch.ones(3, dtype=torch.float64)/3
        assert torch.autograd.gradcheck(lambda m: layer(m, factor, previous), (mu,),
                                       eps=1e-4, atol=2e-3, rtol=2e-2)
        weights = layer(mu, factor, previous)
        assert weights.min() >= -1e-7 and weights.max() <= .8+1e-7
        assert abs(weights.sum().item()-1) < 1e-7


def test_no_future_information():
    panel = synthetic_panel()
    _, train, val, _ = next(rolling_splits(panel, '2014-01-01'))
    assert (panel.label_end[train] < np.datetime64('2013-04-01')).all()
    assert (panel.label_end[val] < np.datetime64('2014-01-01')).all()
    before = covariance_factors(panel)[300]
    panel.risk_returns[301:] = 100
    assert torch.equal(before, covariance_factors(panel)[300])


def test_e2e_runs(tmp_path):
    path = tmp_path / "errors.npz"
    result = run(synthetic_panel(), epochs=1, smoke=True,
                 uncertainty_output=path, error_window=20)
    with np.load(path, allow_pickle=False) as saved:
        assert saved["covariance"].shape == (8, 20, 20)
        assert saved["variance"].shape == (8, 20)
        np.testing.assert_allclose(saved["variance"],
                                   np.diagonal(saved["covariance"], axis1=1, axis2=2))
        assert np.isfinite(saved["covariance"]).all()
    assert result.observations.iloc[0] == 8
    assert np.isfinite(result.select_dtypes(include="number").to_numpy()).all()


def test_historical_errors_timing_covariance_and_gradient():
    import pytest
    from uncertainty import historical_errors

    panel = synthetic_panel(periods=40, assets=3)
    predictions = torch.tensor(panel.features[:, :, 0], dtype=torch.float64)
    estimate = historical_errors(panel, predictions, t=15, lookback=5)
    # The target at index 13 becomes observable at signal 15; index 14 does not.
    np.testing.assert_array_equal(estimate.indices, np.arange(9, 14))
    errors = panel.targets[9:14] - predictions[9:14].numpy()
    np.testing.assert_allclose(estimate.covariance, np.cov(errors, rowvar=False, ddof=0))
    weights = torch.tensor([.2, .3, .5], dtype=torch.float64)
    projected = estimate.residuals @ weights
    torch.testing.assert_close(weights @ estimate.covariance @ weights,
                               projected.var(unbiased=False))

    panel.targets[14:] = 999
    predictions[14:] = -999
    after = historical_errors(panel, predictions, t=15, lookback=5)
    torch.testing.assert_close(after.covariance, estimate.covariance)
    with pytest.raises(ValueError, match="observed errors"):
        historical_errors(panel, predictions, t=3, lookback=5)

    scale = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
    def covariance_for_scale(value):
        return historical_errors(panel, predictions * value, 15, 5).covariance
    assert torch.autograd.gradcheck(covariance_for_scale, (scale,))
    assert covariance_for_scale(scale).sum().requires_grad


def test_error_predictions_preserve_model_state_and_gradients():
    from uncertainty import predict_history
    from wang_e2e import ReturnPredictor

    panel = synthetic_panel(periods=10, assets=12)
    model = ReturnPredictor(6).double().train()
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    predictions = predict_history(model, panel, 5)
    repeated = predict_history(model, panel, 5)
    torch.testing.assert_close(predictions, repeated)
    assert model.training
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, buffers[name])
    predictions.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
