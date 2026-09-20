"""Long-only Markowitz allocation and a historical-mean baseline."""
import argparse
import math
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer
from data import load_data, covariance_factors, experiment_splits, summarize, save_results


class PortfolioLayer(torch.nn.Module):
    def __init__(self, assets, risk_aversion=1.0, cap=0.1, risk="variance", cost=0.0):
        super().__init__()
        if risk not in {"variance", "sd"}:
            raise ValueError("risk must be variance or sd")
        if not 0 < cap <= 1 or assets*cap < 1:
            raise ValueError("cap must be in (0,1] and assets*cap >= 1")
        if any(not math.isfinite(v) or v < 0 for v in (risk_aversion, cost)):
            raise ValueError("risk aversion and cost must be finite and nonnegative")
        self.risk_aversion, self.risk, self.cost = risk_aversion, risk, cost
        w = cp.Variable(assets)
        mu = cp.Parameter(assets)
        factor = cp.Parameter((assets, assets))
        previous = cp.Parameter(assets)
        # Factor.T @ factor = covariance makes the problem DPP-compliant.
        risk_term = cp.sum_squares(factor @ w) if risk == "variance" else cp.norm(factor @ w)
        objective = -mu @ w + risk_aversion*risk_term + cost/2*cp.norm1(w-previous)
        problem = cp.Problem(cp.Minimize(objective), [w >= 0, cp.sum(w) == 1, w <= cap])
        self.solver = CvxpyLayer(problem, parameters=[mu, factor, previous], variables=[w])

    def forward(self, mu, factor, previous):
        weights, = self.solver(mu, factor, previous, solver_args={
            "solve_method": "Clarabel", "tol_gap_abs": 1e-9,
            "tol_gap_rel": 1e-9, "tol_feas": 1e-9})
        return weights

    def loss(self, weights, realized, factor, previous):
        exposure = factor @ weights
        risk = exposure.square().sum() if self.risk == "variance" else exposure.norm()
        # Wang eq. (21): substitute realized returns in the upper loss.
        return -weights @ realized + self.risk_aversion*risk + self.cost/2*(weights-previous).abs().sum()


def run(panel, *, risk_aversion=1.0, first_test="2014-01-01", smoke=False):
    factors = covariance_factors(panel)
    layer = PortfolioLayer(len(panel.assets), risk_aversion)
    previous = torch.full((len(panel.assets),), 1/len(panel.assets), dtype=torch.float64)
    records = []
    for _, _, _, test in experiment_splits(panel, first_test, factors, smoke):
        with torch.no_grad():
            for t in test:
                # Match the one-year risk lookback; these returns are already observable.
                mu = torch.tensor(panel.risk_returns[t-251:t+1].mean(axis=0), dtype=torch.float64)
                weights = layer(mu, factors[t], previous)
                records.append((t, weights.numpy(), 0.0))
                previous = weights
    return summarize(panel, records).assign(model="Mean-variance", risk="variance",
                                          risk_aversion=risk_aversion, cost=0.0, smoke=smoke)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", help="Shared NPZ panel; omit only with --smoke")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--risk-aversion", type=float, nargs="+", default=[1.0])
    parser.add_argument("--output", default="outputs/mean_variance.csv")
    parser.add_argument("--first-test", default="2014-01-01")
    args = parser.parse_args()
    panel = load_data(args.panel, args.smoke)
    results = [run(panel, risk_aversion=eta, first_test=args.first_test, smoke=args.smoke)
               for eta in args.risk_aversion]
    save_results(results, args.output, args.panel or "synthetic-seed-7")
