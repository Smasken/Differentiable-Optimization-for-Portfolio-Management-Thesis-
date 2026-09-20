# Portfolio optimization thesis

Initial Wang et al. reproduction; synthetic examples run, but real-data validation
is ongoing. A fixed-uncertainty Black–Litterman layer is available; learned uncertainty is pending.

## Run

From the repository root:

```sh
uv sync
uv run python src/mean_variance.py --smoke --risk-aversion 0.1 1 10 --output outputs/comparison/mean_variance.csv
uv run python src/wang_e2e.py --smoke --risk-aversion 0.1 1 10 --output outputs/comparison/wang_e2e.csv
uv run python src/visualize.py outputs/comparison --output outputs/frontier.png
uv run pytest -q
```

The plotter accepts a results folder or one CSV. Replace `--smoke` with
`--panel data/panel.npz` for real data; set `--first-test YYYY-MM-DD` to a quarter
start. E2E also accepts `--epochs`, `--cost`, and `--risk variance` (default: `sd`).
Each risk-aversion value trains separately. Smoke runs are brief synthetic checks.

## French industry data

Place the unzipped daily industry and Fama–French factors CSVs in `data/`, then run:

```sh
uv run python src/prepare_french.py --factors data/F-F_Research_Data_Factors_daily.csv
uv run python src/mean_variance.py --panel data/panel.npz --first-test 2015-01-01
uv run python src/wang_e2e.py --panel data/panel.npz --first-test 2015-01-01
```

Preparation defaults to 2010–2024 (`--start`, `--end`, and `--output` are configurable).
It uses 30 value-weighted industries and six features: trailing 1/5/21/63-day
compounded returns and 21/63-day daily-return volatility. Earlier source history
initializes features. Targets are single-day returns two trading dates after the
signal, relative to the US market (`Mkt-RF + RF`); historical risk uses same-date
relative returns. The end date limits target dates. These are pilot features,
not Wang's original stock characteristics. The model runs above are full experiments.

## Methodology

- **Mean–variance:** historical average returns feed a constrained Markowitz optimizer.
- **Wang E2E:** a shared stock-level NN (32/16/8 hidden units) predicts returns,
  starts with MSE training, then learns through the differentiable CVXPYlayers/Clarabel
  optimizer by minimizing realized negative utility. Default risk is standard deviation,
  following Wang's empirical experiment; `--risk variance` uses mean–variance.
- **Shared data:** `src/data.py` supplies 252-day diagonal risk estimates and Wang's
  rolling 39/9/3-month training/validation/test splits, excluding unavailable labels.
  Both portfolios are fully invested, long-only, with a 10% per-asset cap.
- **Features:** point-in-time stock characteristics shaped `[dates, assets, features]`.
  Smoke data uses six random features; French pilot features are described above.
  The shared NPZ also needs `dates`, `label_end`, `assets`, `targets`, `risk_returns`,
  and `raw_returns` (see `Panel` in `src/data.py`). Returns are decimal; forward
  targets must not enter same-date historical risk estimates.

Plots show realized benchmark-relative return versus raw-return volatility across
risk preferences, inspired by Wang Figure 2. They are not ex-ante efficient frontiers.

## Historical-error uncertainty

Costa–Iyengar's [reference implementation](https://github.com/Iyengar-Lab/E2E-DRO)
recomputes past predictions with the current model and forms observed return minus
prediction. `src/uncertainty.py` implements this component, preserving gradients.
Only targets observed by the signal close enter the window. The nominal covariance
is centered and divided by the number of observations (not `T - 1`).

To save estimates alongside a Wang run:

```sh
uv run python src/wang_e2e.py --panel data/panel.npz --first-test 2015-01-01 --uncertainty-output outputs/errors.npz --error-window 504
```

This is a full training run. A quick synthetic check uses `--smoke --epochs 1`
instead of `--panel` and `--first-test`. The NPZ contains test signal dates,
label-end dates, asset names, predictions, mean errors, full residual covariance,
and its per-asset variance diagonal. Our default 504 daily observations approximates
their 104-week window; it is a frequency adaptation, not an exact replication.

These estimates do not yet change portfolio allocations. They are the nominal
historical-error component, not the full DRO model, a learned variance head, or a
BL update. Residuals use the current fitted model, not archived out-of-sample
forecasts, so their dispersion can understate forecast error. It is not automatically
the uncertainty of the expected return. The helper also returns the raw scenarios
for a future robust layer; callers must ensure the model itself is fitted without
future information. Historical predictions use evaluation mode to avoid dropout
noise and BatchNorm updates, while retaining gradients when enabled.

## Fixed-uncertainty BL layer

`src/bl.py` implements `prior + C @ solve(C + diag(variance), forecast - prior)`
with `C = tau * covariance` and one view per asset. It runs between prediction
and allocation during E2E training, validation, and testing. MSE pretraining is
unchanged. `--bl off` (default) runs the original baseline.

```sh
uv run python src/wang_e2e.py --smoke --epochs 1 --risk variance --bl constant --bl-variance 0.0001 --output outputs/bl_smoke.csv
```

For real data, first supply a market-weight CSV with a `date` column and one column
per panel asset, in decimal weights summing to one. Dates mean **availability dates**;
the latest available row is carried forward, never backward. Assets must match
exactly (for the current panel: Food, Beer, ..., Other). Supply history covering
training as well as testing. There is no equal-weight fallback.

```sh
uv run python src/wang_e2e.py --panel data/panel.npz --first-test 2015-01-01 --risk variance --bl historical --bl-market-weights data/market_weights.csv --bl-factors data/F-F_Research_Data_Factors_daily.csv --bl-tau 0.05 --bl-delta 2.5 --output outputs/wang_bl.csv
```

The prior is `delta * Sigma * market_weights`, with full sample covariance of
252 observable risk-free-excess daily returns. We subtract the trailing mean
market premium to express this prior in the network's market-relative units;
that benchmark estimate is treated as fixed, without extra uncertainty propagation.
The BL covariance is this full excess-return covariance; the portfolio objective
retains its existing diagonal market-relative risk estimate for baseline comparison.
The supplied market weights define the equilibrium universe and should represent
that universe, rather than an arbitrary reference portfolio.

`--bl historical` freezes per-asset residual variances after each quarter's MSE
pretraining, using training history only (`--error-window`, default 504; floor 1e-8).
`--bl constant` uses `--bl-variance` for every asset: 0.0001 means 1% daily standard
deviation. These values and tau/delta are pilot settings, not tuned results.
Synthetic smoke runs use a zero prior and synthetic covariance, not actual CAPM data.
The layer accepts a differentiable variance tensor for the later learned-head phase;
no two-output head is trained yet. The optimizer is CVXPYlayers/Clarabel, and
`--risk variance` selects the QP. BL results default to `outputs/wang_bl.csv`.
