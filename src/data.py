"""Explicit timing contract for a first, fixed-universe daily panel."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch


@dataclass
class Panel:
    dates: np.ndarray             # signal-availability dates
    label_end: np.ndarray         # dates when realized targets become observable
    features: np.ndarray         # [dates, assets, features], already point-in-time
    targets: np.ndarray          # forward benchmark-relative returns
    risk_returns: np.ndarray     # trailing benchmark-relative returns known on dates
    raw_returns: np.ndarray      # forward raw returns for wealth/drawdown
    assets: np.ndarray

    def __post_init__(self):
        self.dates = np.asarray(self.dates, dtype="datetime64[D]")
        self.label_end = np.asarray(self.label_end, dtype="datetime64[D]")
        t, n, _ = self.features.shape
        if n < 2 or len(set(self.assets)) != n:
            raise ValueError("at least two distinct assets are required")
        if self.dates.shape != (t,) or self.label_end.shape != (t,):
            raise ValueError("date shape mismatch")
        if np.isnat(self.dates).any() or np.isnat(self.label_end).any():
            raise ValueError("missing dates")
        if not np.all(np.diff(self.dates) > np.timedelta64(0, 'D')):
            raise ValueError("signal dates must be strictly increasing")
        if not np.all(self.label_end > self.dates):
            raise ValueError("labels must become observable after signal dates")
        for a in (self.targets, self.risk_returns, self.raw_returns):
            if a.shape != (t, n):
                raise ValueError("return shape mismatch")
        if not all(np.isfinite(a).all() for a in
                   (self.features, self.targets, self.risk_returns, self.raw_returns)):
            raise ValueError("missing/nonfinite data: supply an explicit preprocessing policy")
        if (self.raw_returns <= -1).any():
            raise ValueError("raw simple returns must exceed -1")

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(**{key: data[key] for key in cls.__dataclass_fields__})

    def save(self, path):
        np.savez_compressed(path, **vars(self))


def covariance_factors(panel, lookback=252, floor=1e-8):
    """Wang Section 4.4: one-year diagonal risk, using only known returns."""
    if lookback < 2 or floor <= 0:
        raise ValueError("lookback >= 2 and positive variance floor required")
    result = {}
    for t in range(lookback - 1, len(panel.dates)):
        variance = panel.risk_returns[t-lookback+1:t+1].var(axis=0, ddof=1)
        result[t] = torch.diag(torch.tensor(np.maximum(variance, floor)**0.5, dtype=torch.float64))
    return result


def rolling_splits(panel, first_test, last_test=None):
    """Wang et al., Section 4.1: 39/9/3 months; purge unavailable labels."""
    start = pd.Timestamp(first_test)
    if start.day != 1 or start.month not in (1, 4, 7, 10):
        raise ValueError("first_test must be a calendar quarter start")
    end = pd.Timestamp(last_test) if last_test else pd.Timestamp(panel.dates[-1])
    while start <= end:
        train_start = np.datetime64(start - pd.DateOffset(months=48), 'D')
        val_start = np.datetime64(start - pd.DateOffset(months=9), 'D')
        test_start = np.datetime64(start, 'D')
        test_end = np.datetime64(start + pd.DateOffset(months=3), 'D')
        d, label = panel.dates, panel.label_end
        train = np.flatnonzero((d >= train_start) & (d < val_start) & (label < val_start))
        val = np.flatnonzero((d >= val_start) & (d < test_start) & (label < test_start))
        test = np.flatnonzero((d >= test_start) & (d < test_end))
        if all(len(x) for x in (train, val, test)):
            yield str(start.date()), train, val, test
        start += pd.DateOffset(months=3)


def synthetic_panel(seed=7, periods=1400, assets=20, features=6):
    """Synthetic diagnostics only: targets end two business dates after the signal."""
    rng = np.random.default_rng(seed)
    calendar = pd.bdate_range('2010-01-01', periods=periods + 2).values.astype('datetime64[D]')
    x = rng.normal(size=(periods, assets, features))
    target = 0.003 * x[:, :, 0] + rng.normal(0, 0.015, (periods, assets))
    market = rng.normal(0.0002, 0.008, (periods, 1))
    # At date t, the label generated at t-2 has just become known.
    known = np.concatenate([rng.normal(0, .015, (2, assets)), target[:-2]])
    return Panel(calendar[:-2], calendar[2:], x, target, known,
                 target + market, np.array([f'SYN{i:03}' for i in range(assets)]))


def load_data(path=None, smoke=False):
    if smoke and path is None:
        return synthetic_panel()
    if path is not None and not smoke:
        return Panel.load(path)
    raise ValueError("choose --panel PATH or --smoke")


def experiment_splits(panel, first_test, factors, smoke=False):
    found = False
    for quarter, train, val, test in rolling_splits(panel, first_test):
        train, val, test = [np.array([t for t in ids if t in factors]) for ids in (train, val, test)]
        if smoke:
            train, val, test = train[-16:], val[-8:], test[:8]
        if not all(len(ids) for ids in (train, val, test)):
            raise ValueError("not enough history for the requested split")
        found = True
        yield quarter, train, val, test
        if smoke:
            break
    if not found:
        raise ValueError("no usable splits; check first-test and panel dates")


def summarize(panel, records):
    if len(records) < 2:
        raise ValueError("need at least two test observations")
    raw = np.array([weights @ panel.raw_returns[t] - cost for t, weights, cost in records])
    excess = np.array([weights @ panel.targets[t] - cost for t, weights, cost in records])
    if not np.isfinite(raw).all() or np.any(raw <= -1):
        raise ValueError("invalid portfolio returns")
    wealth = np.r_[1.0, np.cumprod(1+raw)]
    return pd.DataFrame([{
        "observations": len(raw),
        "annualized_return": raw.mean()*252,
        "annualized_excess_return": excess.mean()*252,
        "annualized_volatility": raw.std(ddof=1)*np.sqrt(252),
        "test_start": str(panel.dates[records[0][0]]),
        "test_end": str(panel.label_end[records[-1][0]]),
        "information_ratio": excess.mean()/excess.std(ddof=1)*np.sqrt(252) if excess.std(ddof=1)>0 else np.nan,
        "max_drawdown": (1-wealth/np.maximum.accumulate(wealth)).max(),
    }])


def save_results(results, path, source):
    result = pd.concat(results, ignore_index=True).assign(source=source)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    print(result.to_string(index=False))
    print(f"Saved {path}")
