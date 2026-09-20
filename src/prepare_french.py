"""Prepare daily French industry returns for both portfolio models."""
import argparse
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from data import Panel


FEATURE_NAMES = ["return_1d", "return_5d", "return_21d", "return_63d",
                 "volatility_21d", "volatility_63d"]


def read_daily_table(path, header_prefix):
    """Read one daily table, excluding French's preamble and later tables."""
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    header = next((i for i, line in enumerate(lines)
                   if line.strip().startswith(header_prefix)), None)
    if header is None:
        raise ValueError(f"Cannot find {header_prefix!r} table in {path}")
    end = header + 1
    while end < len(lines):
        date = lines[end].split(",")[0].strip()
        if len(date) != 8 or not date.isdigit():
            break
        end += 1
    table = pd.read_csv(StringIO("\n".join(lines[header:end])), index_col=0)
    table.index = pd.to_datetime(table.index.astype(str), format="%Y%m%d")
    table.columns = table.columns.str.strip()
    if table.empty or not table.index.is_unique or not table.index.is_monotonic_increasing:
        raise ValueError(f"Expected unique, increasing daily dates in {path}")
    # French reports percentages and uses these values for missing observations.
    return table.astype(float).replace([-99.99, -999.0], np.nan) / 100


def make_panel(returns, factors, start, end):
    if returns.shape[1] != 30:
        raise ValueError("Expected 30 industry columns")
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError("start must be on or before end")
    market = (factors["Mkt-RF"] + factors["RF"]).reindex(returns.index)
    relative_returns = returns.sub(market, axis=0)

    # These six history-based features are a pilot choice, not Wang's characteristics.
    feature_tables = [returns]
    for window in (5, 21, 63):
        compounded = (1 + returns).rolling(window).apply(np.prod, raw=True) - 1
        feature_tables.append(compounded)
    for window in (21, 63):
        feature_tables.append(returns.rolling(window).std(ddof=1))
    features = np.stack([table.to_numpy() for table in feature_tables], axis=-1)

    # Match our Wang pipeline's execution delay: signal t predicts the day t+2.
    # Shift before filtering, so removing rows cannot change the target horizon.
    label_end = pd.Series(returns.index, index=returns.index).shift(-2)
    raw_returns = returns.shift(-2)
    targets = relative_returns.shift(-2)
    selected = ((returns.index >= pd.Timestamp(start))
                & (label_end <= pd.Timestamp(end)))
    positions = np.flatnonzero(selected)
    if len(positions) == 0:
        raise ValueError("No dates in the requested interval")
    if positions[0] < 62:
        raise ValueError("Supply at least 62 trading days before start for feature history")

    # Fail on missing data rather than silently compressing the trading calendar.
    return Panel(
        dates=returns.index.to_numpy()[positions],
        label_end=label_end.to_numpy()[positions],
        features=features[positions],
        targets=targets.to_numpy()[positions],
        risk_returns=relative_returns.to_numpy()[positions],
        raw_returns=raw_returns.to_numpy()[positions],
        assets=returns.columns.to_numpy(dtype=str),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--industries", type=Path,
                        default=Path("data/30_Industry_Portfolios_Daily.csv"))
    parser.add_argument("--factors", type=Path, required=True,
                        help="Unzipped daily Fama-French factors CSV (Mkt-RF and RF)")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2024-12-31",
                        help="Latest allowed target date")
    parser.add_argument("--output", type=Path, default=Path("data/panel.npz"))
    args = parser.parse_args()
    if args.output.suffix != ".npz":
        parser.error("output must end in .npz")

    # The first industry table is value weighted; the second is equal weighted.
    returns = read_daily_table(args.industries, ",Food")
    factors = read_daily_table(args.factors, ",Mkt-RF")
    panel = make_panel(returns, factors, args.start, args.end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(args.output)
    print(f"Saved {args.output}: features {panel.features.shape}")
    print(f"Signal dates: {panel.dates[0]} to {panel.dates[-1]}")
    print(f"Last target date: {panel.label_end[-1]}")
    print(f"Feature order: {', '.join(FEATURE_NAMES)}")


if __name__ == "__main__":
    main()
