"""Plot saved model results as a realized return–volatility frontier (Wang Figure 2)."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def plot_results(path, output=None):
    path = Path(path)
    files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"No CSV results found in {path}")
    frames = []
    required = {"model", "risk", "risk_aversion", "cost", "smoke", "source",
                "test_start", "test_end", "observations",
                "annualized_excess_return", "annualized_volatility"}
    for file in files:
        frame = pd.read_csv(file)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{file}: missing columns {sorted(missing)}; rerun the model with --output")
        if frame.empty or frame[list(required)].isna().any().any():
            raise ValueError(f"{file}: empty or incomplete results")
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    sample = ["source", "test_start", "test_end", "observations", "smoke"]
    if len(data[sample].drop_duplicates()) != 1:
        raise ValueError("Choose results from the same data source, test period and smoke/full setting")
    numeric = data[["risk_aversion", "cost", "annualized_excess_return", "annualized_volatility"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Plot values must be finite")
    groups = ["model", "risk", "cost"]
    if data.duplicated(groups + ["risk_aversion"]).any():
        raise ValueError("Duplicate model/risk/cost/eta results; select one run per setting")
    if (data[["risk_aversion", "cost", "annualized_volatility"]] < 0).any().any():
        raise ValueError("Risk aversion, cost and volatility must be nonnegative")

    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    model_keys = list(data[["model", "risk"]].drop_duplicates().itertuples(index=False, name=None))
    colors = {key: f"C{i % 10}" for i, key in enumerate(model_keys)}
    for (model, risk, cost), rows in data.groupby(groups, sort=False):
        # Figure 2 connects realized portfolios across risk preferences, not an ex-ante envelope.
        rows = rows.sort_values("risk_aversion", ascending=False)
        label = f"{model} ({risk}; {'gross' if cost == 0 else f'net, {cost*10000:g} bps'})"
        ax.plot(rows.annualized_volatility, rows.annualized_excess_return,
                marker="o", linestyle="--" if cost == 0 else "-",
                color=colors[(model, risk)], label=label)
        for index, row in enumerate(rows.itertuples()):
            offset = [(7, 9), (7, -14), (-7, 9)][index % 3]
            ax.annotate(f"η={row.risk_aversion:g}",
                        (row.annualized_volatility, row.annualized_excess_return),
                        xytext=offset, ha="right" if offset[0] < 0 else "left",
                        textcoords="offset points", fontsize=8)
    smoke = str(data.smoke.iloc[0]).lower() == "true"
    title = "Realized portfolio frontier"
    if smoke:
        title += " — synthetic smoke test"
    ax.set(title=title, xlabel="Annualized standard deviation (raw portfolio returns)",
           ylabel="Annualized benchmark-relative return")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.2)
    ax.margins(.15)
    ax.legend(fontsize=9)
    output = Path(output) if output else (path / "frontier.png" if path.is_dir() else path.with_suffix(".png"))
    if output.suffix.lower() not in {".png", ".pdf", ".svg"}:
        raise ValueError("Output must be .png, .pdf or .svg")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="One model CSV, or a directory of model CSVs to compare")
    parser.add_argument("--output", help="Figure path; defaults beside the input")
    args = parser.parse_args()
    print(f"Saved {plot_results(args.path, args.output)}")
