"""Plot a saved portfolio CSV against buy-and-hold benchmarks.

Run ``python plot_results.py`` to chart the original experiment without rerunning
Monte Carlo simulations. Both old indexed CSVs and new backtest outputs work.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT = Path(__file__).resolve().parent / "results" / "original_backtest.csv"
COLORS = {
    "Strategy": "#b64530",
    "Hold S&P series": "#346b9a",
    "Hold Nasdaq series": "#16806a",
    "50/50 buy and hold": "#7756a1",
}


def load_results(path):
    """Read only the columns needed for plotting and validate observations."""
    data = pd.read_csv(path)
    required = ["ts", "pv", "spy_price", "qs_price"]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Missing results columns: {', '.join(sorted(missing))}")
    data = data[required].copy()
    data["ts"] = pd.to_datetime(data["ts"], errors="raise")
    if data["ts"].isna().any() or data["ts"].duplicated().any():
        raise ValueError("Result timestamps must be non-null and unique.")
    if len(data) < 2:
        raise ValueError("At least two portfolio observations are required.")
    values = data[["pv", "spy_price", "qs_price"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(
            "Portfolio values and asset prices must be finite and positive."
        )
    data[["pv", "spy_price", "qs_price"]] = values
    return data.sort_values("ts").reset_index(drop=True)


def benchmark_curves(data):
    """Build comparable portfolios from recorded prices, with no trading costs.

    Equal-weight buy and hold allocates half the initial capital to each asset
    and allows weights to drift. It does not rebalance at each observation.
    """
    initial = data["pv"].iloc[0]
    leader = initial * data["spy_price"] / data["spy_price"].iloc[0]
    follower = initial * data["qs_price"] / data["qs_price"].iloc[0]
    return pd.DataFrame(
        {
            "Strategy": data["pv"].to_numpy(),
            "Hold S&P series": leader.to_numpy(),
            "Hold Nasdaq series": follower.to_numpy(),
            "50/50 buy and hold": ((leader + follower) / 2).to_numpy(),
        },
        index=pd.DatetimeIndex(data["ts"], name="Timestamp"),
    )


def create_chart(data, output):
    """Save value and observed-drawdown panels, then return the output path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as dates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    curves = benchmark_curves(data)
    drawdowns = curves / curves.cummax() - 1
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#c5cbd1",
            "axes.labelcolor": "#354251",
            "xtick.color": "#566372",
            "ytick.color": "#566372",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    ):
        fig, (value_ax, drawdown_ax) = plt.subplots(
            2,
            1,
            figsize=(12, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [2.3, 1], "hspace": 0.16},
        )
        try:
            fig.subplots_adjust(top=0.77, bottom=0.15, left=0.09, right=0.97)
            fig.text(
                0.09,
                0.95,
                "Allocation backtest versus simple benchmarks",
                fontsize=20,
                weight="bold",
                color="#1e2c3b",
            )
            first, last = curves.index[0], curves.index[-1]
            fig.text(
                0.09,
                0.91,
                f"{first:%b %d, %Y} to {last:%b %d, %Y}  |  Saved portfolio observations",
                fontsize=11,
                color="#566372",
            )
            for name, color in COLORS.items():
                strategy = name == "Strategy"
                total_return = curves[name].iloc[-1] / curves[name].iloc[0] - 1
                value_ax.plot(
                    curves.index,
                    curves[name],
                    color=color,
                    linewidth=2.7 if strategy else 1.6,
                    zorder=4 if strategy else 2,
                    label=f"{name}  {total_return:+.2%}",
                )
                drawdown_ax.plot(
                    drawdowns.index,
                    drawdowns[name],
                    color=color,
                    linewidth=2 if strategy else 1.1,
                    zorder=4 if strategy else 2,
                )
            drawdown_ax.fill_between(
                drawdowns.index,
                drawdowns["Strategy"],
                0,
                color=COLORS["Strategy"],
                alpha=0.09,
            )
            handles, labels = value_ax.get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(0.08, 0.88),
                ncol=2,
                frameon=False,
                columnspacing=2,
                handlelength=2.8,
            )
            value_ax.set_ylabel("Portfolio value")
            value_ax.yaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{value:,.0f}")
            )
            drawdown_ax.set_ylabel("Drawdown")
            drawdown_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
            locator = dates.AutoDateLocator(minticks=5, maxticks=8)
            drawdown_ax.xaxis.set_major_locator(locator)
            drawdown_ax.xaxis.set_major_formatter(dates.DateFormatter("%b %d"))
            for ax in (value_ax, drawdown_ax):
                ax.grid(axis="y", color="#e5e9ee", linewidth=0.8)
                ax.set_axisbelow(True)
                ax.margins(x=0)
            fig.text(
                0.09,
                0.075,
                f"{len(data)} saved observations. Drawdowns omit movements between observations.\n"
                "Fees, slippage, financing, and futures contract mechanics are excluded.",
                fontsize=9,
                color="#566372",
                linespacing=1.6,
            )
            fig.savefig(output, dpi=180, facecolor="white")
        finally:
            plt.close(fig)
    return output


def main(argv=None):
    """Generate a chart from original results or a newly produced portfolio CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, help="Saved portfolio CSV"
    )
    parser.add_argument(
        "--output", type=Path, help="PNG output (default: beside the input CSV)"
    )
    args = parser.parse_args(argv)
    output = args.output or args.input.with_name("results_chart.png")
    path = create_chart(load_results(args.input), output)
    print(f"Saved chart: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
