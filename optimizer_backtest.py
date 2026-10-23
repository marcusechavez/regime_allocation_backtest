"""Walk-forward portfolio allocation with regime-conditioned Monte Carlo paths.

Run ``python optimizer_backtest.py --help`` for the command-line interface.
"""

import argparse
import logging
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from backtest_data import load_indicator_data, split_test_train

LOGGER = logging.getLogger(__name__)
REQUIRED_COLUMNS = {
    "timestamp",
    "close",
    "high",
    "low",
    "ath",
    "path",
    "last",
    "latest",
}


def pseudo_balance_portfolio(
    spy_alloc, spy_data, qs_alloc, qs_data, bt_pv_list, initial_value=1000.0
):
    """Mark previous holdings to market, then record new target allocations.

    The model is frictionless and fully invested. Legacy spy/qs column names
    and rounding at each rebalance are retained for existing analyses.
    """
    weights = np.array([spy_alloc, qs_alloc], dtype=float)
    if (
        not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.isclose(weights.sum(), 1)
    ):
        raise ValueError("Allocations must be nonnegative and sum to one.")
    prices = [spy_data["close"], qs_data["close"]]
    if not all(np.isfinite(price) and price > 0 for price in prices):
        raise ValueError("Asset prices must be finite and positive.")
    if not np.isfinite(initial_value) or initial_value <= 0:
        raise ValueError("Initial portfolio value must be finite and positive.")
    if bt_pv_list:
        previous = bt_pv_list[-1]
        value = previous["pv"] * (
            previous["spy_frac"] * prices[0] / previous["spy_price"]
            + previous["qs_frac"] * prices[1] / previous["qs_price"]
        )
        start_spy = bt_pv_list[0]["spy_price"]
        start_qs = bt_pv_list[0]["qs_price"]
    else:
        value = initial_value
        start_spy, start_qs = prices
    bt_pv_list.append(
        {
            "ts": spy_data.get("ts", spy_data.get("timestamp")),
            "spy_frac": spy_alloc,
            "spy_price": prices[0],
            "spy_change": prices[0] / start_spy - 1,
            "qs_frac": qs_alloc,
            "qs_price": prices[1],
            "qs_change": prices[1] / start_qs - 1,
            "pv": round(value, 2),
        }
    )
    return bt_pv_list


def get_change_points(df):
    """Return the first observation and each change in last-extremum state."""
    return df.loc[df["last"].ne(df["last"].shift()), "timestamp"].tolist()


def run_backtest(
    leader,
    follower,
    start_train,
    start_test,
    end_test,
    *,
    paths=4000,
    horizon_hours=96,
    initial_value=1000.0,
    output_dir="outputs",
    save_simulations=False,
):
    """Rebalance at leader regime changes, using strictly earlier training rows.

    Require simultaneous observations instead of selecting a future follower
    price. The final portfolio value is measured at the last rebalance.
    """
    if (
        paths <= 0
        or horizon_hours <= 0
        or not np.isfinite(initial_value)
        or initial_value <= 0
    ):
        raise ValueError("Paths, horizon, and initial value must be positive.")
    for name, data in (("leader", leader), ("follower", follower)):
        missing = REQUIRED_COLUMNS - set(data.columns)
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")
        if data["timestamp"].isna().any() or data["timestamp"].duplicated().any():
            raise ValueError(f"{name}: timestamps must be non-null and unique.")
    leader = leader.sort_values("timestamp")
    follower = follower.sort_values("timestamp")
    _, test = split_test_train(leader, start_train, start_test, end_test)
    points = get_change_points(test)
    if not points:
        raise ValueError("No leader observations in the requested test period.")
    from optimizer_mc import flagship_mc
    from size_by_beta import beta_to_prices, get_size_by_lgbm

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for timestamp in points:
        leader_train, leader_test = split_test_train(
            leader, start_train, timestamp, end_test
        )
        follower_train, follower_test = split_test_train(
            follower, start_train, timestamp, end_test
        )
        aligned_follower = follower_test.loc[follower_test["timestamp"].eq(timestamp)]
        if leader_train.empty or follower_train.empty or aligned_follower.empty:
            raise ValueError(
                f"Insufficient history or missing aligned price at {timestamp}."
            )
        current = leader_test.iloc[0].to_dict()
        follower_row = aligned_follower.iloc[0].to_dict()
        mc_df, beta_df, _, _ = flagship_mc(
            leader_train.copy(),
            follower_train.copy(),
            current,
            current["ath"],
            timestamp,
            paths,
            horizon_hours,
        )
        if mc_df.empty or beta_df.empty:
            raise ValueError(f"No simulation paths generated at {timestamp}.")
        if save_simulations:
            label = timestamp.strftime("%Y%m%dT%H%M%S%f")
            mc_df.to_csv(output / f"mc_{label}.csv", index=False)
            beta_df.to_csv(output / f"beta_{label}.csv", index=False)
        allocations = get_size_by_lgbm(
            mc_df,
            beta_to_prices(beta_df, follower_row["close"]),
            min_allocation=0,
            max_allocation=1,
            step=0.05,
        ).iloc[0]
        pseudo_balance_portfolio(
            allocations["leader_frac"],
            current,
            allocations["follower_frac"],
            follower_row,
            records,
            initial_value,
        )
        LOGGER.info(
            "%s | leader %.0f%% | portfolio %.2f",
            timestamp,
            allocations["leader_frac"] * 100,
            records[-1]["pv"],
        )
    result = pd.DataFrame(records)
    result["spread"] = (
        result["spy_price"].pct_change() - result["qs_price"].pct_change()
    )
    result["captured"] = ((result["spread"] > 0) & (result["spy_frac"].shift() > 0)) | (
        (result["spread"] < 0) & (result["qs_frac"].shift() > 0)
    )
    result.to_csv(output / "portfolio.csv", index=False)
    return result


def main(argv=None):
    """Parse portable file paths and execute the backtest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--leader", required=True, type=Path, help="Leader indicator CSV"
    )
    parser.add_argument(
        "--follower", required=True, type=Path, help="Follower indicator CSV"
    )
    parser.add_argument("--start-train", default="2020-01-01")
    parser.add_argument("--start-test", required=True)
    parser.add_argument("--end-test", required=True)
    parser.add_argument("--paths", type=int, default=4000)
    parser.add_argument("--horizon-hours", type=int, default=96)
    parser.add_argument("--initial-value", type=float, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--save-simulations", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    started = perf_counter()
    result = run_backtest(
        load_indicator_data(args.leader),
        load_indicator_data(args.follower),
        pd.Timestamp(args.start_train),
        pd.Timestamp(args.start_test),
        pd.Timestamp(args.end_test),
        paths=args.paths,
        horizon_hours=args.horizon_hours,
        initial_value=args.initial_value,
        output_dir=args.output_dir,
        save_simulations=args.save_simulations,
    )
    LOGGER.info(
        "Final value at last rebalance: %.2f | elapsed %.1fs",
        result.iloc[-1]["pv"],
        perf_counter() - started,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
