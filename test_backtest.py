"""Deterministic accounting and orchestration tests; no market downloads."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from backtest_data import load_indicator_data, split_test_train
from optimizer_backtest import get_change_points, pseudo_balance_portfolio, run_backtest
from size_by_beta import beta_to_prices, get_size_by_lgbm


def indicator_frame():
    timestamps = pd.date_range("2024-01-01", periods=5, freq="h")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": [100, 100, 110, 100, 105],
            "high": 110,
            "low": 90,
            "ath": 120,
            "path": 0.8,
            "last": [0.8, 0.8, 0.7, 0.7, 0.9],
            "latest": 0.9,
        }
    )


class BacktestTests(unittest.TestCase):
    def test_split_boundaries_and_independent_copies(self):
        data = indicator_frame()
        train, test = split_test_train(
            data, data.timestamp[0], data.timestamp[2], data.timestamp[4]
        )
        self.assertEqual(train.index.tolist(), [0, 1])
        self.assertEqual(test.index.tolist(), [2, 3])
        train.loc[0, "close"] = 500
        self.assertEqual(data.loc[0, "close"], 100)
        with self.assertRaises(ValueError):
            split_test_train(
                data, data.timestamp[2], data.timestamp[0], data.timestamp[4]
            )

    def test_loader_accepts_clean_csv_and_legacy_columns(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.csv"
            source = indicator_frame().rename(columns={"last": "last_extrema_path"})
            source.iloc[::-1].to_csv(path, index=False)
            loaded = load_indicator_data(path)
            self.assertTrue(loaded.timestamp.is_monotonic_increasing)
            pd.testing.assert_series_equal(
                loaded["last"], loaded["last_extrema_path"], check_names=False
            )
            pd.concat([source, source.iloc[:1]]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "unique"):
                load_indicator_data(path)

    def test_change_points(self):
        data = indicator_frame()
        self.assertEqual(
            get_change_points(data), data.timestamp.iloc[[0, 2, 4]].tolist()
        )

    def test_previous_allocation_earns_return_before_rebalance(self):
        records = []
        pseudo_balance_portfolio(
            1, {"close": 100, "timestamp": 1}, 0, {"close": 50}, records
        )
        pseudo_balance_portfolio(
            0, {"close": 110, "timestamp": 2}, 1, {"close": 25}, records
        )
        self.assertEqual(records[-1]["pv"], 1100)
        pseudo_balance_portfolio(
            0.5, {"close": 55, "timestamp": 3}, 0.5, {"close": 50}, records
        )
        self.assertEqual(records[-1]["pv"], 2200)
        self.assertEqual(records[-1]["ts"], 3)
        with self.assertRaises(ValueError):
            pseudo_balance_portfolio(0.8, {"close": 1}, 0.8, {"close": 1}, [])

    def test_beta_compounding_does_not_mutate_input(self):
        source = pd.DataFrame({"ts": [0, 1, 2], "1": [123, 0.1, -0.1]})
        result = beta_to_prices(source, 100)
        np.testing.assert_allclose(result["1"], [100, 110, 99])
        self.assertEqual(source.loc[0, "1"], 123)

    def test_allocator_prefers_consistently_better_asset(self):
        leader = pd.DataFrame(
            {"ts": [0, 1], **{str(i): [100, 120] for i in range(100)}}
        )
        follower = pd.DataFrame(
            {"ts": [0, 1], **{str(i): [100, 90] for i in range(100)}}
        )
        ranked = get_size_by_lgbm(leader, follower, max_allocation=1)
        self.assertGreaterEqual(ranked.iloc[0].leader_frac, 0.95)

    def test_orchestration_excludes_current_and_future_training_rows(self):
        leader = indicator_frame()
        follower = indicator_frame()
        calls = []

        def simulate(train, train2, current, ath, timestamp, paths, horizon):
            self.assertTrue((train.timestamp < timestamp).all())
            self.assertTrue((train2.timestamp < timestamp).all())
            calls.append(timestamp)
            return (
                pd.DataFrame({"ts": [0, 1], "1": [100, 110]}),
                pd.DataFrame({"ts": [0, 1], "1": [0, 0.1]}),
                None,
                1,
            )

        allocations = pd.DataFrame({"leader_frac": [1.0], "follower_frac": [0.0]})
        with tempfile.TemporaryDirectory() as folder:
            with (
                patch("optimizer_mc.flagship_mc", side_effect=simulate),
                patch("size_by_beta.get_size_by_lgbm", return_value=allocations),
            ):
                result = run_backtest(
                    leader,
                    follower,
                    leader.timestamp[0],
                    leader.timestamp[1],
                    leader.timestamp[4],
                    output_dir=folder,
                )
            self.assertEqual(calls, leader.timestamp.iloc[[1, 2]].tolist())
            self.assertEqual(result.pv.tolist(), [1000, 1100])
            saved = pd.read_csv(Path(folder) / "portfolio.csv")
            self.assertEqual(len(saved), 2)
            self.assertNotIn("Unnamed: 0", saved)

    def test_missing_aligned_price_fails_before_simulation(self):
        data = indicator_frame()
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("optimizer_mc.flagship_mc") as simulate,
        ):
            with self.assertRaisesRegex(ValueError, "aligned"):
                run_backtest(
                    data,
                    data.drop(index=1),
                    data.timestamp[0],
                    data.timestamp[1],
                    data.timestamp[4],
                    output_dir=folder,
                )
            simulate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
