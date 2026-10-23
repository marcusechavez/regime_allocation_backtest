"""Checks for regime matching, path conventions, and simulation integration."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import optimizer_mc as simulation
import simulation_utils as sampling
from regime_matching import knn_vectorized, knn_vectorized_1
from size_by_beta import get_size_arrays


class SerialExecutor:
    """Exercise real numerical workers without process-pool randomness."""

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def map(self, function, batches):
        return map(function, batches)


def synthetic_history():
    """Provide enough varied history for the engine's 60-day exclusion."""
    random = np.random.default_rng(912)
    timestamps = pd.date_range("2024-01-01", periods=1800, freq="h")
    prices = 100 + 2 * np.sin(np.arange(1800) / 4) + random.normal(0, 0.1, 1800)
    leader = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": prices,
            "high": prices + 0.1,
            "low": prices - 0.1,
            "ath": 110.0,
            "path": prices / 110,
            "latest": 0.93,
            "last": 0.89,
        }
    )
    follower = leader.copy()
    follower["close"] = (
        200 + 2 * np.sin(np.arange(1800) / 5) + random.normal(0, 0.2, 1800)
    )
    return leader, follower


class SimulationTests(unittest.TestCase):
    def test_distance_metrics_have_distinct_cutoffs(self):
        timestamps = pd.date_range("2024-01-01", periods=2, freq="h").to_numpy()
        features = np.array([1.0, 1.02])
        strict, _ = knn_vectorized(
            features, features, features, timestamps, 1, 1, 1, 0.025
        )
        relaxed, distances = knn_vectorized_1(
            features, features, features, timestamps, 1, 1, 1, 0.025
        )
        self.assertEqual(strict, [pd.Timestamp(timestamps[0])])
        self.assertEqual(len(relaxed), 2)
        np.testing.assert_allclose(distances, [0, 0.02])

    def test_matching_fallback_and_original_minimum_boundary(self):
        timestamps = pd.date_range("2024-01-01", periods=31, freq="h").to_numpy()
        features = np.full(31, 1.02)
        current = {"path": 1, "latest": 1, "last": 1}
        matched = sampling.get_similar_dates(
            timestamps, features, features, features, current
        )
        self.assertEqual(len(matched), 31)
        # Exactly 30 matches still returns empty under the historical rule.
        boundary = sampling.get_similar_dates(
            timestamps[:30], features[:30], features[:30], features[:30], current
        )
        self.assertTrue(boundary.empty)

    def test_regime_windows_include_endpoints_and_drop_nan(self):
        timestamps = pd.date_range("2024-01-01", periods=8, freq="h").to_numpy()
        dates = pd.DataFrame({"ts": timestamps[[4, 1]]})
        returns = np.arange(8, dtype=float)
        returns[2] = np.nan
        selected = simulation.get_similar_ohlc(dates, timestamps, returns, window=2)
        np.testing.assert_array_equal(selected, [1, 3, 4, 5, 6])
        coefficients = simulation.get_similar_beta(dates, timestamps, returns, window=2)
        np.testing.assert_array_equal(coefficients, [1, 4])

    def test_extrema_state_rejects_weaker_same_type_candidates(self):
        state = []
        for item in ((0, 100, 0), (1, 99, 0), (2, 101, 0), (3, 97, 1)):
            sampling.update_structure_fast(state, *item)
        self.assertEqual(state, [(2, 101, 0), (3, 97, 1)])

    def test_kde_price_paths_start_at_initial_price(self):
        state = np.random.get_state()
        try:
            np.random.seed(42)
            paths = sampling.mc_wrapper(
                100, np.array([-0.02, -0.01, 0.01, 0.03]), paths=3, steps=4
            )
        finally:
            np.random.set_state(state)
        self.assertEqual(paths.shape, (3, 5))
        np.testing.assert_array_equal(paths[:, 0], 100)
        self.assertTrue(np.isfinite(paths).all())
        self.assertTrue((paths > 0).all())
        timestamps = sampling.generate_mc_timestamps(np.datetime64("2024-01-01T00"), 4)
        self.assertEqual(len(timestamps), 5)
        self.assertEqual(timestamps[-1], np.datetime64("2024-01-01T04"))

    def test_relative_returns_preserve_sign_zero_and_clipping(self):
        data = pd.DataFrame(
            {"high": [100.0, 110.0, 110.0, 99.0], "beta": [-10, 0, 2, -1]}
        )
        prices, increments = simulation._correlated_path_returns(data)
        np.testing.assert_array_equal(prices, [100, 110, 110, 99])
        np.testing.assert_allclose(
            increments, [np.nan, -1 / 11, 0.0000025, 1 / 18], equal_nan=True
        )

    def test_allocation_pairs_columns_by_name(self):
        leader = pd.DataFrame({"ts": [0, 1], "a": [100, 110], "b": [100, 90]})
        follower = pd.DataFrame({"ts": [0, 1], "b": [200, 240], "a": [200, 180]})
        weights, complements, returns = get_size_arrays(
            leader, follower, np.array([0.0, 1.0])
        )
        np.testing.assert_array_equal(weights, [0, 0, 1, 1])
        np.testing.assert_array_equal(complements, [1, 1, 0, 0])
        np.testing.assert_allclose(returns, [-0.1, 0.2, 0.1, -0.1])

    def test_paired_pipeline_produces_complete_aligned_paths(self):
        leader, follower = synthetic_history()
        current = leader.iloc[-1].to_dict()
        random_state = np.random.get_state()
        try:
            np.random.seed(314)
            with patch("concurrent.futures.ProcessPoolExecutor", SerialExecutor):
                prices, coefficients, array, fraction = simulation.flagship_mc(
                    leader.iloc[:-1].copy(),
                    follower.iloc[:-1].copy(),
                    current,
                    current["ath"],
                    current["timestamp"],
                    paths=4,
                    steps=3,
                    order=2,
                )
        finally:
            np.random.set_state(random_state)
        self.assertEqual(prices.shape, (4, 5))
        self.assertEqual(coefficients.shape, prices.shape)
        self.assertEqual(array.shape, (4, 4))
        self.assertEqual(fraction, 1)
        pd.testing.assert_series_equal(prices["ts"], coefficients["ts"])
        np.testing.assert_allclose(prices.drop(columns="ts").iloc[0], current["close"])
        self.assertTrue(np.isfinite(prices.drop(columns="ts")).all().all())
        self.assertTrue(np.isnan(array[:, 0]).all())


if __name__ == "__main__":
    unittest.main()
