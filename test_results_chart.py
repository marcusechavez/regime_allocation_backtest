"""Check benchmark math and input validation independently of plotting style."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from plot_results import benchmark_curves, load_results


class ResultsChartTests(unittest.TestCase):
    def test_benchmarks_buy_and_hold_without_rebalancing(self):
        data = pd.DataFrame(
            {
                "ts": pd.date_range("2024-01-01", periods=3),
                "pv": [1000, 1050, 900],
                "spy_price": [100, 200, 100],
                "qs_price": [200, 100, 200],
            }
        )
        curves = benchmark_curves(data)
        np.testing.assert_allclose(curves["Hold S&P series"], [1000, 2000, 1000])
        np.testing.assert_allclose(curves["Hold Nasdaq series"], [1000, 500, 1000])
        np.testing.assert_allclose(curves["50/50 buy and hold"], [1000, 1250, 1000])
        drawdown = curves["Strategy"] / curves["Strategy"].cummax() - 1
        np.testing.assert_allclose(drawdown, [0, 0, 900 / 1050 - 1])

    def test_loader_sorts_legacy_indexed_output_and_rejects_bad_prices(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "portfolio.csv"
            data = pd.DataFrame(
                {
                    "ts": ["2024-01-02", "2024-01-01"],
                    "pv": [1010, 1000],
                    "spy_price": [110, 100],
                    "qs_price": [200, 200],
                }
            )
            data.to_csv(path)
            result = load_results(path)
            self.assertEqual(result["pv"].tolist(), [1000, 1010])
            self.assertNotIn("Unnamed: 0", result)
            data.loc[0, "spy_price"] = 0
            data.to_csv(path)
            with self.assertRaisesRegex(ValueError, "positive"):
                load_results(path)
