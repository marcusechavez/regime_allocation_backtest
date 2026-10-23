"""CSV normalization and non-overlapping chronological train/test splits."""

from pathlib import Path

import pandas as pd


def split_test_train(data_df, start_train, end_train, end_test):
    """Return independent, sorted frames for [start, split) and [split, end)."""
    if not start_train < end_train < end_test:
        raise ValueError("Dates must satisfy start_train < end_train < end_test.")
    timestamps = data_df["timestamp"]
    train = data_df.loc[(timestamps >= start_train) & (timestamps < end_train)]
    test = data_df.loc[(timestamps >= end_train) & (timestamps < end_test)]
    return (train.sort_values("timestamp").copy(), test.sort_values("timestamp").copy())


def load_indicator_data(path: str | Path) -> pd.DataFrame:
    """Load a CSV, normalize legacy extrema names, and validate timestamps."""
    data = pd.read_csv(path).drop(columns=["Unnamed: 0", "index"], errors="ignore")
    if "timestamp" not in data:
        raise ValueError(f"{path}: missing required column 'timestamp'.")
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    if data["timestamp"].isna().any() or data["timestamp"].duplicated().any():
        raise ValueError(f"{path}: timestamps must be non-null and unique.")
    for short, legacy in (
        ("last", "last_extrema_path"),
        ("latest", "latest_extrema_path"),
    ):
        if legacy in data:
            data[short] = data[legacy]
        if short in data:
            data[legacy] = data[short]
    return data.sort_values("timestamp").reset_index(drop=True)
