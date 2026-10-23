"""Feature distances used to match historical observations to current regimes."""

import numpy as np
import pandas as pd


def knn_vectorized_1(
    path_arr,
    path_change_arr,
    trend_delta_arr,
    ts_arr,
    path,
    path_change,
    trend_delta,
    cutoff=0.01,
):
    """Match timestamps using mean absolute normalized feature distance."""
    diff1, diff2, diff3 = _normalized_feature_differences(
        path_arr, path_change_arr, trend_delta_arr, path, path_change, trend_delta
    )
    distance = (diff1 + diff2 + diff3) / 3
    matches = distance < cutoff
    return list(pd.to_datetime(ts_arr[matches])), distance[matches]


def knn_vectorized(
    path_arr,
    path_change_arr,
    trend_delta_arr,
    ts_arr,
    path,
    path_change,
    trend_delta,
    cutoff=0.01,
):
    """Match timestamps using Euclidean normalized feature distance."""
    diff1, diff2, diff3 = _normalized_feature_differences(
        path_arr, path_change_arr, trend_delta_arr, path, path_change, trend_delta
    )
    distance = ((diff1**2) + (diff2**2) + (diff3**2)) ** (1 / 2)
    matches = distance < cutoff
    return list(pd.to_datetime(ts_arr[matches])), distance[matches]


def _normalized_feature_differences(
    path_arr, path_change_arr, trend_delta_arr, path, path_change, trend_delta
):
    """Normalize absolute deviations only for positive reference features."""
    diff1 = np.abs(path_arr - path)
    diff2 = np.abs(path_change_arr - path_change)
    diff3 = np.abs(trend_delta_arr - trend_delta)
    if path > 0:
        diff1 /= path
    if path_change > 0:
        diff2 /= path_change
    if trend_delta > 0:
        diff3 /= trend_delta
    return diff1, diff2, diff3
