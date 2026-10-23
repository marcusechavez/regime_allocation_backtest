"""KDE path sampling, historical regime windows, and extrema state updates."""

import numpy as np
import pandas as pd


def update_structure_fast(structure, idx, value, typ):
    """Append an accepted extremum and retain at most two entries.

    Type 0 denotes a high and type 1 a low. Same-type candidates are
    accepted only when more extreme than the most recent entry.
    """
    if not structure:
        structure.append((idx, value, typ))
        return

    last_idx, last_val, last_typ = structure[-1]

    # Same type - check if more extreme
    if typ == last_typ:
        if (typ == 0 and value <= last_val) or (typ == 1 and value >= last_val):
            return

    # Append new extrema
    structure.append((idx, value, typ))

    # Maintain max size of 2
    if len(structure) > 2:
        structure.pop(0)


def get_returns(r):
    """Remove NaN observations while preserving order."""
    r = r[~np.isnan(r)]

    return r


def get_similar_dates(
    knn_ts, knn_path, knn_latest, knn_last, base_data, min_trade_count=30, cutoff=0.025
):
    """Match rounded regime features with Euclidean then mean distances.

    The fallback runs only below min_trade_count; matches are returned
    only strictly above it. This boundary preserves the original experiment.
    """
    from regime_matching import knn_vectorized, knn_vectorized_1

    round_val = 3

    # Raw features
    similar_trades_dates, distance_list = knn_vectorized(
        knn_path,
        knn_latest,
        knn_last,
        knn_ts,
        round(base_data["path"], round_val),
        round(base_data["latest"], round_val),
        round(base_data["last"], round_val),
        cutoff,
    )

    trade_count = len(set(similar_trades_dates))

    if trade_count < min_trade_count:

        # Raw features
        similar_trades_dates, distance_list = knn_vectorized_1(
            knn_path,
            knn_latest,
            knn_last,
            knn_ts,
            round(base_data["path"], round_val),
            round(base_data["latest"], round_val),
            round(base_data["last"], round_val),
            cutoff,
        )

        trade_count = len(set(similar_trades_dates))

    if trade_count > min_trade_count:

        df = pd.DataFrame(
            {"ts": pd.to_datetime(similar_trades_dates), "distance": distance_list}
        )

        return df

    return pd.DataFrame([])


def generate_mc_timestamps(start_ts, steps):
    """
    Generate hourly timestamps for MC paths.

    Parameters
    ----------
    start_ts : np.datetime64
        Starting timestamp for simulation (S0)
    steps : int
        Number of hours / MC steps

    Returns
    -------
    mc_ts_arr : np.ndarray of np.datetime64
        Timestamps for each MC step
    """
    mc_ts_arr = start_ts + np.arange(steps + 1) * np.timedelta64(1, "h")
    return mc_ts_arr


def mc_wrapper(S0, returns_arr, paths=1000, steps=24):
    """Sample KDE increments and exponentiate cumulative sums into price paths.

    The first column is S0. The existing engine treats sampled increments
    as log returns even though upstream uses a custom return definition.
    """
    all_paths = get_paths(returns_arr, paths=paths, steps=steps)

    # prepend zero return so paths start at S0
    all_paths = np.hstack([np.zeros((paths, 1)), all_paths])

    cumulative_returns = np.cumsum(all_paths, axis=1)

    price_paths = S0 * np.exp(cumulative_returns)

    return price_paths


def get_paths(returns_arr, paths=1000, steps=24):
    """Sample a (paths, steps) increment matrix from a Gaussian KDE."""
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(returns_arr)

    return kde.resample(steps * paths).reshape(paths, steps)


def mc_wrapper_one_path(S0, returns_arr, steps=24):
    """
    Generate a single Monte Carlo price path.

    Parameters
    ----------
    S0 : float
        Starting price.
    returns_arr : np.ndarray
        Historical returns to sample from.
    steps : int
        Number of steps to simulate.

    Returns
    -------
    np.ndarray
        Price path of length steps+1, starting with S0.
    """
    # Sample returns
    sampled_returns = get_path_one(returns_arr, steps)

    # Prepend 0 return so path starts at S0
    sampled_returns = np.insert(sampled_returns, 0, 0.0)

    # Cumulative sum and exponentiation for log returns
    cumulative_returns = np.cumsum(sampled_returns)
    price_path = S0 * np.exp(cumulative_returns)

    return price_path


def get_path_one(returns_arr, steps=24):
    """Sample one vector of KDE increments."""
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(returns_arr)
    sampled = kde.resample(steps).reshape(steps)  # only 1 path
    return sampled


def get_similar_ohlc(dates_df, price_ts, diffs, window=24):
    """Select non-NaN returns in closed, fixed-length regime windows.

    Sorting window starts allows binary search of the latest preceding
    window for each price timestamp.
    """
    import datetime

    dates_df = dates_df.copy()
    dates_df["end_ts"] = dates_df["ts"] + datetime.timedelta(hours=window)

    start_ts = dates_df["ts"].values
    end_ts = dates_df["end_ts"].values

    # sort by start_ts
    order = np.argsort(start_ts)
    start_ts = start_ts[order]
    end_ts = end_ts[order]

    # binary search
    idx = np.searchsorted(start_ts, price_ts, side="right") - 1

    valid = idx >= 0
    valid_rows = np.zeros(len(price_ts), dtype=bool)
    valid_rows[valid] = price_ts[valid] <= end_ts[idx[valid]]

    returns = diffs[valid_rows]
    return get_returns(returns)


def prep_batch(paths):
    """Partition path indices into roughly ten batches, preserving order."""
    step = max(1, paths // 10)
    return [list(range(i, min(i + step, paths))) for i in range(0, paths, step)]
