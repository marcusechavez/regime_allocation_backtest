"""Regime-conditioned simulation of paired leader and relative-return paths.

The original return signs, clipping bounds, sampling, and fallback behavior are
retained so cleanup does not reinterpret the historical experiment.
"""

import logging

import numpy as np
import pandas as pd

from simulation_utils import (
    generate_mc_timestamps,
    get_similar_dates,
    get_similar_ohlc,
    mc_wrapper,
    mc_wrapper_one_path,
    update_structure_fast,
)

LOGGER = logging.getLogger(__name__)


def find_extrema_wrapper_fast_v3(
    thresh_ts, high_arr, low_arr, ath_arr, beta_arr, ts_arr, order, structure=None
):
    """Track the last two extrema and stop after a new post-threshold extremum.

    Metrics keep unprocessed rows as NaN. Right-edge detection and strict
    comparisons follow the original research algorithm.
    """
    observation_count = len(high_arr)

    # Pre-allocate metrics array
    metrics = np.empty((observation_count, 9), dtype=np.float64)
    metrics[:] = np.nan

    # Normalized paths - compute once
    normalized_highs = high_arr / ath_arr
    normalized_lows = low_arr / ath_arr

    # Initialize structure
    if structure is None:
        structure = []
        first_index = order
    else:
        first_index = max(order, structure[-1][0] - order)

    # Cache left-window extrema before the sequential scan.
    # Left windows
    rolling_high_max = np.empty(observation_count, dtype=np.float64)
    rolling_low_min = np.empty(observation_count, dtype=np.float64)

    for i in range(observation_count):
        left = max(0, i - order)
        rolling_high_max[i] = high_arr[left:i].max() if i > left else -np.inf
        rolling_low_min[i] = low_arr[left:i].min() if i > left else np.inf

    # --- Main loop with early exit ---
    for t in range(first_index, observation_count):
        start = max(first_index, t - order)

        for i in range(start, t + 1):
            center_high = high_arr[i]
            center_low = low_arr[i]

            # --- HIGH extrema check ---
            if center_high > rolling_high_max[i]:
                # Check right side
                right_end = min(i + order + 1, t)
                if right_end > i + 1:
                    right_max = high_arr[i + 1 : right_end].max()
                    is_high = center_high > right_max
                else:
                    is_high = True

                if is_high:
                    update_structure_fast(structure, i, center_high, 0)

            # --- LOW extrema check ---
            if center_low < rolling_low_min[i]:
                # Check right side
                right_end = min(i + order + 1, t)
                if right_end > i + 1:
                    right_min = low_arr[i + 1 : right_end].min()
                    is_low = center_low < right_min
                else:
                    is_low = True

                if is_low:
                    update_structure_fast(structure, i, center_low, 1)

        # Update metrics if we have enough structure
        if len(structure) >= 2:
            i2, v2, p2 = structure[-2]
            i1, v1, p1 = structure[-1]

            # Direct array assignment (faster than tuple unpacking)
            metrics[t, 0] = ts_arr[t]
            metrics[t, 1] = high_arr[t]
            metrics[t, 2] = v1
            metrics[t, 3] = v2
            metrics[t, 4] = ts_arr[i1]
            metrics[t, 5] = ts_arr[i2]
            metrics[t, 6] = normalized_highs[i1] if p1 == 0 else normalized_lows[i1]
            metrics[t, 7] = normalized_highs[i2] if p2 == 0 else normalized_lows[i2]
            metrics[t, 8] = beta_arr[t]

            # Early exit check
            if thresh_ts < ts_arr[i1]:
                return structure, pd.DataFrame(
                    metrics,
                    columns=[
                        "timestamp",
                        "high",
                        "latest_price",
                        "last_price",
                        "current_ts",
                        "last_ts",
                        "latest",
                        "last",
                        "beta",
                    ],
                )

    return structure, pd.DataFrame(
        metrics,
        columns=[
            "timestamp",
            "high",
            "latest_price",
            "last_price",
            "current_ts",
            "last_ts",
            "latest",
            "last",
            "beta",
        ],
    )


def get_beta_kde(boot_strap_beta, remaining_steps):
    """Sample one row of relative-return coefficients from a Gaussian KDE."""
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(boot_strap_beta)

    return kde.resample(remaining_steps)


def get_beta_kde_matrix(boot_strap_beta, paths, steps):
    """Sample a (paths, steps) matrix of relative-return coefficients."""
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(boot_strap_beta)

    return kde.resample(steps * paths).reshape(paths, steps)


def get_similar_beta(dates_df, price_ts, diffs, window=24):
    """Select non-NaN coefficients at matched timestamps; window is legacy."""
    selected = diffs[np.isin(price_ts, dates_df["ts"].values)]
    return selected[~np.isnan(selected)]


def flagship_mc(
    train_df,
    train_df_2,
    base_data,
    ath,
    ts,
    paths=1000,
    steps=24,
    order=120,
    epsilon=0.000001,
):
    """Simulate paired paths using historical observations from similar regimes.

    Returns leader paths, relative-return paths, a relative-return array,
    and the completion fraction. A no-match result retains the legacy
    empty frames and sentinel value 50. Input training frames are modified.
    The engine excludes the most recent 60 days from initial regime matches.
    """
    # Start price
    start_price = base_data["close"]

    # For MC
    train_df["diff"] = (train_df["close"].shift(1) - train_df["close"]) / (
        train_df["close"] + epsilon
    )
    # Protect from invalid denominator
    train_df["diff"] = train_df["diff"].replace(0, 0.000001)

    # Match secondary df and join
    train_df_2["diff_2"] = (train_df_2["close"].shift(1) - train_df_2["close"]) / (
        train_df_2["close"] + epsilon
    )
    filtered_train_2 = train_df_2[["timestamp", "diff_2"]].copy()
    train_df = pd.merge(train_df, filtered_train_2, on="timestamp", how="inner")
    train_df["beta"] = (train_df["diff"] - train_df["diff_2"]) / (
        train_df["diff"] + epsilon
    )

    # Preserve the original relative-return clipping bounds.
    train_df["beta"] = train_df["beta"].clip(lower=-2, upper=2)

    # Historical observations available for regime-conditioned sampling.
    price_ts = train_df["timestamp"].values
    diffs = train_df["diff"].values
    beta = train_df["beta"].values

    # Use these to find similar regimes
    knn_df = train_df.copy()
    history_cutoff = base_data["timestamp"] - pd.Timedelta(days=60)
    knn_df = knn_df[knn_df["timestamp"] < history_cutoff].copy()
    feature_decimals = 3
    knn_path_arr = np.round(knn_df["path"].to_numpy(), decimals=feature_decimals)
    knn_ts_arr = np.array(knn_df["timestamp"])
    knn_last_arr = np.array(knn_df["last"])
    knn_latest_arr = np.array(knn_df["latest"])

    # Match earlier observations with similar normalized extrema.
    similar_dates_df = get_similar_dates(
        knn_ts_arr, knn_path_arr, knn_latest_arr, knn_last_arr, base_data, cutoff=0.025
    )

    if len(similar_dates_df) > 0:
        # Sample increments from matched historical windows.

        returns_arr = get_similar_ohlc(similar_dates_df, price_ts, diffs, steps)
        matched_coefficients = get_similar_beta(similar_dates_df, price_ts, beta, steps)

        # Last timestamp
        max_ts = ts + pd.Timedelta(hours=steps)

        price_paths = mc_wrapper(start_price, returns_arr, paths, steps)
        kde_beta = get_beta_kde_matrix(matched_coefficients, paths, steps + 1)

        mc_data_df = train_df[
            ["timestamp", "high", "low", "ath", "latest", "last"]
        ].copy()

        mc_data_df = mc_data_df[
            mc_data_df["timestamp"] >= ts - pd.Timedelta(days=60)
        ].copy()

        beta_df = train_df[["timestamp", "beta"]].copy()
        beta_df = beta_df[beta_df["timestamp"] <= ts - pd.Timedelta(days=60)].copy()

        mc_df, beta_df, mc_arr, pct = mc_parallel_wrapper_batched(
            price_paths,
            kde_beta,
            mc_data_df,
            beta_df,
            returns_arr,
            ath,
            ts,
            max_ts,
            paths,
            steps,
            order,
            knn_ts_arr,
            knn_path_arr,
            knn_latest_arr,
            knn_last_arr,
            price_ts,
            diffs,
        )

        return mc_df, beta_df, mc_arr, pct

    else:
        LOGGER.warning("No historical regimes matched the current observation.")

    return pd.DataFrame([]), pd.DataFrame([]), np.array([[]]), 50


def mc_parallel_wrapper_batched(
    price_paths,
    kde_beta,
    mc_data_df,
    beta_df,
    returns_arr,
    ath,
    ts,
    max_ts,
    paths,
    steps,
    order,
    knn_ts_arr,
    knn_path_arr,
    knn_latest_arr,
    knn_last_arr,
    price_ts,
    diffs,
):
    """Collect complete simulated paths from batches in submission order."""
    import functools
    from concurrent.futures import ProcessPoolExecutor

    all_arr_list = []
    all_kde_list = []
    count = 0

    from simulation_utils import prep_batch

    batches = prep_batch(paths)

    worker = functools.partial(
        batch_wrapper,
        price_paths=price_paths,
        kde_beta=kde_beta,
        mc_data_df=mc_data_df,
        beta_df=beta_df,
        returns_arr=returns_arr,
        ath=ath,
        ts=ts,
        max_ts=max_ts,
        paths=paths,
        steps=steps,
        order=order,
        knn_ts_arr=knn_ts_arr,
        knn_path_arr=knn_path_arr,
        knn_latest_arr=knn_latest_arr,
        knn_last_arr=knn_last_arr,
        price_ts=price_ts,
        diffs=diffs,
    )

    with ProcessPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(worker, batches))

    for batch_count, paths_list, beta_kde_list in results:
        all_arr_list.extend(paths_list)
        all_kde_list.extend(beta_kde_list)
        count += batch_count

    new_ts_arr = generate_mc_timestamps(ts, steps)

    mc_df = pd.DataFrame(all_arr_list).T
    mc_df.columns = range(1, mc_df.shape[1] + 1)
    mc_df.insert(0, "ts", new_ts_arr)

    arr = np.vstack(all_kde_list)
    beta_df = pd.DataFrame(all_kde_list).T
    beta_df.columns = range(1, beta_df.shape[1] + 1)
    beta_df.insert(0, "ts", new_ts_arr)

    return mc_df, beta_df, arr, count / paths


def get_new_data_forced(
    price_path,
    kde_beta,
    train_df,
    returns_arr,
    ath,
    ts,
    strc,
    paths=1000,
    steps=24,
    order=120,
):
    """Extend historical prices and coefficients, then recompute extrema.

    Return the new segment, full extended frame, and updated extrema state.
    """
    mc_ts_arr = generate_mc_timestamps(ts, steps)

    np_ts = np.datetime64(ts)

    # Sort once if not guaranteed
    train_df = train_df.sort_values("timestamp")

    train_df = train_df[train_df["timestamp"] < ts].copy()

    # Carry forward the last ATH before the window
    pre_max = max(train_df["ath"])

    # Combine highs
    train_high = np.array(train_df["high"])
    combined_high = np.concatenate((train_high, price_path))

    # Combine highs
    train_beta = np.array(train_df["beta"])
    combined_beta = np.concatenate((train_beta, kde_beta))

    # Seeded cummax
    ath = np.maximum.accumulate(np.concatenate(([pre_max], combined_high)))[1:]

    # Build final df
    data_df = pd.DataFrame(
        {
            "timestamp": np.concatenate((train_df["timestamp"].values, mc_ts_arr)),
            "high": combined_high,
            "low": combined_high,
            "beta": combined_beta,
            "ath": ath,
        }
    )

    strc, new_data = find_extrema_wrapper_fast_v3(
        np_ts,
        np.array(data_df["high"]),
        np.array(data_df["low"]),
        np.array(data_df["ath"]),
        np.array(data_df["beta"]),
        np.array(data_df["timestamp"]),
        order,
        strc,
    )

    new_data["timestamp"] = pd.to_datetime(new_data["timestamp"])

    new_data = new_data[new_data["timestamp"] >= ts]
    new_data = new_data[["timestamp", "latest", "last", "high", "beta"]]

    data_df.set_index("timestamp", inplace=True)
    update_data = new_data.set_index("timestamp")

    data_df.update(update_data[["latest", "last"]])

    data_df.reset_index(inplace=True)

    return new_data, data_df, strc


def run_one_parallel_forced(
    price_path,
    kde_beta,
    mc_data_df,
    beta_df,
    returns_arr,
    ath,
    ts,
    max_ts,
    paths,
    steps,
    order,
    knn_ts_arr,
    knn_path_arr,
    knn_latest_arr,
    knn_last_arr,
    price_ts,
    diffs,
    debug_bool=False,
):
    """Extend a paired path across regime changes until its horizon.

    If no matching continuation regime exists, keep the most recent full
    simulated segment, as in the original implementation.
    """
    extrema_state = None

    mc_data_df["beta"] = -1

    # new data only
    data_df, full_mc_data_df, extrema_state = get_new_data_forced(
        price_path,
        kde_beta,
        mc_data_df,
        returns_arr,
        ath,
        ts,
        extrema_state,
        paths,
        steps,
        order,
    )
    mc_data_df = full_mc_data_df[
        full_mc_data_df["timestamp"] <= max(data_df["timestamp"])
    ].copy()

    max_mc_ts = max(data_df["timestamp"])

    beta_ts = beta_df["timestamp"].values
    beta = beta_df["beta"].values

    if max_mc_ts < max_ts:

        while True:
            base_data = data_df.iloc[-1]

            change_ts = base_data["timestamp"]

            remaining_steps = int((max_ts - change_ts).total_seconds() // 3600)

            if remaining_steps <= 0:
                if debug_bool:
                    LOGGER.debug("Simulation reached its requested horizon.")
                returns_df = mc_data_df[mc_data_df["timestamp"] >= ts].copy()

                return _correlated_path_returns(returns_df)

            new_price = base_data["high"]
            new_latest = base_data["latest"]
            new_last = base_data["last"]

            path = new_price / ath

            current_regime = {
                "close": new_price,
                "latest": new_latest,
                "last": new_last,
                "path": path,
                "timestamp": change_ts,
            }

            new_similar_dates_df = get_similar_dates(
                knn_ts_arr,
                knn_path_arr,
                knn_latest_arr,
                knn_last_arr,
                current_regime,
                cutoff=0.025,
            )

            if len(new_similar_dates_df) > 0:

                mc_arr = get_similar_ohlc(
                    new_similar_dates_df, price_ts, diffs, remaining_steps
                )
                boot_strap_beta = get_similar_beta(
                    new_similar_dates_df, beta_ts, beta, remaining_steps
                )

                # Add extra step to match new_price
                beta_kde = get_beta_kde(boot_strap_beta, remaining_steps + 1)
                beta_kde = beta_kde[0]

                price_path = mc_wrapper_one_path(new_price, mc_arr, remaining_steps)

                # MC_data_df is longer than steps +1 bc it needs to hold order data+ more historical data
                data_df, full_mc_data_df, extrema_state = get_new_data_forced(
                    price_path,
                    beta_kde,
                    mc_data_df,
                    mc_arr,
                    ath,
                    change_ts,
                    extrema_state,
                    paths,
                    remaining_steps,
                    order,
                )

                # Truncate outside of the function now
                mc_data_df = full_mc_data_df[
                    full_mc_data_df["timestamp"] <= max(data_df["timestamp"])
                ].copy()

            else:

                if debug_bool:
                    LOGGER.debug(
                        "No matching continuation regime; using existing path."
                    )

                # If we fail to find similar conditions, default to the full simulated df from the last sim
                mc_data_df = full_mc_data_df[full_mc_data_df["timestamp"] >= ts].copy()
                break

    else:
        mc_data_df = mc_data_df[mc_data_df["timestamp"] >= ts].copy()

    return _correlated_path_returns(mc_data_df)


def _correlated_path_returns(data):
    """Preserve the research return definition and coefficient clipping.

    Return leader prices and follower-relative increments in row order.
    This mutates only the working frame passed by the simulation worker.
    """
    data["diff"] = (data["high"].shift(1) - data["high"]) / data["high"]
    data["diff"] = data["diff"].replace(0, 0.000001)
    data["beta"] = data["beta"].clip(lower=-0.5, upper=1.5)
    data["beta"] = data["beta"] + 1
    data["cor_move"] = data["diff"] * data["beta"]
    return np.array(data["high"]), np.array(data["cor_move"])


def batch_wrapper(
    batch,
    price_paths,
    kde_beta,
    mc_data_df,
    beta_df,
    returns_arr,
    ath,
    ts,
    max_ts,
    paths,
    steps,
    order,
    knn_ts_arr,
    knn_path_arr,
    knn_latest_arr,
    knn_last_arr,
    price_ts,
    diffs,
):
    """Run a batch and collect only paths with the full requested length."""
    count = 0
    all_arr_list = []
    beta_arr_list = []
    for i in batch:
        mc_new_path, kde_new_path = run_one_parallel_forced(
            price_paths[i],
            kde_beta[i],
            mc_data_df,
            beta_df,
            returns_arr,
            ath,
            ts,
            max_ts,
            paths,
            steps,
            order,
            knn_ts_arr,
            knn_path_arr,
            knn_latest_arr,
            knn_last_arr,
            price_ts,
            diffs,
        )

        if len(mc_new_path) == steps + 1:
            all_arr_list.append(mc_new_path)
            beta_arr_list.append(kde_new_path)
            count += 1

    return count, all_arr_list, beta_arr_list
