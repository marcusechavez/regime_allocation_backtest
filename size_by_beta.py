"""Rank two-asset allocations and convert simulated relative returns to prices."""

import numpy as np
import pandas as pd


def get_size_by_lgbm(
    leader_df, follower_df, min_allocation=0, max_allocation=0.5, step=0.05
):
    """Rank candidate leader weights by LightGBM-predicted terminal return.

    Each candidate is paired with its complementary follower weight.
    Training targets are simulated portfolio returns, not realized returns.
    """
    allocation_list = np.arange(min_allocation, max_allocation + step / 2, step)

    # Leader frac, follower frac, net returns
    leader_fracs, follower_fracs, net_returns = get_size_arrays(
        leader_df, follower_df, allocation_list
    )

    features = pd.DataFrame(
        {"leader_frac": leader_fracs, "follower_frac": follower_fracs}
    )
    targets = net_returns

    from lightgbm import LGBMRegressor

    model = LGBMRegressor(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=6,
        objective="regression",
        random_state=42,
        force_row_wise=True,
        verbosity=-1,
    )
    model.fit(features, targets)

    # Evaluate all candidate allocations
    candidates = np.column_stack((allocation_list, 1 - allocation_list))

    predictions = model.predict(pd.DataFrame(candidates, columns=features.columns))

    results_df = pd.DataFrame(
        {
            "leader_frac": candidates[:, 0],
            "follower_frac": candidates[:, 1],
            "pred_return": predictions,
        }
    )

    results_df = results_df.sort_values("pred_return", ascending=False).reset_index(
        drop=True
    )

    return results_df


def get_size_arrays(leader_df, follower_df, allocation_list):
    """Flatten paired-path terminal returns across a candidate allocation grid."""
    leader_returns, follower_returns = _paired_terminal_returns(leader_df, follower_df)

    # Broadcast allocation grid across MC paths
    allocation_count = len(allocation_list)
    path_count = len(leader_returns)

    leader_fracs = np.broadcast_to(
        allocation_list[:, None], (allocation_count, path_count)
    )

    follower_fracs = 1 - leader_fracs

    net_returns = (
        leader_fracs * leader_returns[None, :]
        + follower_fracs * follower_returns[None, :]
    )

    return (leader_fracs.ravel(), follower_fracs.ravel(), net_returns.ravel())


def beta_to_prices(beta_df, start_price):
    """Compound relative-return paths into prices without changing the input.

    The first row is an initial observation, so its increment is zeroed.
    """
    beta_df = beta_df.copy()

    cols = beta_df.columns.drop("ts")

    # First row is the starting price, not a return
    beta_df.loc[0, cols] = 0

    # Compound percentage changes into price paths
    beta_df[cols] = (1 + beta_df[cols]).cumprod() * start_price

    return beta_df


def _paired_terminal_returns(leader_df, follower_df):
    """Extract matched-path terminal returns in the leader's column order."""
    columns = leader_df.columns.drop("ts")
    leader = leader_df[columns].to_numpy(dtype=float)
    follower = follower_df[columns].to_numpy(dtype=float)
    return leader[-1] / leader[0] - 1, follower[-1] / follower[0] - 1
