"""End-to-end smoke test / runnable example for rolling holdout selection.

Run from the repository root with::

    python -m experiments.rolling_selection.demo

It builds a small synthetic hierarchy, deliberately makes under-forecasts
more expensive than over-forecasts, and runs the full rolling process with
three reconciliation candidates, one of which enters late to demonstrate the
common-support handling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hierarchicalforecast.methods import BottomUp, MinTrace, TopDown
from hierarchicalforecast.utils import aggregate

from .cost import AsymmetricStockoutCost
from .rolling import Candidate, common_support_score, rolling_holdout_selection


def build_hierarchy(n_periods: int = 60, seed: int = 7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n_periods, freq="MS")
    structure = {
        ("AU", "NSW", "Sydney"): 100.0,
        ("AU", "NSW", "Newcastle"): 55.0,
        ("AU", "VIC", "Melbourne"): 80.0,
        ("AU", "VIC", "Geelong"): 45.0,
    }
    rows = []
    for (country, state, region), base in structure.items():
        level = base + rng.normal(0, 6, n_periods).cumsum() * 0.15
        season = 8 * np.sin(np.linspace(0, 6 * np.pi, n_periods))
        noise = rng.normal(0, 5, n_periods)
        for ds, value in zip(dates, base + level - base + season + noise):
            rows.append(
                {"Country": country, "State": state, "Region": region,
                 "ds": ds, "y": max(value, 1.0)}
            )
    df = pd.DataFrame(rows)
    spec = [
        ["Country"],
        ["Country", "State"],
        ["Country", "State", "Region"],
    ]
    Y_df, S_df, tags = aggregate(df, spec)
    return Y_df, S_df, tags


def make_naive_forecaster(bias: float = 0.0, window: int = 6):
    """Rolling-mean base forecaster, refit fresh on every cutoff prefix.

    ``bias`` deliberately shifts forecasts down so that stockout behavior is
    visible; a real deployment would plug in statsforecast / mlforecast /
    neuralforecast pipelines here, refit on the same prefix.
    """

    def forecast_fn(train: np.ndarray, horizon: int):
        # Naive seasonal-ish level: trailing mean minus a downward bias.
        level = train[:, -window:].mean(axis=1, keepdims=True)
        drift = np.diff(train[:, -window:], axis=1).mean(axis=1, keepdims=True)
        steps = np.arange(1, horizon + 1)[None, :]
        y_hat = np.maximum(level + drift * steps - bias, 0.0)
        # Crude in-sample fitted values (trailing rolling mean) for MinTrace.
        y_hat_insample = train.copy()
        if train.shape[1] > window:
            csum = np.cumsum(np.pad(train, ((0, 0), (window, 0)), mode="edge"), axis=1)
            roll = (csum[:, window:] - csum[:, :-window]) / window
            y_hat_insample[:, : roll.shape[1]] = roll
        return y_hat.astype(np.float64), y_hat_insample.astype(np.float64)

    return forecast_fn


def main() -> None:
    Y_df, S_df, tags = build_hierarchy()
    unique_ids = Y_df["unique_id"].unique()
    n_series = len(unique_ids)
    n_time = Y_df["ds"].nunique()

    pivot = Y_df.pivot(index="unique_id", columns="ds", values="y").loc[unique_ids]
    Y = pivot.to_numpy(dtype=np.float64)
    ds = pivot.columns.to_numpy()
    # S_df carries a 'unique_id' label column plus one column per bottom series.
    S = S_df.drop(columns=["unique_id"]).to_numpy(dtype=np.float64)

    # Key region = the two city SKUs get a 4x unit stockout penalty.
    c_under = np.ones(n_series)
    key_regions = [i for i, uid in enumerate(unique_ids) if uid.endswith(("Sydney", "Melbourne"))]
    c_under[key_regions] = 4.0
    c_over = np.ones(n_series) * 0.5

    cost_fn = AsymmetricStockoutCost(c_under=c_under, c_over=c_over)
    print("Implied service-level quantiles:",
          np.round(cost_fn.quantile_level(n_series), 3))

    # aggregate() names tag levels by joined hierarchy columns:
    # ['Country', 'Country/State', 'Country/State/Region']
    level_weights = {lvl: w for lvl, w in zip(tags, [0.25, 0.35, 0.40])}

    base = make_naive_forecaster(bias=3.0)
    candidates = [
        Candidate("bottom_up", lambda: BottomUp(), base),
        Candidate(
            "mint_shrink",
            lambda: MinTrace(method="mint_shrink"),
            base,
        ),
        Candidate(
            "top_down_fp",
            lambda: TopDown(method="forecast_proportions"),
            base,
        ),
    ]

    horizon = 4
    cutoffs = list(range(30, n_time - horizon - 1, 4))
    result = rolling_holdout_selection(
        Y=Y,
        ds=ds,
        S=S,
        tags=tags,
        candidates=candidates,
        cost_fn=cost_fn,
        cutoffs=cutoffs,
        horizon=horizon,
        series_labels=list(unique_ids),
        level_weights=level_weights,
        selection_rule="cumulative",
        stockout_budget=None,
        min_switch_gain=0.02,
    )

    pd.set_option("display.width", 140)
    print("\nPer-cutoff weighted asymmetric cost:")
    print(result.leaderboard().to_string(index=False))
    print("\nWinners:")
    print(result.winners().to_string(index=False))
    print("\nCumulative common-support scores:")
    print(result.cumulative_paired_scores().to_string(index=False))

    a, b, n = common_support_score(result.records, "bottom_up", "mint_shrink")
    print(f"\nPaired common-support comparison bottom_up={a:.3f} vs "
          f"mint_shrink={b:.3f} over {n} (series, step) cells")

    # --- Late-entering candidate demonstration ---------------------------
    # A method whose data source only has history covering the second half of
    # the cutoffs. entry_cutoff makes the absence structural: it is simply not
    # fitted or scored before that point, rather than producing fake NaNs.
    late_base = make_naive_forecaster(bias=1.0)
    late_candidate = Candidate(
        "new_method_late",
        lambda: MinTrace(method="ols"),
        late_base,
        entry_cutoff=len(cutoffs) // 2,
    )
    late_result = rolling_holdout_selection(
        Y=Y, ds=ds, S=S, tags=tags,
        candidates=candidates + [late_candidate],
        cost_fn=cost_fn, cutoffs=cutoffs, horizon=horizon,
        series_labels=list(unique_ids),
        level_weights=level_weights, selection_rule="cumulative",
    )
    early = late_result.records[: len(cutoffs) // 2]
    assert all("new_method_late" not in rec.support for rec in early)
    a, b, n = common_support_score(
        late_result.records, "new_method_late", "mint_shrink"
    )
    print("\nLate entrant vs incumbent on INTERSECTED support only:")
    print(f"  new_method_late={a:.3f}  mint_shrink={b:.3f}  comparable_cells={n}")
    assert n > 0, "Expected overlapping support in later cutoffs"
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
