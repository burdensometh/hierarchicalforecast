"""Tests for rolling-origin selection and asymmetric stockout costs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hierarchicalforecast.methods import BottomUp, MinTrace
from hierarchicalforecast.utils import aggregate

from experiments.rolling_selection import (
    AsymmetricStockoutCost,
    Candidate,
    common_support_score,
    rolling_holdout_selection,
)


@pytest.fixture(scope="module")
def hierarchy():
    rng = np.random.default_rng(11)
    n_periods = 48
    dates = pd.date_range("2022-01-01", periods=n_periods, freq="MS")
    rows = []
    for country, state, region, base in [
        ("AU", "NSW", "Sydney", 100.0),
        ("AU", "NSW", "Newcastle", 50.0),
        ("AU", "VIC", "Melbourne", 80.0),
        ("AU", "VIC", "Geelong", 40.0),
    ]:
        values = base + rng.normal(0, 4, n_periods)
        for ds, y in zip(dates, values):
            rows.append(
                {"Country": country, "State": state, "Region": region,
                 "ds": ds, "y": max(y, 1.0)}
            )
    Y_df, S_df, tags = aggregate(
        pd.DataFrame(rows),
        [["Country"], ["Country", "State"], ["Country", "State", "Region"]],
    )
    unique_ids = Y_df["unique_id"].unique()
    pivot = Y_df.pivot(index="unique_id", columns="ds", values="y").loc[unique_ids]
    return {
        "Y": pivot.to_numpy(dtype=np.float64),
        "ds": pivot.columns.to_numpy(),
        "S": S_df.drop(columns=["unique_id"]).to_numpy(dtype=np.float64),
        "tags": tags,
        "labels": list(unique_ids),
    }


def constant_forecaster(value):
    def forecast_fn(train, horizon):
        y_hat = np.full((train.shape[0], horizon), value, dtype=np.float64)
        return y_hat, train.copy()

    return forecast_fn


def test_asymmetric_cost_matches_pinball():
    # Symmetric unit costs reduce to scaled absolute error.
    cost = AsymmetricStockoutCost(c_under=2.0, c_over=2.0)
    y = np.array([[10.0, 20.0]])
    y_hat = np.array([[12.0, 15.0]])
    expected = 2.0 * (2.0 + 5.0)
    assert cost.monetary_total(y, y_hat) == pytest.approx(expected)

    # Under-forecast is weighted by c_under, over by c_over.
    cost2 = AsymmetricStockoutCost(c_under=4.0, c_over=1.0)
    # under: 20 - 15 = 5 units -> 20; over: 12 - 10 = 2 units -> 2
    assert cost2.monetary_total(y, y_hat) == pytest.approx(22.0)
    assert cost2.stockout_component(y, y_hat) == pytest.approx(20.0)
    assert cost2.quantile_level(1) == pytest.approx([0.8])


def test_score_hierarchy_weighting():
    # 2 series, one level, distinct unit costs.
    y = np.array([[10.0], [10.0]])
    y_hat = np.array([[8.0], [10.0]])  # series 0 under by 2, series 1 exact
    cost = AsymmetricStockoutCost(c_under=np.array([3.0, 1.0]), c_over=1.0)
    tags = {"lvl": np.array([0, 1])}
    # per-series means: 6 and 0 -> mean 3
    assert cost.score(y, y_hat, tags=tags) == pytest.approx(3.0)
    # With series weights concentrated on the exact series, score is 0.
    weighted = AsymmetricStockoutCost(
        c_under=np.array([3.0, 1.0]), c_over=1.0,
        series_weights=np.array([0.0, 1.0]),
    )
    assert weighted.score(y, y_hat, tags=tags) == pytest.approx(0.0)


def test_invalid_costs_rejected():
    with pytest.raises(ValueError):
        AsymmetricStockoutCost(c_under=-1.0, c_over=1.0).per_cell(
            np.zeros((1, 1)), np.zeros((1, 1))
        )
    with pytest.raises(ValueError):
        AsymmetricStockoutCost(c_under=0.0, c_over=0.0).quantile_level(1)


def test_rolling_selection_runs_and_is_leakage_free(hierarchy):
    seen_prefixes = []

    def spying_forecaster(train, horizon):
        seen_prefixes.append(train.shape[1])
        return constant_forecaster(train[:, -1:].mean())(train, horizon)

    candidates = [
        Candidate("bu", lambda: BottomUp(), spying_forecaster),
        Candidate("mint_ols", lambda: MinTrace(method="ols"), spying_forecaster),
    ]
    n_time = hierarchy["Y"].shape[1]
    cutoffs = [30, 34, 38]
    result = rolling_holdout_selection(
        Y=hierarchy["Y"],
        ds=hierarchy["ds"],
        S=hierarchy["S"],
        tags=hierarchy["tags"],
        series_labels=hierarchy["labels"],
        candidates=candidates,
        cost_fn=AsymmetricStockoutCost(c_under=3.0, c_over=1.0),
        cutoffs=cutoffs,
        horizon=4,
        selection_rule="per_cutoff",
    )
    assert len(result.records) == 3
    # Every fit -- evaluation fits plus the winner's publication refit --
    # happens only on prefixes ending at an actual cutoff, never on the future.
    legal = {t + 1 for t in cutoffs}
    assert set(seen_prefixes) <= legal
    assert legal <= set(seen_prefixes)
    # Every published forecast covers all series and is finite.
    for rec in result.records:
        assert rec.published.shape == (hierarchy["Y"].shape[0],)
        assert np.isfinite(rec.published).all()
        assert rec.winner in {"bu", "mint_ols"}


def test_label_tags_require_series_labels(hierarchy):
    candidates = [Candidate("bu", BottomUp, constant_forecaster(50.0))]
    with pytest.raises(ValueError, match="series_labels"):
        rolling_holdout_selection(
            Y=hierarchy["Y"], ds=hierarchy["ds"], S=hierarchy["S"],
            tags=hierarchy["tags"], candidates=candidates,
            cost_fn=AsymmetricStockoutCost(1.0, 1.0),
            cutoffs=[30], horizon=4,
        )


def test_stockout_budget_excludes_breaching_candidate(hierarchy):
    # A low forecaster under-predicts -> stockout; a high one never stocks out.
    low = Candidate("low", lambda: BottomUp(), constant_forecaster(1.0))
    high = Candidate("high", lambda: BottomUp(), constant_forecaster(1_000.0))
    result = rolling_holdout_selection(
        Y=hierarchy["Y"], ds=hierarchy["ds"], S=hierarchy["S"],
        tags=hierarchy["tags"], series_labels=hierarchy["labels"],
        candidates=[low, high],
        cost_fn=AsymmetricStockoutCost(c_under=10.0, c_over=0.01),
        cutoffs=[30, 34], horizon=4,
        stockout_budget=1.0,
    )
    # The low candidate's stockout cost dwarfs the budget at every cutoff.
    for rec in result.records:
        assert rec.stockout_cost["low"] > 1.0
        assert rec.winner == "high"


def test_min_switch_gain_keeps_incumbent(hierarchy):
    # Two near-identical candidates: tiny differences must not cause churn.
    candidates = [
        Candidate("a", lambda: BottomUp(), constant_forecaster(60.0)),
        Candidate("b", lambda: MinTrace(method="ols"), constant_forecaster(60.5)),
    ]
    result = rolling_holdout_selection(
        Y=hierarchy["Y"], ds=hierarchy["ds"], S=hierarchy["S"],
        tags=hierarchy["tags"], series_labels=hierarchy["labels"],
        candidates=candidates,
        cost_fn=AsymmetricStockoutCost(c_under=3.0, c_over=1.0),
        cutoffs=[30, 34, 38, 42], horizon=3,
        selection_rule="cumulative", min_switch_gain=0.5, incumbent="a",
    )
    assert all(rec.winner == "a" for rec in result.records)


def test_common_support_intersection(hierarchy):
    good = Candidate("good", lambda: BottomUp(), constant_forecaster(60.0))
    # The late entrant only exists from the second cutoff onward.
    late = Candidate(
        "late", lambda: MinTrace(method="ols"), constant_forecaster(62.0),
        entry_cutoff=1,
    )
    result = rolling_holdout_selection(
        Y=hierarchy["Y"], ds=hierarchy["ds"], S=hierarchy["S"],
        tags=hierarchy["tags"], series_labels=hierarchy["labels"],
        candidates=[good, late],
        cost_fn=AsymmetricStockoutCost(c_under=3.0, c_over=1.0),
        cutoffs=[30, 34, 38], horizon=4,
    )
    assert "late" not in result.records[0].support
    assert "late" in result.records[1].support

    a, b, n = common_support_score(result.records, "good", "late")
    # Only the last two cutoffs (2 * 4 steps * n_series cells) are compared.
    n_series = hierarchy["Y"].shape[0]
    assert n == 2 * 4 * n_series
    assert np.isfinite(a) and np.isfinite(b)


def test_entry_cutoff_validation(hierarchy):
    candidates = [
        Candidate("bu", BottomUp, constant_forecaster(50.0), entry_cutoff=9),
    ]
    with pytest.raises(ValueError, match="entry_cutoff"):
        rolling_holdout_selection(
            Y=hierarchy["Y"], ds=hierarchy["ds"], S=hierarchy["S"],
            tags=hierarchy["tags"], series_labels=hierarchy["labels"],
            candidates=candidates,
            cost_fn=AsymmetricStockoutCost(1.0, 1.0),
            cutoffs=[30, 34], horizon=4,
        )


def test_bad_cutoff_rejected(hierarchy):
    candidates = [Candidate("bu", BottomUp, constant_forecaster(50.0))]
    n_time = hierarchy["Y"].shape[1]
    with pytest.raises(ValueError):
        rolling_holdout_selection(
            Y=hierarchy["Y"], ds=hierarchy["ds"], S=hierarchy["S"],
            tags=hierarchy["tags"], series_labels=hierarchy["labels"],
            candidates=candidates,
            cost_fn=AsymmetricStockoutCost(1.0, 1.0),
            cutoffs=[n_time - 1], horizon=4,
        )
