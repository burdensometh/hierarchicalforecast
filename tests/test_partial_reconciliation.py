"""Tests for reconciliation of partially delivered base forecasts.

Channel forecasts often arrive with only a national total and a few stores
forecasted. Missing values must never be read as zero, and reconciliation must
be performed per date on the rows that actually carry a forecast.
"""

import numpy as np
import pandas as pd
import pytest

from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import PartialMinTrace, partial_reconcile
from hierarchicalforecast.utils import aggregate


def _two_state_two_store_data(n_dates=10):
    rng = np.random.default_rng(0)
    rows = []
    for state in ["A", "B"]:
        for store in ["1", "2"]:
            base = rng.uniform(10, 20)
            for ds in range(n_dates):
                rows.append(
                    {
                        "State": state,
                        "Store": store,
                        "ds": ds,
                        "y": base + rng.normal(0, 0.5),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def partial_hierarchy():
    df = _two_state_two_store_data()
    Y_df, S_df, tags = aggregate(df, [["State"], ["State", "Store"]])
    S = S_df.drop(columns=["unique_id"]).to_numpy(dtype=np.float64)
    ids = S_df["unique_id"].tolist()
    return {"S": S, "ids": ids, "tags": tags, "Y_df": Y_df, "S_df": S_df}


def _coherent_column(S, bottom):
    return S @ np.asarray(bottom, dtype=np.float64)


# ---------------------------------------------------------------------------
# Unit tests against the reconciler / helper.
# ---------------------------------------------------------------------------


def test_complete_delivery_recovers_coherent_forecasts(partial_hierarchy):
    S = partial_hierarchy["S"]
    bottoms = np.array([[1.0, 5.0, 2.0, 7.0], [3.0, 1.0, 4.0, 2.0]])
    y_hat = np.column_stack([_coherent_column(S, b) for b in bottoms])

    out = partial_reconcile(S, y_hat)

    assert out["publishable_dates"] == [0, 1]
    assert out["unpublishable_dates"] == []
    np.testing.assert_allclose(out["mean"], y_hat)


def test_missing_single_leaf_is_recovered_from_aggregates(partial_hierarchy):
    """A missing leaf can be filled using its parent and siblings."""
    S = partial_hierarchy["S"]
    ids = partial_hierarchy["ids"]
    bottom = np.array([1.0, 5.0, 2.0, 7.0])
    y_complete = _coherent_column(S, bottom)

    mask = np.ones(S.shape[0], dtype=bool)
    mask[ids.index("A/2")] = False  # drop one leaf
    y_partial = y_complete.copy()
    y_partial[~mask] = np.nan

    out = partial_reconcile(S, y_partial[:, None])

    assert out["publishable_dates"] == [0]
    np.testing.assert_allclose(out["mean"][:, 0], y_complete, atol=1e-10)


def test_missing_values_are_not_treated_as_zero(partial_hierarchy):
    """A missing leaf must be inferred, not assumed to be 0."""
    S = partial_hierarchy["S"]
    ids = partial_hierarchy["ids"]
    bottom = np.array([2.0, 8.0, 4.0, 6.0])
    y_complete = _coherent_column(S, bottom)

    y_partial = y_complete.copy()
    y_partial[ids.index("B/1")] = np.nan  # truly nonzero, but missing

    out = partial_reconcile(S, y_partial[:, None])

    filled = out["mean"][ids.index("B/1"), 0]
    assert filled == pytest.approx(bottom[2])
    assert filled != 0.0


def test_only_national_total_is_not_identified(partial_hierarchy):
    """Total-only delivery has many leaf solutions: not publishable."""
    S = partial_hierarchy["S"]
    ids = partial_hierarchy["ids"]
    y_partial = np.full(S.shape[0], np.nan)
    y_partial[ids.index("A")] = 42.0  # only one aggregate delivered

    out = partial_reconcile(S, y_partial[:, None], allow_unidentified=True)

    assert out["publishable_dates"] == []
    assert out["unpublishable_dates"] == [0]
    assert np.isnan(out["mean"]).all()


def test_total_plus_few_stores_not_identified(partial_hierarchy):
    """The motivating case: national total and a couple of stores."""
    S = partial_hierarchy["S"]
    ids = partial_hierarchy["ids"]
    y_partial = np.full(S.shape[0], np.nan)
    y_partial[ids.index("A")] = 30.0
    y_partial[ids.index("A/1")] = 12.0
    y_partial[ids.index("B/2")] = 5.0

    out = partial_reconcile(S, y_partial[:, None], allow_unidentified=True)
    assert out["unpublishable_dates"] == [0]
    assert np.isnan(out["mean"][:, 0]).all()


def test_unidentified_raises_when_disallowed(partial_hierarchy):
    S = partial_hierarchy["S"]
    ids = partial_hierarchy["ids"]
    y_partial = np.full((S.shape[0], 1), np.nan)
    y_partial[ids.index("A")] = 10.0

    with pytest.raises(ValueError, match="do not determine the bottom-level"):
        partial_reconcile(S, y_partial, allow_unidentified=False)


def test_publishable_and_unpublishable_dates_are_separated(partial_hierarchy):
    """Each date is judged on its own delivered rows."""
    S = partial_hierarchy["S"]
    ids = partial_hierarchy["ids"]
    bottom = np.array([1.0, 2.0, 3.0, 4.0])
    complete = _coherent_column(S, bottom)

    y_hat = np.full((S.shape[0], 3), np.nan)
    y_hat[:, 0] = complete  # fully delivered
    y_hat[ids.index("A"), 1] = 9.0  # only the national total -> not identified
    y_hat[:, 2] = complete * 3  # fully delivered

    out = partial_reconcile(S, y_hat, allow_unidentified=True)

    assert out["publishable_dates"] == [0, 2]
    assert out["unpublishable_dates"] == [1]
    np.testing.assert_allclose(out["mean"][:, 0], complete)
    np.testing.assert_allclose(out["mean"][:, 2], complete * 3)
    assert np.isnan(out["mean"][:, 1]).all()


def test_ridge_stabiliser_is_optional(partial_hierarchy):
    S = partial_hierarchy["S"]
    bottom = np.array([1.0, 2.0, 3.0, 4.0])
    y_hat = _coherent_column(S, bottom)[:, None]

    plain = partial_reconcile(S, y_hat)["mean"]
    ridge = partial_reconcile(S, y_hat, ridge=1e-6)["mean"]

    np.testing.assert_allclose(plain, y_hat)
    # A tiny ridge perturbs the recovered leaves only slightly.
    np.testing.assert_allclose(ridge, y_hat, atol=1e-3)


def test_partial_min_trace_reports_dates(partial_hierarchy):
    S = partial_hierarchy["S"]
    ids = partial_hierarchy["ids"]
    bottom = np.array([1.0, 2.0, 3.0, 4.0])
    y_hat = np.full((S.shape[0], 2), np.nan)
    y_hat[:, 0] = _coherent_column(S, bottom)
    y_hat[ids.index("B")] = np.nan
    y_hat[ids.index("B"), 1] = 11.0

    rec = PartialMinTrace(allow_unidentified=True)
    res = rec.fit_predict(S=S, y_hat=y_hat)

    assert rec.publishable_dates == [0]
    assert rec.unpublishable_dates == [1]
    assert res["mean"].shape == y_hat.shape


def test_partial_min_trace_rejects_intervals(partial_hierarchy):
    S = partial_hierarchy["S"]
    y_hat = np.zeros((S.shape[0], 1))
    rec = PartialMinTrace()
    with pytest.raises(NotImplementedError, match="point forecasts"):
        rec.fit_predict(S=S, y_hat=y_hat, level=[80])


# ---------------------------------------------------------------------------
# Integration through `HierarchicalReconciliation.reconcile`.
# ---------------------------------------------------------------------------


def _sparse_delivery(Y_hat_df, S_df, drop):
    """Drop `(unique_id, ds)` rows selected by the predicate `drop`."""
    mask = drop(Y_hat_df)
    return Y_hat_df.loc[~mask].copy()


def test_reconcile_accepts_partial_dataframe(partial_hierarchy):
    Y_df = partial_hierarchy["Y_df"]
    S_df = partial_hierarchy["S_df"]
    tags = partial_hierarchy["tags"]

    Y_df = Y_df.copy()
    Y_df["m"] = Y_df["y"]
    Y_hat_df = Y_df.groupby("unique_id").tail(3).copy()
    Y_hat_df["m"] = Y_hat_df["m"] + np.random.default_rng(1).uniform(
        -0.5, 0.5, len(Y_hat_df)
    )

    last_ds = Y_hat_df["ds"].max()
    # On the final date deliver only the state aggregates; leaves are absent.
    partial = _sparse_delivery(
        Y_hat_df,
        S_df,
        lambda d: (d["ds"] == last_ds) & (d["unique_id"].str.count("/") == 1),
    )

    with pytest.warns(UserWarning, match="could not publish"):
        hrec = HierarchicalReconciliation([PartialMinTrace(allow_unidentified=True)])
        out = hrec.reconcile(
            Y_hat_df=partial[["unique_id", "ds", "m"]], S_df=S_df, tags=tags
        )

    rec_name = [c for c in out.columns if c not in ["unique_id", "ds", "m"]]
    assert len(rec_name) == 1
    rec_col = rec_name[0]

    # Series that were entirely omitted still appear on the full grid.
    assert set(out["unique_id"]) == set(S_df["unique_id"])

    # The unpublishable date is NaN everywhere, not zero.
    unpublished = out[out["ds"] == last_ds]
    assert unpublished[rec_col].isna().all()

    # Published dates are coherent: aggregates equal the sum of their leaves.
    S = S_df.drop(columns=["unique_id"]).to_numpy(dtype=np.float64)
    ids = S_df["unique_id"].tolist()
    published_dates = [d for d in sorted(out["ds"].unique()) if d != last_ds]
    for ds in published_dates:
        rows = out[out["ds"] == ds].set_index("unique_id").loc[ids, rec_col]
        vec = rows.to_numpy(dtype=np.float64)
        np.testing.assert_allclose(vec, S @ vec[-4:], atol=1e-8)


def test_reconcile_partial_reports_unpublishable_dates(partial_hierarchy):
    Y_df = partial_hierarchy["Y_df"]
    S_df = partial_hierarchy["S_df"]
    tags = partial_hierarchy["tags"]

    Y_df = Y_df.copy()
    Y_df["m"] = Y_df["y"]
    Y_hat_df = Y_df.groupby("unique_id").tail(2).copy()
    first_ds = Y_hat_df["ds"].min()
    partial = _sparse_delivery(
        Y_hat_df,
        S_df,
        lambda d: (d["ds"] == first_ds) & (d["unique_id"].str.count("/") == 1),
    )

    with pytest.warns(UserWarning):
        hrec = HierarchicalReconciliation([PartialMinTrace(allow_unidentified=True)])
        hrec.reconcile(Y_hat_df=partial[["unique_id", "ds", "m"]], S_df=S_df, tags=tags)

    rec_name = next(iter(hrec.unpublishable_dates))
    assert hrec.unpublishable_dates[rec_name] == [first_ds]
    assert first_ds not in hrec.publishable_dates[rec_name]


def test_reconcile_rejects_mixing_partial_and_complete(partial_hierarchy):
    Y_df = partial_hierarchy["Y_df"]
    S_df = partial_hierarchy["S_df"]
    tags = partial_hierarchy["tags"]
    from hierarchicalforecast.methods import BottomUp

    Y_df = Y_df.copy()
    Y_df["m"] = Y_df["y"]
    Y_hat_df = Y_df.groupby("unique_id").tail(2).copy()

    hrec = HierarchicalReconciliation([PartialMinTrace(), BottomUp()])
    with pytest.raises(NotImplementedError, match="Cannot mix partial"):
        hrec.reconcile(
            Y_hat_df=Y_hat_df[["unique_id", "ds", "m"]], S_df=S_df, tags=tags
        )


def test_reconcile_rejects_unknown_series_in_partial(partial_hierarchy):
    Y_df = partial_hierarchy["Y_df"]
    S_df = partial_hierarchy["S_df"]
    tags = partial_hierarchy["tags"]

    Y_df = Y_df.copy()
    Y_df["m"] = Y_df["y"]
    Y_hat_df = Y_df.groupby("unique_id").tail(2).copy()
    Y_hat_df.loc[Y_hat_df.index[0], "unique_id"] = "Unknown/Store"

    hrec = HierarchicalReconciliation([PartialMinTrace(allow_unidentified=True)])
    with pytest.raises(ValueError, match="not in S_df"):
        hrec.reconcile(
            Y_hat_df=Y_hat_df[["unique_id", "ds", "m"]], S_df=S_df, tags=tags
        )


def test_reconcile_rejects_duplicate_grid_points(partial_hierarchy):
    S_df = partial_hierarchy["S_df"]
    tags = partial_hierarchy["tags"]
    Y_df = partial_hierarchy["Y_df"].copy()
    Y_df["m"] = Y_df["y"]
    Y_hat_df = Y_df.groupby("unique_id").tail(2)[["unique_id", "ds", "m"]].copy()
    duplicated = pd.concat([Y_hat_df, Y_hat_df.iloc[[0]]], ignore_index=True)

    hrec = HierarchicalReconciliation([PartialMinTrace()])
    with pytest.raises(ValueError, match="repeated"):
        hrec.reconcile(Y_hat_df=duplicated, S_df=S_df, tags=tags)


def test_reconcile_rejects_partial_with_intervals(partial_hierarchy):
    Y_df = partial_hierarchy["Y_df"].copy()
    S_df = partial_hierarchy["S_df"]
    tags = partial_hierarchy["tags"]
    Y_df["m"] = Y_df["y"]
    Y_hat_df = Y_df.groupby("unique_id").tail(2)[["unique_id", "ds", "m"]].copy()

    hrec = HierarchicalReconciliation([PartialMinTrace()])
    with pytest.raises(NotImplementedError, match="Prediction intervals"):
        hrec.reconcile(Y_hat_df=Y_hat_df, S_df=S_df, tags=tags, level=[80])


def test_reconcile_rejects_partial_with_y_df(partial_hierarchy):
    Y_df = partial_hierarchy["Y_df"].copy()
    S_df = partial_hierarchy["S_df"]
    tags = partial_hierarchy["tags"]
    Y_df["m"] = Y_df["y"]
    Y_hat_df = Y_df.groupby("unique_id").tail(2)[["unique_id", "ds", "m"]].copy()
    Y_train = Y_df.drop(Y_hat_df.index)

    hrec = HierarchicalReconciliation([PartialMinTrace()])
    with pytest.raises(NotImplementedError, match="together with `Y_df`"):
        hrec.reconcile(Y_hat_df=Y_hat_df, S_df=S_df, tags=tags, Y_df=Y_train)


def test_reconcile_partial_with_smatrix(partial_hierarchy):
    """Partial forecasts also work with a sparse `SMatrix` summing matrix."""
    Y_df = partial_hierarchy["Y_df"].copy()
    tags = partial_hierarchy["tags"]
    Y_df["m"] = Y_df["y"]
    Y_hat_df = Y_df.groupby("unique_id").tail(2).copy()
    _, S_matrix, _ = aggregate(
        _two_state_two_store_data(),
        [["State"], ["State", "Store"]],
        sparse_s=True,
    )

    last_ds = Y_hat_df["ds"].max()
    partial = _sparse_delivery(
        Y_hat_df,
        S_matrix,
        lambda d: (d["ds"] == last_ds) & (d["unique_id"].str.count("/") == 1),
    )

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hrec = HierarchicalReconciliation([PartialMinTrace()])
        out = hrec.reconcile(
            Y_hat_df=partial[["unique_id", "ds", "m"]], S_df=S_matrix, tags=tags
        )
    assert len(out) == S_matrix.sparse_shape[0] * Y_hat_df["ds"].nunique()
