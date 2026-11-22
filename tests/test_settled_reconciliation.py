"""Tests for point reconciliation with settled (closed) leaf values.

Settled values pin specific bottom-level series at specific timestamps. The
remaining free leaves must stay as close as possible to the base forecasts,
and every level of the hierarchy has to re-aggregate coherently.
"""

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import pytest

from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import (
    BottomUp,
    ConstrainedReconciliation,
    MinTrace,
)
from hierarchicalforecast.utils import aggregate

REC_COL = "y_model/ConstrainedReconciliation"


def _strict_hierarchy_fixture(horizon: int = 3):
    """Strict hierarchy: Country -> State -> Region (4 leaves, 7 series)."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=horizon + 3, freq="MS")
    structure = {
        ("NSW", "Sydney"): 100.0,
        ("NSW", "Newcastle"): 50.0,
        ("VIC", "Melbourne"): 80.0,
        ("VIC", "Geelong"): 40.0,
    }
    rows = []
    for (state, region), base in structure.items():
        for ds in dates:
            rows.append(
                {
                    "Country": "AU",
                    "State": state,
                    "Region": region,
                    "ds": ds,
                    "y": base + rng.normal(scale=5),
                }
            )
    df = pd.DataFrame(rows)
    spec = [
        ["Country"],
        ["Country", "State"],
        ["Country", "State", "Region"],
    ]
    Y_df, S_df, tags = aggregate(df, spec)
    Y_df["y_model"] = Y_df["y"] + rng.normal(scale=3, size=len(Y_df))
    Y_hat_df = Y_df.groupby("unique_id").tail(horizon).reset_index(drop=True)
    return Y_hat_df, S_df, tags


def _grouped_hierarchy_fixture(horizon: int = 4):
    """Grouped hierarchy with cross-cutting Region / Category groupings."""
    rng = np.random.default_rng(7)
    rows = []
    for region in ["R1", "R2"]:
        for category in ["C1", "C2"]:
            for ds in range(horizon + 4):
                rows.append(
                    {
                        "Country": "AU",
                        "Region": region,
                        "Category": category,
                        "ds": ds,
                        "y": rng.uniform(20, 90),
                    }
                )
    df = pd.DataFrame(rows)
    spec = [
        ["Country"],
        ["Country", "Region"],
        ["Country", "Category"],
        ["Country", "Region", "Category"],
    ]
    Y_df, S_df, tags = aggregate(df, spec)
    Y_df["y_model"] = Y_df["y"] + rng.normal(scale=2, size=len(Y_df))
    Y_hat_df = Y_df.groupby("unique_id").tail(horizon).reset_index(drop=True)
    return Y_hat_df, S_df, tags


def _s_matrix(S_df):
    return S_df.drop(columns=["unique_id"]).to_numpy(dtype=np.float64)


def _pivot(result, S_df, column):
    return (
        result.pivot(index="unique_id", columns="ds", values=column)
        .loc[S_df["unique_id"]]
        .to_numpy()
    )


def _leaf_ids(S_df):
    n_bottom = _s_matrix(S_df).shape[1]
    return list(S_df["unique_id"])[-n_bottom:]


@pytest.fixture
def strict_case():
    return _strict_hierarchy_fixture()


@pytest.fixture
def grouped_case():
    return _grouped_hierarchy_fixture()


def test_settled_leaves_are_pinned_and_output_is_coherent(strict_case):
    """Fixed leaves keep exact settled values; the whole table re-aggregates."""
    Y_hat_df, S_df, tags = strict_case
    S = _s_matrix(S_df)
    n_bottom = S.shape[1]
    leaf_ids = _leaf_ids(S_df)
    horizon_dates = sorted(Y_hat_df["ds"].unique())

    settled_df = pd.DataFrame(
        [
            {"unique_id": leaf_ids[0], "ds": horizon_dates[0], "y": 123.45},
            {"unique_id": leaf_ids[2], "ds": horizon_dates[0], "y": 77.7},
            {"unique_id": leaf_ids[0], "ds": horizon_dates[1], "y": 110.0},
        ]
    )

    hrec = HierarchicalReconciliation([ConstrainedReconciliation()])
    result = hrec.reconcile(
        Y_hat_df=Y_hat_df, S_df=S_df, tags=tags, settled_df=settled_df
    )

    # Coherence: every series equals S applied to the reconciled leaves.
    reconciled = _pivot(result, S_df, REC_COL)
    np.testing.assert_allclose(reconciled, S @ reconciled[-n_bottom:], atol=1e-10)

    # Settled values are preserved exactly at the leaf level.
    pinned = result.pivot(index="unique_id", columns="ds", values=REC_COL)
    for record in settled_df.itertuples():
        assert pinned.loc[record.unique_id, record.ds] == pytest.approx(
            record.y, abs=1e-12
        )


def test_free_leaves_solve_the_constrained_least_squares(strict_case):
    """Free leaves are the analytical closest-to-base constrained optimum."""
    Y_hat_df, S_df, tags = strict_case
    S = _s_matrix(S_df)
    n_bottom = S.shape[1]
    leaf_ids = _leaf_ids(S_df)
    horizon_dates = sorted(Y_hat_df["ds"].unique())

    settled_df = pd.DataFrame(
        [
            {"unique_id": leaf_ids[1], "ds": horizon_dates[0], "y": 60.0},
            {"unique_id": leaf_ids[3], "ds": horizon_dates[0], "y": 30.0},
        ]
    )
    hrec = HierarchicalReconciliation([ConstrainedReconciliation()])
    result = hrec.reconcile(
        Y_hat_df=Y_hat_df, S_df=S_df, tags=tags, settled_df=settled_df
    )

    base = _pivot(Y_hat_df, S_df, "y_model")
    reconciled = _pivot(result, S_df, REC_COL)
    bottom = reconciled[-n_bottom:]
    fixed_mask = np.zeros(n_bottom, dtype=bool)
    fixed_mask[[1, 3]] = True
    free_idx = np.flatnonzero(~fixed_mask)
    fixed_values = np.array([60.0, 30.0])

    residual = base[:, 0] - S[:, fixed_mask] @ fixed_values
    expected_free = np.linalg.solve(
        S[:, free_idx].T @ S[:, free_idx], S[:, free_idx].T @ residual
    )
    np.testing.assert_allclose(bottom[free_idx, 0], expected_free, atol=1e-10)
    np.testing.assert_allclose(bottom[fixed_mask, 0], fixed_values, atol=1e-12)


def test_fully_settled_preserves_all_actuals(strict_case):
    """When every leaf is settled, all actuals survive and upper levels sum."""
    Y_hat_df, S_df, tags = strict_case
    S = _s_matrix(S_df)
    n_bottom = S.shape[1]
    leaf_ids = _leaf_ids(S_df)
    horizon_dates = sorted(Y_hat_df["ds"].unique())
    rng = np.random.default_rng(11)

    settled_records = [
        {"unique_id": uid, "ds": ds, "y": float(value)}
        for uid in leaf_ids
        for ds, value in zip(
            horizon_dates, rng.uniform(10, 90, len(horizon_dates)), strict=False
        )
    ]
    settled_df = pd.DataFrame(settled_records)

    hrec = HierarchicalReconciliation([ConstrainedReconciliation()])
    result = hrec.reconcile(
        Y_hat_df=Y_hat_df, S_df=S_df, tags=tags, settled_df=settled_df
    )

    pinned = result.pivot(index="unique_id", columns="ds", values=REC_COL)
    for record in settled_df.itertuples():
        assert pinned.loc[record.unique_id, record.ds] == pytest.approx(
            record.y, abs=1e-12
        )
    reconciled = _pivot(result, S_df, REC_COL)
    np.testing.assert_allclose(reconciled, S @ reconciled[-n_bottom:], atol=1e-10)
    assert int(result["is_settled"].sum()) == n_bottom * len(horizon_dates)


def test_is_settled_marker_only_flags_settled_leaf_rows(strict_case):
    """The publication table marks exactly the settled leaf cells."""
    Y_hat_df, S_df, tags = strict_case
    leaf_ids = _leaf_ids(S_df)
    horizon_dates = sorted(Y_hat_df["ds"].unique())
    settled_df = pd.DataFrame(
        [
            {"unique_id": leaf_ids[0], "ds": horizon_dates[0], "y": 10.0},
            {"unique_id": leaf_ids[2], "ds": horizon_dates[2], "y": 20.0},
        ]
    )

    hrec = HierarchicalReconciliation([ConstrainedReconciliation()])
    result = hrec.reconcile(
        Y_hat_df=Y_hat_df, S_df=S_df, tags=tags, settled_df=settled_df
    )

    assert result["is_settled"].dtype == bool
    marker = result.pivot(index="unique_id", columns="ds", values="is_settled")
    aggregate_ids = [uid for uid in S_df["unique_id"] if uid not in set(leaf_ids)]
    assert not marker.loc[aggregate_ids].to_numpy().any()
    assert bool(marker.loc[leaf_ids[0], horizon_dates[0]]) is True
    assert bool(marker.loc[leaf_ids[2], horizon_dates[2]]) is True
    assert bool(marker.loc[leaf_ids[0], horizon_dates[1]]) is False
    assert int(result["is_settled"].sum()) == len(settled_df)


def test_no_settled_column_when_settled_df_absent(strict_case):
    """Without settled values the output schema is unchanged apart from models."""
    Y_hat_df, S_df, tags = strict_case
    hrec = HierarchicalReconciliation([MinTrace(method="ols")])
    result = hrec.reconcile(Y_hat_df=Y_hat_df, S_df=S_df, tags=tags)
    assert "is_settled" not in result.columns


def test_constrained_without_settled_raises(strict_case):
    """The constrained reconciler must receive settled values."""
    Y_hat_df, S_df, tags = strict_case
    hrec = HierarchicalReconciliation([ConstrainedReconciliation()])
    with pytest.raises(ValueError, match="requires `settled`"):
        hrec.reconcile(Y_hat_df=Y_hat_df, S_df=S_df, tags=tags)


def test_settled_keys_must_be_leaf_series(strict_case):
    """Aggregate-level and unknown keys are rejected with the offending key."""
    Y_hat_df, S_df, tags = strict_case
    first_ds = Y_hat_df["ds"].iloc[0]
    hrec = HierarchicalReconciliation([ConstrainedReconciliation()])

    aggregate_key = next(uid for uid in S_df["unique_id"] if uid.count("/") < 2)
    bad_aggregate = pd.DataFrame(
        [{"unique_id": aggregate_key, "ds": first_ds, "y": 1.0}]
    )
    with pytest.raises(ValueError, match="not bottom-level"):
        hrec.reconcile(
            Y_hat_df=Y_hat_df, S_df=S_df, tags=tags, settled_df=bad_aggregate
        )

    bad_unknown = pd.DataFrame(
        [{"unique_id": "AU/NSW/Nowhere", "ds": first_ds, "y": 1.0}]
    )
    with pytest.raises(ValueError, match="not bottom-level"):
        hrec.reconcile(Y_hat_df=Y_hat_df, S_df=S_df, tags=tags, settled_df=bad_unknown)


def test_settled_timestamps_must_match_forecast_grid(strict_case):
    """Unknown timestamps are reported instead of silently dropped."""
    Y_hat_df, S_df, tags = strict_case
    leaf_ids = _leaf_ids(S_df)
    bad_ds = pd.DataFrame(
        [
            {
                "unique_id": leaf_ids[0],
                "ds": pd.Timestamp("1999-01-01"),
                "y": 1.0,
            }
        ]
    )
    hrec = HierarchicalReconciliation([ConstrainedReconciliation()])
    with pytest.raises(ValueError, match="timestamps not present"):
        hrec.reconcile(Y_hat_df=Y_hat_df, S_df=S_df, tags=tags, settled_df=bad_ds)


def test_settled_df_validates_schema_nulls_and_duplicates(strict_case):
    """Missing columns, null values and duplicate keys fail fast."""
    Y_hat_df, S_df, tags = strict_case
    leaf_ids = _leaf_ids(S_df)
    first_ds = Y_hat_df["ds"].iloc[0]
    hrec = HierarchicalReconciliation([ConstrainedReconciliation()])

    with pytest.raises(ValueError, match="must be"):
        hrec.reconcile(
            Y_hat_df=Y_hat_df,
            S_df=S_df,
            tags=tags,
            settled_df=pd.DataFrame([{"unique_id": leaf_ids[0], "ds": first_ds}]),
        )

    with pytest.raises(ValueError, match="null values"):
        hrec.reconcile(
            Y_hat_df=Y_hat_df,
            S_df=S_df,
            tags=tags,
            settled_df=pd.DataFrame(
                [{"unique_id": leaf_ids[0], "ds": first_ds, "y": np.nan}]
            ),
        )

    with pytest.raises(ValueError, match="duplicated"):
        hrec.reconcile(
            Y_hat_df=Y_hat_df,
            S_df=S_df,
            tags=tags,
            settled_df=pd.DataFrame(
                [
                    {"unique_id": leaf_ids[0], "ds": first_ds, "y": 1.0},
                    {"unique_id": leaf_ids[0], "ds": first_ds, "y": 2.0},
                ]
            ),
        )


def test_settled_reconciliation_on_grouped_hierarchy(grouped_case):
    """The constrained solution also works for non-strict grouped structures."""
    Y_hat_df, S_df, tags = grouped_case
    S = _s_matrix(S_df)
    n_bottom = S.shape[1]
    leaf_ids = _leaf_ids(S_df)
    horizon_dates = sorted(Y_hat_df["ds"].unique())

    settled_df = pd.DataFrame(
        [
            {"unique_id": leaf_ids[0], "ds": horizon_dates[0], "y": 55.5},
            {"unique_id": leaf_ids[3], "ds": horizon_dates[0], "y": 88.8},
            {"unique_id": leaf_ids[0], "ds": horizon_dates[1], "y": 60.0},
        ]
    )
    hrec = HierarchicalReconciliation([ConstrainedReconciliation()])
    result = hrec.reconcile(
        Y_hat_df=Y_hat_df, S_df=S_df, tags=tags, settled_df=settled_df
    )

    reconciled = _pivot(result, S_df, REC_COL)
    np.testing.assert_allclose(reconciled, S @ reconciled[-n_bottom:], atol=1e-10)
    pinned = result.pivot(index="unique_id", columns="ds", values=REC_COL)
    for record in settled_df.itertuples():
        assert pinned.loc[record.unique_id, record.ds] == pytest.approx(
            record.y, abs=1e-12
        )


def test_settled_reconciliation_with_sparse_smatrix(grouped_case):
    """Sparse SMatrix inputs produce the same constrained solution."""
    from scipy import sparse as sp

    from hierarchicalforecast.utils import SMatrix

    Y_hat_df, S_df, tags = grouped_case
    S_dense = _s_matrix(S_df)
    leaf_ids = _leaf_ids(S_df)
    horizon_dates = sorted(Y_hat_df["ds"].unique())
    settled_df = pd.DataFrame(
        [{"unique_id": leaf_ids[1], "ds": horizon_dates[0], "y": 42.0}]
    )

    smat = SMatrix(
        sparse_matrix=sp.csc_matrix(S_dense),
        row_labels=S_df["unique_id"].to_numpy(),
        col_labels=np.array(leaf_ids),
    )

    hrec = HierarchicalReconciliation([ConstrainedReconciliation()])
    result_dense = hrec.reconcile(
        Y_hat_df=Y_hat_df, S_df=S_df, tags=tags, settled_df=settled_df
    )
    result_sparse = hrec.reconcile(
        Y_hat_df=Y_hat_df, S_df=smat, tags=tags, settled_df=settled_df
    )
    np.testing.assert_allclose(
        _pivot(result_sparse, S_df, REC_COL),
        _pivot(result_dense, S_df, REC_COL),
        atol=1e-10,
    )
    assert int(result_sparse["is_settled"].sum()) == 1


def test_mixed_reconcilers_share_settled_marker(strict_case):
    """Unconstrained reconcilers run alongside the constrained one."""
    Y_hat_df, S_df, tags = strict_case
    leaf_ids = _leaf_ids(S_df)
    first_ds = Y_hat_df["ds"].iloc[0]
    settled_df = pd.DataFrame([{"unique_id": leaf_ids[0], "ds": first_ds, "y": 33.3}])

    hrec = HierarchicalReconciliation([BottomUp(), ConstrainedReconciliation()])
    result = hrec.reconcile(
        Y_hat_df=Y_hat_df, S_df=S_df, tags=tags, settled_df=settled_df
    )
    assert "y_model/BottomUp" in result.columns
    assert REC_COL in result.columns
    assert "is_settled" in result.columns


def test_settled_rejects_intervals_and_temporal(strict_case):
    """Leaf constraints are a point-forecast feature."""
    Y_hat_df, S_df, tags = strict_case
    leaf_ids = _leaf_ids(S_df)
    first_ds = Y_hat_df["ds"].iloc[0]
    settled_df = pd.DataFrame([{"unique_id": leaf_ids[0], "ds": first_ds, "y": 1.0}])
    hrec = HierarchicalReconciliation([ConstrainedReconciliation()])

    with pytest.raises(NotImplementedError, match="point forecasts"):
        hrec.reconcile(
            Y_hat_df=Y_hat_df,
            S_df=S_df,
            tags=tags,
            settled_df=settled_df,
            level=[80],
        )
    with pytest.raises(NotImplementedError, match="temporal reconciliation"):
        hrec.reconcile(
            Y_hat_df=Y_hat_df,
            S_df=S_df,
            tags=tags,
            settled_df=settled_df,
            temporal=True,
        )


def test_settled_values_round_trip_through_polars(strict_case):
    """The settled path is backend-agnostic (polars inputs)."""
    pl = pytest.importorskip("polars")
    Y_hat_df, S_df, tags = strict_case
    leaf_ids = _leaf_ids(S_df)
    first_ds = Y_hat_df["ds"].iloc[0]
    settled_df = pd.DataFrame([{"unique_id": leaf_ids[2], "ds": first_ds, "y": 71.25}])

    hrec = HierarchicalReconciliation([ConstrainedReconciliation()])
    result = hrec.reconcile(
        Y_hat_df=pl.from_pandas(Y_hat_df),
        S_df=pl.from_pandas(S_df),
        tags=tags,
        settled_df=pl.from_pandas(settled_df),
    )
    result_nw = nw.from_native(result)
    pinned = (
        result_nw.filter(
            (nw.col("unique_id") == leaf_ids[2]) & (nw.col("ds") == nw.lit(first_ds))
        )
        .select(REC_COL, "is_settled")
        .to_native()
    )
    assert float(pinned[REC_COL][0]) == pytest.approx(71.25, abs=1e-12)
    assert bool(pinned["is_settled"][0]) is True


def test_constrained_reconciler_validates_settled_shape(strict_case):
    """Direct class usage rejects settled arrays of the wrong shape."""
    Y_hat_df, S_df, tags = strict_case
    S = _s_matrix(S_df)
    base = _pivot(Y_hat_df, S_df, "y_model")
    with pytest.raises(ValueError, match="shape"):
        ConstrainedReconciliation()(
            S=S, y_hat=base, settled=np.zeros((2, base.shape[1]))
        )


def test_unconstrained_special_case_matches_ols(strict_case):
    """With an all-NaN settled mask the solution equals OLS reconciliation."""
    Y_hat_df, S_df, tags = strict_case
    S = _s_matrix(S_df)
    n_bottom = S.shape[1]
    base = _pivot(Y_hat_df, S_df, "y_model")

    reconciler = ConstrainedReconciliation()
    out = reconciler(
        S=S,
        y_hat=base,
        settled=np.full((n_bottom, base.shape[1]), np.nan),
    )
    ols_beta = np.linalg.solve(S.T @ S, S.T @ base)
    np.testing.assert_allclose(out["mean"], S @ ols_beta, atol=1e-10)
