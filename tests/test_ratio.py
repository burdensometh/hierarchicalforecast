"""Tests for reconciling a ratio from separately reconciled numerator/denominator."""

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import pytest

from hierarchicalforecast.core import (
    HierarchicalReconciliation,
    reconcile_ratio,
)
from hierarchicalforecast.methods import BottomUp, MinTrace
from hierarchicalforecast.utils import aggregate

try:  # polars is optional; its cases are skipped when it is not installed.
    import polars as pl

    HAS_POLARS = True
except ImportError:  # pragma: no cover - depends on the environment
    pl = None
    HAS_POLARS = False

requires_polars = pytest.mark.skipif(not HAS_POLARS, reason="polars is not installed")
LIBS = [
    pytest.param("pandas", id="pandas"),
    pytest.param("polars", id="polars", marks=requires_polars),
]


# Stores carry very different visit volumes so that an unweighted average of
# store conversion rates differs from the true (pooled) regional rate.
STORES = [
    ("r1", "s1", 90, 9),   # 10%
    ("r1", "s2", 10, 9),   # 90%
    ("r2", "s3", 50, 25),  # 50%
    ("r2", "s4", 50, 5),   # 10%
]
SPEC = [["region"], ["region", "store"]]


def _rows(periods=8):
    rows = []
    for region, store, visits, orders in STORES:
        for ds in range(1, periods + 1):
            rows.append(
                {
                    "region": region,
                    "store": store,
                    "ds": ds,
                    "visits": visits,
                    "orders": orders,
                }
            )
    return rows


def _frame(rows, lib):
    return pd.DataFrame(rows) if lib == "pandas" else pl.DataFrame(rows)


def _to_backend(frame, lib):
    """Return ``frame`` in the requested backend, leaving native frames as is."""
    if lib == "pandas":
        return frame
    if isinstance(frame, pl.DataFrame):
        return frame
    return pl.from_pandas(frame)


def _make_data(lib="pandas"):
    """Build coherent totals and store forecasts for the requested backend."""
    df = _frame(_rows(), lib)
    Y_df, S_df, tags = aggregate(df, SPEC, target_cols=["orders", "visits"])
    return Y_df, S_df, tags


def _split_hat(Y_df, h, lib, periods=8):
    """Split the last ``h`` periods of every series into the forecast frame."""
    Y_nw = nw.from_native(Y_df)
    cutoff = periods - h
    Y_hat = Y_nw.filter(nw.col("ds") > cutoff).to_native()
    Y_train = Y_nw.filter(nw.col("ds") <= cutoff).to_native()
    return _to_backend(Y_hat, lib), _to_backend(Y_train, lib)


def _split_sides(Y_hat, lib):
    numerator = _to_backend(
        Y_hat[["unique_id", "ds", "orders"]].to_native()
        if isinstance(Y_hat, nw.DataFrame)
        else Y_hat[["unique_id", "ds", "orders"]],
        lib,
    )
    denominator = _to_backend(
        Y_hat[["unique_id", "ds", "visits"]].to_native()
        if isinstance(Y_hat, nw.DataFrame)
        else Y_hat[["unique_id", "ds", "visits"]],
        lib,
    )
    return numerator, denominator


@pytest.mark.parametrize("lib", LIBS)
def test_ratio_report_structure(lib):
    Y_df, S_df, tags = _make_data(lib)
    Y_hat, Y_train = _split_hat(Y_df, h=4, lib=lib)
    numerator, denominator = _split_sides(Y_hat, lib)

    out = reconcile_ratio(
        numerator,
        denominator,
        reconcilers=[BottomUp()],
        S_df=_to_backend(S_df, lib),
        tags=tags,
        numerator_col="orders",
        denominator_col="visits",
        Y_df=Y_train,
    )

    out_nw = nw.from_native(out)
    assert out_nw.columns == [
        "level",
        "unique_id",
        "ds",
        "model",
        "numerator",
        "denominator",
        "rate",
    ]
    # One row per (series, timestamp) for the single reconciler. Each series
    # belongs to exactly one level, so the rows cover every series.
    n_series = nw.from_native(Y_hat)["unique_id"].n_unique()
    assert len(out_nw) == n_series * 4
    assert set(out_nw["level"].unique().to_list()) == set(tags)
    # The report always exposes both the numerator and the denominator.
    assert out_nw["numerator"].null_count() == 0
    assert out_nw["denominator"].null_count() == 0


@pytest.mark.parametrize("lib", LIBS)
def test_ratio_level_rate_is_pooled_not_averaged(lib):
    Y_df, S_df, tags = _make_data(lib)
    Y_hat, Y_train = _split_hat(Y_df, h=4, lib=lib)
    numerator, denominator = _split_sides(Y_hat, lib)

    out = reconcile_ratio(
        numerator,
        denominator,
        reconcilers=[BottomUp()],
        S_df=_to_backend(S_df, lib),
        tags=tags,
        numerator_col="orders",
        denominator_col="visits",
        Y_df=Y_train,
    )
    out_nw = nw.from_native(out)

    ts = 5
    region_r1 = out_nw.filter(
        (nw.col("level") == "region")
        & (nw.col("unique_id") == "r1")
        & (nw.col("ds") == ts)
    )
    # Pooled rate: total orders over total visits for the region.
    expected_pooled = (9 + 9) / (90 + 10)
    assert region_r1["rate"].to_list()[0] == pytest.approx(expected_pooled)

    # An unweighted average of the two store rates would be 0.5, which the
    # pooled rate must not reproduce.
    store_rates = (
        out_nw.filter((nw.col("level") == "region/store") & (nw.col("ds") == ts))
        .filter(nw.col("unique_id").is_in(["r1/s1", "r1/s2"]))["rate"]
        .to_list()
    )
    assert store_rates[0] == pytest.approx(0.1)
    assert store_rates[1] == pytest.approx(0.9)
    assert region_r1["rate"].to_list()[0] != pytest.approx(np.mean(store_rates))

    # Aggregate numerator and denominator are the sums of the children.
    store_rows = out_nw.filter(
        (nw.col("level") == "region/store")
        & (nw.col("ds") == ts)
        & nw.col("unique_id").is_in(["r1/s1", "r1/s2"])
    )
    assert region_r1["numerator"].to_list()[0] == pytest.approx(
        sum(store_rows["numerator"].to_list())
    )
    assert region_r1["denominator"].to_list()[0] == pytest.approx(
        sum(store_rows["denominator"].to_list())
    )


@pytest.mark.parametrize("lib", LIBS)
def test_ratio_zero_denominator_is_missing(lib):
    Y_df, S_df, tags = _make_data(lib)
    Y_hat, Y_train = _split_hat(Y_df, h=4, lib=lib)
    numerator, denominator = _split_sides(Y_hat, lib)
    # A store with no visits at all: the rate is undefined, not zero.
    numerator = nw.from_native(numerator).to_native()
    denominator = nw.from_native(denominator)
    denominator = denominator.with_columns(
        nw.when(nw.col("unique_id") == "r1/s2")
        .then(nw.lit(0.0))
        .otherwise(nw.col("visits"))
        .alias("visits")
    ).to_native()
    numerator = nw.from_native(numerator)
    numerator = numerator.with_columns(
        nw.when(nw.col("unique_id") == "r1/s2")
        .then(nw.lit(0.0))
        .otherwise(nw.col("orders"))
        .alias("orders")
    ).to_native()

    out = reconcile_ratio(
        numerator,
        denominator,
        reconcilers=[BottomUp()],
        S_df=_to_backend(S_df, lib),
        tags=tags,
        numerator_col="orders",
        denominator_col="visits",
        Y_df=Y_train,
    )
    out_nw = nw.from_native(out)
    no_visits = out_nw.filter(nw.col("unique_id") == "r1/s2")
    assert no_visits["denominator"].to_list() == [0.0] * len(no_visits)
    assert no_visits["rate"].null_count() == len(no_visits)


@pytest.mark.parametrize("lib", LIBS)
def test_ratio_missing_series_raises_before_reconciliation(lib):
    Y_df, S_df, tags = _make_data(lib)
    Y_hat, Y_train = _split_hat(Y_df, h=4, lib=lib)
    numerator, denominator = _split_sides(Y_hat, lib)
    # Drop a store that is present in the denominator.
    numerator = nw.from_native(numerator)
    numerator = numerator.filter(nw.col("unique_id") != "r1/s2").to_native()

    hrec = HierarchicalReconciliation([BottomUp()])
    with pytest.raises(ValueError, match="same \\(series, timestamp\\) pairs"):
        hrec.reconcile_ratio(
            numerator,
            denominator,
            S_df=_to_backend(S_df, lib),
            tags=tags,
            numerator_col="orders",
            denominator_col="visits",
            Y_df=Y_train,
        )
    # Reconciliation never ran, so no model columns were produced.
    assert getattr(hrec, "model_names", None) in (None, [])


@pytest.mark.parametrize("lib", LIBS)
def test_ratio_missing_timestamp_raises(lib):
    Y_df, S_df, tags = _make_data(lib)
    Y_hat, Y_train = _split_hat(Y_df, h=4, lib=lib)
    numerator, denominator = _split_sides(Y_hat, lib)
    # One series misses a single timestamp on the denominator side only.
    denominator = nw.from_native(denominator)
    denominator = denominator.filter(
        ~((nw.col("unique_id") == "r1/s1") & (nw.col("ds") == 5))
    ).to_native()

    with pytest.raises(ValueError, match="same \\(series, timestamp\\) pairs"):
        reconcile_ratio(
            numerator,
            denominator,
            reconcilers=[BottomUp()],
            S_df=_to_backend(S_df, lib),
            tags=tags,
            numerator_col="orders",
            denominator_col="visits",
            Y_df=Y_train,
        )


def test_ratio_repeated_keys_raise():
    Y_df, S_df, tags = _make_data("pandas")
    Y_hat, Y_train = _split_hat(Y_df, h=4, lib="pandas")
    numerator, denominator = _split_sides(Y_hat, "pandas")
    numerator = pd.concat([numerator, numerator.head(1)], ignore_index=True)

    with pytest.raises(ValueError, match="repeated \\(series, timestamp\\) pairs"):
        reconcile_ratio(
            numerator,
            denominator,
            reconcilers=[BottomUp()],
            S_df=S_df,
            tags=tags,
            numerator_col="orders",
            denominator_col="visits",
            Y_df=Y_train,
        )


@pytest.mark.parametrize("lib", LIBS)
def test_ratio_pairs_columns_per_reconciler(lib):
    Y_df, S_df, tags = _make_data(lib)
    Y_hat, Y_train = _split_hat(Y_df, h=4, lib=lib)
    numerator, denominator = _split_sides(Y_hat, lib)

    out = reconcile_ratio(
        numerator,
        denominator,
        reconcilers=[BottomUp(), MinTrace(method="ols")],
        S_df=_to_backend(S_df, lib),
        tags=tags,
        numerator_col="orders",
        denominator_col="visits",
        Y_df=Y_train,
    )
    out_nw = nw.from_native(out)
    assert set(out_nw["model"].unique().to_list()) == {
        "BottomUp",
        "MinTrace_method-ols",
    }
    # Every reconcile method yields a rate built from its own two columns.
    assert out_nw["rate"].null_count() == 0
    assert np.isfinite(out_nw["rate"].to_numpy()).all()


def test_ratio_infers_single_value_column():
    Y_df, S_df, tags = _make_data("pandas")
    Y_hat, Y_train = _split_hat(Y_df, h=4, lib="pandas")
    numerator, denominator = _split_sides(Y_hat, "pandas")

    out = reconcile_ratio(
        numerator,
        denominator,
        reconcilers=[BottomUp()],
        S_df=S_df,
        tags=tags,
        Y_df=Y_train,
    )
    out_nw = nw.from_native(out)
    assert len(out_nw) > 0
    assert out_nw["rate"].null_count() == 0


def test_ratio_ambiguous_value_column_raises():
    Y_df, S_df, tags = _make_data("pandas")
    Y_hat, Y_train = _split_hat(Y_df, h=4, lib="pandas")
    numerator, denominator = _split_sides(Y_hat, "pandas")
    numerator["orders_alt"] = numerator["orders"]

    with pytest.raises(ValueError, match="Could not infer the value column"):
        reconcile_ratio(
            numerator,
            denominator,
            reconcilers=[BottomUp()],
            S_df=S_df,
            tags=tags,
            Y_df=Y_train,
        )


def test_ratio_missing_history_column_raises():
    Y_df, S_df, tags = _make_data("pandas")
    Y_hat, Y_train = _split_hat(Y_df, h=4, lib="pandas")
    numerator, denominator = _split_sides(Y_hat, "pandas")
    Y_train_missing = Y_train.drop(columns=["visits"])

    with pytest.raises(ValueError, match="must contain the denominator value column"):
        reconcile_ratio(
            numerator,
            denominator,
            reconcilers=[BottomUp()],
            S_df=S_df,
            tags=tags,
            numerator_col="orders",
            denominator_col="visits",
            Y_df=Y_train_missing,
        )


@pytest.mark.parametrize("lib", LIBS)
def test_ratio_allows_shared_value_column_name(lib):
    Y_df, S_df, tags = _make_data(lib)
    Y_hat, _ = _split_hat(Y_df, h=4, lib=lib)
    # Both sides expose their value under the same name, which must not clash.
    Y_hat_nw = nw.from_native(Y_hat)
    numerator = Y_hat_nw.select(
        "unique_id", "ds", nw.col("orders").alias("y")
    ).to_native()
    denominator = Y_hat_nw.select(
        "unique_id", "ds", nw.col("visits").alias("y")
    ).to_native()

    out = reconcile_ratio(
        numerator,
        denominator,
        reconcilers=[BottomUp()],
        S_df=_to_backend(S_df, lib),
        tags=tags,
    )
    out_nw = nw.from_native(out)
    assert out_nw["rate"].null_count() == 0
    region = out_nw.filter(
        (nw.col("level") == "region")
        & (nw.col("unique_id") == "r1")
        & (nw.col("ds") == 5)
    )
    assert region["rate"].to_list()[0] == pytest.approx((9 + 9) / (90 + 10))


def test_ratio_missing_key_column_raises():
    Y_df, S_df, tags = _make_data("pandas")
    Y_hat, Y_train = _split_hat(Y_df, h=4, lib="pandas")
    numerator, denominator = _split_sides(Y_hat, "pandas")
    denominator = denominator.drop(columns=["unique_id"])

    with pytest.raises(ValueError, match="missing required column"):
        reconcile_ratio(
            numerator,
            denominator,
            reconcilers=[BottomUp()],
            S_df=S_df,
            tags=tags,
            numerator_col="orders",
            denominator_col="visits",
            Y_df=Y_train,
        )


def test_ratio_requires_tags():
    Y_df, S_df, _ = _make_data("pandas")
    Y_hat, _ = _split_hat(Y_df, h=4, lib="pandas")
    numerator, denominator = _split_sides(Y_hat, "pandas")

    with pytest.raises(ValueError, match="`tags` is required"):
        reconcile_ratio(
            numerator,
            denominator,
            reconcilers=[BottomUp()],
            S_df=S_df,
        )


def test_ratio_custom_rate_column_name():
    Y_df, S_df, tags = _make_data("pandas")
    Y_hat, Y_train = _split_hat(Y_df, h=4, lib="pandas")
    numerator, denominator = _split_sides(Y_hat, "pandas")

    out = reconcile_ratio(
        numerator,
        denominator,
        reconcilers=[BottomUp()],
        S_df=S_df,
        tags=tags,
        numerator_col="orders",
        denominator_col="visits",
        ratio_col="conversion",
        Y_df=Y_train,
    )
    assert nw.from_native(out).columns[-1] == "conversion"


def test_ratio_explicit_unknown_value_column_raises():
    Y_df, S_df, tags = _make_data("pandas")
    Y_hat, _ = _split_hat(Y_df, h=4, lib="pandas")
    numerator, denominator = _split_sides(Y_hat, "pandas")

    with pytest.raises(ValueError, match="was not found in the numerator"):
        reconcile_ratio(
            numerator,
            denominator,
            reconcilers=[BottomUp()],
            S_df=S_df,
            tags=tags,
            numerator_col="not_a_column",
            denominator_col="visits",
        )


def test_ratio_non_numeric_value_column_raises():
    Y_df, S_df, tags = _make_data("pandas")
    Y_hat, _ = _split_hat(Y_df, h=4, lib="pandas")
    numerator, denominator = _split_sides(Y_hat, "pandas")
    numerator["orders"] = numerator["orders"].astype(str)

    with pytest.raises(ValueError, match="must be numeric"):
        reconcile_ratio(
            numerator,
            denominator,
            reconcilers=[BottomUp()],
            S_df=S_df,
            tags=tags,
            numerator_col="orders",
            denominator_col="visits",
        )


@pytest.mark.parametrize("lib", LIBS)
def test_ratio_accepts_single_frame_with_both_columns(lib):
    Y_df, S_df, tags = _make_data(lib)
    Y_hat, Y_train = _split_hat(Y_df, h=4, lib=lib)
    combined = Y_hat  # both `orders` and `visits` in one frame

    out = reconcile_ratio(
        combined,
        reconcilers=[BottomUp()],
        S_df=_to_backend(S_df, lib),
        tags=tags,
        numerator_col="orders",
        denominator_col="visits",
        Y_df=Y_train,
    )
    out_nw = nw.from_native(out)
    assert out_nw["rate"].null_count() == 0
    region = out_nw.filter(
        (nw.col("level") == "region")
        & (nw.col("unique_id") == "r1")
        & (nw.col("ds") == 5)
    )
    assert region["rate"].to_list()[0] == pytest.approx((9 + 9) / (90 + 10))


def test_ratio_single_frame_requires_denominator_col():
    Y_df, S_df, tags = _make_data("pandas")
    Y_hat, _ = _split_hat(Y_df, h=4, lib="pandas")

    with pytest.raises(ValueError, match="Could not infer the value column"):
        reconcile_ratio(
            Y_hat,
            reconcilers=[BottomUp()],
            S_df=S_df,
            tags=tags,
            numerator_col="orders",
        )


def test_ratio_empty_tags_raises():
    Y_df, S_df, _ = _make_data("pandas")
    Y_hat, _ = _split_hat(Y_df, h=4, lib="pandas")
    numerator, denominator = _split_sides(Y_hat, "pandas")

    with pytest.raises(ValueError, match="`tags` is required"):
        reconcile_ratio(
            numerator,
            denominator,
            reconcilers=[BottomUp()],
            S_df=S_df,
            tags={},
        )


def test_ratio_grouped_hierarchy_levels_pooled():
    """Grouped hierarchies expose each level with its own pooled rate."""
    rows = []
    for region, channel, store, visits, orders in [
        ("r1", "web", "s1", 90, 9),
        ("r1", "app", "s2", 10, 9),
        ("r2", "web", "s3", 50, 25),
        ("r2", "app", "s4", 50, 5),
    ]:
        for ds in (1, 2):
            rows.append(
                {
                    "region": region,
                    "channel": channel,
                    "store": store,
                    "ds": ds,
                    "visits": visits,
                    "orders": orders,
                }
            )
    df = pd.DataFrame(rows)
    spec = [["region"], ["channel"], ["region", "channel", "store"]]
    Y_df, S_df, tags = aggregate(df, spec, target_cols=["orders", "visits"])
    numerator = Y_df[["unique_id", "ds", "orders"]]
    denominator = Y_df[["unique_id", "ds", "visits"]]

    out = reconcile_ratio(
        numerator,
        denominator,
        reconcilers=[BottomUp()],
        S_df=S_df,
        tags=tags,
    )
    out_nw = nw.from_native(out)
    assert set(out_nw["level"].unique().to_list()) == set(tags)
    # The `channel` level has a web node and an app node; each pools its stores.
    web = out_nw.filter(
        (nw.col("level") == "channel")
        & (nw.col("unique_id") == "web")
        & (nw.col("ds") == 1)
    )
    assert web["numerator"].to_list()[0] == pytest.approx(9 + 25)
    assert web["denominator"].to_list()[0] == pytest.approx(90 + 50)
    assert web["rate"].to_list()[0] == pytest.approx((9 + 25) / (90 + 50))
