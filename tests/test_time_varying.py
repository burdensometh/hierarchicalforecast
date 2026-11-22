import numpy as np
import pandas as pd
import pytest

from hierarchicalforecast.methods import MinTrace
from hierarchicalforecast.time_varying import (
    DuplicateOwnershipError,
    MissingOwnershipError,
    TimeVaryingHierarchyError,
    reconcile_time_varying,
)

pytestmark = pytest.mark.parametrize("lib", ["pandas", "polars"])


SWITCH_DATE = pd.Timestamp("2026-07-01")
DATES = pd.date_range("2026-06-01", periods=4, freq="MS")  # Jun, Jul, Aug, Sep


def _maybe_polars(df, lib):
    if lib == "polars":
        import polars as pl

        return pl.from_pandas(df)
    return df


@pytest.fixture
def ownership_df():
    # Outlet A switches from region R1 to R2 on July 1st; outlet B stays in R1.
    return pd.DataFrame(
        {
            "unique_id": ["A", "A", "B"],
            "region": ["R1", "R2", "R1"],
            "valid_from": pd.to_datetime(["2026-06-01", "2026-07-01", "2026-06-01"]),
            "valid_to": pd.to_datetime(["2026-06-30", pd.NaT, pd.NaT]),
        }
    )


@pytest.fixture
def Y_hat_df():
    rng = np.random.default_rng(42)
    rows = []
    for uid, base in [("A", 10.0), ("B", 20.0)]:
        for ds in DATES:
            rows.append({"unique_id": uid, "ds": ds, "y_hat": base + rng.normal()})
    # Aggregate-level base forecasts: deliberately wrong, they must be
    # overwritten by the bottom-up aggregates.
    for region in ["R1", "R2"]:
        for ds in DATES:
            rows.append({"unique_id": region, "ds": ds, "y_hat": 999.0})
    return pd.DataFrame(rows)


def _reconcile(lib, Y_hat_df, ownership_df, **kwargs):
    return reconcile_time_varying(
        Y_hat_df=_maybe_polars(Y_hat_df, lib),
        ownership_df=_maybe_polars(ownership_df, lib),
        parent_cols=["region"],
        **kwargs,
    )


def test_splits_horizon_at_boundary_and_builds_two_segments(
    lib, Y_hat_df, ownership_df
):
    rec = _reconcile(lib, Y_hat_df, ownership_df)
    segments = reconcile_time_varying.segments_

    assert len(segments) == 2
    assert segments[0]["start"] == DATES[0]
    assert segments[0]["end"] == DATES[0]  # June alone, before the switch
    assert segments[0]["n_nodes"] == 3  # R1 + outlets A, B
    assert segments[0]["n_bottom"] == 2
    assert segments[1]["start"] == SWITCH_DATE
    assert segments[1]["end"] == DATES[-1]
    assert segments[1]["n_nodes"] == 4  # R1, R2 + outlets A, B
    assert segments[1]["n_bottom"] == 2

    # Before the switch R2 does not exist as a node.
    rec_pd = rec if isinstance(rec, pd.DataFrame) else rec.to_pandas()
    june_nodes = set(rec_pd.loc[rec_pd["ds"] == DATES[0], "unique_id"])
    assert june_nodes == {"A", "B", "R1"}
    post_nodes = set(rec_pd.loc[rec_pd["ds"] == SWITCH_DATE, "unique_id"])
    assert post_nodes == {"A", "B", "R1", "R2"}


def test_bottom_level_forecasts_are_preserved(lib, Y_hat_df, ownership_df):
    rec = _reconcile(lib, Y_hat_df, ownership_df)
    rec_pd = rec if isinstance(rec, pd.DataFrame) else rec.to_pandas()
    out_col = "y_hat/BottomUp"

    for uid in ["A", "B"]:
        base = (
            Y_hat_df[Y_hat_df["unique_id"] == uid]
            .sort_values("ds")
            .set_index("ds")["y_hat"]
        )
        got = (
            rec_pd[rec_pd["unique_id"] == uid]
            .sort_values("ds")
            .set_index("ds")[out_col]
        )
        pd.testing.assert_series_equal(base, got, check_names=False)


def test_aggregates_follow_the_owner_switch(lib, Y_hat_df, ownership_df):
    rec = _reconcile(lib, Y_hat_df, ownership_df)
    rec_pd = rec if isinstance(rec, pd.DataFrame) else rec.to_pandas()
    out_col = "y_hat/BottomUp"
    lookup = rec_pd.set_index(["unique_id", "ds"])[out_col]

    # June: both outlets roll up under R1; R2 is absent.
    june = DATES[0]
    assert np.isclose(lookup[("R1", june)], lookup[("A", june)] + lookup[("B", june)])
    assert ("R2", june) not in lookup.index

    # From July: A rolls up to R2, B stays under R1.
    for ds in DATES[DATES >= SWITCH_DATE]:
        assert np.isclose(lookup[("R2", ds)], lookup[("A", ds)])
        assert np.isclose(lookup[("R1", ds)], lookup[("B", ds)])

    # The stale aggregate base forecasts (999.0) never survive reconciliation.
    assert (rec_pd[out_col] < 100).all()


def test_result_is_stitched_back_with_dates(lib, Y_hat_df, ownership_df):
    rec = _reconcile(lib, Y_hat_df, ownership_df)
    rec_pd = rec if isinstance(rec, pd.DataFrame) else rec.to_pandas()

    # Every bottom node appears on every forecast date.
    counts = (
        rec_pd[rec_pd["unique_id"].isin(["A", "B"])]
        .groupby("unique_id")["ds"]
        .nunique()
    )
    assert set(counts) == {len(DATES)}
    # No duplicated (node, date) rows after stitching.
    assert not rec_pd.duplicated(["unique_id", "ds"]).any()
    # Output backend matches the input.
    if lib == "polars":
        import polars as pl

        assert isinstance(rec, pl.DataFrame)
    else:
        assert isinstance(rec, pd.DataFrame)


def test_missing_ownership_is_located(lib, Y_hat_df, ownership_df):
    bad = ownership_df.copy()
    # Outlet B has no interval covering June.
    bad.loc[bad["unique_id"] == "B", "valid_from"] = SWITCH_DATE
    with pytest.raises(MissingOwnershipError, match="B.*2026-06-01"):
        _reconcile(lib, Y_hat_df, bad)


def test_duplicate_ownership_on_same_day_is_located(lib, Y_hat_df, ownership_df):
    bad = pd.concat(
        [
            ownership_df,
            pd.DataFrame(
                [
                    {
                        "unique_id": "A",
                        "region": "R1",
                        "valid_from": pd.Timestamp("2026-06-15"),
                        "valid_to": pd.Timestamp("2026-07-15"),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(DuplicateOwnershipError, match="A.*2026-07-01"):
        _reconcile(lib, Y_hat_df, bad)


def test_closed_intervals_touching_at_boundary_do_not_overlap(
    lib, Y_hat_df, ownership_df
):
    # Adjacent closed intervals (old ends the day before the new starts) are
    # legal and must not be flagged as duplicates.
    pre_switch = (ownership_df["unique_id"] == "A") & (
        ownership_df["valid_from"] == pd.Timestamp("2026-06-01")
    )
    ownership_df.loc[pre_switch, "valid_to"] = pd.Timestamp("2026-06-30")
    rec = _reconcile(lib, Y_hat_df, ownership_df)
    rec_pd = rec if isinstance(rec, pd.DataFrame) else rec.to_pandas()
    assert len(rec_pd) > 0


def test_open_ended_interval_covers_the_rest_of_the_horizon(
    lib, Y_hat_df, ownership_df
):
    # Drop the explicit valid_to on B's single row (already open) and on A's
    # post-switch row: the last segment must still cover September.
    assert ownership_df["valid_to"].isna().sum() == 2
    rec = _reconcile(lib, Y_hat_df, ownership_df)
    rec_pd = rec if isinstance(rec, pd.DataFrame) else rec.to_pandas()
    assert rec_pd["ds"].max() == DATES[-1]


def test_string_dates_are_accepted(lib, Y_hat_df, ownership_df):
    Y_hat_str = Y_hat_df.copy()
    own_str = ownership_df.copy()
    Y_hat_str["ds"] = Y_hat_str["ds"].dt.strftime("%Y-%m-%d")
    own_str["valid_from"] = own_str["valid_from"].dt.strftime("%Y-%m-%d")
    own_str["valid_to"] = own_str["valid_to"].dt.strftime("%Y-%m-%d")
    rec = _reconcile(lib, Y_hat_str, own_str)
    rec_pd = rec if isinstance(rec, pd.DataFrame) else rec.to_pandas()
    assert set(rec_pd["ds"].astype(str).str[:10]) == {
        "2026-06-01",
        "2026-07-01",
        "2026-08-01",
        "2026-09-01",
    }


def test_multiple_forecast_columns_are_all_reconciled(lib, Y_hat_df, ownership_df):
    multi = Y_hat_df.copy()
    multi["y_hat2"] = multi["y_hat"] * 2.0
    rec = _reconcile(lib, multi, ownership_df)
    rec_pd = rec if isinstance(rec, pd.DataFrame) else rec.to_pandas()
    assert "y_hat/BottomUp" in rec_pd.columns
    assert "y_hat2/BottomUp" in rec_pd.columns
    lookup = rec_pd.set_index(["unique_id", "ds"])["y_hat2/BottomUp"]
    june = DATES[0]
    assert np.isclose(
        lookup[("R1", june)],
        rec_pd.set_index(["unique_id", "ds"])
        .loc[[("A", june), ("B", june)], "y_hat2/BottomUp"]
        .sum(),
    )


def test_prediction_interval_columns_are_not_treated_as_models(
    lib, Y_hat_df, ownership_df
):
    pi = Y_hat_df.copy()
    pi["y_hat-lo-90"] = pi["y_hat"] - 1.0
    pi["y_hat-hi-90"] = pi["y_hat"] + 1.0
    rec = _reconcile(lib, pi, ownership_df)
    rec_pd = rec if isinstance(rec, pd.DataFrame) else rec.to_pandas()
    assert "y_hat/BottomUp" in rec_pd.columns
    assert "y_hat-lo-90/BottomUp" not in rec_pd.columns


def test_alternative_non_insample_reconciler(lib, Y_hat_df, ownership_df):
    rec = _reconcile(lib, Y_hat_df, ownership_df, reconciler=MinTrace(method="ols"))
    rec_pd = rec if isinstance(rec, pd.DataFrame) else rec.to_pandas()
    assert "y_hat/MinTrace_method-ols" in rec_pd.columns
    # MinTrace OLS is also coherent: check the post-switch R2 aggregate.
    col = "y_hat/MinTrace_method-ols"
    lookup = rec_pd.set_index(["unique_id", "ds"])[col]
    for ds in DATES[DATES >= SWITCH_DATE]:
        assert np.isclose(lookup[("R2", ds)], lookup[("A", ds)])


def test_insample_reconciler_is_rejected(lib, Y_hat_df, ownership_df):
    from hierarchicalforecast.methods import TopDown

    with pytest.raises(TimeVaryingHierarchyError, match="insample"):
        _reconcile(
            lib,
            Y_hat_df,
            ownership_df,
            reconciler=TopDown(method="average_proportions"),
        )


def test_missing_forecast_date_raises(lib, Y_hat_df, ownership_df):
    incomplete = Y_hat_df[
        ~((Y_hat_df["unique_id"] == "A") & (Y_hat_df["ds"] == DATES[1]))
    ]
    with pytest.raises(TimeVaryingHierarchyError, match="every forecast date"):
        _reconcile(lib, incomplete, ownership_df)


def test_unknown_bottom_series_in_ownership_raises(lib, Y_hat_df, ownership_df):
    bad = ownership_df.copy()
    bad.loc[0, "unique_id"] = "Z"
    with pytest.raises(TimeVaryingHierarchyError, match="absent from"):
        _reconcile(lib, Y_hat_df, bad)


def test_reversed_interval_raises(lib, Y_hat_df, ownership_df):
    bad = ownership_df.copy()
    bad.loc[0, "valid_from"] = pd.Timestamp("2026-07-15")
    bad.loc[0, "valid_to"] = pd.Timestamp("2026-06-01")
    with pytest.raises(TimeVaryingHierarchyError, match="valid_to earlier"):
        _reconcile(lib, Y_hat_df, bad)


def test_two_level_hierarchy_switches_intermediate_parent(lib):
    # Country -> Region -> Outlet, with the outlet switching region.
    ownership = pd.DataFrame(
        {
            "unique_id": ["A", "A", "B"],
            "country": ["C1", "C1", "C1"],
            "region": ["R1", "R2", "R1"],
            "valid_from": pd.to_datetime(["2026-06-01", "2026-07-01", "2026-06-01"]),
            "valid_to": pd.to_datetime(["2026-06-30", pd.NaT, pd.NaT]),
        }
    )
    rng = np.random.default_rng(1)
    rows = [
        {"unique_id": u, "ds": d, "y_hat": float(rng.normal())}
        for u in ["A", "B"]
        for d in DATES
    ]
    Y_hat = pd.DataFrame(rows)
    rec = reconcile_time_varying(
        Y_hat_df=_maybe_polars(Y_hat, lib),
        ownership_df=_maybe_polars(ownership, lib),
        parent_cols=["country", "region"],
    )
    rec_pd = rec if isinstance(rec, pd.DataFrame) else rec.to_pandas()
    lookup = rec_pd.set_index(["unique_id", "ds"])["y_hat/BottomUp"]

    # Country total always equals A + B.
    for ds in DATES:
        assert np.isclose(lookup[("C1", ds)], lookup[("A", ds)] + lookup[("B", ds)])
    # June region R1 holds both; post-switch R2 holds only A.
    assert np.isclose(
        lookup[("R1", DATES[0])],
        lookup[("A", DATES[0])] + lookup[("B", DATES[0])],
    )
    for ds in DATES[DATES >= SWITCH_DATE]:
        assert np.isclose(lookup[("R2", ds)], lookup[("A", ds)])
        assert np.isclose(lookup[("R1", ds)], lookup[("B", ds)])


def test_boundary_between_forecast_dates(lib):
    # Switch on July 15th while forecasts are monthly: July's forecast is
    # already produced under the new ownership, so the segment break sits at
    # the first forecast date on or after the boundary.
    dates = pd.date_range("2026-06-01", periods=4, freq="MS")
    ownership = pd.DataFrame(
        {
            "unique_id": ["A", "A", "B"],
            "region": ["R1", "R2", "R1"],
            "valid_from": pd.to_datetime(["2026-01-01", "2026-07-15", "2026-01-01"]),
            "valid_to": pd.to_datetime(["2026-07-14", pd.NaT, pd.NaT]),
        }
    )
    # July 1st is still inside the old interval; July 15th (between forecast
    # dates) is the handover, so the segment break is placed at August 1st.
    Y_hat = pd.DataFrame(
        [
            {"unique_id": u, "ds": d, "y_hat": float(i)}
            for i, (u, d) in enumerate((u, d) for u in ("A", "B") for d in dates)
        ]
    )
    rec = reconcile_time_varying(
        Y_hat_df=_maybe_polars(Y_hat, lib),
        ownership_df=_maybe_polars(ownership, lib),
        parent_cols=["region"],
    )
    segments = reconcile_time_varying.segments_
    assert len(segments) == 2
    assert segments[0]["end"] == pd.Timestamp("2026-07-01")
    assert segments[1]["start"] == pd.Timestamp("2026-08-01")

    rec_pd = rec if isinstance(rec, pd.DataFrame) else rec.to_pandas()
    lookup = rec_pd.set_index(["unique_id", "ds"])["y_hat/BottomUp"]
    # June and July forecasts still roll A up under R1; from August (the
    # first forecast date on or after the July 15th handover) A is under R2.
    for ds in [pd.Timestamp("2026-06-01"), pd.Timestamp("2026-07-01")]:
        assert np.isclose(lookup[("R1", ds)], lookup[("A", ds)] + lookup[("B", ds)])
        assert ("R2", ds) not in lookup.index
    for ds in [pd.Timestamp("2026-08-01"), pd.Timestamp("2026-09-01")]:
        assert np.isclose(lookup[("R2", ds)], lookup[("A", ds)])
        assert np.isclose(lookup[("R1", ds)], lookup[("B", ds)])


def test_distinct_bottom_column_name(lib):
    # The ownership frame names the bottom key ``outlet`` while the forecast
    # frame keeps the series id column ``unique_id``.
    ownership = pd.DataFrame(
        {
            "outlet": ["A", "A", "B"],
            "region": ["R1", "R2", "R1"],
            "valid_from": pd.to_datetime(["2026-06-01", "2026-07-01", "2026-06-01"]),
            "valid_to": pd.to_datetime(["2026-06-30", pd.NaT, pd.NaT]),
        }
    )
    Y_hat = pd.DataFrame(
        [
            {"unique_id": u, "ds": d, "y_hat": float(i)}
            for i, (u, d) in enumerate((u, d) for u in ("A", "B") for d in DATES)
        ]
    )
    rec = reconcile_time_varying(
        Y_hat_df=_maybe_polars(Y_hat, lib),
        ownership_df=_maybe_polars(ownership, lib),
        parent_cols=["region"],
        bottom_col="outlet",
    )
    rec_pd = rec if isinstance(rec, pd.DataFrame) else rec.to_pandas()
    lookup = rec_pd.set_index(["unique_id", "ds"])["y_hat/BottomUp"]
    june, july = DATES[0], DATES[1]
    assert np.isclose(lookup[("R1", june)], lookup[("A", june)] + lookup[("B", june)])
    assert np.isclose(lookup[("R2", july)], lookup[("A", july)])
    assert np.isclose(lookup[("R1", july)], lookup[("B", july)])


def test_no_boundary_inside_horizon_yields_single_segment(lib, Y_hat_df):
    # Everything stays under R1 for the whole horizon: no switch inside it.
    ownership = pd.DataFrame(
        {
            "unique_id": ["A", "B"],
            "region": ["R1", "R1"],
            "valid_from": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "valid_to": pd.to_datetime([pd.NaT, pd.NaT]),
        }
    )
    rec = reconcile_time_varying(
        Y_hat_df=_maybe_polars(Y_hat_df, lib),
        ownership_df=_maybe_polars(ownership, lib),
        parent_cols=["region"],
    )
    assert len(reconcile_time_varying.segments_) == 1
    rec_pd = rec if isinstance(rec, pd.DataFrame) else rec.to_pandas()
    lookup = rec_pd.set_index(["unique_id", "ds"])["y_hat/BottomUp"]
    for ds in DATES:
        assert np.isclose(lookup[("R1", ds)], lookup[("A", ds)] + lookup[("B", ds)])


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "mutate, match",
    [
        ("drop_ownership_col", "missing required columns"),
        ("null_region", "null values"),
        ("duplicate_row", "duplicated ownership rows"),
        ("bad_ownership_date", "cannot be parsed"),
        ("null_valid_from", "null values"),
        ("empty_parent_cols", "at least one hierarchy dimension"),
        ("drop_id_col", "must contain columns"),
        ("no_forecast_col", "no numeric forecast columns"),
        ("non_numeric_forecast", "must be numeric"),
        ("null_forecast", "null values"),
        ("bad_ds", "cannot be parsed"),
        ("null_ds", "null or unparseable dates"),
    ],
)
def test_input_validation_branches(lib, Y_hat_df, ownership_df, mutate, match):
    if lib == "polars" and mutate in {
        "non_numeric_forecast",
        "bad_ownership_date",
        "bad_ds",
        "null_ds",
    }:
        # Polars frames are strictly typed: a datetime/numeric column rejects
        # a plain string/null write. These schema-error paths are covered on pandas.
        pytest.skip("strict polars schema rejects the mutation at assignment time")

    Y_hat = Y_hat_df.copy()
    ownership = ownership_df.copy()

    if mutate == "drop_ownership_col":
        ownership = ownership.drop(columns=["region"])
    elif mutate == "null_region":
        ownership.loc[0, "region"] = None
    elif mutate == "duplicate_row":
        ownership = pd.concat([ownership, ownership.iloc[[0]]], ignore_index=True)
    elif mutate == "bad_ownership_date":
        ownership["valid_from"] = ownership["valid_from"].astype(str)
        ownership.loc[0, "valid_from"] = "not-a-date"
    elif mutate == "null_valid_from":
        ownership.loc[0, "valid_from"] = None
    elif mutate == "drop_id_col":
        Y_hat = Y_hat.drop(columns=["unique_id"])
    elif mutate == "no_forecast_col":
        Y_hat = Y_hat[["unique_id", "ds"]]
    elif mutate == "non_numeric_forecast":
        Y_hat["y_hat"] = Y_hat["y_hat"].astype(str)
    elif mutate == "null_forecast":
        Y_hat.loc[0, "y_hat"] = None
    elif mutate == "bad_ds":
        Y_hat["ds"] = Y_hat["ds"].astype(str)
        Y_hat.loc[0, "ds"] = "not-a-date"
    elif mutate == "null_ds":
        Y_hat.loc[0, "ds"] = None

    kwargs = {"parent_cols": [] if mutate == "empty_parent_cols" else ["region"]}

    with pytest.raises(TimeVaryingHierarchyError, match=match):
        reconcile_time_varying(
            Y_hat_df=_maybe_polars(Y_hat, lib),
            ownership_df=_maybe_polars(ownership, lib),
            **kwargs,
        )
