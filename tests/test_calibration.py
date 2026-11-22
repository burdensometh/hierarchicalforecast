import warnings

import numpy as np
import pandas as pd
import pytest

from hierarchicalforecast.calibration import (
    LeadTimeCalibration,
    extract_lead_time_calibration,
)
from hierarchicalforecast.probabilistic_methods import Conformal, Normality


def _make_vintage_df(
    ids=("agg", "b1", "b2"),
    n_vintages=10,
    max_lead=4,
    start="2026-01-01",
    step=pd.Timedelta(days=1),
):
    """Full-grid vintage calibration data.

    For every vintage (issue date) and every lead time 1..max_lead there is
    one row per series. Residuals follow a deterministic pattern:
    residual(series j, vintage v) = 0.1 * (v + 1) * (j + 1) * (-1)^j.
    """
    rows = []
    base = pd.Timestamp(start)
    for v in range(n_vintages):
        issue = base + v * step
        for lead in range(1, max_lead + 1):
            target = issue + lead * step
            for j, uid in enumerate(ids):
                y_hat = 10.0 * (j + 1) + lead
                y = y_hat + 0.1 * (v + 1) * (j + 1) * (1 if j % 2 == 0 else -1)
                rows.append((uid, issue, target, y_hat, y))
    return pd.DataFrame(
        rows, columns=["unique_id", "issue_date", "target_date", "y_hat", "y"]
    )


# 10 vintages starting 2026-01-01; current issue date is the last vintage.
CURRENT_ISSUE_DATE = "2026-01-10"


@pytest.fixture
def vintage_df():
    return _make_vintage_df()


def test_basic_extraction_per_lead_time(vintage_df):
    cal = extract_lead_time_calibration(
        vintage_df, current_issue_date=CURRENT_ISSUE_DATE, horizon=4
    )
    assert isinstance(cal, LeadTimeCalibration)
    assert cal.lead_times == [1, 2, 3, 4]
    # lead l keeps the vintages whose target date is already observable:
    # issue + l <= current -> 10 - l complete batches
    expected_batches = {1: 9, 2: 8, 3: 7, 4: 6}
    for lead, n_batches in expected_batches.items():
        residuals = cal.residuals_by_lead[lead]
        assert residuals.shape == (3, n_batches)
        assert cal.forecasts_by_lead[lead].shape == (3, n_batches)
        assert cal.actuals_by_lead[lead].shape == (3, n_batches)
        assert len(cal.issue_dates_by_lead[lead]) == n_batches

    # series order follows the sorted ids of the input
    assert list(cal.series_ids) == ["agg", "b1", "b2"]

    # residual pattern: 0.1 * (v + 1) * (j + 1) * (-1)^j, vintage v is column v
    np.testing.assert_allclose(
        cal.residuals_by_lead[1][0, :], 0.1 * np.arange(1, 10)
    )
    np.testing.assert_allclose(
        cal.residuals_by_lead[1][1, :], -0.2 * np.arange(1, 10)
    )
    # residuals are consistent with forecasts and actuals
    np.testing.assert_allclose(
        cal.residuals_by_lead[2],
        cal.actuals_by_lead[2] - cal.forecasts_by_lead[2],
    )

    # batch labels are the issue dates of the usable vintages
    expected_issues = [
        np.datetime64("2026-01-01") + np.timedelta64(v, "D") for v in range(9)
    ]
    assert cal.issue_dates_by_lead[1] == expected_issues
    # lead 4 loses the 4 most recent vintages
    assert cal.issue_dates_by_lead[4][-1] == np.datetime64("2026-01-06")


def test_rows_with_invisible_actuals_are_excluded(vintage_df):
    cal = extract_lead_time_calibration(
        vintage_df, current_issue_date=CURRENT_ISSUE_DATE, horizon=4
    )
    report = cal.report.set_index("lead_time")
    # 30 rows per lead time (10 vintages x 3 series); l vintages per lead
    # have target dates later than the current issue date
    for lead in [1, 2, 3, 4]:
        assert report.loc[lead, "n_rows"] == 30
        assert report.loc[lead, "n_excluded_not_visible"] == 3 * lead
        assert report.loc[lead, "n_batches"] == 10 - lead
        assert report.loc[lead, "n_complete_batches"] == 10 - lead
        assert report.loc[lead, "n_incomplete_batches"] == 0
        assert report.loc[lead, "is_sufficient"]
    # no excluded row leaks into the batches: the most recent usable vintage
    # for lead 2 is current_issue_date - 2 periods
    assert cal.issue_dates_by_lead[2][-1] == np.datetime64("2026-01-08")


def test_availability_lag_postpones_visibility(vintage_df):
    cal = extract_lead_time_calibration(
        vintage_df,
        current_issue_date=CURRENT_ISSUE_DATE,
        horizon=2,
        availability_lag=1,
    )
    # with lag=1, target + 1 <= current -> one fewer batch per lead
    assert cal.residuals_by_lead[1].shape[1] == 8
    assert cal.residuals_by_lead[2].shape[1] == 7
    report = cal.report.set_index("lead_time")
    assert report.loc[1, "n_excluded_not_visible"] == 3 * 2
    assert report.loc[2, "n_excluded_not_visible"] == 3 * 3


def test_incomplete_batch_is_dropped(vintage_df):
    # remove a single row: the (2026-01-05, lead 1) batch misses series b2
    mask = ~(
        (vintage_df["unique_id"] == "b2")
        & (vintage_df["issue_date"] == "2026-01-05")
        & (vintage_df["target_date"] == "2026-01-06")
    )
    cal = extract_lead_time_calibration(
        vintage_df[mask], current_issue_date=CURRENT_ISSUE_DATE, horizon=1
    )
    assert cal.residuals_by_lead[1].shape[1] == 8
    report = cal.report.set_index("lead_time")
    assert report.loc[1, "n_batches"] == 9
    assert report.loc[1, "n_complete_batches"] == 8
    assert report.loc[1, "n_incomplete_batches"] == 1
    assert report.loc[1, "n_missing_entries"] == 1
    # the incomplete vintage is absent from the batch labels
    assert np.datetime64("2026-01-05") not in cal.issue_dates_by_lead[1]


def test_missing_whole_series_is_diagnosed(vintage_df):
    df = vintage_df[vintage_df["unique_id"] != "b2"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cal = extract_lead_time_calibration(
            df,
            current_issue_date=CURRENT_ISSUE_DATE,
            lead_times=[1],
            expected_ids=["agg", "b1", "b2"],
        )
    # every batch misses one series value, so no batch is complete
    assert cal.residuals_by_lead[1].shape == (3, 0)
    message = str(caught[0].message)
    assert "missing series value" in message
    report = cal.report.set_index("lead_time")
    assert report.loc[1, "n_missing_entries"] == 9


def test_nan_actual_marks_batch_incomplete(vintage_df):
    mask = (vintage_df["issue_date"] == "2026-01-04") & (
        vintage_df["target_date"] == "2026-01-05"
    )
    vintage_df.loc[mask, "y"] = np.nan
    cal = extract_lead_time_calibration(
        vintage_df, current_issue_date=CURRENT_ISSUE_DATE, horizon=1
    )
    report = cal.report.set_index("lead_time")
    assert report.loc[1, "n_rows_missing_actual"] == 3
    assert report.loc[1, "n_complete_batches"] == 8
    assert report.loc[1, "n_incomplete_batches"] == 1


def test_nan_forecast_marks_batch_incomplete(vintage_df):
    mask = (vintage_df["issue_date"] == "2026-01-04") & (
        vintage_df["target_date"] == "2026-01-05"
    )
    vintage_df.loc[mask, "y_hat"] = np.nan
    cal = extract_lead_time_calibration(
        vintage_df, current_issue_date=CURRENT_ISSUE_DATE, horizon=1
    )
    report = cal.report.set_index("lead_time")
    assert report.loc[1, "n_rows_missing_forecast"] == 3
    assert report.loc[1, "n_complete_batches"] == 8


def test_warns_specifically_when_too_few_batches(vintage_df):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cal = extract_lead_time_calibration(
            vintage_df,
            current_issue_date=CURRENT_ISSUE_DATE,
            horizon=4,
            min_batches=8,
        )
    messages = [str(w.message) for w in caught]
    # leads 1..4 have 9, 8, 7, 6 batches: leads 3 and 4 are below 8
    assert len(messages) == 2
    assert "Lead time 3" in messages[0]
    assert "only 7 complete calibration batch(es)" in messages[0]
    assert "min_batches=8" in messages[0]
    assert "not yet visible" in messages[0]
    assert "Lead time 4" in messages[1]
    assert "only 6 complete calibration batch(es)" in messages[1]
    report = cal.report.set_index("lead_time")
    assert report.loc[1, "is_sufficient"]
    assert not report.loc[3, "is_sufficient"]
    assert not report.loc[4, "is_sufficient"]


def test_warns_when_no_batches_available(vintage_df):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cal = extract_lead_time_calibration(
            vintage_df,
            current_issue_date=CURRENT_ISSUE_DATE,
            lead_times=[12],
        )
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "Lead time 12" in message
    assert "only 0 complete calibration batch(es)" in message
    assert "No calibration data available" in message
    assert cal.residuals_by_lead[12].shape == (3, 0)
    assert cal.issue_dates_by_lead[12] == []


def test_no_warning_when_enough_batches(vintage_df):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        extract_lead_time_calibration(
            vintage_df, current_issue_date=CURRENT_ISSUE_DATE, horizon=4
        )
    assert not caught


def test_integer_period_columns():
    df = _make_vintage_df()
    base = pd.Timestamp("2026-01-01")
    df["issue_date"] = (df["issue_date"] - base).dt.days + 1
    df["target_date"] = (df["target_date"] - base).dt.days + 1
    cal = extract_lead_time_calibration(
        df, current_issue_date=10, lead_times=[1, 2]
    )
    assert cal.residuals_by_lead[1].shape == (3, 9)
    assert cal.residuals_by_lead[2].shape == (3, 8)
    assert cal.issue_dates_by_lead[1] == list(range(1, 10))


def test_expected_ids_define_row_order(vintage_df):
    cal = extract_lead_time_calibration(
        vintage_df,
        current_issue_date=CURRENT_ISSUE_DATE,
        horizon=1,
        expected_ids=["b2", "agg", "b1"],
    )
    assert list(cal.series_ids) == ["b2", "agg", "b1"]
    # row 0 is now 'b2' (j=2 in the fixture: residual +0.3 * (v + 1))
    np.testing.assert_allclose(
        cal.residuals_by_lead[1][0, :], 0.3 * np.arange(1, 10)
    )
    # row 1 is 'agg' (j=0 in the fixture: residual +0.1 * (v + 1))
    np.testing.assert_allclose(
        cal.residuals_by_lead[1][1, :], 0.1 * np.arange(1, 10)
    )


def test_unknown_ids_raise_with_expected_ids(vintage_df):
    with pytest.raises(ValueError, match="not present in `expected_ids`"):
        extract_lead_time_calibration(
            vintage_df,
            current_issue_date=CURRENT_ISSUE_DATE,
            horizon=1,
            expected_ids=["agg", "b1"],
        )


def test_horizon_and_lead_times_are_mutually_exclusive(vintage_df):
    with pytest.raises(ValueError, match="exactly one of `horizon` or `lead_times`"):
        extract_lead_time_calibration(
            vintage_df, current_issue_date=CURRENT_ISSUE_DATE
        )
    with pytest.raises(ValueError, match="exactly one of `horizon` or `lead_times`"):
        extract_lead_time_calibration(
            vintage_df,
            current_issue_date=CURRENT_ISSUE_DATE,
            horizon=2,
            lead_times=[1],
        )


def test_invalid_arguments_raise(vintage_df):
    with pytest.raises(ValueError, match="`horizon` must be a positive integer"):
        extract_lead_time_calibration(
            vintage_df, current_issue_date=CURRENT_ISSUE_DATE, horizon=0
        )
    with pytest.raises(ValueError, match="`lead_times` must be non-negative"):
        extract_lead_time_calibration(
            vintage_df, current_issue_date=CURRENT_ISSUE_DATE, lead_times=[-1]
        )
    with pytest.raises(ValueError, match="`min_batches` must be a positive integer"):
        extract_lead_time_calibration(
            vintage_df,
            current_issue_date=CURRENT_ISSUE_DATE,
            horizon=1,
            min_batches=0,
        )
    with pytest.raises(ValueError, match="`availability_lag` must be a non-negative"):
        extract_lead_time_calibration(
            vintage_df,
            current_issue_date=CURRENT_ISSUE_DATE,
            horizon=1,
            availability_lag=-1,
        )


def test_duplicated_rows_raise(vintage_df):
    df = pd.concat([vintage_df, vintage_df.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicated rows"):
        extract_lead_time_calibration(
            df, current_issue_date=CURRENT_ISSUE_DATE, horizon=1
        )


def test_missing_columns_raise(vintage_df):
    df = vintage_df.drop(columns=["target_date"])
    with pytest.raises(ValueError, match="target_date"):
        extract_lead_time_calibration(
            df, current_issue_date=CURRENT_ISSUE_DATE, horizon=1
        )


def test_null_dates_raise(vintage_df):
    vintage_df.loc[0, "issue_date"] = pd.NaT
    with pytest.raises(ValueError, match="`issue_date` contains null values"):
        extract_lead_time_calibration(
            vintage_df, current_issue_date=CURRENT_ISSUE_DATE, horizon=1
        )


def test_non_numeric_values_raise(vintage_df):
    vintage_df["y_hat"] = vintage_df["y_hat"].astype(str)
    with pytest.raises(ValueError, match="`y_hat` contains non-numeric values"):
        extract_lead_time_calibration(
            vintage_df, current_issue_date=CURRENT_ISSUE_DATE, horizon=1
        )


def test_weekly_frequency():
    # weekly grid: lead 1 means one week ahead
    df = _make_vintage_df(
        n_vintages=6, max_lead=2, start="2026-01-05", step=pd.Timedelta(weeks=1)
    )
    cal = extract_lead_time_calibration(
        df,
        current_issue_date=pd.Timestamp("2026-01-05") + pd.Timedelta(weeks=5),
        horizon=2,
        freq="W",
    )
    assert cal.residuals_by_lead[1].shape == (3, 5)
    assert cal.residuals_by_lead[2].shape == (3, 4)
    assert cal.issue_dates_by_lead[1][0] == np.datetime64("2026-01-05")


def test_non_fixed_frequency_raises(vintage_df):
    with pytest.raises(ValueError, match="Only fixed-size frequencies"):
        extract_lead_time_calibration(
            vintage_df,
            current_issue_date=CURRENT_ISSUE_DATE,
            horizon=1,
            freq="MS",
        )


def test_current_issue_date_type_must_match_columns(vintage_df):
    with pytest.raises(ValueError, match="must be date-like"):
        extract_lead_time_calibration(
            vintage_df, current_issue_date=10, horizon=1
        )
    df = _make_vintage_df()
    df["issue_date"] = 1
    df["target_date"] = 2
    with pytest.raises(ValueError, match="must be an integer period"):
        extract_lead_time_calibration(
            df, current_issue_date="2026-01-10", horizon=1
        )


def test_report_structure(vintage_df):
    cal = extract_lead_time_calibration(
        vintage_df, current_issue_date=CURRENT_ISSUE_DATE, horizon=2
    )
    report = cal.report
    assert isinstance(report, pd.DataFrame)
    assert list(report.columns) == [
        "lead_time",
        "n_rows",
        "n_excluded_not_visible",
        "n_missing_entries",
        "n_rows_missing_actual",
        "n_rows_missing_forecast",
        "n_batches",
        "n_complete_batches",
        "n_incomplete_batches",
        "min_batches",
        "is_sufficient",
        "complete_issue_dates",
    ]
    assert report["lead_time"].tolist() == [1, 2]
    assert len(report["complete_issue_dates"].iloc[0]) == 9
    assert "LeadTimeCalibration" in repr(cal)
    assert "complete_batches_by_lead={1: 9, 2: 8}" in repr(cal)


def test_extracted_batches_feed_probabilistic_samplers(vintage_df):
    """The extracted per-lead-time batches plug into coherent intervals."""
    cal = extract_lead_time_calibration(
        vintage_df, current_issue_date=CURRENT_ISSUE_DATE, horizon=4
    )
    # rows of the matrices follow cal.series_ids = ['agg', 'b1', 'b2']
    S = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    P = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    y_hat = np.arange(12, dtype=np.float64).reshape(3, 4) + 10

    # Conformal consumes the actuals/forecasts of one lead time
    conformal = Conformal(
        S=S,
        P=P,
        y_hat=y_hat,
        y_cal=cal.actuals_by_lead[1],
        y_hat_cal=cal.forecasts_by_lead[1],
    )
    res = conformal.get_prediction_levels({"mean": S @ P @ y_hat}, level=[80])
    assert res["lo-80"].shape == (3, 4)
    assert res["hi-80"].shape == (3, 4)

    # Normality consumes the joint residuals of one lead time
    normality = Normality(
        S=S,
        P=P,
        y_hat=y_hat,
        sigmah=np.ones((3, 4)),
        covariance_type="shrink",
        residuals=cal.residuals_by_lead[1],
    )
    samples = normality.get_samples(num_samples=10)
    assert samples.shape == (3, 4, 10)
