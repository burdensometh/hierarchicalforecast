__all__ = ['LeadTimeCalibration', 'extract_lead_time_calibration']


import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import narwhals.stable.v2 as nw
import numpy as np
from narwhals.typing import Frame


# Fixed-size frequency aliases mapped to numpy datetime64 units. Lead times are
# computed by integer subtraction of period indexes, which requires a constant
# period length, so non-fixed calendars (e.g. month-start) are not supported.
_FREQ_TO_NUMPY_UNIT = {
    "ns": "ns",
    "us": "us",
    "ms": "ms",
    "s": "s",
    "S": "s",
    "min": "m",
    "T": "m",
    "m": "m",
    "h": "h",
    "H": "h",
    "D": "D",
    "W": "W",
}


def _resolve_freq_unit(freq: str) -> str:
    """Map a fixed-size frequency alias to its numpy datetime64 unit."""
    try:
        return _FREQ_TO_NUMPY_UNIT[freq]
    except KeyError:
        valid = sorted(_FREQ_TO_NUMPY_UNIT)
        raise ValueError(
            f"Unsupported freq `{freq}`. Only fixed-size frequencies are "
            f"supported: {valid}. For non-fixed calendars (e.g. month-end), "
            "convert the dates to integer period indexes before calling."
        ) from None


def _to_period_index(
    values: np.ndarray, freq: str, col: str
) -> tuple[np.ndarray, str, str | None]:
    """Convert a date/period column to integer period indexes.

    Args:
        values: Column values as a numpy array.
        freq: Fixed-size frequency used to index datetime columns.
        col: Column name, used in error messages.

    Returns:
        Tuple of (int64 period indexes, kind, unit) where kind is 'datetime'
        or 'integer' and unit is the numpy datetime64 unit (None for integer).
    """
    if np.issubdtype(values.dtype, np.datetime64):
        unit = _resolve_freq_unit(freq)
        periods = values.astype(f"datetime64[{unit}]").astype("int64")
        return periods, "datetime", unit
    if np.issubdtype(values.dtype, np.integer):
        return values.astype("int64"), "integer", None
    if np.issubdtype(values.dtype, np.floating):
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Column `{col}` contains NaN or infinite period values."
            )
        if not np.all(values == np.floor(values)):
            raise ValueError(
                f"Column `{col}` must contain integer period values, "
                "got fractional ones."
            )
        return values.astype("int64"), "integer", None
    if values.dtype == object:
        try:
            return values.astype("int64"), "integer", None
        except (ValueError, TypeError):
            pass
        unit = _resolve_freq_unit(freq)
        try:
            periods = values.astype(f"datetime64[{unit}]").astype("int64")
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Column `{col}` must contain datetimes or integer period "
                f"indexes, got object dtype with values like {values.flat[0]!r}."
            ) from e
        return periods, "datetime", unit
    raise ValueError(
        f"Column `{col}` must contain datetimes or integer period indexes, "
        f"got dtype {values.dtype}."
    )


def _to_period_scalar(
    value: Any, kind: str, unit: str | None, name: str
) -> int:
    """Convert a scalar date/period to a period index of the given kind."""
    if kind == "datetime":
        if isinstance(value, (int, np.integer, float, np.floating)):
            raise ValueError(
                f"`{name}` must be date-like because the calibration columns "
                f"are datetimes, got {value!r}."
            )
        try:
            return int(np.datetime64(value, unit).astype("int64"))
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Could not parse `{name}`={value!r} as a date."
            ) from e
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return int(value)
        raise ValueError(
            f"`{name}` must be an integer period because the calibration "
            f"columns are integer periods, got {value!r}."
        )
    return int(value)


@dataclass
class LeadTimeCalibration:
    """Calibration batches of joint residuals, extracted per forecast lead time.

    This is the output of
    [extract_lead_time_calibration](calibration.html#extract_lead_time_calibration).
    All matrices share the same row order, given by `series_ids`, and have one
    column per complete calibration batch (a forecast vintage, identified by its
    issue date). Column `j` of the lead-`l` matrices corresponds to
    `issue_dates_by_lead[l][j]`.

    Attributes:
        series_ids (np.ndarray): Series identifiers defining the row order of
            every matrix, aligned with the rows of the summing matrix `S`.
        lead_times (list[int]): Requested forecast lead times (in periods).
        current_issue_date (Any): Issue date of the current forecast, as passed
            to the extraction.
        residuals_by_lead (dict[int, np.ndarray]): Joint residuals
            `y - y_hat` of shape (`n_series`, `n_complete_batches`) per lead
            time. Directly usable as the `residuals` argument of
            `Normality(covariance_type='full'|'shrink')`.
        forecasts_by_lead (dict[int, np.ndarray]): Base forecasts `y_hat` of the
            kept batches, same shape as `residuals_by_lead`.
        actuals_by_lead (dict[int, np.ndarray]): Observed values `y` of the kept
            batches, same shape as `residuals_by_lead`. Together with
            `forecasts_by_lead` usable as `y_cal`/`y_hat_cal` of `Conformal`.
        issue_dates_by_lead (dict[int, list]): Issue dates of the complete
            calibration batches per lead time, in matrix column order.
        report (Frame): Per-lead-time summary DataFrame (same backend as the
            input) with the counts of rows, excluded/incomplete batches and the
            available complete batches.
        min_batches (int): Minimum number of complete batches per lead time
            used for the sufficiency flag and warnings.
    """

    series_ids: np.ndarray
    lead_times: list[int]
    current_issue_date: Any
    residuals_by_lead: dict[int, np.ndarray]
    forecasts_by_lead: dict[int, np.ndarray]
    actuals_by_lead: dict[int, np.ndarray]
    issue_dates_by_lead: dict[int, list]
    report: Frame
    min_batches: int

    def __repr__(self) -> str:
        counts = {
            lead: residuals.shape[1]
            for lead, residuals in self.residuals_by_lead.items()
        }
        return (
            f"LeadTimeCalibration(n_series={len(self.series_ids)}, "
            f"current_issue_date={self.current_issue_date!r}, "
            f"complete_batches_by_lead={counts}, "
            f"min_batches={self.min_batches})"
        )


def extract_lead_time_calibration(
    df: Frame,
    current_issue_date: Any,
    horizon: int | None = None,
    lead_times: Sequence[int] | None = None,
    availability_lag: int = 0,
    min_batches: int = 2,
    expected_ids: Sequence | None = None,
    freq: str = "D",
    id_col: str = "unique_id",
    issue_col: str = "issue_date",
    target_col: str = "target_date",
    forecast_col: str = "y_hat",
    actual_col: str = "y",
) -> LeadTimeCalibration:
    r"""Extract joint residual calibration batches per forecast lead time.

    Delivery-style forecast systems store a *vintage* of base forecasts at every
    issuance: the same target (delivery) date appears once per issue date, each
    time at a different lead time `target_date - issue_date`. This function
    organizes such vintage calibration data into per-lead-time batches of joint
    (all-series) residuals, so that coherent prediction intervals can be
    calibrated against errors observed at the *same* lead time as the forecast
    being issued.

    A calibration row is only usable when its actual value is already
    observable at the current issuance, i.e. when
    `target_date + availability_lag <= current_issue_date`; rows whose actuals
    are not yet visible are excluded. For each requested lead time, the
    remaining rows are grouped by issue date into calibration batches, and a
    batch is kept only when it is *complete*: every expected series has a
    non-NaN forecast and a non-NaN actual for that issue date and lead time.

    A `UserWarning` is issued for every lead time with fewer than `min_batches`
    complete calibration batches, detailing how many batches are available, why
    rows/batches were lost (invisible actuals, missing series entries, NaN
    values), and possible remedies.

    Args:
        df (Frame): DataFrame (pandas or polars) with vintage calibration data,
            one row per (`id_col`, `issue_col`, `target_col`), containing the
            base forecast saved at issuance (`forecast_col`) and the observed
            actual (`actual_col`, possibly NaN when not yet observed).
        current_issue_date (Any): Issue date of the forecast being reconciled.
            Must be date-like when the date columns are datetimes, or an
            integer period when they are integers.
        horizon (int, optional): Forecast horizon. When given, calibration is
            extracted for lead times 1..`horizon`. Exactly one of `horizon` and
            `lead_times` must be provided.
        lead_times (Sequence[int], optional): Explicit non-negative lead times
            (in periods) to extract. Exactly one of `horizon` and `lead_times`
            must be provided.
        availability_lag (int, optional): Periods after the target date at
            which its actual becomes visible. A row is usable when
            `target_date + availability_lag <= current_issue_date`.
            Default is 0.
        min_batches (int, optional): Minimum number of complete calibration
            batches required per lead time. A specific `UserWarning` is issued
            for each lead time below this threshold. Default is 2.
        expected_ids (Sequence, optional): Series identifiers expected in every
            complete batch, defining the row order of the output matrices
            (e.g. the row order of the summing matrix `S`). If None, uses the
            sorted unique ids of `df`. Default is None.
        freq (str, optional): Fixed-size frequency of the date columns, used to
            convert dates to period indexes. One of 'ns', 'us', 'ms', 's',
            'min', 'h', 'D', 'W' (and common aliases). Ignored for integer
            period columns. Default is 'D'.
        id_col (str, optional): Column that identifies each series.
            Default is 'unique_id'.
        issue_col (str, optional): Column with the forecast issue (vintage)
            date. Default is 'issue_date'.
        target_col (str, optional): Column with the target (delivery) date of
            each forecast. Default is 'target_date'.
        forecast_col (str, optional): Column with the base forecast saved at
            issuance. Default is 'y_hat'.
        actual_col (str, optional): Column with the observed actual value.
            Default is 'y'.

    Returns:
        LeadTimeCalibration: Per-lead-time joint residuals, forecasts, actuals
            and batch labels, plus a report DataFrame showing, for each lead
            time, the complete calibration batches available and the rows
            excluded because their actuals were not yet visible.

    Raises:
        ValueError: If both or neither of `horizon`/`lead_times` are given.
        ValueError: If required columns are missing, contain nulls (id/date
            columns) or non-numeric values (forecast/actual columns).
        ValueError: If (`id_col`, `issue_col`, `target_col`) combinations are
            duplicated.
        ValueError: If `df` contains ids not present in `expected_ids`.
        ValueError: If `freq` is not a fixed-size frequency and the date
            columns are datetimes.

    Warnings:
        UserWarning: For each lead time with fewer than `min_batches` complete
            calibration batches.

    Examples:
        >>> import pandas as pd
        >>> from hierarchicalforecast.calibration import extract_lead_time_calibration
        >>> df = pd.DataFrame(
        ...     {
        ...         "unique_id": ["a", "b"] * 2,
        ...         "issue_date": pd.to_datetime(["2026-01-01"] * 2 + ["2026-01-02"] * 2),
        ...         "target_date": pd.to_datetime(["2026-01-02"] * 2 + ["2026-01-03"] * 2),
        ...         "y_hat": [1.0, 2.0, 1.5, 2.5],
        ...         "y": [1.1, 1.9, 1.4, 2.6],
        ...     }
        ... )
        >>> cal = extract_lead_time_calibration(
        ...     df, current_issue_date="2026-01-03", lead_times=[1]
        ... )
        >>> cal.residuals_by_lead[1].round(2)
        array([[ 0.1, -0.1],
               [-0.1,  0.1]])
    """
    # ----------------------------- Argument validation ----------------------
    if (horizon is None) == (lead_times is None):
        raise ValueError(
            "Provide exactly one of `horizon` or `lead_times` to select the "
            "lead times of the current forecast."
        )
    if horizon is not None:
        if (
            isinstance(horizon, bool)
            or not isinstance(horizon, (int, np.integer))
            or horizon < 1
        ):
            raise ValueError(f"`horizon` must be a positive integer, got {horizon!r}.")
        leads = list(range(1, int(horizon) + 1))
    else:
        assert lead_times is not None  # guaranteed by the check above
        leads = []
        for lead in lead_times:
            if isinstance(lead, bool) or not isinstance(lead, (int, np.integer)):
                raise ValueError(
                    f"`lead_times` must contain integers, got {lead!r}."
                )
            if lead < 0:
                raise ValueError(
                    f"`lead_times` must be non-negative, got {lead}."
                )
            leads.append(int(lead))
        if not leads:
            raise ValueError("`lead_times` must contain at least one lead time.")
        leads = sorted(set(leads))

    if (
        isinstance(availability_lag, bool)
        or not isinstance(availability_lag, (int, np.integer))
        or availability_lag < 0
    ):
        raise ValueError(
            f"`availability_lag` must be a non-negative integer, got {availability_lag!r}."
        )
    if (
        isinstance(min_batches, bool)
        or not isinstance(min_batches, (int, np.integer))
        or min_batches < 1
    ):
        raise ValueError(
            f"`min_batches` must be a positive integer, got {min_batches!r}."
        )

    # ----------------------------- Input validation -------------------------
    df_nw = nw.from_native(df)
    required_cols = [id_col, issue_col, target_col, forecast_col, actual_col]
    missing_cols = set(required_cols) - set(df_nw.columns)
    if missing_cols:
        raise ValueError(
            f"Check the calibration DataFrame columns, "
            f"{sorted(missing_cols)} must be present."
        )
    if len(df_nw) == 0:
        raise ValueError("The calibration DataFrame is empty.")

    for col in [id_col, issue_col, target_col]:
        if df_nw[col].is_null().any():
            raise ValueError(f"Column `{col}` contains null values.")
    for col in [forecast_col, actual_col]:
        if not df_nw.schema[col].is_numeric():
            raise ValueError(
                f"Column `{col}` contains non-numeric values. Make sure it "
                "only contains numeric values (NaN allowed)."
            )

    # Cast to float64 so missing values are represented as NaN
    df_nw = df_nw.with_columns(
        nw.col(forecast_col).cast(nw.Float64),
        nw.col(actual_col).cast(nw.Float64),
    )

    ids = df_nw[id_col].to_numpy()
    issue_raw = df_nw[issue_col].to_numpy()
    issue_periods, kind, unit = _to_period_index(issue_raw, freq, issue_col)
    target_periods, target_kind, _ = _to_period_index(
        df_nw[target_col].to_numpy(), freq, target_col
    )
    if target_kind != kind:
        raise ValueError(
            f"Columns `{issue_col}` and `{target_col}` must have the same "
            f"kind (both datetimes or both integer periods)."
        )
    y_hat = df_nw[forecast_col].to_numpy()
    y = df_nw[actual_col].to_numpy()

    # Batch labels are the issue dates as they appear in the input data
    label_of_period: dict[int, Any] = {}
    for period, raw in zip(issue_periods, issue_raw, strict=True):
        label_of_period.setdefault(int(period), raw)

    current_period = _to_period_scalar(current_issue_date, kind, unit, "current_issue_date")

    # ----------------------------- Series ordering --------------------------
    if expected_ids is None:
        expected = np.unique(ids)
    else:
        expected = np.asarray(list(expected_ids))
        if expected.size == 0:
            raise ValueError("`expected_ids` must contain at least one id.")
        if np.unique(expected).size != expected.size:
            raise ValueError("`expected_ids` contains duplicated ids.")
        unknown = sorted(set(ids.tolist()) - set(expected.tolist()))
        if unknown:
            raise ValueError(
                f"The calibration DataFrame contains ids not present in "
                f"`expected_ids`: {unknown}."
            )
    n_series = len(expected)
    row_of_id = {v: i for i, v in enumerate(expected.tolist())}
    series_row = np.array([row_of_id[v] for v in ids.tolist()])

    # Duplicate (id, issue, target) rows are ambiguous
    keys = np.stack([series_row, issue_periods, target_periods], axis=1)
    if np.unique(keys, axis=0).shape[0] != len(keys):
        raise ValueError(
            f"Duplicated rows for the same (`{id_col}`, `{issue_col}`, "
            f"`{target_col}`) combination. Keep one row per series, issue "
            "date and target date."
        )

    # --------------------- Lead times and actual visibility -----------------
    lead_arr = target_periods - issue_periods
    # A row's actual is only usable if it is observable at the current issuance
    visible = (target_periods + availability_lag) <= current_period

    # ------------------------- Per-lead-time batches ------------------------
    residuals_by_lead: dict[int, np.ndarray] = {}
    forecasts_by_lead: dict[int, np.ndarray] = {}
    actuals_by_lead: dict[int, np.ndarray] = {}
    issue_dates_by_lead: dict[int, list] = {}
    report_data: dict[str, list] = {
        "lead_time": [],
        "n_rows": [],
        "n_excluded_not_visible": [],
        "n_missing_entries": [],
        "n_rows_missing_actual": [],
        "n_rows_missing_forecast": [],
        "n_batches": [],
        "n_complete_batches": [],
        "n_incomplete_batches": [],
        "min_batches": [],
        "is_sufficient": [],
        "complete_issue_dates": [],
    }

    for lead in leads:
        at_lead = lead_arr == lead
        n_rows = int(at_lead.sum())
        usable = at_lead & visible
        idx = np.nonzero(usable)[0]
        n_excluded = n_rows - int(idx.size)
        n_missing_actual = int(np.isnan(y[idx]).sum())
        n_missing_forecast = int(np.isnan(y_hat[idx]).sum())

        # Group the usable rows by issue date: each issue date is one batch
        batches: dict[int, list[int]] = {}
        for i in idx:
            batches.setdefault(int(issue_periods[i]), []).append(int(i))

        kept_residuals = []
        kept_forecasts = []
        kept_actuals = []
        kept_issue_dates = []
        n_incomplete = 0
        n_missing_entries = 0
        for issue_period in sorted(batches):
            rows = batches[issue_period]
            rows_idx = np.asarray(rows)
            batch_residuals = np.full(n_series, np.nan)
            batch_forecasts = np.full(n_series, np.nan)
            batch_actuals = np.full(n_series, np.nan)
            batch_rows = series_row[rows_idx]
            batch_actuals[batch_rows] = y[rows_idx]
            batch_forecasts[batch_rows] = y_hat[rows_idx]
            batch_residuals[batch_rows] = y[rows_idx] - y_hat[rows_idx]
            if np.isnan(batch_residuals).any():
                # Incomplete batch: missing series or NaN values
                n_incomplete += 1
                n_missing_entries += n_series - len(rows)
                continue
            kept_residuals.append(batch_residuals)
            kept_forecasts.append(batch_forecasts)
            kept_actuals.append(batch_actuals)
            kept_issue_dates.append(label_of_period[issue_period])

        n_complete = len(kept_residuals)
        if n_complete:
            residuals_by_lead[lead] = np.stack(kept_residuals, axis=1)
            forecasts_by_lead[lead] = np.stack(kept_forecasts, axis=1)
            actuals_by_lead[lead] = np.stack(kept_actuals, axis=1)
        else:
            empty = np.empty((n_series, 0))
            residuals_by_lead[lead] = empty
            forecasts_by_lead[lead] = empty.copy()
            actuals_by_lead[lead] = empty.copy()
        issue_dates_by_lead[lead] = kept_issue_dates

        report_data["lead_time"].append(lead)
        report_data["n_rows"].append(n_rows)
        report_data["n_excluded_not_visible"].append(n_excluded)
        report_data["n_missing_entries"].append(n_missing_entries)
        report_data["n_rows_missing_actual"].append(n_missing_actual)
        report_data["n_rows_missing_forecast"].append(n_missing_forecast)
        report_data["n_batches"].append(len(batches))
        report_data["n_complete_batches"].append(n_complete)
        report_data["n_incomplete_batches"].append(n_incomplete)
        report_data["min_batches"].append(int(min_batches))
        report_data["is_sufficient"].append(n_complete >= min_batches)
        report_data["complete_issue_dates"].append(kept_issue_dates)

        if n_complete < min_batches:
            zero_note = (
                " No calibration data available for this lead time."
                if n_complete == 0
                else ""
            )
            warnings.warn(
                f"Lead time {lead}: only {n_complete} complete calibration "
                f"batch(es) available, fewer than min_batches={min_batches}."
                f"{zero_note} {n_excluded} row(s) were excluded because their "
                f"actuals are not yet visible at the current issue date "
                f"({current_issue_date!r}) with availability_lag={availability_lag}; "
                f"{n_incomplete} batch(es) were dropped as incomplete "
                f"({n_missing_entries} missing series value(s), "
                f"{n_missing_actual} row(s) with missing actuals, "
                f"{n_missing_forecast} row(s) with missing forecasts). "
                "Consider collecting more forecast vintages, shortening the "
                "forecast horizon, or lowering `min_batches`.",
                UserWarning,
                stacklevel=2,
            )

    backend = nw.get_native_namespace(df)
    report = nw.from_dict(report_data, backend=backend).to_native()

    return LeadTimeCalibration(
        series_ids=expected,
        lead_times=leads,
        current_issue_date=current_issue_date,
        residuals_by_lead=residuals_by_lead,
        forecasts_by_lead=forecasts_by_lead,
        actuals_by_lead=actuals_by_lead,
        issue_dates_by_lead=issue_dates_by_lead,
        report=report,
        min_batches=int(min_batches),
    )
