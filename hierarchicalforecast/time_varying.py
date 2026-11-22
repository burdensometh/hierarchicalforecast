"""Time-varying hierarchical reconciliation.

A standard hierarchy assumes the parent of every bottom-level series is fixed
for the whole forecast horizon. In practice, organisational boundaries move:
an outlet may be handed over from one region to another on a known date while
a quarterly forecast spans that date. This module reconciles such forecasts
by splitting the horizon at the ownership boundary dates, building the
hierarchy that was *in effect* on each segment, reconciling every segment
independently, and stitching the dated results back together.

The ownership information is supplied as non-overlapping validity intervals
per bottom-level series (see :func:`reconcile_time_varying`). Two data
problems are detected explicitly:

* a forecast date on which a bottom-level series has no valid ownership row
  (:class:`MissingOwnershipError`), and
* a date on which more than one ownership row is valid for the same series
  (:class:`DuplicateOwnershipError`).
"""

__all__ = [
    "TimeVaryingHierarchyError",
    "MissingOwnershipError",
    "DuplicateOwnershipError",
    "reconcile_time_varying",
]

from typing import Any

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
from narwhals.typing import Frame, FrameT

from .core import HierarchicalReconciliation
from .methods import BottomUp
from .utils import aggregate


class TimeVaryingHierarchyError(ValueError):
    """Base class for time-varying hierarchy validation errors."""


class MissingOwnershipError(TimeVaryingHierarchyError):
    """Raised when a forecast date is not covered by any ownership interval."""


class DuplicateOwnershipError(TimeVaryingHierarchyError):
    """Raised when several ownership intervals cover the same forecast date."""


def _as_pandas(df: Frame, name: str) -> pd.DataFrame:
    """Convert a pandas/polars (narwhals-compatible) frame to pandas."""
    if not isinstance(df, pd.DataFrame):
        try:
            df = nw.from_native(df).to_pandas()
        except Exception as exc:  # pragma: no cover - defensive
            raise TypeError(
                f"`{name}` must be a pandas or polars DataFrame, got {type(df)!r}."
            ) from exc
    return df


def _coerce_date(series: pd.Series, col: str, frame: str) -> pd.Series:
    """Coerce a column to pandas datetimes, rejecting unparseable values."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    parsed = pd.to_datetime(series, errors="coerce")
    bad = series[parsed.isna() & series.notna()]
    if len(bad) > 0:
        example = repr(bad.iloc[0])
        raise TimeVaryingHierarchyError(
            f"Column `{col}` in `{frame}` contains values that cannot be parsed "
            f"as dates, e.g. {example}."
        )
    return parsed


def _validate_ownership_columns(
    ownership: pd.DataFrame,
    bottom_col: str,
    parent_cols: list[str],
    valid_from_col: str,
    valid_to_col: str,
) -> None:
    required = [bottom_col, *parent_cols, valid_from_col, valid_to_col]
    missing = [c for c in required if c not in ownership.columns]
    if missing:
        raise TimeVaryingHierarchyError(
            f"`ownership_df` is missing required columns: {missing}. "
            f"Available columns: {ownership.columns.tolist()}."
        )
    null_cols = [c for c in required if ownership[c].isna().any() and c != valid_to_col]
    if null_cols:
        raise TimeVaryingHierarchyError(
            f"`ownership_df` contains null values in columns {null_cols}. "
            f"Only `{valid_to_col}` may be null (an open-ended interval)."
        )
    duplicated_rows = ownership.duplicated(subset=required)
    if duplicated_rows.any():
        raise TimeVaryingHierarchyError(
            "`ownership_df` contains duplicated ownership rows: "
            f"{ownership[duplicated_rows].head().to_dict('records')}."
        )


def _detect_ownership_problems(
    ownership: pd.DataFrame,
    forecast_dates: pd.Series,
    bottom_col: str,
    valid_from_col: str,
    valid_to_col: str,
) -> None:
    """Locate dates with missing or duplicate ownership for each bottom node.

    Intervals are treated as closed on both ends: a row with
    ``[valid_from, valid_to]`` is active on every forecast date ``d`` with
    ``valid_from <= d <= valid_to``; a null ``valid_to`` means the interval
    stays open until the end of the horizon.
    """
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(forecast_dates.unique())))
    missing: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []

    for bottom_id, group in ownership.groupby(bottom_col, sort=True):
        starts = group[valid_from_col].to_numpy(dtype="datetime64[ns]")
        ends = group[valid_to_col].to_numpy(dtype="datetime64[ns]")
        open_end = group[valid_to_col].isna().to_numpy()
        date_np = dates.to_numpy(dtype="datetime64[ns]")

        # coverage[i, j] == 1 when row i covers forecast date j
        coverage = (starts[:, None] <= date_np[None, :]) & (
            open_end[:, None] | (ends[:, None] >= date_np[None, :])
        )
        n_cover = coverage.sum(axis=0)

        miss_idx = np.flatnonzero(n_cover == 0)
        for j in miss_idx:
            missing.append({"bottom_id": bottom_id, "ds": dates[j]})

        dup_idx = np.flatnonzero(n_cover > 1)
        for j in dup_idx:
            rows = group.iloc[np.flatnonzero(coverage[:, j])]
            duplicates.append(
                {
                    "bottom_id": bottom_id,
                    "ds": dates[j],
                    "rows": rows.to_dict("records"),
                }
            )

    if duplicates:
        first = duplicates[0]
        parts = []
        for row in first["rows"]:
            start = pd.Timestamp(row[valid_from_col]).date()
            end = row[valid_to_col]
            end = "open" if pd.isna(end) else pd.Timestamp(end).date()
            parts.append(f"{start}..{end}")
        raise DuplicateOwnershipError(
            f"Bottom-level series {first['bottom_id']!r} has multiple ownership "
            f"intervals active on {pd.Timestamp(first['ds']).date()} "
            f"({'; '.join(parts)}). Total duplicate dates: {len(duplicates)}."
        )
    if missing:
        detail = ", ".join(
            f"{m['bottom_id']!r} on {pd.Timestamp(m['ds']).date()}" for m in missing[:5]
        )
        more = "" if len(missing) <= 5 else f" (and {len(missing) - 5} more)"
        raise MissingOwnershipError(
            f"No valid ownership interval covers the forecast date(s): {detail}{more}."
        )


def _build_segment_hierarchy(
    segment_dates: pd.DatetimeIndex,
    ownership: pd.DataFrame,
    Y_hat: pd.DataFrame,
    bottom_col: str,
    parent_cols: list[str],
    id_col: str,
    time_col: str,
    target_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """Build the hierarchy in effect on a segment and aggregate forecasts.

    Returns the aggregated forecast frame, summing matrix, tags, and the
    bottom-level frame used to build them.
    """
    first_date = segment_dates[0]
    from_col = "__valid_from__"
    to_col = "__valid_to__"
    # Ownership is constant inside a segment, so the rows active on its first
    # date define the hierarchy for the whole segment.
    active = ownership[
        (ownership[from_col] <= first_date)
        & (ownership[to_col].isna() | (ownership[to_col] >= first_date))
    ]

    dim_cols = [bottom_col, *parent_cols]
    active_dims = active[dim_cols].drop_duplicates()

    # Bottom-level forecasts restricted to this segment, joined with the
    # dimensions that are in effect on it. ``aggregate`` always *creates* the
    # id column itself by concatenating the spec columns, so the input frame
    # must not carry an id column: keep the bottom identity in a dedicated dim
    # and drop any pre-existing id column before aggregating. Bottom node ids
    # are reconstructed afterwards from the original bottom identifiers.
    internal_bottom = "__bottom_dim__"
    seg_fc = Y_hat[
        Y_hat[time_col].isin(segment_dates)
        & Y_hat[id_col].isin(active_dims[bottom_col])
    ].copy()

    # Rename the bottom identifier in the dimension frame so the merge keeps
    # the forecast frame's id column and the bottom dimension column apart,
    # regardless of whether bottom_col and id_col have the same name.
    active_dims = active_dims.rename(columns={bottom_col: internal_bottom})
    bottom_fc = seg_fc.merge(
        active_dims, left_on=id_col, right_on=internal_bottom, how="left"
    )

    # Composite bottom id (built by aggregate) -> original series identifier.
    composite = (
        bottom_fc[parent_cols].astype(str).agg("/".join, axis=1)
        + "/"
        + bottom_fc[internal_bottom].astype(str)
    )
    bottom_id_map = (
        pd.DataFrame({"composite": composite, id_col: bottom_fc[id_col]})
        .drop_duplicates("composite")
        .set_index("composite")[id_col]
        .to_dict()
    )
    bottom_fc = bottom_fc.drop(columns=[id_col])

    spec = [[c] for c in parent_cols] + [[*parent_cols, internal_bottom]]
    Y_seg, S_seg, tags_seg = aggregate(
        bottom_fc,
        spec=spec,
        id_col=id_col,
        time_col=time_col,
        target_cols=target_cols,
    )
    # ``aggregate`` names bottom nodes by joining every dimension in the
    # bottom spec (e.g. "R1/A"). Map those composite ids, used both as row
    # labels and as bottom-column labels of S, back to the original bottom
    # identifiers. The bottom block stays an identity matrix because the
    # renaming is a pure bijection.
    Y_seg[id_col] = Y_seg[id_col].replace(bottom_id_map)
    S_seg = S_seg.copy()
    S_seg[id_col] = S_seg[id_col].replace(bottom_id_map)
    S_seg = S_seg.rename(columns=bottom_id_map)
    internal_bottom_tag = "/".join([*parent_cols, internal_bottom])
    bottom_tag = "/".join([*parent_cols, bottom_col])
    tags_seg[bottom_tag] = np.asarray(
        [bottom_id_map.get(nid, nid) for nid in tags_seg.pop(internal_bottom_tag)]
    )
    return Y_seg, S_seg, tags_seg, bottom_fc


def reconcile_time_varying(
    Y_hat_df: Frame,
    ownership_df: Frame,
    parent_cols: list[str],
    bottom_col: str = "unique_id",
    id_col: str = "unique_id",
    time_col: str = "ds",
    valid_from_col: str = "valid_from",
    valid_to_col: str = "valid_to",
    reconciler: Any | None = None,
) -> FrameT:
    """Reconcile forecasts across a hierarchy that changes on known dates.

    The forecast horizon is split at the ownership boundary dates. On each
    resulting segment the hierarchy in effect is built from
    ``ownership_df``, the base forecasts are aggregated bottom-up, and
    :class:`~hierarchicalforecast.core.HierarchicalReconciliation` is run with
    a bottom-up reconciler. Reconciled segments are then concatenated back
    into a single dated frame.

    Args:
        Y_hat_df: Base forecasts. Long frame with at least ``id_col`` and
            ``time_col`` plus one or more numeric forecast columns. It must
            contain the bottom-level series; aggregate-level rows are allowed
            as additional base forecasts but will be overwritten by the
            reconciled aggregates. Every bottom-level series must carry a
            forecast on every forecast date.
        ownership_df: Non-overlapping ownership intervals for the
            bottom-level series. Must contain ``bottom_col``, every column in
            ``parent_cols``, ``valid_from_col`` and ``valid_to_col``.
            Intervals are closed on both ends; a null ``valid_to`` marks an
            open-ended interval. Intervals for one bottom-level series must
            not overlap and must jointly cover every forecast date.
        parent_cols: Dimension columns defining the hierarchy, from the
            coarsest to the finest level above the bottom, e.g.
            ``["country", "region"]``.
        bottom_col: Column in ``ownership_df`` holding the bottom-level
            series identifier. Defaults to ``"unique_id"``.
        id_col: Column in ``Y_hat_df`` identifying each series. Defaults to
            ``"unique_id"``.
        time_col: Column identifying each timestamp. Defaults to ``"ds"``.
        valid_from_col: Column with the inclusive interval start. Defaults to
            ``"valid_from"``.
        valid_to_col: Column with the inclusive interval end (null = open).
            Defaults to ``"valid_to"``.
        reconciler: An optional non-insample reconciler instance. Defaults to
            :class:`~hierarchicalforecast.methods.BottomUp`, which matches the
            bottom-up coordination requested. Any reconciler passed here must
            work without ``Y_df`` (``insample=False``).

    Returns:
        FrameT: Reconciled forecasts with the same backend (pandas/polars) as
        ``Y_hat_df``, containing one row per ``(id_col, time_col)`` pair for
        every node that exists on at least one segment, and a
        ``<model>/<reconciler>`` column for every base forecast column.

    Raises:
        MissingOwnershipError: If a bottom-level series has no valid
            ownership row on any forecast date.
        DuplicateOwnershipError: If more than one ownership row is valid for
            the same series on the same forecast date.
        TimeVaryingHierarchyError: For any other malformed input.

    Note:
        After a successful call, ``reconcile_time_varying.segments_`` holds a
        list of dictionaries describing the segments that were reconciled
        (boundary dates, node counts, and the tags of the in-effect hierarchy).
    """
    if reconciler is None:
        reconciler = BottomUp()
    if getattr(reconciler, "insample", False):
        raise TimeVaryingHierarchyError(
            f"Reconciler {type(reconciler).__name__} requires in-sample data "
            "(insample=True), which is not supported across time-varying segments."
        )

    Y_hat = _as_pandas(Y_hat_df, "Y_hat_df")
    ownership = _as_pandas(ownership_df, "ownership_df")
    parent_cols = list(parent_cols)
    if not parent_cols:
        raise TimeVaryingHierarchyError(
            "`parent_cols` must contain at least one hierarchy dimension column."
        )

    # ----- frame schema checks ------------------------------------------------
    if id_col not in Y_hat.columns or time_col not in Y_hat.columns:
        raise TimeVaryingHierarchyError(
            f"`Y_hat_df` must contain columns {id_col!r} and {time_col!r}, "
            f"found {Y_hat.columns.tolist()}."
        )
    target_cols = [
        c
        for c in Y_hat.columns
        if c not in (id_col, time_col)
        and not ("-lo" in c or "-hi" in c or "-median" in c or "-sample" in c)
    ]
    if not target_cols:
        raise TimeVaryingHierarchyError(
            "`Y_hat_df` has no numeric forecast columns to reconcile."
        )
    for col in target_cols:
        if not pd.api.types.is_numeric_dtype(Y_hat[col]):
            raise TimeVaryingHierarchyError(f"Forecast column `{col}` must be numeric.")
        if Y_hat[col].isna().any():
            raise TimeVaryingHierarchyError(
                f"Forecast column `{col}` contains null values."
            )

    _validate_ownership_columns(
        ownership, bottom_col, parent_cols, valid_from_col, valid_to_col
    )
    ownership = ownership.copy()
    ownership["__valid_from__"] = _coerce_date(
        ownership[valid_from_col], valid_from_col, "ownership_df"
    )
    ownership["__valid_to__"] = _coerce_date(
        ownership[valid_to_col].fillna(pd.NaT), valid_to_col, "ownership_df"
    )
    Y_hat = Y_hat.copy()
    Y_hat[time_col] = _coerce_date(Y_hat[time_col], time_col, "Y_hat_df")
    if Y_hat[time_col].isna().any():
        raise TimeVaryingHierarchyError(
            f"`Y_hat_df` column `{time_col}` contains null or unparseable dates."
        )

    closed_rows = ownership["__valid_to__"].notna()
    reversed_intervals = (
        ownership.loc[closed_rows, "__valid_to__"]
        < ownership.loc[closed_rows, "__valid_from__"]
    )
    if reversed_intervals.any():
        raise TimeVaryingHierarchyError(
            "`ownership_df` contains intervals with valid_to earlier than valid_from."
        )

    forecast_dates = pd.Series(Y_hat[time_col].unique())
    unknown_owners = set(ownership[bottom_col].unique()) - set(Y_hat[id_col].unique())
    if unknown_owners:
        raise TimeVaryingHierarchyError(
            f"Ownership rows reference bottom-level series absent from `Y_hat_df`: "
            f"{sorted(unknown_owners)}."
        )

    # Validate coverage (missing / duplicate ownership) on the forecast dates.
    _detect_ownership_problems(
        ownership=ownership,
        forecast_dates=forecast_dates,
        bottom_col=bottom_col,
        valid_from_col="__valid_from__",
        valid_to_col="__valid_to__",
    )

    # Every bottom series must have a forecast row on every forecast date.
    bottom_id_set = set(ownership[bottom_col].unique())
    pivot_counts = (
        Y_hat[Y_hat[id_col].isin(bottom_id_set)].groupby(id_col)[time_col].nunique()
    )
    n_dates = forecast_dates.nunique()
    incomplete = pivot_counts[pivot_counts != n_dates]
    if len(incomplete) > 0:
        raise TimeVaryingHierarchyError(
            "Every bottom-level series needs a forecast on every forecast date. "
            f"Series with missing dates: {incomplete.index.tolist()}."
        )

    # ----- split the horizon at ownership boundaries --------------------------
    # A boundary may fall between forecast dates (e.g. a switch on July 15th
    # with monthly forecasts). The new segment then starts at the first
    # forecast date on or after the boundary.
    all_dates = pd.DatetimeIndex(sorted(forecast_dates.unique()))
    break_dates = set()
    for boundary in ownership["__valid_from__"].unique():
        after = all_dates[all_dates >= boundary]
        if len(after) > 0:
            break_dates.add(after[0])
    break_dates = pd.DatetimeIndex(sorted(break_dates))

    segments: list[pd.DatetimeIndex] = []
    remaining = all_dates
    for break_date in break_dates:
        head = remaining[remaining < break_date]
        if len(head) > 0:
            segments.append(head)
        remaining = remaining[remaining >= break_date]
    if len(remaining) > 0:
        segments.append(remaining)
    if not segments:  # pragma: no cover - defensive; unreachable for non-empty input
        raise TimeVaryingHierarchyError("No forecast dates to reconcile.")

    # ----- reconcile segment by segment --------------------------------------
    hrec = HierarchicalReconciliation(reconcilers=[reconciler])
    reconciled_parts: list[pd.DataFrame] = []
    segment_info: list[dict[str, Any]] = []

    for segment_dates in segments:
        Y_seg, S_seg, tags_seg, bottom_fc = _build_segment_hierarchy(
            segment_dates=segment_dates,
            ownership=ownership,
            Y_hat=Y_hat,
            bottom_col=bottom_col,
            parent_cols=parent_cols,
            id_col=id_col,
            time_col=time_col,
            target_cols=target_cols,
        )
        rec_seg = hrec.reconcile(
            Y_hat_df=Y_seg,
            S_df=S_seg,
            tags=tags_seg,
            id_col=id_col,
            time_col=time_col,
        )
        rec_cols = [
            c
            for c in rec_seg.columns
            if c not in (id_col, time_col) and c not in target_cols
        ]
        reconciled_parts.append(rec_seg[[id_col, time_col, *rec_cols]])
        segment_info.append(
            {
                "start": segment_dates[0],
                "end": segment_dates[-1],
                "n_dates": len(segment_dates),
                "n_nodes": len(S_seg),
                "n_bottom": S_seg.shape[1] - 1,  # exclude the id label column
                "tags": {k: list(v) for k, v in tags_seg.items()},
            }
        )

    reconciled_df = pd.concat(reconciled_parts, ignore_index=True)
    reconciled_df = reconciled_df.sort_values([id_col, time_col]).reset_index(drop=True)

    # Preserve the input backend (pandas vs polars).
    if not isinstance(Y_hat_df, pd.DataFrame):
        native_backend = str(nw.from_native(Y_hat_df).implementation).lower()
        if "polars" in native_backend:
            import polars as pl

            reconciled_df = pl.from_pandas(reconciled_df)
        else:  # pragma: no cover - other backends default to pandas
            reconciled_df = nw.from_native(reconciled_df).to_pandas()

    # Expose segment bookkeeping for inspection / debugging.
    reconcile_time_varying.segments_ = segment_info
    return reconciled_df
