__all__ = ['aggregate', 'aggregate_temporal', 'aggregate_temporal_calendar', 'make_future_dataframe', 'get_cross_temporal_tags', 'HierarchicalPlot', 'set_num_threads', 'get_num_threads', 'SMatrix']


import itertools

# Default OpenMP to physical cores (logical / 2) to avoid oversubscription,
# matching PyTorch's convention. This matters when running reconciliation
# inside multiprocessing.Pool — without it each worker silently spawns
# all-CPU OpenMP threads.
import os as _os
import reprlib
import sys
import timeit
from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import narwhals.stable.v2 as nw
import numpy as np
import utilsforecast.processing as ufp
import utilsforecast.validation as ufv
from narwhals.typing import Frame, FrameT
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

from hierarchicalforecast._lib import reconciliation as _lib_recon

_default_threads = int(
    _os.environ.get(
        "OMP_NUM_THREADS", max(1, (_os.cpu_count() or 1) // 2)
    )
)
_lib_recon.set_num_threads(_default_threads)
del _os, _default_threads


def set_num_threads(n: int) -> None:
    """Set the number of OpenMP threads used by the C++ backend.

    Args:
        n (int): Number of threads. Must be >= 1.
    """
    _lib_recon.set_num_threads(n)


def get_num_threads() -> int:
    """Return the current maximum number of OpenMP threads.

    Returns:
        int: Number of threads.
    """
    return _lib_recon.get_num_threads()


class CodeTimer:
    def __init__(self, name=None, verbose=True):
        self.name = " '" + name + "'" if name else ""
        self.verbose = verbose

    def __enter__(self):
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = timeit.default_timer() - self.start
        if self.verbose:
            print(
                "Code block"
                + self.name
                + f" took:\t{self.took:.5f}"
                + " seconds"
            )


def _is_bottom_identity(B: np.ndarray, n: int) -> bool:
    """Check that the bottom block of a dense summing matrix is an identity matrix.

    Args:
        B (np.ndarray): Bottom block of a summing matrix S.
        n (int): Number of bottom-level series.

    Returns:
        (bool): True if `B` is an `n` x `n` identity matrix.
    """
    if B.shape != (n, n):
        return False
    # Fast path: `aggregate` emits an exact 0/1 matrix, so this settles the
    # common case without allocating the n x n temporaries `np.allclose` needs.
    if np.count_nonzero(B) == n and np.all(np.diagonal(B) == 1.0):
        return True
    # Fall back to a tolerant comparison, so summing matrices built by the user
    # that carry floating point noise stay acceptable.
    return bool(np.allclose(B, np.eye(n)))


def _construct_adjacency_matrix(
    S: sparse.csr_matrix, tags: dict[str, np.ndarray]
) -> sparse.csr_matrix:
    """Construct a disaggregation adjacency matrix.

    Args:
        S (sparse.csr_matrix): A summing matrix for hierarchical or grouped time series.
        tags (dict[str, np.ndarray]): A mapping of level name to node indices.

    Returns:
        sparse.csr_matrix: The disaggregation adjacency matrix for the structure.
    """
    # Get the nodes in each level.
    l = sorted(tags.values(), key=lambda x: len(x))
    # Get the number of aggregation nodes.
    n_a = S.shape[0] - S.shape[1]
    # Copy and cast the summing matrix to bool.
    S = S.astype(bool)
    # Precompute the transpose of the boolean summing matrix.
    S_T = S.T
    # Find the affinity, i.e., connectivity, between nodes in successive
    # levels, construct a sparse block diagonal matrix from these blocks, and
    # return the truncated disaggregation adjacency matrix.
    return sparse.hstack(
        (
            sparse.csr_matrix((n_a, len(l[0])), dtype=bool),
            sparse.block_diag(
                [S[l[i]] * S_T[:, l[i + 1]] for i in range(len(l) - 1)], "csr"
            ),
        )
    )


def is_strictly_hierarchical(S: np.ndarray, tags: dict[str, np.ndarray]) -> bool:
    # main idea:
    # if S represents a strictly hierarchical structure
    # the number of paths before the bottom level
    # should be equal to the number of nodes
    # of the previuos level
    levels_ = dict(sorted(tags.items(), key=lambda x: len(x[1])))
    # removing bottom level
    levels_.popitem()
    # making S categorical
    hiers = [np.argmax(S[idx], axis=0) + 1 for _, idx in levels_.items()]
    hiers = np.vstack(hiers)
    paths = np.unique(hiers, axis=1).shape[1]
    nodes = levels_.popitem()[1].size
    return paths == nodes


def _is_strictly_hierarchical(
    A: sparse.csr_matrix, tags: dict[str, np.ndarray]
) -> bool:
    """Check if a disaggregation structure is strictly hierarchical.

    The nodes in a strictly hierarchical disaggregation structure, except for
    the root node(s), should have exactly one incoming edge.

    Args:
        A (sparse.csr_matrix): A disaggregation adjacency matrix.
        tags (dict[str, np.ndarray]): A mapping of level name to node indices.

    Returns:
        bool: `True` if strictly hierarchical, otherwise `False`.
    """
    return np.all(A.sum(axis=0).A1[len(next(iter(tags.values()))) :] == 1)


def _to_upper_hierarchy(
    bottom_split: list[str], bottom_values: str, upper_key: str
) -> list[str]:
    upper_split = upper_key.split("/")
    upper_idxs = [bottom_split.index(i) for i in upper_split]

    def join_upper(bottom_value):
        bottom_parts = bottom_value.split("/")
        return "/".join(bottom_parts[i] for i in upper_idxs)

    return [join_upper(val) for val in bottom_values]


def aggregate(
    df: Frame,
    spec: list[list[str]],
    exog_vars: dict[str, str | list[str]] | None = None,
    sparse_s: bool = False,
    id_col: str = "unique_id",
    time_col: str = "ds",
    id_time_col: str | None = None,
    target_cols: Sequence[str] = ("y",),
) -> tuple[FrameT, "FrameT | SMatrix", dict]:
    """Utils Aggregation Function.

    Aggregates bottom level series contained in the DataFrame `df` according
    to levels defined in the `spec` list.

    Args:
        df (Frame): Dataframe with columns `[time_col, *target_cols]`, columns to aggregate and optionally exog_vars.
        spec (list[list[str]]): list of levels. Each element of the list should contain a list of columns of `df` to aggregate.
        exog_vars (Optional[dict[str, Union[str, list[str]]]], optional): dictionary of string keys & values that can either be a list of strings or a single string
            keys correspond to column names and the values represent the aggregation(s) that will be applied to each column. Accepted values are those from Pandas or Polars aggregation Functions, check the respective docs for guidance. Default is None.
        sparse_s (bool, optional): Return `S_df` as an `SMatrix` (sparse summing matrix wrapper) instead of a dense DataFrame. Works with both Pandas and Polars inputs. Default is False.
        id_col (str, optional): Column that will identify each serie after aggregation. Default is "unique_id".
        time_col (str, optional): Column that identifies each timestep, its values can be timestamps or integers. Default is "ds".
        id_time_col (Optional[str], optional): Column that will identify each timestep after temporal aggregation. If provided, aggregate will operate temporally. Default is None.
        target_cols (Sequence[str], optional): list of columns that contains the targets to aggregate. Default is ("y",).

    Returns:
        tuple[FrameT, FrameT | SMatrix, dict]: Y_df, S_df, tags
            Y_df: Hierarchically structured series.
            S_df: Summing dataframe. When ``sparse_s=True``, returns an
                :class:`SMatrix` instead of a DataFrame.
            tags: Aggregation indices.
    """
    # To Narwhals
    target_cols = list(target_cols)
    df_nw = nw.from_native(df)
    backend = df_nw.implementation

    # Check if we do temporal aggregation
    if id_time_col is not None:
        temporal_agg = True
        group_col = id_col
        _id_col = id_time_col
    else:
        temporal_agg = False
        group_col = time_col
        _id_col = id_col

    # Checks
    # Generate order-preserving list of unique cols based on spec
    spec_cols = list(dict.fromkeys([col for cols in spec for col in cols]))
    # Check if last level in spec contains all levels
    missing_cols_in_bottom_spec = set(spec_cols) - set(spec[-1])
    if missing_cols_in_bottom_spec and not temporal_agg:
        raise ValueError(
            f"Check the last (bottom) level of spec, it has missing columns: {reprlib.repr(missing_cols_in_bottom_spec)}"
        )

    # sparse_s is supported for both Pandas and Polars via SMatrix

    for col in df_nw.columns:
        if df_nw[col].is_null().any():
            raise ValueError(
                f"Column {col} contains null values. Make sure no column in the DataFrame contains null values."
            )

    # Check whether all columns in the spec are in the df
    aggregation_cols_in_spec = list(
        dict.fromkeys([col for cols in spec for col in cols])
    )
    for col in aggregation_cols_in_spec:
        if col not in df_nw.columns:
            raise ValueError(f"Column {col} in spec not present in df")

    # Prepare the aggregation dictionary
    agg_dict = dict(
        zip(target_cols, tuple(zip(target_cols, len(target_cols) * ["sum"], strict=False)), strict=False)
    )

    # Check if exog_vars are present in df & add to the aggregation dictionary if it is not None
    exog_var_names = []
    if exog_vars is not None:
        missing_vars = [var for var in exog_vars.keys() if var not in df.columns]
        if missing_vars:
            raise ValueError(
                f"The following exogenous variables are not present in the DataFrame: {', '.join(missing_vars)}"
            )
        else:
            # Update agg_dict to handle multiple aggregations for each exog_vars key
            for key, agg_func in exog_vars.items():
                # Ensure agg_func is a list
                if isinstance(
                    agg_func, str
                ):  # If it's a single string, convert to list
                    agg_func = [agg_func]
                elif not isinstance(agg_func, list):  # Raise an error if it's neither
                    raise ValueError(
                        f"Aggregation functions for '{key}' must be a string or a list of strings."
                    )

                for func in agg_func:
                    agg_dict[f"{key}_{func}"] = (
                        key,
                        func,
                    )  # Update the agg_dict with the new naming structure
                    exog_var_names.append(f"{key}_{func}")

    # compute aggregations and tags
    spec = sorted(spec, key=len)

    tags = {}
    Y_nws = []
    category_list = []
    level_sep = "/"
    # Perform the aggregation
    for level in spec:
        level_name = level_sep.join(level)

        # Create Y_df
        Y_level = df_nw.with_columns(
            *[
                getattr(nw.col(col), agg)().over(level + [group_col]).alias(col_name)
                for col_name, (col, agg) in agg_dict.items()
            ]
        )
        Y_level = Y_level.select(
            nw.concat_str([nw.col(col) for col in level], separator=level_sep).alias(
                _id_col
            ),
            nw.all(),
        )
        # For temporal aggregation, we need to keep the time column
        if temporal_agg:
            Y_level = Y_level.sort(by=[id_col, time_col])
            Y_level = Y_level.unique(
                subset=level + [group_col], maintain_order=temporal_agg, keep="last"
            ).select([_id_col, group_col, time_col, *target_cols] + exog_var_names)
            tags[level_name] = (
                Y_level[_id_col].unique(maintain_order=temporal_agg).to_numpy()
            )
        else:
            Y_level = Y_level.unique(
                subset=level + [group_col], maintain_order=temporal_agg, keep="last"
            ).select([_id_col, group_col, *target_cols] + exog_var_names)
            Y_level = Y_level.sort(by=[_id_col, group_col])
            tags[level_name] = (
                Y_level[_id_col].unique(maintain_order=temporal_agg).sort().to_numpy()
            )

        Y_nws.append(Y_level)
        category_list.extend(tags[level_name])

    Y_nw = nw.concat(Y_nws, how="vertical")
    Y_nw = nw.maybe_reset_index(Y_nw)
    Y_df = Y_nw.to_native()

    # construct S
    if temporal_agg:
        bottom = spec_cols
        bottom_levels = (
            df_nw.select(
                nw.concat_str(
                    [nw.col(col) for col in bottom], separator=level_sep
                ).alias(_id_col)
            )
            .unique(subset=[_id_col], maintain_order=temporal_agg, keep="last")[_id_col]
            .to_numpy()
        )
        categories = list([v for k, v in tags.items() if k != level_name])
        categories += [bottom_levels]
    else:
        bottom = spec[-1]
        bottom_levels = tags[level_name]
        categories = list(tags.values())

    S = np.empty((len(bottom_levels), len(spec)), dtype=object)

    for j, levels in enumerate(spec[:-1]):
        S[:, j] = _to_upper_hierarchy(bottom, bottom_levels, level_sep.join(levels))
    S[:, -1] = bottom_levels

    encoder = OneHotEncoder(
        categories=categories, sparse_output=sparse_s, dtype=np.float64
    )
    S_dum = encoder.fit_transform(S)

    if not sparse_s:
        S_nw = nw.from_dict(
            {
                **{_id_col: category_list},
                **dict(zip(tags[level_name], S_dum, strict=False)),
            },
            backend=backend,
        )
        S_nw = nw.maybe_reset_index(S_nw)
        S_df = S_nw.to_native()
    else:
        S_df = SMatrix(
            sparse_matrix=S_dum.T,
            row_labels=np.asarray(category_list),
            col_labels=np.asarray(list(bottom_levels)),
            id_col=_id_col,
            backend=backend,
        )
        # Built by one-hot encoding the bottom level, so the bottom block is an
        # identity matrix by construction and needs no further validation.
        S_df._bottom_identity_verified = True

    return Y_df, S_df, tags


def aggregate_temporal(
    df: Frame,
    spec: dict[str, int],
    exog_vars: dict[str, str | list[str]] | None = None,
    sparse_s: bool = False,
    id_col: str = "unique_id",
    time_col: str = "ds",
    id_time_col: str = "temporal_id",
    target_cols: Sequence[str] = ("y",),
    aggregation_type: str = "local",
) -> tuple[FrameT, FrameT, dict]:
    """Utils Aggregation Function for Temporal aggregations.

    Aggregates bottom level timesteps contained in the DataFrame `df` according
    to temporal levels defined in the `spec` list.

    Args:
        df (Frame): Dataframe with columns `[time_col, target_cols]` and columns to aggregate.
        spec (dict[str, int]): Dictionary of temporal levels. Each key should be a string with the value representing the number of bottom-level timesteps contained in the aggregation.
        exog_vars (Optional[dict[str, Union[str, list[str]]]], optional): dictionary of string keys & values that can either be a list of strings or a single string
            keys correspond to column names and the values represent the aggregation(s) that will be applied to each column. Accepted values are those from Pandas or Polars aggregation Functions, check the respective docs for guidance. Default is None.
        sparse_s (bool, optional): Return `S_df` as an `SMatrix` (sparse summing matrix wrapper) instead of a dense DataFrame. Works with both Pandas and Polars inputs. Default is False.
        id_col (str, optional): Column that will identify each serie after aggregation. Default is 'unique_id'.
        time_col (str, optional): Column that identifies each timestep, its values can be timestamps or integers. Default is 'ds'.
        id_time_col (str, optional): Column that will identify each timestep after aggregation. Default is 'temporal_id'.
        target_cols (Sequence[str], optional): List of columns that contain the targets to aggregate. Default is ('y',).
        aggregation_type (str, optional): If 'local' the aggregation will be performed on the timestamps of each timeseries independently. If 'global' the aggregation will be performed on the unique timestamps of all timeseries. Default is 'local'.

    Returns:
        tuple[FrameT, FrameT, dict]: Y_df, S_df, tags
            Y_df: Temporally hierarchically structured series.
            S_df: Temporal summing dataframe.
            tags: Temporal aggregation indices.
    """
    # Check if ds column is a timestamp or integer, if not raise an error
    target_cols = list(target_cols)
    df = ufv.ensure_time_dtype(df=df, time_col=time_col)
    df = ufp.ensure_sorted(df=df, id_col=id_col, time_col=time_col)
    df_nw = nw.from_native(df)

    # We add a cumulative count column to the dataframe to be able to compute the aggregations
    if aggregation_type == "local":
        df_nw = df_nw.with_columns(
            nw.col(time_col).cum_count().over([id_col]).alias(f"{time_col}_count")
        )
    elif aggregation_type == "global":
        unique_ts = df_nw.select(nw.col(time_col)).unique(maintain_order=True)
        unique_ts = unique_ts.with_columns(
            nw.col(time_col).cum_count().alias(f"{time_col}_count")
        )
        df_nw = df_nw.join(unique_ts, on=[time_col], how="left")
        df_nw = df_nw.sort(by=[id_col, time_col])
    else:
        raise ValueError("aggregation_type must be either 'local' or 'global'.")
    df_nw = nw.maybe_reset_index(df_nw)

    # Check spec that lowest level with seasonality of 1 has been defined
    if 1 not in spec.values():
        raise ValueError(
            "The spec must contain a level with a seasonality of 1. This represents the lowest level in the temporal aggregation."
        )

    # Check for duplicate values in spec
    if len(spec) != len(set(spec.values())):
        raise ValueError(
            "The spec must contain unique values for each level. Each value represents the seasonality of the aggregation."
        )

    # Check for duplicate keys in spec
    if len(spec) != len(set(spec.keys())):
        raise ValueError(
            "The spec must contain unique keys for each level. Each key represents the name of the aggregation."
        )

    # Loop over the spec and create the aggregation columns
    spec_agg: list = []
    for agg, seasonality in spec.items():
        df_nw = df_nw.with_columns(
            nw.concat_str(
                [
                    nw.lit(agg),
                    nw.lit("-"),
                    (((nw.col(f"{time_col}_count") - 1) // seasonality) + 1).cast(
                        nw.String
                    ),
                ]
            ).alias(agg)
        )
        spec_agg.append([agg])

    # If target_cols is not in df, we add a placeholder column so that we can compute the aggregations
    add_placeholder = False
    if set(df.columns) == set([time_col, id_col]):
        add_placeholder = True
        df_nw = df_nw.with_columns(nw.lit(0).alias("y"))

    # Create the aggregation
    df = df_nw.to_native()
    Y_df, S_df, tags = aggregate(
        df=df,
        spec=spec_agg,
        exog_vars=exog_vars,
        sparse_s=sparse_s,
        id_col=id_col,
        time_col=time_col,
        target_cols=target_cols,
        id_time_col=id_time_col,
    )
    Y_nw = nw.from_native(Y_df)

    # Drop the placeholder column if it was added
    if add_placeholder:
        Y_nw = Y_nw.drop("y")

    Y_df = Y_nw.to_native()

    return Y_df, S_df, tags


def _py_scalar(value):
    """Convert numpy / pandas scalars to plain Python objects.

    Used when rebuilding a frame in a different dataframe backend, where
    ``pandas.Timestamp`` objects would require pyarrow to cross into polars.
    """
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    if hasattr(value, "to_pylist"):
        return value.to_pylist()
    return value


def _calendar_to_backend(frame_nw, backend):
    """Return ``frame_nw`` using ``backend``, converting through Python values."""
    if frame_nw.implementation == backend:
        return frame_nw
    data = frame_nw.to_dict(as_series=False)
    data = {
        key: [_py_scalar(value) for value in values]
        for key, values in data.items()
    }
    return nw.from_dict(data, backend=backend)


def _calendar_label_expr(column: str, schema, time_fmt: str):
    """Build a string label expression for a calendar column.

    Temporal columns are formatted with ``time_fmt``; string and numeric
    columns are cast so the label round-trips the value the caller supplied.
    """
    expr = nw.col(column)
    dtype = schema[column]
    if dtype.is_temporal():
        return expr.dt.to_string(time_fmt)
    return expr.cast(nw.String)


def _regular_period_length(expected: list[int]) -> int:
    """Return the habitual number of business days for a level.

    The modal length is used (ties resolved to the larger value), so periods
    that are shorter than usual can be flagged as short without depending on
    the calendar's first/last period being complete.
    """
    counts: dict[int, int] = {}
    for value in expected:
        counts[int(value)] = counts.get(int(value), 0) + 1
    best = max(counts.values())
    return max(length for length, count in counts.items() if count == best)


def aggregate_temporal_calendar(
    df: Frame,
    calendar: Frame,
    spec: dict[str, str],
    id_col: str = "unique_id",
    time_col: str = "ds",
    id_time_col: str = "temporal_id",
    target_cols: Sequence[str] = ("y",),
    sparse_s: bool = False,
    bottom_level: str = "day",
    level_sep: str = "//",
    time_fmt: str = "%Y-%m-%d",
) -> tuple[FrameT, "FrameT | SMatrix", dict[str, np.ndarray], FrameT]:
    """Build a temporal hierarchy from an explicit fiscal/business calendar.

    Unlike :func:`aggregate_temporal`, which derives temporal levels from a
    fixed number of bottom-level timesteps per period, this function takes the
    period assignment of every business day from ``calendar``. That makes it
    possible to model real fiscal calendars where weeks and fiscal months have
    an irregular number of business days, where the first/last week is short,
    and where a fiscal week spans two natural months.

    Each series is aggregated to every level in ``spec`` and to the
    bottom-level business day. A group (a fiscal week, a fiscal month, or a
    business day) only participates in the hierarchy when **every** series
    covers **all** of the business days the calendar assigns to it. Groups
    with missing data are dropped, so the resulting summing matrix is coherent
    and can be used with ``HierarchicalReconciliation.reconcile(temporal=True)``.

    The fiscal period labels supplied by ``calendar`` are preserved verbatim in
    ``id_time_col`` as ``"<level><level_sep><label>"``; periods are never
    renumbered. A coverage report is returned alongside the hierarchy so
    downstream reports can tell a *short* period (fewer business days than
    usual, fully observed) from an *incomplete* period (some business day is
    missing for at least one series).

    Args:
        df (Frame): Daily data with columns ``[id_col, time_col, *target_cols]``.
            Rows sharing ``(id_col, time_col)`` are summed.
        calendar (Frame): Business-day calendar with one row per business day.
            It must contain ``time_col`` (unique) and one column per entry in
            ``spec`` giving the period that the day belongs to.
        spec (dict[str, str]): Mapping of level name to the ``calendar`` column
            that holds its period label, e.g.
            ``{"week": "fiscal_week", "month": "fiscal_month"}``. The level
            order in the hierarchy is derived from the observed period length
            (coarsest first); ties follow the order in ``spec``.
        id_col (str, optional): Column that identifies each series. Default is
            "unique_id".
        time_col (str, optional): Column that identifies each business day.
            Default is "ds".
        id_time_col (str, optional): Name of the output period identifier
            column. Default is "temporal_id".
        target_cols (Sequence[str], optional): Target columns to aggregate.
            Default is ``("y",)``.
        sparse_s (bool, optional): Return ``S_df`` as an :class:`SMatrix`
            instead of a dense DataFrame. Default is False.
        bottom_level (str, optional): Name given to the bottom (business day)
            level. Default is "day".
        level_sep (str, optional): Separator between the level name and the
            calendar label in ``id_time_col``. Default is "//".
        time_fmt (str, optional): ``strftime`` format used to label business
            days and any temporal calendar column. Default is "%Y-%m-%d".

    Returns:
        tuple[FrameT, FrameT | SMatrix, dict[str, np.ndarray], FrameT]:
            Y_df, S_df, tags, coverage
                Y_df: Temporally aggregated series for the participating
                    groups, with columns ``[id_time_col, id_col, time_col,
                    *target_cols]``. The ``time_col`` of an aggregated row is
                    the last business day of its period.
                S_df: Temporal summing matrix (dense DataFrame or
                    :class:`SMatrix`).
                tags: Mapping of level name to the participating
                    ``id_time_col`` identifiers, coarsest level first and
                    bottom level last.
                coverage: One row per calendar group (levels and business
                    days) describing how much of the period was observed.
    """
    target_cols = list(target_cols)
    if not spec:
        raise ValueError("`spec` must define at least one temporal level.")
    if len(set(spec)) != len(spec):
        raise ValueError("`spec` level names must be unique.")
    if len(set(spec.values())) != len(spec.values()):
        raise ValueError("`spec` must reference distinct calendar columns.")
    if bottom_level in spec or bottom_level in spec.values():
        raise ValueError(
            f"The bottom level name {bottom_level!r} clashes with a level in `spec`."
        )
    if time_col in spec.values():
        raise ValueError(
            f"The calendar column {time_col!r} cannot be used as a temporal level."
        )

    df = ufv.ensure_time_dtype(df=df, time_col=time_col)
    df = ufp.ensure_sorted(df=df, id_col=id_col, time_col=time_col)
    calendar = ufv.ensure_time_dtype(df=calendar, time_col=time_col)

    df_nw = nw.from_native(df)
    backend = df_nw.implementation
    cal_nw = _calendar_to_backend(nw.from_native(calendar), backend)

    required = [time_col, *spec.values()]
    missing = set(required) - set(cal_nw.columns)
    if missing:
        raise ValueError(
            f"`calendar` is missing required columns: {reprlib.repr(missing)}"
        )
    for column in required:
        if cal_nw[column].is_null().any():
            raise ValueError(
                f"Column {column!r} contains null values in `calendar`."
            )
    if cal_nw.select(time_col).unique().shape[0] != len(cal_nw):
        raise ValueError(
            f"Column {time_col!r} in `calendar` must contain one row per business day."
        )
    if len(cal_nw) == 0:
        raise ValueError("`calendar` contains no business days.")
    if id_time_col in df_nw.columns:
        raise ValueError(
            f"Column {id_time_col!r} already exists in `df`; choose another `id_time_col`."
        )

    # Align time dtype so the calendar join is exact, then order by time.
    cal_nw = cal_nw.with_columns(nw.col(time_col).cast(df_nw.schema[time_col]))
    cal_nw = cal_nw.sort(by=[time_col])

    # Per-day strings, shared by the aggregated and bottom-level identifiers.
    cal_nw = cal_nw.with_columns(
        _calendar_label_expr(time_col, cal_nw.schema, time_fmt).alias("__day_label")
    )
    day_label_by_name = {
        column: _calendar_label_expr(column, cal_nw.schema, time_fmt)
        for column in spec.values()
    }
    cal_nw = cal_nw.with_columns(
        *[
            day_label_by_name[column].alias(f"__label_{column}")
            for column in spec.values()
        ]
    )

    day_order = cal_nw[time_col].to_list()
    cal_records = cal_nw.select(
        [time_col, "__day_label", *[f"__label_{c}" for c in spec.values()]]
    ).to_dict(as_series=False)

    # Unique labels keep the bottom block an identity matrix; a coarse
    # `time_fmt` on a finer time column could otherwise collapse two days.
    day_labels = [str(label) for label in cal_records["__day_label"]]
    if len(set(day_labels)) != len(day_labels):
        raise ValueError(
            f"`time_fmt`={time_fmt!r} produces duplicate business-day labels; "
            "choose a format that distinguishes every day in `calendar`."
        )

    # bottom-level identifier for every calendar day
    bottom_by_day = {
        day: f"{bottom_level}{level_sep}{day_labels[i]}"
        for i, day in enumerate(day_order)
    }

    # ------------------------------------------------------- period metadata
    levels: dict[str, dict] = {}
    for name, column in spec.items():
        label_column = f"__label_{column}"
        per = cal_nw.group_by(column).agg(
            nw.col(time_col).min().alias("__first"),
            nw.col(time_col).max().alias("__last"),
            nw.len().alias("__expected"),
            nw.col(label_column).first().alias("__label"),
        )
        per = per.sort(by=["__first"])
        per = per.with_columns(
            nw.concat_str(
                [nw.lit(f"{name}{level_sep}"), nw.col("__label")]
            ).alias(id_time_col)
        )
        levels[name] = {
            "column": column,
            "frame": per,
            "expected": [int(v) for v in per["__expected"].to_list()],
            "temporal_ids": [str(v) for v in per[id_time_col].to_list()],
            "periods": per[column].to_list(),
            "regular": _regular_period_length(
                [int(v) for v in per["__expected"].to_list()]
            ),
        }

    # Coarsest (largest) periods first; ties keep the `spec` order.
    level_order = sorted(
        spec, key=lambda name: -float(np.mean(levels[name]["expected"]))
    )

    # --------------------------------------------------------- attach periods
    dfj = df_nw.join(
        cal_nw.select([time_col, *spec.values()]), on=time_col, how="left"
    )
    unmatched = dfj.filter(nw.col(spec[level_order[0]]).is_null())
    if len(unmatched) > 0:
        bad = unmatched.select(time_col).unique().head(5)[time_col].to_list()
        raise ValueError(
            f"`df` contains {time_col} values absent from `calendar`: "
            f"{reprlib.repr(bad)}"
        )

    n_series = df_nw.select(id_col).unique().shape[0]
    if n_series == 0:
        raise ValueError("`df` contains no series.")

    sums = [nw.col(target).sum().alias(target) for target in target_cols]

    # ------------------------------------------------------------- coverage
    # For each level, a period participates only when every series covers all
    # the business days the calendar assigns to it.
    participating: dict[str, list[str]] = {}
    coverage_rows: list[dict] = []
    coverage_columns: list[str] = []
    for name in level_order:
        info = levels[name]
        column = info["column"]

        cells = dfj.group_by([id_col, column]).agg(
            nw.col(time_col).n_unique().alias("__covered")
        )
        cells = cells.join(
            info["frame"].select(
                [column, id_time_col, "__expected", "__first", "__last"]
            ),
            on=column,
            how="left",
        )
        cells = cells.with_columns(
            (nw.col("__covered") == nw.col("__expected")).alias("__ok")
        )
        # Simple aggregations only (pandas cannot vectorise n_unique in a
        # group-by): one row per present (series, period), then count rows.
        present = cells.group_by(column).agg(
            nw.len().alias("__n_present"),
            nw.col("__covered").min().alias("__covered_min"),
            nw.col("__covered").max().alias("__covered_max"),
        )
        complete_counts = (
            cells.filter(nw.col("__ok"))
            .group_by(column)
            .agg(nw.len().alias("__n_complete"))
        )
        stats_frame = present.join(complete_counts, on=column, how="left")
        stats_frame = stats_frame.with_columns(
            nw.col("__n_complete").fill_null(0).cast(nw.Int64)
        )
        stats_frame = stats_frame.with_columns(
            *[nw.col(c).cast(nw.Int64) for c in
              ("__n_present", "__covered_min", "__covered_max")]
        )
        stats_records = stats_frame.to_dict(as_series=False)
        stats = {
            stats_records[column][i]: {
                "n_present": int(stats_records["__n_present"][i]),
                "n_complete": int(stats_records["__n_complete"][i]),
                "covered_min": int(stats_records["__covered_min"][i]),
                "covered_max": int(stats_records["__covered_max"][i]),
            }
            for i in range(len(stats_records[column]))
        }

        meta = info["frame"].select(
            [column, id_time_col, "__expected", "__first", "__last"]
        )
        meta_records = meta.to_dict(as_series=False)
        meta_by_period = {
            meta_records[column][i]: (
                meta_records[id_time_col][i],
                int(meta_records["__expected"][i]),
                meta_records["__first"][i],
                meta_records["__last"][i],
            )
            for i in range(len(meta_records[column]))
        }

        active: list[str] = []
        for period in info["periods"]:
            temporal_id, expected, first_ds, last_ds = meta_by_period[period]
            period_stats = stats.get(
                period,
                {"n_present": 0, "n_complete": 0, "covered_min": 0, "covered_max": 0},
            )
            n_present = period_stats["n_present"]
            n_complete = period_stats["n_complete"]
            complete = n_complete == n_series
            is_short = expected < info["regular"]
            if complete:
                active.append(str(temporal_id))
            row = {
                "level": name,
                "period": str(temporal_id)[len(name) + len(level_sep):],
                id_time_col: str(temporal_id),
                "expected_days": expected,
                "first_ds": first_ds,
                "last_ds": last_ds,
                "n_series": n_series,
                "n_series_covered": n_complete,
                "covered_days_min": period_stats["covered_min"],
                "covered_days_max": period_stats["covered_max"],
                "is_short": bool(is_short),
                "is_complete": bool(complete),
                "participates": bool(complete),
                "status": (
                    "incomplete"
                    if not complete
                    else ("short" if is_short else "complete")
                ),
            }
            if not coverage_columns:
                coverage_columns = list(row)
            coverage_rows.append(row)
        participating[name] = active

    # -------------------------------------------------------- bottom coverage
    day_totals = dfj.group_by([id_col, time_col]).agg(*sums)
    day_counts = day_totals.group_by(time_col).agg(
        nw.col(id_col).n_unique().alias("__n_present")
    )
    day_counts_records = day_counts.to_dict(as_series=False)
    day_present = {
        day: int(day_counts_records["__n_present"][i])
        for i, day in enumerate(day_counts_records[time_col])
    }
    bottom_days = {
        day for day, n_present in day_present.items() if n_present == n_series
    }
    bottom_ids = [
        bottom_by_day[day] for day in day_order if day in bottom_days
    ]
    participating[bottom_level] = bottom_ids

    for day in day_order:
        n_present = day_present.get(day, 0)
        complete = day in bottom_days
        coverage_rows.append(
            {
                "level": bottom_level,
                "period": bottom_by_day[day][len(bottom_level) + len(level_sep):],
                id_time_col: bottom_by_day[day],
                "expected_days": 1,
                "first_ds": day,
                "last_ds": day,
                "n_series": n_series,
                "n_series_covered": n_present,
                "covered_days_min": 1 if n_present > 0 else 0,
                "covered_days_max": 1 if n_present > 0 else 0,
                "is_short": False,
                "is_complete": bool(complete),
                "participates": bool(complete),
                "status": "complete" if complete else "incomplete",
            }
        )

    # ----------------------------------------------------------- build Y_df
    frames = []
    for name in level_order:
        info = levels[name]
        column = info["column"]
        active = participating[name]
        if not active:
            continue
        level_df = dfj.group_by([id_col, column]).agg(*sums)
        level_df = level_df.join(
            info["frame"].select([column, id_time_col, "__last"]),
            on=column,
            how="left",
        )
        level_df = level_df.filter(nw.col(id_time_col).is_in(active))
        level_df = level_df.rename({"__last": time_col})
        level_df = level_df.select([id_time_col, id_col, time_col, *target_cols])
        frames.append(level_df)

    if bottom_ids:
        bottom_df = day_totals.filter(nw.col(time_col).is_in(list(bottom_days)))
        bottom_df = bottom_df.with_columns(
            nw.concat_str(
                [
                    nw.lit(f"{bottom_level}{level_sep}"),
                    _calendar_label_expr(time_col, bottom_df.schema, time_fmt),
                ]
            ).alias(id_time_col)
        )
        bottom_df = bottom_df.select([id_time_col, id_col, time_col, *target_cols])
        frames.append(bottom_df)

    if frames:
        Y_nw = nw.concat(frames, how="vertical").sort(by=[id_col, time_col])
    else:
        Y_nw = nw.from_dict(
            {
                id_time_col: [],
                id_col: [],
                time_col: [],
                **{target: [] for target in target_cols},
            },
            backend=backend,
        )
    Y_nw = nw.maybe_reset_index(Y_nw)

    # ------------------------------------------------------------- build S
    # Membership of a period is read straight from the calendar: a business
    # day belongs to the period its calendar row assigns to it.
    membership: dict[str, set[str]] = {}
    for column in spec.values():
        for i, day in enumerate(day_order):
            period = cal_records[f"__label_{column}"][i]
            membership.setdefault(f"{column}::{period}", set()).add(
                bottom_by_day[day]
            )

    row_ids: list[str] = []
    row_members: list[set[str]] = []
    tags: dict[str, np.ndarray] = {}
    for name in level_order:
        info = levels[name]
        active_set = set(participating[name])
        level_ids: list[str] = []
        meta_records = info["frame"].select([info["column"], id_time_col]).to_dict(
            as_series=False
        )
        for i in range(len(meta_records[info["column"]])):
            temporal_id = str(meta_records[id_time_col][i])
            if temporal_id not in active_set:
                continue
            period = meta_records[info["column"]][i]
            key = f"{info['column']}::{period}"
            row_ids.append(temporal_id)
            row_members.append(membership.get(key, set()))
            level_ids.append(temporal_id)
        tags[name] = np.asarray(level_ids, dtype=object)

    row_ids.extend(bottom_ids)
    for bottom_id in bottom_ids:
        row_members.append({bottom_id})
    tags[bottom_level] = np.asarray(bottom_ids, dtype=object)

    n_rows = len(row_ids)
    n_bottom = len(bottom_ids)
    bottom_position = {bottom_id: j for j, bottom_id in enumerate(bottom_ids)}
    matrix = np.zeros((n_rows, n_bottom), dtype=np.float64)
    for i, members in enumerate(row_members):
        for member in members:
            position = bottom_position.get(member)
            if position is not None:
                matrix[i, position] = 1.0

    if sparse_s:
        S_df = SMatrix(
            sparse_matrix=sparse.csr_matrix(matrix),
            row_labels=np.asarray(row_ids, dtype=object),
            col_labels=np.asarray(bottom_ids, dtype=object),
            id_col=id_time_col,
            backend=backend,
        )
        S_df._bottom_identity_verified = True
    else:
        S_nw = nw.from_dict(
            {
                id_time_col: row_ids,
                **{
                    bottom_id: matrix[:, j]
                    for j, bottom_id in enumerate(bottom_ids)
                },
            },
            backend=backend,
        )
        S_df = nw.maybe_reset_index(S_nw).to_native()

    coverage = nw.maybe_reset_index(
        nw.from_dict(
            {
                column: [row[column] for row in coverage_rows]
                for column in coverage_columns
            },
            backend=backend,
        )
    ).to_native()

    return Y_nw.to_native(), S_df, tags, coverage


def make_future_dataframe(
    df: Frame,
    freq: str | int,
    h: int,
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> FrameT:
    """Create future dataframe for forecasting.

    Args:
        df (Frame): Dataframe with ids, times and values for the exogenous regressors.
        freq (Union[str, int]): Frequency of the data. Must be a valid pandas or polars offset alias, or an integer.
        h (int): Forecast horizon.
        id_col (str, optional): Column that identifies each serie. Default is 'unique_id'.
        time_col (str, optional): Column that identifies each timestep, its values can be timestamps or integers. Default is 'ds'.

    Returns:
        FrameT: DataFrame with future values.
    """
    times_by_id = ufp.group_by_agg(df, id_col, {time_col: "max"}, maintain_order=True)
    times_by_id = ufp.sort(times_by_id, id_col)
    future_df = ufp.make_future_dataframe(
        uids=times_by_id[id_col],
        last_times=times_by_id[time_col],
        freq=freq,
        h=h,
        id_col=id_col,
        time_col=time_col,
    )
    return future_df


def get_cross_temporal_tags(
    df: Frame,
    tags_cs: dict[str, np.ndarray],
    tags_te: dict[str, np.ndarray],
    sep: str = "//",
    id_col: str = "unique_id",
    id_time_col: str = "temporal_id",
    cross_temporal_id_col: str = "cross_temporal_id",
) -> tuple[FrameT, dict[str, np.ndarray]]:
    """Get cross-temporal tags.

    Args:
        df (Frame): DataFrame with temporal ids.
        tags_cs (dict[str, np.ndarray]): Tags for the cross-sectional hierarchies.
        tags_te (dict[str, np.ndarray]): Tags for the temporal hierarchies.
        sep (str, optional): Separator for the cross-temporal tags. Default is "//".
        id_col (str, optional): Column that identifies each series. Default is 'unique_id'.
        id_time_col (str, optional): Column that identifies each (aggregated) timestep. Default is 'temporal_id'.
        cross_temporal_id_col (str, optional): Column that will identify each cross-temporal aggregation. Default is 'cross_temporal_id'.

    Returns:
        tuple[FrameT, dict[str, np.ndarray]]: df, tags_ct
            df: DataFrame with cross-temporal ids.
            tags_ct: Tags for the cross-temporal hierarchies.
    """
    df_nw = nw.from_native(df)

    # Check if relevant columns are present
    if id_col not in df_nw.columns:
        raise ValueError(f"Column '{id_col}' not present in df")
    if id_time_col not in df_nw.columns:
        raise ValueError(f"Column '{id_time_col}' not present in df")

    # Create cross-temporal tags
    tags_ct = {}
    for key_cs, value_cs in tags_cs.items():
        for key_te, value_te in tags_te.items():
            key_ct = key_cs + sep + key_te
            value_ct = list(sep.join(s) for s in itertools.product(value_cs, value_te))
            tags_ct[key_ct] = value_ct

    df_nw = df_nw.with_columns(
        **{cross_temporal_id_col: df_nw[id_col] + sep + df_nw[id_time_col]}
    )
    df = df_nw.to_native()

    return df, tags_ct


class HierarchicalPlot:
    r"""Hierarchical Plot

    This class contains a collection of matplotlib visualization methods, suited for small
    to medium sized hierarchical series.

    Args:
        S (Frame): DataFrame with summing matrix of size `(base, bottom)`, see [aggregate function](./utils.html#aggregate).
        tags (dict[str, np.ndarray]): hierarchical aggregation indexes, where
            each key is a level and its value contains tags associated to that level.
        S_id_col (str, optional): column that identifies each aggregation. Default is 'unique_id'.
    """

    def __init__(
        self,
        S: Frame,
        tags: dict[str, np.ndarray],
        S_id_col: str = "unique_id",
    ):

        self.S = nw.from_native(S)
        S_cols_ex_id_col = self.S.columns
        S_cols_ex_id_col.remove(S_id_col)
        self.S_cols_ex_id_col = S_cols_ex_id_col
        self.tags = tags

    def plot_summing_matrix(self):
        r"""Summation Constraints plot

        This method simply plots the hierarchical aggregation
        constraints matrix $\mathbf{S}$.

        Returns:
            fig (matplotlib.figure.Figure): figure object containing the plot of the summing matrix.
        """
        fig = plt.figure(num=1, figsize=(4, 6), dpi=80, facecolor="w")
        plt.spy(self.S[self.S_cols_ex_id_col].to_numpy())
        plt.show()
        return fig

    def plot_series(
        self,
        series: str,
        Y_df: Frame,
        models: list[str] | None = None,
        level: list[int] | None = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ):
        r"""Single Series plot

        Args:
            series (str): string identifying the `'unique_id'` any-level series to plot.
            Y_df (Frame): hierarchically structured series ($\mathbf{y}_{[a,b]}$).
                    It contains columns `['unique_id', 'ds', 'y']`, it may have `'models'`.
            models (Optional[list[str]], optional): string identifying filtering model columns. Default is None.
            level (Optional[list[int]], optional): confidence levels for prediction intervals available in `Y_df`. Default is None.
            id_col (str, optional): column that identifies each series. Default is 'unique_id'.
            time_col (str, optional): column that identifies each timestep, its values can be timestamps or integers. Default is 'ds'.
            target_col (str, optional): column that contains the target. Default is 'y'.

        Returns:
            fig (matplotlib.figure.Figure): figure object containing the plot of the single series.
        """
        Y_nw = nw.from_native(Y_df)

        if series not in self.S[id_col]:
            raise Exception(f"time series {series} not found")
        fig, ax = plt.subplots(1, 1, figsize=(20, 7))
        df_plot = Y_nw.filter(nw.col(id_col) == series)
        cols = (
            models
            if models is not None
            else [col for col in df_plot.columns if col not in [id_col, time_col]]
        )
        cols_wo_levels = [
            col for col in cols if ("-lo-" not in col and "-hi-" not in col)
        ]
        try:
            cmap = plt.get_cmap("tab10", 10)
        except AttributeError:
            cmap = plt.cm.get_cmap("tab10", 10)
        cmap = [cmap(i) for i in range(10)][: len(cols_wo_levels)]
        cmap_dict = dict(zip(cols_wo_levels, cmap, strict=False))
        for col in cols_wo_levels:
            ax.plot(
                df_plot[time_col].to_numpy(),
                df_plot[col].to_numpy(),
                linewidth=2,
                label=col,
                color=cmap_dict[col],
            )
            if level is not None and col != target_col:
                for lv in level:
                    if f"{col}-lo-{lv}" not in df_plot.columns:
                        # if model
                        # doesnt have levels
                        continue
                    ax.fill_between(
                        df_plot.select(nw.col(time_col))[time_col].to_numpy(),
                        df_plot.select(nw.col(f"{col}-lo-{lv}"))[
                            f"{col}-lo-{lv}"
                        ].to_numpy(),
                        df_plot.select(nw.col(f"{col}-hi-{lv}"))[
                            f"{col}-hi-{lv}"
                        ].to_numpy(),
                        alpha=-lv / 100 + 1,
                        color=cmap_dict[col],
                        label=f"{col}_level_{lv}",
                    )
        ax.set_title(f"{series} Forecast", fontsize=22)
        ax.set_xlabel("Timestamp [t]", fontsize=20)
        ax.legend(prop={"size": 15})
        ax.grid()
        ax.xaxis.set_major_locator(plt.MaxNLocator(min(max(len(df_plot) // 10, 1), 10)))
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(20)

        plt.show()
        return fig

    def plot_hierarchically_linked_series(
        self,
        bottom_series: str,
        Y_df: Frame,
        models: list[str] | None = None,
        level: list[int] | None = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ):
        r"""Hierarchically Linked Series plot

        Args:
            bottom_series (str): string identifying the `'unique_id'` bottom-level series to plot.
            Y_df (Frame): hierarchically structured series ($\mathbf{y}_{[a,b]}$).
                    It contains columns ['unique_id', 'ds', 'y'] and models.
            models (Optional[list[str]], optional): string identifying filtering model columns. Default is None.
            level (Optional[list[int]], optional): confidence levels for prediction intervals available in `Y_df`. Default is None.
            id_col (str, optional): column that identifies each series. Default is 'unique_id'.
            time_col (str, optional): column that identifies each timestep, its values can be timestamps or integers. Default is 'ds'.
            target_col (str, optional): column that contains the target. Default is 'y'.

        Returns:
            fig (matplotlib.figure.Figure): figure object containing the plots of the hierarchically linked series.
        """
        Y_nw = nw.from_native(Y_df)

        if bottom_series not in self.S.columns:
            raise Exception(f"bottom time series {bottom_series} not found")

        linked_series = (
            self.S[[id_col, bottom_series]]
            .filter(nw.col(bottom_series) == 1)[id_col]
            .to_numpy()
        )
        fig, axs = plt.subplots(
            len(linked_series), 1, figsize=(20, 2 * len(linked_series))
        )
        cols = (
            models
            if models is not None
            else [col for col in Y_nw.columns if col not in [id_col, time_col]]
        )
        cols_wo_levels = [
            col for col in cols if ("-lo-" not in col and "-hi-" not in col)
        ]
        cmap = plt.cm.get_cmap("tab10", 10)
        cmap = [cmap(i) for i in range(10)][: len(cols_wo_levels)]
        cmap_dict = dict(zip(cols_wo_levels, cmap, strict=False))
        for idx, series in enumerate(linked_series):
            df_plot = Y_nw.filter(nw.col(id_col) == series)
            for col in cols_wo_levels:
                axs[idx].plot(
                    df_plot[time_col].to_numpy(),
                    df_plot[col].to_numpy(),
                    linewidth=2,
                    label=col,
                    color=cmap_dict[col],
                )
                if level is not None and col != target_col:
                    for lv in level:
                        if f"{col}-lo-{lv}" not in df_plot.columns:
                            # if model
                            # doesnt have levels
                            continue
                        axs[idx].fill_between(
                            df_plot.select(nw.col(time_col))[time_col].to_numpy(),
                            df_plot.select(nw.col(f"{col}-lo-{lv}"))[
                                f"{col}-lo-{lv}"
                            ].to_numpy(),
                            df_plot.select(nw.col(f"{col}-hi-{lv}"))[
                                f"{col}-hi-{lv}"
                            ].to_numpy(),
                            alpha=-lv / 100 + 1,
                            color=cmap_dict[col],
                            label=f"{col}_level_{lv}",
                        )
            axs[idx].set_title(f"{series}", fontsize=10)
            axs[idx].grid()
            axs[idx].get_xaxis().label.set_visible(False)
            axs[idx].legend().set_visible(False)
            axs[idx].xaxis.set_major_locator(
                plt.MaxNLocator(min(max(len(df_plot) // 10, 1), 10))
            )
            for label in axs[idx].get_xticklabels() + axs[idx].get_yticklabels():
                label.set_fontsize(10)
        plt.subplots_adjust(hspace=0.4)
        handles, labels = axs[0].get_legend_handles_labels()
        kwargs = dict(
            loc="lower center", prop={"size": 10}, bbox_to_anchor=(0, 0.05, 1, 1)
        )
        if sys.version_info.minor > 7:  # noqa: YTT204
            kwargs["ncols"] = np.max([2, np.ceil(len(labels) / 2)])
        fig.legend(handles, labels, **kwargs)
        return fig

    def plot_hierarchical_predictions_gap(
        self,
        Y_df: Frame,
        models: list[str] | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ):
        r"""Hierarchically Predictions Gap plot

        Args:
            Y_df (Frame): hierarchically structured series ($\mathbf{y}_{[a,b]}$).
                    It contains columns ['unique_id', 'ds', 'y'] and models.
            models (Optional[list[str]], optional): string identifying filtering model columns. Default is None.
            xlabel (Optional[str], optional): string for the plot's x axis label. Default is None.
            ylabel (Optional[str], optional): string for the plot's y axis label. Default is None.
            id_col (str, optional): column that identifies each series. Default is 'unique_id'.
            time_col (str, optional): column that identifies each timestep, its values can be timestamps or integers. Default is 'ds'.
            target_col (str, optional): column that contains the target. Default is 'y'.

        Returns:
            fig (matplotlib.figure.Figure): figure object containing the plot of the aggregated predictions at different levels of the hierarchical structure.
        """
        Y_nw = nw.from_native(Y_df)

        # Parse predictions dataframe
        horizon_dates = Y_nw["ds"].unique().to_numpy()
        cols = (
            models
            if models is not None
            else [col for col in Y_nw.columns if col not in [id_col, time_col]]
        )

        # Plot predictions across tag levels
        fig, ax = plt.subplots(figsize=(8, 5))

        if target_col in Y_nw.columns:
            idx_top = (
                self.S.with_columns(sum_cols=nw.sum_horizontal(cols))
                .sort(by="sum_cols", descending=True)[0][id_col]
                .to_numpy()
            )
            y_plot = Y_nw.filter(nw.col(id_col) == idx_top)[target_col].to_numpy()
            plt.plot(horizon_dates, y_plot, label="True")

        ys = []
        for tag in self.tags:
            y_plot = sum(
                [
                    Y_nw.filter(nw.col(id_col) == idx)[cols].to_numpy()
                    for idx in self.tags[tag]
                ]
            )
            plt.plot(horizon_dates, y_plot, label=f"Level: {tag}")

            ys.append(y_plot[:, None])

        plt.title("Predictions Accumulated Difference")
        if ylabel is not None:
            plt.ylabel(ylabel)
        if xlabel is not None:
            plt.xlabel(xlabel)

        plt.legend()
        plt.grid()
        plt.show()
        return fig


# convert levels to output quantile names
def level_to_outputs(level: list[int]) -> tuple[list[float], list[str]]:
    """Converts list of levels into output names matching StatsForecast and NeuralForecast methods.

    Args:
        level (list[int]): Probability levels for prediction intervals [0,100].

    Returns:
        tuple[list[float], list[str]]: quantiles and output_names
            quantiles: quantiles derived from levels.
            output_names: String list with output column names.
    """
    qs = sum([[50 - l / 2, 50 + l / 2] for l in level], [])
    output_names = sum([[f"-lo-{l}", f"-hi-{l}"] for l in level], [])

    sort_idx = np.argsort(qs)
    quantiles = np.array(qs)[sort_idx]

    # Add default median
    quantiles = np.concatenate([np.array([50]), quantiles]) / 100
    output_names = list(np.array(output_names)[sort_idx])
    output_names.insert(0, "-median")

    return quantiles, output_names


# convert quantiles to output quantile names
def quantiles_to_outputs(quantiles: list[float]) -> tuple[list[float], list[str]]:
    """Converts list of quantiles into output names matching StatsForecast and NeuralForecast methods.

    Args:
        quantiles (list[float]): Alternative to level, quantiles to estimate from y distribution [0., 1.].

    Returns:
        tuple[list[float], list[str]]: quantiles and output_names
            quantiles: quantiles to estimate from y distribution.
            output_names: String list with output column names.
    """
    output_names = []
    for q in quantiles:
        if q < 0.50:
            output_names.append(f"-lo-{np.round(100-200*q,2)}")
        elif q > 0.50:
            output_names.append(f"-hi-{np.round(100-200*(1-q),2)}")
        else:
            output_names.append("-median")
    return quantiles, output_names


# given input array of sample forecasts and inptut quantiles/levels,
# output a Pandas Dataframe with columns of quantile predictions
def samples_to_quantiles_df(
    samples: np.ndarray,
    unique_ids: Sequence[str],
    dates: list[str],
    quantiles: list[float] | None = None,
    level: list[int] | None = None,
    model_name: str = "model",
    id_col: str = "unique_id",
    time_col: str = "ds",
    backend: str = "pandas",
) -> tuple[list[float], FrameT]:
    """Transform Random Samples into HierarchicalForecast input.

    Auxiliary function to create compatible HierarchicalForecast input `Y_hat_df` dataframe.

    Args:
        samples (np.ndarray): Samples from forecast distribution of shape [n_series, n_samples, horizon].
        unique_ids (Sequence[str]): Unique identifiers for each time series.
        dates (list[str]): list of forecast dates.
        quantiles (Optional[list[float]], optional): Alternative to level, quantiles to estimate from y distribution [0., 1.]. Default is None.
        level (Optional[list[int]], optional): Probability levels for prediction intervals [0,100]. Default is None.
        model_name (str, optional): Name of forecasting model. Default is "model".
        id_col (str, optional): column that identifies each serie. Default is 'unique_id'.
        time_col (str, optional): column that identifies each timestep, its values can be timestamps or integers. Default is 'ds'.
        backend (str, optional): backend to use for the output dataframe, either 'pandas' or 'polars'. Default is 'pandas'.

    Returns:
        tuple[list[float], FrameT]: quantiles and Y_hat_df
            quantiles: quantiles to estimate from y distribution [0., 1.].
            Y_hat_df: DataFrame with base quantile forecasts with columns ds and models to reconcile indexed by unique_id.
    """

    # Get the shape of the array
    n_series, n_samples, horizon = samples.shape

    if n_series != len(unique_ids):
        raise ValueError(
            f"Number of unique_ids ({len(unique_ids)}) must match the number of series ({n_series})."
        )
    if horizon != len(dates):
        raise ValueError(
            f"Number of dates ({len(dates)}) must match third dimension of samples array ({horizon})."
        )
    if not ((quantiles is None) ^ (level is None)):
        raise ValueError("Either quantiles or level must be provided, but not both.")

    namespace = sys.modules.get(backend, None)
    if namespace is None:
        raise ValueError(f"DataFrame backend {backend} not installed.")

    # create initial dictionary
    forecasts_mean = np.mean(samples, axis=1).flatten()
    unique_ids = np.repeat(unique_ids, horizon)
    ds = np.tile(dates, n_series)

    # create quantiles and quantile names
    if level is not None:
        _quantiles, quantile_names = level_to_outputs(level)
    elif quantiles is not None:
        _quantiles, quantile_names = quantiles_to_outputs(quantiles)

    percentiles = [quantile * 100 for quantile in _quantiles]
    col_names = np.array(
        [model_name + quantile_name for quantile_name in quantile_names]
    )

    # add quantiles to dataframe
    forecasts_quantiles = np.percentile(samples, percentiles, axis=1)

    forecasts_quantiles = np.transpose(
        forecasts_quantiles, (1, 2, 0)
    )  # [Q,H,N] -> [N,H,Q]
    forecasts_quantiles = forecasts_quantiles.reshape(-1, len(_quantiles))

    df_nw = nw.from_dict(
        {
            **{id_col: unique_ids, time_col: ds, model_name: forecasts_mean},
            **dict(zip(col_names, forecasts_quantiles.T, strict=False)),
        },
        backend=backend,
    )

    return _quantiles, df_nw.to_native()


# Masked empirical covariance matrix
def _ma_cov(residuals: np.ndarray, not_nan_mask: np.ndarray):
    """Masked empirical covariance matrix.

    :meta private:
    """
    return _lib_recon._ma_cov(residuals, not_nan_mask)


# Shrunk covariance matrix using the Schafer-Strimmer method
def _shrunk_covariance_schaferstrimmer_no_nans(
    residuals: np.ndarray, mint_shr_ridge: float
):
    """Shrink empirical covariance (Schäfer-Strimmer, no NaN handling).

    :meta private:
    """
    return _lib_recon._shrunk_covariance_schaferstrimmer_no_nans(
        residuals, mint_shr_ridge
    )


def _shrunk_covariance_schaferstrimmer_with_nans(
    residuals: np.ndarray, not_nan_mask: np.ndarray, mint_shr_ridge: float
):
    """Shrink empirical covariance (Schäfer-Strimmer, with NaN masking).

    :meta private:
    """
    return _lib_recon._shrunk_covariance_schaferstrimmer_with_nans(
        residuals, not_nan_mask, mint_shr_ridge
    )


# Lasso cyclic coordinate descent
def _lasso(
    X: np.ndarray,
    y: np.ndarray,
    lambda_reg: float,
    max_iters: int = 1_000,
    tol: float = 1e-4,
):
    """Lasso cyclic coordinate descent.

    :meta private:
    """
    return _lib_recon._lasso(np.asfortranarray(X), y, lambda_reg, max_iters, tol)


class SMatrix:
    """Lightweight wrapper around a scipy.sparse summing matrix with labels.

    Stores the hierarchical summing matrix S natively as a
    ``scipy.sparse.csc_matrix``.  Dense NumPy arrays and
    Pandas/Polars DataFrames are materialised only on demand via
    :meth:`to_dense` and :meth:`to_frame`.

    Parameters
    ----------
    sparse_matrix : scipy.sparse.spmatrix
        Summing matrix of shape ``(n_series, n_bottom)``.
    row_labels : np.ndarray
        Unique-id labels for every row (all hierarchical series).
    col_labels : np.ndarray
        Unique-id labels for every column (bottom-level series).
    id_col : str
        Name of the identifier column (default ``"unique_id"``).
    backend : str or nw.Implementation
        Narwhals backend hint (``"pandas"`` or ``"polars"``).
    """

    def __init__(
        self,
        sparse_matrix: sparse.spmatrix,
        row_labels: np.ndarray,
        col_labels: np.ndarray,
        id_col: str = "unique_id",
        backend: Any = "pandas",
    ):
        self._sparse = sparse.csc_matrix(sparse_matrix)
        self.row_labels = np.asarray(row_labels)
        self.col_labels = np.asarray(col_labels)
        self.id_col = id_col
        self.backend = backend
        self._col_index = {name: i for i, name in enumerate(col_labels)}
        self._bottom_identity_verified = False
        self._bottom_identity_cacheable = True
        # Lazily cached representations
        self._dense: np.ndarray | None = None
        self._csr: sparse.csr_matrix | None = None
        self._frames: dict = {}

    # ------------------------------------------------------------------
    # Core accessors
    # ------------------------------------------------------------------

    def check_bottom_identity(self) -> bool:
        """Check that the bottom n_bottom x n_bottom block of S is an identity matrix.

        Uses sparse operations to avoid dense allocations. Successful checks
        are cached, and matrices built by `aggregate` are trusted by
        construction, because an ``SMatrix`` is commonly reused across calls
        to `HierarchicalReconciliation.reconcile`. Once :meth:`to_sparse`
        exposes the mutable backing matrix, caching is disabled and every call
        revalidates the bottom block.
        """
        cacheable = getattr(self, "_bottom_identity_cacheable", True)
        if cacheable and getattr(self, "_bottom_identity_verified", False):
            return True

        n_bottom = self._sparse.shape[1]
        bottom_block = self._sparse[-n_bottom:, :]
        bottom_coo = bottom_block.tocoo()
        is_identity = (
            bottom_coo.shape[0] == bottom_coo.shape[1]
            and bottom_coo.nnz == n_bottom
            and np.allclose(bottom_coo.data, 1.0)
            and np.array_equal(bottom_coo.row, bottom_coo.col)
        )
        self._bottom_identity_verified = is_identity if cacheable else False
        return is_identity

    def clear_cache(self) -> None:
        """Release cached representations and identity verification.

        Call this after reconciliation to free memory when the
        ``SMatrix`` is long-lived but cached representations are no
        longer needed. This does not re-enable identity caching after the
        mutable matrix returned by :meth:`to_sparse` has been exposed.
        """
        self._dense = None
        self._csr = None
        self._frames = {}
        self._bottom_identity_verified = False

    def to_sparse(self) -> sparse.csc_matrix:
        """Return the underlying sparse matrix (zero-copy).

        Because the returned matrix is mutable, calling this method permanently
        disables bottom-identity caching for this ``SMatrix``. Subsequent
        reconciliations revalidate the bottom block, including mutations made
        through references retained by the caller.
        """
        self._bottom_identity_verified = False
        self._bottom_identity_cacheable = False
        return self._sparse

    def to_csr(self) -> sparse.csr_matrix:
        """Return the matrix as CSR (efficient for row slicing), cached."""
        if self._csr is None:
            self._csr = sparse.csr_matrix(self._sparse)
        return self._csr

    def to_dense(self) -> np.ndarray:
        """Return a dense ``float64`` NumPy array, cached after first call."""
        if self._dense is None:
            self._dense = self._sparse.toarray().astype(np.float64, copy=False)
        return self._dense

    def to_frame(self, backend: str | None = None):
        """Materialise S as a Pandas or Polars DataFrame.

        The result is cached on the first call (per backend).

        Parameters
        ----------
        backend : str, optional
            ``"pandas"`` or ``"polars"``.  Defaults to the backend that
            was used to create the original input data.

        Returns
        -------
        DataFrame
            A native Pandas or Polars DataFrame with the id column
            followed by one column per bottom-level series.
        """
        backend = backend or self.backend
        if backend in self._frames:
            return self._frames[backend]

        dense = self.to_dense()
        data = {self.id_col: self.row_labels}
        data.update(zip(self.col_labels, dense.T, strict=False))

        frame_nw = nw.from_dict(data, backend=backend)
        frame_nw = nw.maybe_reset_index(frame_nw)
        self._frames[backend] = frame_nw.to_native()
        return self._frames[backend]

    # ------------------------------------------------------------------
    # Backward-compatible properties
    # ------------------------------------------------------------------

    @property
    def sparse_shape(self) -> tuple[int, int]:
        """``(n_series, n_bottom)`` — shape of the underlying sparse matrix.

        Alias for :attr:`shape`.
        """
        return self._sparse.shape

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_series, n_bottom)`` — shape of the underlying sparse matrix."""
        return self._sparse.shape

    @property
    def frame_shape(self) -> tuple[int, int]:
        """``(n_series, n_bottom + 1)`` — shape matching the DataFrame representation (includes id column)."""
        rows, cols = self._sparse.shape
        return (rows, cols + 1)

    @property
    def columns(self):
        """Column names matching the DataFrame representation.

        Returns a pandas Index for compatibility with DataFrame.columns API.
        """
        import pandas as pd

        return pd.Index([self.id_col] + list(self.col_labels))

    def __len__(self) -> int:
        return self._sparse.shape[0]

    def __getitem__(self, key):
        """Support ``S_matrix['unique_id']`` and ``S_matrix[col_name]``."""
        if isinstance(key, str):
            if key == self.id_col:
                return self.row_labels
            idx = self._col_index.get(key)
            if idx is not None:
                return self._sparse[:, idx].toarray().ravel()
            raise KeyError(key)
        # Fall back to frame for complex indexing
        return self.to_frame()[key]

    def __repr__(self) -> str:
        nnz = self._sparse.nnz
        density = nnz / max(1, self._sparse.shape[0] * self._sparse.shape[1])
        return (
            f"SMatrix(shape={self.sparse_shape}, nnz={nnz}, "
            f"density={density:.4f}, id_col='{self.id_col}')"
        )

    def _repr_html_(self) -> str:
        """Jupyter notebook display — show truncated DataFrame."""
        return self.to_frame()._repr_html_()

    def __array__(self, dtype=None):
        """Support ``np.asarray(S_matrix)`` — returns dense values without id col."""
        arr = self.to_dense()
        if dtype is not None:
            return arr.astype(dtype, copy=False)
        return arr
