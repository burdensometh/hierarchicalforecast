__all__ = ['aggregate', 'aggregate_temporal', 'share_summing_matrix', 'make_future_dataframe', 'get_cross_temporal_tags', 'HierarchicalPlot', 'set_num_threads', 'get_num_threads', 'SMatrix']


import itertools
import math
import numbers

# Default OpenMP to physical cores (logical / 2) to avoid oversubscription,
# matching PyTorch's convention. This matters when running reconciliation
# inside multiprocessing.Pool — without it each worker silently spawns
# all-CPU OpenMP threads.
import os as _os
import reprlib
import sys
import timeit
from collections.abc import Mapping, Sequence
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


def _parse_share_mapping(
    shares: Any,
    leaf_col: str,
    region_col: str,
    share_col: str,
) -> list[tuple[Any, Any, Any]]:
    """Normalise a share mapping into an ordered list of ``(leaf, region, weight)``.

    Accepted inputs are a nested mapping ``{leaf: {region: weight}}``, a
    mapping keyed by ``(leaf, region)`` pairs, a sequence of
    ``(leaf, region, weight)`` triples, or a dataframe with one row per
    assignment.  The returned entries preserve input order.
    """
    entries: list[tuple[Any, Any, Any]] = []

    if isinstance(shares, Mapping):
        values = list(shares.values())
        if values and all(isinstance(value, Mapping) for value in values):
            for leaf, region_weights in shares.items():
                for region, weight in region_weights.items():
                    entries.append((leaf, region, weight))
        elif values and not any(isinstance(value, Mapping) for value in values):
            for key, weight in shares.items():
                if not (isinstance(key, tuple) and len(key) == 2):
                    raise ValueError(
                        "Share mappings with scalar values must be keyed by "
                        "(leaf, region) pairs or nested as {leaf: {region: weight}}. "
                        f"Got key {key!r}."
                    )
                leaf, region = key
                entries.append((leaf, region, weight))
        elif values:
            raise ValueError(
                "Share mapping mixes nested and scalar values. Use either "
                "{leaf: {region: weight}} or {(leaf, region): weight}."
            )
        return entries

    def _as_triple(item: Any) -> tuple[Any, Any, Any]:
        if isinstance(item, (str, bytes)) or isinstance(item, Mapping):
            raise ValueError(
                "Share assignments must be (leaf, region, weight) triples. "
                f"Got {item!r}."
            )
        try:
            values = list(item)
        except TypeError as err:
            raise ValueError(
                "Share assignments must be (leaf, region, weight) triples. "
                f"Got {item!r}."
            ) from err
        if len(values) != 3:
            raise ValueError(
                "Share assignments must be (leaf, region, weight) triples. "
                f"Got {item!r}."
            )
        return values[0], values[1], values[2]

    if isinstance(shares, np.ndarray):
        if shares.ndim == 2 and shares.shape[1] == 3:
            entries.extend(_as_triple(row) for row in shares)
            return entries
        if shares.ndim == 1 and shares.shape[0] == 3:
            return [_as_triple(shares)]
        raise ValueError(
            "Array share mappings must have shape (n, 3) of "
            f"(leaf, region, weight) rows. Got shape {shares.shape}."
        )

    if isinstance(shares, Sequence) and not isinstance(shares, (str, bytes)):
        # A lone triple is accepted as well as a sequence of triples.
        if len(shares) == 3 and not any(
            isinstance(item, (tuple, list, np.ndarray)) for item in shares
        ):
            return [_as_triple(shares)]
        for item in shares:
            entries.append(_as_triple(item))
        return entries

    try:
        frame = nw.from_native(shares, eager_only=True)
    except TypeError as err:
        raise ValueError(
            "`shares` must be a mapping, a sequence of (leaf, region, weight) "
            "triples, or a dataframe."
        ) from err

    missing = {leaf_col, region_col, share_col} - set(frame.columns)
    if missing:
        raise ValueError(
            f"`shares` is missing required columns: {reprlib.repr(missing)}."
        )
    leaves = frame[leaf_col].to_list()
    regions = frame[region_col].to_list()
    weights = frame[share_col].to_list()
    entries.extend(zip(leaves, regions, weights, strict=True))
    return entries


def share_summing_matrix(
    leaves: Sequence[Any],
    regions: Sequence[Any],
    shares: Any,
    national_label: Any = "National",
    leaf_col: str = "leaf",
    region_col: str = "region",
    share_col: str = "share",
    share_tolerance: float = 1e-8,
    id_col: str = "unique_id",
    backend: Any = "pandas",
) -> "tuple[SMatrix, dict[str, np.ndarray]]":
    r"""Build a sparse summing matrix for shared (fractional) region membership.

    A single leaf (for example a store) can contribute to several aggregates
    (for example business regions) at once, with a share that splits its sales
    across those regions.  The resulting summing matrix ``S`` has one column per
    leaf and three blocks of rows:

    - one **national** row summing every leaf once (weight ``1``), so the
      national total never double counts a store that belongs to several
      regions;
    - one **region** row per region, holding each leaf's share for that region;
    - one **leaf** row per store, holding the identity block.

    Leaves occupy the last rows, so the bottom ``n_leaves x n_leaves`` block is
    the identity matrix and the result can be passed directly to
    reconciliation methods (``BottomUp``, ``MinTrace``, ...).

    All validation runs before the matrix is built, so malformed input raises
    instead of producing a matrix: illegal weights (non-numeric, non-finite,
    ``<= 0`` or ``> 1``), leaves without any share, references to unknown
    leaves/regions, duplicated ``(leaf, region)`` assignments, and leaves whose
    shares do not conserve to ``1``.

    Args:
        leaves (Sequence): Ordered, unique leaf identifiers (e.g. stores).
        regions (Sequence): Ordered, unique region identifiers.
        shares (Any): Share mapping. One of a nested mapping
            ``{leaf: {region: weight}}``, a mapping keyed by
            ``(leaf, region)`` pairs, a sequence of
            ``(leaf, region, weight)`` triples, or a dataframe with columns
            ``[leaf_col, region_col, share_col]``.
        national_label (Any, optional): Identifier for the independent national
            row. Must not collide with a leaf or region identifier.
            Default is ``"National"``.
        leaf_col (str, optional): Leaf column name for dataframe input.
            Default is ``"leaf"``.
        region_col (str, optional): Region column name for dataframe input.
            Default is ``"region"``.
        share_col (str, optional): Weight column name for dataframe input.
            Default is ``"share"``.
        share_tolerance (float, optional): Absolute tolerance used to decide
            whether a leaf's shares conserve to ``1``. Default is ``1e-8``.
        id_col (str, optional): Identifier column name stored on the returned
            :class:`SMatrix`. Default is ``"unique_id"``.
        backend (Any, optional): Narwhals backend used when the ``SMatrix`` is
            materialised as a frame. Default is ``"pandas"``.

    Returns:
        tuple[SMatrix, dict[str, np.ndarray]]: The sparse summing matrix and the
            ``tags`` mapping (``"national"``, ``"regions"``, ``"leaves"``) ready
            for `HierarchicalReconciliation.reconcile`.

    Examples:
        >>> S_df, tags = share_summing_matrix(
        ...     leaves=["s1", "s2"],
        ...     regions=["north", "south"],
        ...     shares={"s1": {"north": 0.6, "south": 0.4}, "s2": {"south": 1.0}},
        ... )
        >>> S_df.shape
        (5, 2)
    """
    leaf_ids = list(leaves)
    region_ids = list(regions)

    if not leaf_ids:
        raise ValueError("`leaves` must contain at least one leaf identifier.")
    if share_tolerance < 0:
        raise ValueError("`share_tolerance` must be non-negative.")

    def _check_unique(ids: list[Any], name: str) -> None:
        seen = set()
        duplicates = []
        for item in ids:
            try:
                already_seen = item in seen
            except TypeError as err:
                raise ValueError(
                    f"`{name}` identifiers must be hashable, got {item!r}."
                ) from err
            if already_seen:
                duplicates.append(item)
            seen.add(item)
        if duplicates:
            raise ValueError(
                f"`{name}` contains duplicate identifiers: {reprlib.repr(duplicates)}."
            )

    _check_unique(leaf_ids, "leaves")
    _check_unique(region_ids, "regions")

    leaf_set = set(leaf_ids)
    region_set = set(region_ids)
    if leaf_set & region_set:
        raise ValueError(
            "Leaf and region identifiers must be disjoint: "
            f"{reprlib.repr(leaf_set & region_set)}."
        )
    if national_label in leaf_set or national_label in region_set:
        raise ValueError(
            f"`national_label` {national_label!r} collides with a leaf or region identifier."
        )

    entries = _parse_share_mapping(shares, leaf_col, region_col, share_col)

    # Duplicate (leaf, region) assignments are ambiguous: reject them outright.
    seen_pairs: set[tuple[Any, Any]] = set()
    duplicates = []
    for leaf, region, _ in entries:
        pair = (leaf, region)
        if pair in seen_pairs:
            duplicates.append(pair)
        seen_pairs.add(pair)
    if duplicates:
        raise ValueError(
            "Duplicate (leaf, region) share assignments: "
            f"{reprlib.repr(duplicates)}. Each pair may appear only once."
        )

    # Illegal weights: anything that cannot be a probability-like share.
    illegal: list[tuple[Any, Any, Any, str]] = []
    for leaf, region, weight in entries:
        if isinstance(weight, (bool, np.bool_)) or not isinstance(weight, numbers.Real):
            illegal.append((leaf, region, weight, "must be a real number"))
        elif not np.isfinite(weight):
            illegal.append((leaf, region, weight, "must be finite"))
        elif weight <= 0 or weight > 1:
            illegal.append((leaf, region, weight, "must be in the interval (0, 1]"))
    if illegal:
        details = ", ".join(
            f"({leaf!r}, {region!r})={weight!r} {reason}"
            for leaf, region, weight, reason in illegal[:10]
        )
        raise ValueError(
            f"Invalid share weights: {details}"
            + (f" ... ({len(illegal)} total)" if len(illegal) > 10 else "")
        )

    # References to identifiers that were not declared.
    unknown_leaves = sorted(
        {str(leaf) for leaf, _, _ in entries if leaf not in leaf_set}
    )
    unknown_regions = sorted(
        {str(region) for _, region, _ in entries if region not in region_set}
    )
    if unknown_leaves:
        raise ValueError(
            "`shares` references leaves not present in `leaves`: "
            f"{reprlib.repr(unknown_leaves)}."
        )
    if unknown_regions:
        raise ValueError(
            "`shares` references regions not present in `regions`: "
            f"{reprlib.repr(unknown_regions)}."
        )

    # Coverage and conservation, evaluated per leaf.
    weights_by_leaf: dict[Any, list[tuple[Any, float]]] = {leaf: [] for leaf in leaf_ids}
    for leaf, region, weight in entries:
        weights_by_leaf[leaf].append((region, float(weight)))

    uncovered = [leaf for leaf in leaf_ids if not weights_by_leaf[leaf]]
    if uncovered:
        raise ValueError(
            "Leaves without any share assignment: "
            f"{reprlib.repr(uncovered)}. Every leaf must be covered."
        )

    non_conserving = []
    for leaf in leaf_ids:
        total = math.fsum(weight for _, weight in weights_by_leaf[leaf])
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=share_tolerance):
            non_conserving.append((leaf, total))
    if non_conserving:
        details = ", ".join(f"{leaf!r} sums to {total:.12g}" for leaf, total in non_conserving[:10])
        raise ValueError(
            "Shares for each leaf must conserve to 1: "
            + details
            + (f" ... ({len(non_conserving)} total)" if len(non_conserving) > 10 else "")
        )

    # Build the sparse matrix: national row, region rows, then leaf identity.
    n_leaves = len(leaf_ids)
    n_regions = len(region_ids)
    region_pos = {region: i for i, region in enumerate(region_ids)}
    leaf_pos = {leaf: j for j, leaf in enumerate(leaf_ids)}

    row_idx: list[int] = []
    col_idx: list[int] = []
    data: list[float] = []

    # National row: every leaf counted exactly once.
    row_idx.extend([0] * n_leaves)
    col_idx.extend(range(n_leaves))
    data.extend([1.0] * n_leaves)

    # Region rows: each leaf's fractional contribution.
    for leaf, region, weight in entries:
        row_idx.append(1 + region_pos[region])
        col_idx.append(leaf_pos[leaf])
        data.append(float(weight))

    # Leaf rows: identity block (bottom of the matrix).
    for j in range(n_leaves):
        row_idx.append(1 + n_regions + j)
        col_idx.append(j)
        data.append(1.0)

    n_rows = 1 + n_regions + n_leaves
    sparse_matrix = sparse.coo_matrix(
        (np.asarray(data, dtype=np.float64), (row_idx, col_idx)),
        shape=(n_rows, n_leaves),
    ).tocsc()

    def _label_array(labels: list[Any]) -> np.ndarray:
        # ``np.array([...], dtype=object)`` collapses equal-length tuple
        # identifiers into a 2-D array; build a flat 1-D object array instead.
        arr = np.empty(len(labels), dtype=object)
        arr[:] = labels
        return arr

    row_labels = _label_array([national_label, *region_ids, *leaf_ids])
    col_labels = _label_array(leaf_ids)

    S_df = SMatrix(
        sparse_matrix=sparse_matrix,
        row_labels=row_labels,
        col_labels=col_labels,
        id_col=id_col,
        backend=backend,
    )
    # The bottom block is the identity by construction.
    S_df._bottom_identity_verified = True

    tags = {
        "national": row_labels[:1].copy(),
        "regions": row_labels[1 : 1 + n_regions].copy(),
        "leaves": col_labels.copy(),
    }
    return S_df, tags


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
