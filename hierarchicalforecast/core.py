__all__ = ['HierarchicalReconciliation', 'reconcile_ratio']


import re
import reprlib
import time
from inspect import signature

import narwhals.stable.v2 as nw
import numpy as np
from narwhals.typing import Frame, FrameT
from scipy import sparse
from scipy.stats import norm

from .methods import HReconciler
from .utils import (
    SMatrix,
    _construct_adjacency_matrix,
    _is_bottom_identity,
    _is_strictly_hierarchical,
    is_strictly_hierarchical,
)


def _compute_coherence_residual(
    y: np.ndarray,
    S: np.ndarray,
    idx_bottom: np.ndarray,
) -> np.ndarray:
    """Compute coherence residual: y - S @ y_bottom.

    The coherence residual measures how much the forecasts violate
    the hierarchical aggregation constraint. For coherent forecasts,
    this should be zero (or near-zero due to numerical precision).

    Args:
        y: Forecasts array of shape (n_series,) or (n_series, horizon).
        S: Summing matrix of shape (n_series, n_bottom).
        idx_bottom: Indices of bottom-level series.

    Returns:
        Coherence residual array with same shape as y.
    """
    y_bottom = y[idx_bottom]
    y_implied = S @ y_bottom
    return y - y_implied


def _compute_diagnostics_for_level(
    y_before: np.ndarray,
    y_after: np.ndarray,
    residual_before: np.ndarray,
    residual_after: np.ndarray,
    level_indices: np.ndarray,
) -> dict[str, float]:
    """Compute all diagnostic metrics for a single hierarchical level.

    Args:
        y_before: Base forecasts array of shape (n_series, horizon).
        y_after: Reconciled forecasts array of shape (n_series, horizon).
        residual_before: Coherence residuals before reconciliation.
        residual_after: Coherence residuals after reconciliation.
        level_indices: Indices of series belonging to this level.

    Returns:
        Dictionary of metric names to values.
    """
    # Extract level data
    y_before_level = y_before[level_indices].flatten()
    y_after_level = y_after[level_indices].flatten()
    residual_before_level = residual_before[level_indices].flatten()
    residual_after_level = residual_after[level_indices].flatten()

    # Adjustments
    adjustment = y_after_level - y_before_level

    metrics = {
        "coherence_residual_mae_before": np.mean(np.abs(residual_before_level), dtype=np.float64),
        "coherence_residual_rmse_before": np.sqrt(np.mean(residual_before_level**2), dtype=np.float64),
        "coherence_residual_mae_after": np.mean(np.abs(residual_after_level), dtype=np.float64),
        "coherence_residual_rmse_after": np.sqrt(np.mean(residual_after_level**2), dtype=np.float64),
        "adjustment_mae": np.mean(np.abs(adjustment), dtype=np.float64),
        "adjustment_rmse": np.sqrt(np.mean(adjustment**2), dtype=np.float64),
        "adjustment_max": np.max(np.abs(adjustment, dtype=np.float64)),
        "adjustment_mean": np.mean(adjustment, dtype=np.float64),
        "negative_count_before": np.sum(y_before_level < 0, dtype=np.int64),
        "negative_count_after": np.sum(y_after_level < 0, dtype=np.int64),
        "negative_introduced": np.sum((y_before_level >= 0) & (y_after_level < 0), dtype=np.int64),
        "negative_removed": np.sum((y_before_level < 0) & (y_after_level >= 0), dtype=np.int64),
    }
    return metrics


def _aggregate_diagnostics(
    diagnostics_per_model: dict[str, dict[str, dict[str, float]]],
    backend,
) -> Frame:
    """Aggregate per-model, per-level diagnostics into a DataFrame.

    Args:
        diagnostics_per_model: Nested dict {model_name: {level: {metric: value}}}.
        backend: Narwhals backend for DataFrame creation.

    Returns:
        DataFrame with columns [level, metric, model1, model2, ...].
    """
    models = list(diagnostics_per_model.keys())
    first_model_data = next(iter(diagnostics_per_model.values()))
    levels = list(first_model_data.keys())

    # Build output data - iterate through each level's metrics since
    # different levels may have different metrics (e.g., is_coherent only for Overall)
    data: dict[str, list] = {"level": [], "metric": []}
    for model in models:
        data[model] = []

    for level in levels:
        level_metrics = list(first_model_data[level].keys())
        for metric in level_metrics:
            data["level"].append(level)
            data["metric"].append(metric)
            for model in models:
                data[model].append(diagnostics_per_model[model][level][metric])

    df = nw.from_dict(data, backend=backend)
    return df.to_native()


def _build_fn_name(fn) -> str:
    fn_name = type(fn).__name__
    # Use _init_params if available, otherwise fall back to __dict__ for backwards compatibility
    func_params = getattr(fn, "_init_params", None)
    if func_params is None:
        func_params = fn.__dict__

    # Take default parameter out of names
    args_to_remove = ["insample", "num_threads"]
    if not func_params.get("nonnegative", False):
        args_to_remove.append("nonnegative")

    if fn_name == "MinTrace" and func_params.get("method") == "mint_shrink":
        if func_params.get("mint_shr_ridge") == 2e-8:
            args_to_remove.append("mint_shr_ridge")

    func_params = [
        f"{name}-{value}"
        for name, value in func_params.items()
        if name not in args_to_remove
    ]
    if func_params:
        fn_name += "_" + "_".join(func_params)
    return fn_name


def _estimate_sigmah(
    Y_hat_df: Frame,
    y_hat: np.ndarray,
    model_name: str,
    id_col: str = "unique_id",
) -> np.ndarray:
    r"""Estimate forecast standard deviation from paired prediction intervals.

    Uses the interquartile range (IQR) approach: for each pair of prediction
    interval columns (``{model_name}-lo-{level}`` and ``{model_name}-hi-{level}``),
    computes:

    .. math::
        \hat{\sigma}_h = \frac{\text{hi} - \text{lo}}{2z}

    where :math:`z = \Phi^{-1}(0.5 + \text{level}/200)`. When multiple paired
    levels are available, the estimates are averaged for robustness.

    Args:
        Y_hat_df: DataFrame with base forecasts and prediction interval columns.
        y_hat: Point forecasts of shape (``n_series``, ``n_horizon``).
        model_name: Name of the forecast model (used to find PI columns).
        id_col: Column name for series identifiers.

    Returns:
        Estimated forecast standard deviations of shape (``n_series``, ``n_horizon``).

    Raises:
        ValueError: If no paired prediction interval columns are found.
    """
    n_series = Y_hat_df[id_col].n_unique()

    # Find all lo/hi PI columns for this model
    lo_cols = {}
    hi_cols = {}
    for col in Y_hat_df.columns:
        if not col.startswith(model_name + "-"):
            continue
        suffix = col[len(model_name) + 1 :]
        level_match = re.match(r"^(lo|hi)-(.+)$", suffix)
        if level_match is None:
            continue
        direction, level_str = level_match.groups()
        level = float(level_str)
        if direction == "lo":
            lo_cols[level] = col
        else:
            hi_cols[level] = col

    # Collect paired levels
    paired_levels = sorted(lo_cols.keys() & hi_cols.keys())
    if not paired_levels:
        raise ValueError(
            f"No paired prediction interval columns found for `{model_name}`. "
            f"Expected columns like `{model_name}-lo-90` and `{model_name}-hi-90`."
        )

    # Estimate sigma_h from each paired level and average
    sigma_estimates = []
    for level in paired_levels:
        lo = Y_hat_df[lo_cols[level]].to_numpy().reshape(n_series, -1)
        hi = Y_hat_df[hi_cols[level]].to_numpy().reshape(n_series, -1)
        z = norm.ppf(0.5 + level / 200)
        sigma_estimates.append((hi - lo) / (2 * z))

    sigmah = np.mean(sigma_estimates, axis=0)
    sigmah = np.maximum(sigmah, 0)
    return sigmah


def _ratio_pairing_errors(
    numerator_nw: Frame,
    denominator_nw: Frame,
    id_col: str,
    time_col: str,
) -> list[str]:
    """Collect the (series, timestamp) pairs one ratio side is missing.

    Args:
        numerator_nw: Numerator frame with `id_col` and `time_col`.
        denominator_nw: Denominator frame with `id_col` and `time_col`.
        id_col: Column that identifies each series.
        time_col: Column that identifies each timestamp.

    Returns:
        A list with one human readable message per side that has unpaired keys.
        An empty list means both frames describe exactly the same key grid.
    """
    keys = [id_col, time_col]
    numerator_only = numerator_nw.select(keys).join(
        denominator_nw.select(keys), on=keys, how="anti"
    )
    denominator_only = denominator_nw.select(keys).join(
        numerator_nw.select(keys), on=keys, how="anti"
    )
    errors = []
    if len(numerator_only) > 0:
        errors.append(
            f"{len(numerator_only)} row(s) of the numerator have no matching "
            f"(series, timestamp) in the denominator, e.g. "
            f"{reprlib.repr(numerator_only.rows()[:5])}."
        )
    if len(denominator_only) > 0:
        errors.append(
            f"{len(denominator_only)} row(s) of the denominator have no matching "
            f"(series, timestamp) in the numerator, e.g. "
            f"{reprlib.repr(denominator_only.rows()[:5])}."
        )
    return errors


def _validate_ratio_inputs(
    numerator_nw: Frame,
    denominator_nw: Frame,
    id_col: str,
    time_col: str,
) -> None:
    """Validate that numerator and denominator share one key grid.

    The check runs before any reconciliation so that unpaired series or
    timestamps cannot be silently collapsed into a ratio.

    Raises:
        ValueError: If a frame has repeated keys, or if the two frames do not
            cover exactly the same (series, timestamp) pairs.
    """
    keys = [id_col, time_col]
    for name, frame in (
        ("numerator", numerator_nw),
        ("denominator", denominator_nw),
    ):
        repeated = frame.group_by(keys).agg(nw.len().alias("__count"))
        repeated = repeated.filter(nw.col("__count") > 1)
        if len(repeated) > 0:
            raise ValueError(
                f"The {name} has repeated (series, timestamp) pairs, e.g. "
                f"{reprlib.repr(repeated.drop('__count').rows()[:5])}. "
                "Each pair must appear exactly once."
            )
    errors = _ratio_pairing_errors(numerator_nw, denominator_nw, id_col, time_col)
    if errors:
        raise ValueError(
            "The numerator and denominator must cover the same (series, "
            "timestamp) pairs before reconciliation. " + " ".join(errors)
        )


def _build_ratio_report(
    reconciled_nw: Frame,
    model_specs: list[tuple[str, str, str]],
    tags: dict[str, np.ndarray],
    id_col: str,
    time_col: str,
    ratio_col: str,
) -> Frame:
    """Build a per-node rate report from the reconciled ratio columns.

    The numerator and denominator are each reconciled to be coherent, so every
    node carries a usable pair of values. A node's rate is its own reconciled
    numerator over its own reconciled denominator. Because both columns are
    additive, an aggregate node already reports the pooled rate (e.g. total
    orders over total visits), which avoids averaging store rates and the
    distortion that comes with it when store volumes shift.

    Args:
        reconciled_nw: Frame with the reconciled numerator and denominator
            columns, keyed by `id_col` and `time_col`.
        model_specs: `(model_name, numerator_col, denominator_col)` entries.
        tags: Mapping of level name to the series that belong to it.
        id_col: Column that identifies each series.
        time_col: Column that identifies each timestamp.
        ratio_col: Name of the output rate column.

    Returns:
        Frame with one row per (level, series, timestamp, model) holding the
        reconciled numerator, the reconciled denominator and the rate. A zero
        denominator yields a missing rate.
    """
    level_frames = []
    for level_rank, (level_name, level_ids) in enumerate(tags.items()):
        level_df = reconciled_nw.filter(nw.col(id_col).is_in(level_ids))
        for model_name, numerator_col, denominator_col in model_specs:
            rows = level_df.select(
                nw.lit(level_name).alias("level"),
                nw.lit(level_rank).alias("__level_rank"),
                nw.col(id_col),
                nw.col(time_col),
                nw.lit(model_name).alias("model"),
                nw.col(numerator_col).alias("numerator"),
                nw.col(denominator_col).alias("denominator"),
            )
            level_frames.append(rows)
    report = nw.concat(level_frames, how="vertical")
    # A zero denominator has no defined rate; report it as missing. Guarding
    # with `is_finite` also clears 0/0 and infinite ratios.
    report = report.with_columns(
        (nw.col("numerator") / nw.col("denominator")).cast(nw.Float64).alias(ratio_col)
    )
    report = report.with_columns(
        (nw.col("denominator") == 0).alias("__zero_denominator")
    )
    report = report.with_columns(
        nw.when(nw.col(ratio_col).is_finite() & ~nw.col("__zero_denominator"))
        .then(nw.col(ratio_col))
        .otherwise(nw.lit(None, dtype=nw.Float64))
        .alias(ratio_col)
    )
    report = report.drop("__zero_denominator")
    report = report.sort(by=["__level_rank", id_col, time_col, "model"])
    report = report.drop("__level_rank")
    report = report.select(
        "level", id_col, time_col, "model", "numerator", "denominator", ratio_col
    )
    return report


def _resolve_ratio_value_col(
    frame: Frame,
    value_col: str | None,
    id_col: str,
    time_col: str,
    side: str,
    infer: bool = True,
) -> str:
    """Resolve the value column of a ratio input frame.

    Args:
        frame: Input frame with `id_col`, `time_col` and the values.
        value_col: Explicit value column, or None to auto-detect it.
        id_col: Column that identifies each series.
        time_col: Column that identifies each timestamp.
        side: Either `numerator` or `denominator`, used in error messages.
        infer: Whether a missing `value_col` may be auto-detected.

    Returns:
        The name of the value column.

    Raises:
        ValueError: If the column is missing, ambiguous or not numeric.
    """
    candidates = [c for c in frame.columns if c not in (id_col, time_col)]
    if value_col is None:
        numeric = [c for c in candidates if frame.schema[c].is_numeric()]
        if not infer or len(numeric) != 1:
            raise ValueError(
                f"Could not infer the value column of the {side}. Pass "
                f"`{side}_col` explicitly. Numeric candidates: "
                f"{reprlib.repr(numeric)}."
            )
        value_col = numeric[0]
    if value_col not in frame.columns:
        raise ValueError(
            f"`{value_col}` was not found in the {side} columns "
            f"{reprlib.repr(frame.columns)}."
        )
    if not frame.schema[value_col].is_numeric():
        raise ValueError(
            f"The {side} column `{value_col}` must be numeric."
        )
    return value_col


class HierarchicalReconciliation:
    r"""Hierarchical Reconciliation Class.

    The `core.HierarchicalReconciliation` class allows you to efficiently fit multiple
    HierarchicaForecast methods for a collection of time series and base predictions stored in
    pandas DataFrames. The `Y_df` dataframe identifies series and datestamps with the unique_id and ds columns while the
    y column denotes the target time series variable. The `Y_h` dataframe stores the base predictions,
    example ([AutoARIMA](../statsforecast/src/core/models.html#autoarima),
    [ETS](../statsforecast/src/core/models.html#autoets), etc.).

    Args:
        - reconcilers (list[HReconciler]): A list of instantiated classes of the [reconciliation methods](./methods.html) module.

    References:
        - [Rob J. Hyndman and George Athanasopoulos (2018). "Forecasting principles and practice, Hierarchical and Grouped Series"](https://otexts.com/fpp3/hierarchical.html).

    """

    def __init__(self, reconcilers: list[HReconciler]):
        self.reconcilers = reconcilers

    def _prepare_fit(
        self,
        Y_hat_nw: Frame,
        S_nw: Frame,
        Y_nw: Frame | None,
        tags: dict[str, np.ndarray],
        level: list[int] | None = None,
        intervals_method: str = "normality",
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        id_time_col: str = "temporal_id",
        temporal: bool = False,
        skip_identity_check: bool = False,
    ) -> tuple[FrameT, FrameT, FrameT, list[str], str]:
        """Performs preliminary wrangling and protections."""
        Y_hat_nw_cols = Y_hat_nw.columns
        S_nw_cols = S_nw.columns

        # Check if Y_hat_df has the necessary columns for temporal
        if temporal:
            # We don't support insample methods, so Y_df must be None
            if Y_nw is not None:
                raise NotImplementedError(
                    "Temporal reconciliation requires `Y_df` to be None."
                )
            # If Y_nw is None, we need to check if the reconcilers are not insample methods
            for reconciler in self.reconcilers:
                if reconciler.insample:
                    reconciler_name = _build_fn_name(reconciler)
                    raise NotImplementedError(
                        f"Temporal reconciliation is not supported for `{reconciler_name}`."
                    )
            # Hence we also don't support bootstrap or permbu (rely on insample values)
            if intervals_method in ["bootstrap", "permbu"]:
                raise NotImplementedError(
                    f"Temporal reconciliation is not supported for intervals_method=`{intervals_method}`."
                )

            # Validate horizon against max row sum in temporal spec
            S_cols = [c for c in S_nw.columns if c != id_time_col]
            max_agg_factor = S_nw.with_columns(
                sum=nw.sum_horizontal(S_cols)
            )["sum"].max()

           # Compute horizon: number of unique timestamps per unique_id
            # This represents the number of bottom-level periods available for reconciliation
            horizon = Y_hat_nw.group_by(id_col).agg(nw.col(time_col).n_unique().alias("count"))["count"].max()
            if horizon < max_agg_factor:
                raise ValueError(
                    f"The forecast horizon ({horizon}) must be greater than or equal to "
                    f"the maximum aggregation factor in temporal_spec ({max_agg_factor}). "
                    f"Ensure you have at least {max_agg_factor} forecast periods."
                )

            missing_cols_temporal = set([id_col, time_col, id_time_col]) - set(
                Y_hat_nw_cols
            )
            if missing_cols_temporal:
                raise ValueError(
                    f"Check `Y_hat_df` columns, for temporal reconciliation {reprlib.repr(missing_cols_temporal)} must be in `Y_hat_df` columns."
                )
            if id_time_col not in S_nw_cols:
                raise ValueError(
                    f"Check `S_df` columns, {reprlib.repr(id_time_col)} must be in `S_df` columns."
                )
            id_cols = [id_col, time_col, target_col, id_time_col]
            id_col = id_time_col
        else:
            id_cols = [id_col, time_col, target_col]
            if id_col not in S_nw_cols:
                raise ValueError(
                    f"Check `S_df` columns, {reprlib.repr(id_col)} must be in `S_df` columns."
                )

        # Check if Y_hat_df has the right shape
        if len(Y_hat_nw.group_by(id_col).agg(nw.len()).unique(subset="len")) != 1:
            raise ValueError(
                "Check `Y_hat_df`, there are missing timestamps. All series should have the same number of predictions."
            )

        # -------------------------------- Match Y_hat/Y/S index order --------------------------------#
        # TODO: This is now a bit slow as we always sort.
        S_nw = S_nw.with_columns(**{f"{id_col}_id": np.arange(len(S_nw))})

        Y_hat_nw = Y_hat_nw.join(S_nw[[id_col, f"{id_col}_id"]], on=id_col, how="left")
        Y_hat_nw = Y_hat_nw.sort(by=[f"{id_col}_id", time_col])
        Y_hat_nw = Y_hat_nw[Y_hat_nw_cols]

        if Y_nw is not None:
            Y_nw_cols = Y_nw.columns
            Y_nw = Y_nw.join(S_nw[[id_col, f"{id_col}_id"]], on=id_col, how="left")
            Y_nw = Y_nw.sort(by=[f"{id_col}_id", time_col])
            Y_nw = Y_nw[Y_nw_cols]

        S_nw = S_nw[S_nw_cols]

        # ----------------------------------- Check Input's Validity ----------------------------------#

        # Check input's validity
        if intervals_method not in ["normality", "bootstrap", "permbu", "conformal"]:
            raise ValueError(f"Unknown interval method: {intervals_method}")

        # Check absence of Y_nw for insample reconcilers
        if Y_nw is None:
            for reconciler in self.reconcilers:
                if reconciler.insample:
                    reconciler_name = _build_fn_name(reconciler)
                    raise ValueError(
                        f"You need to provide `Y_df` for reconciler {reconciler_name}"
                    )
            if intervals_method in ["bootstrap", "permbu", "conformal"]:
                raise ValueError(
                    f"You need to provide `Y_df` when using intervals_method=`{intervals_method}`."
                )

        # Protect level list
        if level is not None:
            level_outside_domain = not all(0 <= x < 100 for x in level)
            if level_outside_domain and (intervals_method in ["normality", "permbu"]):
                raise ValueError(
                    "Level must be a list containing floating values in the interval [0, 100)."
                )

        # Declare output names
        model_names = [col for col in Y_hat_nw.columns if col not in id_cols]

        # Ensure numeric columns
        for model in model_names:
            if not Y_hat_nw.schema[model].is_numeric():
                raise ValueError(
                    f"Column `{model}` in `Y_hat_df` contains non-numeric values. Make sure no column in `Y_hat_df` contains non-numeric values."
                )
            if Y_hat_nw[model].is_null().any():
                raise ValueError(
                    f"Column `{model}` in `Y_hat_df` contains null values. Make sure no column in `Y_hat_df` contains null values."
                )

        # TODO: Complete y_hat_insample protection
        model_names = [
            name
            for name in model_names
            if not ("-lo" in name or "-hi" in name or "-median" in name)
        ]
        if intervals_method in ["bootstrap", "permbu"] and Y_nw is not None:
            missing_models = set(model_names) - set(Y_nw.columns)
            if missing_models:
                raise ValueError(
                    f"Check `Y_df` columns, {reprlib.repr(missing_models)} must be in `Y_df` columns."
                )

        # Assert S is an identity matrix at the bottom
        S_nw_cols.remove(id_col)
        if not skip_identity_check:
            n = len(S_nw_cols)
            # Slice rows first so only the n x n bottom block is materialised.
            S_bottom_nw = S_nw[-n:][S_nw_cols]
            S_bottom = S_bottom_nw.to_native()
            is_sparse_df = (
                hasattr(S_bottom, "sparse")
                and hasattr(S_bottom, "dtypes")
                and all(str(dtype).startswith("Sparse") for dtype in S_bottom.dtypes)
                # Only a zero fill value can be converted to COO. Any other fill
                # value falls through to the dense path, which reads the same
                # values, rather than raising from `.sparse.to_coo()`.
                and all(dtype.fill_value == 0 for dtype in S_bottom.dtypes)
            )
            if is_sparse_df:
                # Sparse-aware identity check: verify diagonal is 1 and off-diagonal is 0.
                S_bottom_coo = S_bottom.sparse.to_coo()
                is_identity = (
                    S_bottom_coo.shape[0] == S_bottom_coo.shape[1]
                    and S_bottom_coo.nnz == n
                    and np.allclose(S_bottom_coo.data, 1.0)
                    and np.array_equal(S_bottom_coo.row, S_bottom_coo.col)
                )
            else:
                is_identity = _is_bottom_identity(S_bottom_nw.to_numpy(), n)
            if not is_identity:
                raise ValueError(
                    f"The bottom {n}x{n} part of S must be an identity matrix."
                )

        # Check Y_hat_df\S_df series difference
        # TODO: this logic should be method specific
        S_diff = set(S_nw[id_col]) - set(Y_hat_nw[id_col])
        Y_hat_diff = set(Y_hat_nw[id_col]) - set(S_nw[id_col])
        if S_diff:
            raise ValueError(
                f"There are unique_ids in S_df that are not in Y_hat_df: {reprlib.repr(S_diff)}"
            )
        if Y_hat_diff:
            raise ValueError(
                f"There are unique_ids in Y_hat_df that are not in S_df: {reprlib.repr(Y_hat_diff)}"
            )

        if Y_nw is not None:
            Y_diff = set(Y_nw[id_col]) - set(Y_hat_nw[id_col])
            Y_hat_diff = set(Y_hat_nw[id_col]) - set(Y_nw[id_col])
            if Y_diff:
                raise ValueError(
                    f"There are unique_ids in Y_df that are not in Y_hat_df: {reprlib.repr(Y_diff)}"
                )
            if Y_hat_diff:
                raise ValueError(
                    f"There are unique_ids in Y_hat_df that are not in Y_df: {reprlib.repr(Y_hat_diff)}"
                )

        # Same Y_hat_df/S_df/Y_df's unique_ids. Order is guaranteed by sorting.
        # TODO: this logic should be method specific
        unique_ids = Y_hat_nw[id_col].unique().to_numpy()
        S_nw = S_nw.filter(nw.col(id_col).is_in(unique_ids))

        return Y_hat_nw, S_nw, Y_nw, model_names, id_col

    def _prepare_Y(
        self,
        Y_nw: Frame,
        S_nw: Frame,
        is_balanced: bool = True,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> np.ndarray:
        """Prepare Y data."""
        if is_balanced:
            Y = Y_nw[target_col].to_numpy().reshape(len(S_nw), -1)
        else:
            Y_pivot = Y_nw.pivot(
                on=time_col, index=id_col, values=target_col, sort_columns=True
            ).sort(by=id_col)
            Y_pivot_cols_ex_id_col = Y_pivot.columns
            Y_pivot_cols_ex_id_col.remove(id_col)

            # TODO: check if this is the best way to do it - it's reasonably fast to ensure Y_pivot has same order as S_nw
            pos_in_Y = np.searchsorted(
                Y_pivot[id_col].to_numpy(), S_nw[id_col].to_numpy()
            )
            Y_pivot = Y_pivot.select(nw.col(Y_pivot_cols_ex_id_col))
            Y_pivot = Y_pivot[pos_in_Y]
            Y = Y_pivot.to_numpy()

        # TODO: the result is a Fortran contiguous array, see if we can avoid the below copy (I don't think so)
        Y = np.ascontiguousarray(Y, dtype=np.float64)
        return Y

    def reconcile(
        self,
        Y_hat_df: Frame,
        tags: dict[str, np.ndarray],
        S_df: "Frame | SMatrix" = None,
        Y_df: Frame | None = None,
        level: list[int] | None = None,
        intervals_method: str = "normality",
        num_samples: int = -1,
        seed: int = 0,
        is_balanced: bool = False,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        id_time_col: str = "temporal_id",
        temporal: bool = False,
        diagnostics: bool = False,
        diagnostics_atol: float = 1e-6,
    ) -> FrameT:
        r"""Hierarchical Reconciliation Method.

        The `reconcile` method is analogous to SKLearn `fit_predict` method, it
        applies different reconciliation techniques instantiated in the `reconcilers` list.

        Most reconciliation methods can be described by the following convenient
        linear algebra notation:

        ```math
        \tilde{\mathbf{y}}_{[a,b],\\tau} = \mathbf{S}_{[a,b][b]} \mathbf{P}_{[b][a,b]} \hat{\mathbf{y}}_{[a,b],\\tau}
        ```

        where $a, b$ represent the aggregate and bottom levels, $\mathbf{S}_{[a,b][b]}$ contains
        the hierarchical aggregation constraints, and $\mathbf{P}_{[b][a,b]}$ varies across
        reconciliation methods. The reconciled predictions are

        ```math
        \tilde{\mathbf{y}}_{[a,b],\tau}
        ```

        and the base predictions

        ```math
        \hat{\mathbf{y}}_{[a,b],\tau}
        ```

        Args:
            Y_hat_df (Frame): DataFrame, base forecasts with columns ['unique_id', 'ds'] and models to reconcile.
            tags (dict[str, np.ndarray]): Each key is a level and its value contains tags associated to that level.
            S_df (Frame | SMatrix, optional): DataFrame or :class:`~hierarchicalforecast.utils.SMatrix` with summing matrix of size `(base, bottom)`, see [aggregate method](./utils.html#aggregate). Passing an ``SMatrix`` (from ``aggregate(..., sparse_s=True)``) avoids dense materialization. Default is None.
            Y_df (Optional[Frame], optional): DataFrame, training set of base time series with columns `['unique_id', 'ds', 'y']`.
                If a class of `self.reconciles` receives `y_hat_insample`, `Y_df` must include them as columns. Default is None.
            level (Optional[list[int]], optional): positive float list [0,100), confidence levels for prediction intervals. Default is None.
            intervals_method (str, optional): method used to calculate prediction intervals, one of `normality`, `bootstrap`, `permbu`. Default is "normality".
            num_samples (int, optional): if positive return that many probabilistic coherent samples. Default is -1.
            seed (int, optional): random seed for numpy generator's replicability. Default is 0.
            is_balanced (bool, optional): wether `Y_df` is balanced, set it to True to speed things up if `Y_df` is balanced. Default is False.
            id_col (str, optional): column that identifies each serie. Default is "unique_id".
            time_col (str, optional): column that identifies each timestep, its values can be timestamps or integers. Default is "ds".
            target_col (str, optional): column that contains the target. Default is "y".
            id_time_col (str, optional): column that identifies each temporal aggregation level (required when `temporal=True`). Default is "temporal_id".
            temporal (bool, optional): if True, perform temporal reconciliation. Default is False.
            diagnostics (bool, optional): if True, compute coherence diagnostics and store in `self.diagnostics`. Default is False.
            diagnostics_atol (float, optional): absolute tolerance for numerical coherence check. Default is 1e-6.

        Returns:
            (FrameT): DataFrame, with reconciled predictions.

        Note:
            When `diagnostics=True`, after reconciliation completes, `self.diagnostics` will contain
            a DataFrame with coherence metrics per hierarchical level, including:
            - `coherence_residual_mae_before/after`: Mean absolute coherence residual before/after reconciliation
            - `adjustment_mae/rmse/max/mean`: Statistics on the adjustments made by reconciliation
            - `negative_count_before/after`: Count of negative values before/after reconciliation
            - `is_coherent`: Whether reconciled forecasts satisfy aggregation constraints (Overall level only)
            - `coherence_max_violation`: Maximum coherence violation (Overall level only)
        """
        # To Narwhals
        Y_hat_nw = nw.from_native(Y_hat_df)
        # Accept SMatrix or DataFrame for S_df
        self._s_matrix = None
        if isinstance(S_df, SMatrix):
            self._s_matrix = S_df
            # Validate identity property on sparse data (avoids dense allocation)
            n_bottom = S_df.sparse_shape[1]
            if not S_df.check_bottom_identity():
                raise ValueError(
                    f"The bottom {n_bottom}x{n_bottom} part of S must be an identity matrix."
                )
            # Build lightweight id-only frame for _prepare_fit (avoids dense materialisation)
            S_nw = nw.from_dict(
                {S_df.id_col: S_df.row_labels}, backend=S_df.backend
            )
            S_nw = nw.maybe_reset_index(S_nw)
        else:
            S_nw = nw.from_native(S_df)
        if Y_df is not None:
            Y_nw = nw.from_native(Y_df)
        else:
            Y_nw = None

        # Check input's validity and sort dataframes
        Y_hat_nw, S_nw, Y_nw, self.model_names, id_col = self._prepare_fit(
            Y_hat_nw=Y_hat_nw,
            S_nw=S_nw,
            Y_nw=Y_nw,
            tags=tags,
            level=level,
            intervals_method=intervals_method,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            id_time_col=id_time_col,
            temporal=temporal,
            skip_identity_check=self._s_matrix is not None,
        )

        # Initialize reconciler arguments
        _s_matrix = self._s_matrix
        if _s_matrix is not None:
            n_bottom = _s_matrix.sparse_shape[1]
            n_series = _s_matrix.sparse_shape[0]
            S_nw_cols_ex_id_col = list(_s_matrix.col_labels)
        else:
            n_series = len(S_nw)
            S_nw_cols_ex_id_col = S_nw.columns
            S_nw_cols_ex_id_col.remove(id_col)
            n_bottom = len(S_nw_cols_ex_id_col)

        reconciler_args = dict(
            idx_bottom=np.arange(n_series)[-n_bottom:],
            tags={
                key: S_nw.with_columns(nw.col(id_col).is_in(val).alias("in_cols"))[
                    "in_cols"
                ]
                .to_numpy()
                .nonzero()[0]
                for key, val in tags.items()
            },
        )

        any_sparse = any([method.is_sparse_method for method in self.reconcilers])

        # Fast path: when S_df was an SMatrix, extract sparse/dense directly
        if any_sparse:
            if _s_matrix is not None:
                S_for_sparse = _s_matrix.to_csr()
            else:
                try:
                    S_for_sparse = sparse.csr_matrix(
                        S_nw.select(nw.col(S_nw_cols_ex_id_col)).to_native().sparse.to_coo()
                    )
                except AttributeError:
                    S_for_sparse = sparse.csr_matrix(
                        S_nw.select(nw.col(S_nw_cols_ex_id_col))
                        .to_numpy()
                        .astype(np.float64, copy=False)
                    )
        if Y_nw is not None:
            y_insample = self._prepare_Y(
                Y_nw=Y_nw,
                S_nw=S_nw,
                is_balanced=is_balanced,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
            )
            reconciler_args["y_insample"] = y_insample

        # Check if any reconciler requires a strictly hierarchical structure (fail fast)
        # Single traversal to collect all strict reconcilers
        strict_reconcilers = [
            r for r in self.reconcilers
            if getattr(r, 'is_strictly_hierarchical', False)
        ]

        if strict_reconcilers:
            # Check if any strict reconciler uses sparse methods
            has_sparse_strict_method = any(r.is_sparse_method for r in strict_reconcilers)

            # Perform the appropriate hierarchy validation
            is_valid_hierarchy = False
            if has_sparse_strict_method:
                # Use sparse check with adjacency matrix
                A = _construct_adjacency_matrix(S_for_sparse, reconciler_args["tags"])
                is_valid_hierarchy = _is_strictly_hierarchical(A, reconciler_args["tags"])
            else:
                # Use dense check with summing matrix
                if _s_matrix is not None:
                    S_numpy = _s_matrix.to_dense()
                else:
                    S_numpy = (
                        S_nw.select(nw.col(S_nw_cols_ex_id_col))
                        .to_numpy()
                        .astype(np.float64, copy=False)
                    )
                is_valid_hierarchy = is_strictly_hierarchical(S_numpy, reconciler_args["tags"])

            # Raise error if hierarchy is not valid
            if not is_valid_hierarchy:
                strict_method_names = [_build_fn_name(r) for r in strict_reconcilers]
                methods_str = "', '".join(strict_method_names)
                raise ValueError(
                    f"The reconciliation method(s) '{methods_str}' require a strictly hierarchical structure. "
                    f"The provided hierarchy contains nodes with multiple parents (grouped structure), "
                    f"which is not supported by these methods. Please use a different reconciliation method "
                    f"(e.g., BottomUp, MinTrace, or ERM) that supports grouped hierarchies."
                )

        Y_tilde_nw = nw.maybe_reset_index(Y_hat_nw.clone())
        self.execution_times = {}
        self.level_names = {}
        self.sample_names = {}
        for reconciler in self.reconcilers:
            reconcile_fn_name = _build_fn_name(reconciler)

            if reconciler.is_sparse_method:
                reconciler_args["S"] = S_for_sparse
            else:
                if _s_matrix is not None:
                    reconciler_args["S"] = _s_matrix.to_dense()
                else:
                    reconciler_args["S"] = (
                        S_nw.select(nw.col(S_nw_cols_ex_id_col))
                        .to_numpy()
                        .astype(np.float64, copy=False)
                    )

            for model_name in self.model_names:
                start = time.time()
                recmodel_name = f"{model_name}/{reconcile_fn_name}"

                model_cols = [id_col, time_col, model_name]

                # TODO: the below should be method specific
                y_hat = self._prepare_Y(
                    Y_nw=Y_hat_nw[model_cols],
                    S_nw=S_nw,
                    is_balanced=True,
                    id_col=id_col,
                    time_col=time_col,
                    target_col=model_name,
                )
                reconciler_args["y_hat"] = y_hat

                if Y_nw is not None and model_name in Y_nw.columns:
                    y_hat_insample = self._prepare_Y(
                        Y_nw=Y_nw[model_cols],
                        S_nw=S_nw,
                        is_balanced=is_balanced,
                        id_col=id_col,
                        time_col=time_col,
                        target_col=model_name,
                    )
                    reconciler_args["y_hat_insample"] = y_hat_insample

                if level is not None:
                    reconciler_args["intervals_method"] = intervals_method
                    reconciler_args["num_samples"] = 200
                    reconciler_args["seed"] = seed

                    if intervals_method in ["normality", "permbu"]:
                        sigmah = _estimate_sigmah(
                            Y_hat_df=Y_hat_nw,
                            y_hat=y_hat,
                            model_name=model_name,
                            id_col=id_col,
                        )
                        reconciler_args["sigmah"] = sigmah

                # Mean and Probabilistic reconciliation
                kwargs_ls = [
                    key
                    for key in signature(reconciler.fit_predict).parameters
                    if key in reconciler_args.keys()
                ]
                kwargs = {key: reconciler_args[key] for key in kwargs_ls}

                fcsts_model = reconciler(**kwargs, level=level)

                # Validate reconciler state for sampling (fail fast before processing results)
                if num_samples > 0 and level is not None:
                    if not getattr(reconciler, "fitted", False):
                        raise ValueError(
                            f"Reconciler {reconcile_fn_name} does not support sampling. "
                            "Set num_samples=0 or use a different reconciler."
                        )
                    if getattr(reconciler, "sampler", None) is None:
                        raise ValueError(
                            f"Reconciler {reconcile_fn_name} does not have a sampler configured. "
                            "Ensure intervals_method is set correctly."
                        )

                # Parse final outputs
                Y_tilde_nw = Y_tilde_nw.with_columns(
                    **{recmodel_name: fcsts_model["mean"].flatten()}
                )

                if (
                    intervals_method in ["bootstrap", "normality", "permbu", "conformal"]
                    and level is not None
                ):
                    level.sort()
                    lo_names = [f"{recmodel_name}-lo-{lv}" for lv in reversed(level)]
                    hi_names = [f"{recmodel_name}-hi-{lv}" for lv in level]
                    self.level_names[recmodel_name] = lo_names + hi_names
                    sorted_quantiles = np.reshape(
                        fcsts_model["quantiles"], (len(Y_tilde_nw), -1)
                    )
                    y_tilde = dict(
                        zip(self.level_names[recmodel_name], sorted_quantiles.T, strict=False)
                    )
                    Y_tilde_nw = Y_tilde_nw.with_columns(**y_tilde)

                    if num_samples > 0:
                        samples = reconciler.sample(num_samples=num_samples)
                        self.sample_names[recmodel_name] = [
                            f"{recmodel_name}-sample-{i}" for i in range(num_samples)
                        ]
                        samples = np.reshape(samples, (len(Y_tilde_nw), -1))
                        y_tilde = dict(zip(self.sample_names[recmodel_name], samples.T, strict=False))
                        Y_tilde_nw = Y_tilde_nw.with_columns(**y_tilde)

                end = time.time()
                self.execution_times[f"{model_name}/{reconcile_fn_name}"] = end - start

        # Compute diagnostics if requested
        if diagnostics:
            native_namespace = nw.get_native_namespace(Y_hat_nw)

            # Prepare S matrix as dense numpy array for diagnostics
            if _s_matrix is not None:
                S_numpy = _s_matrix.to_dense()
            else:
                S_numpy = (
                    S_nw.select(nw.col(S_nw_cols_ex_id_col))
                    .to_numpy()
                    .astype(np.float64, copy=False)
                )

            # Get indices - note: S_numpy has shape (n_series, n_bottom)
            # idx_bottom should refer to the last n_bottom rows of y
            n_series = S_numpy.shape[0]
            n_bottom = S_numpy.shape[1]
            idx_bottom_diag = np.arange(n_series)[-n_bottom:]
            tags_numeric = reconciler_args["tags"]

            # Add "Overall" level
            tags_with_overall = {
                **tags_numeric,
                "Overall": np.arange(n_series),
            }

            diagnostics_per_model: dict[str, dict[str, dict[str, float]]] = {}

            for recmodel_name in self.execution_times.keys():
                # Extract base model name
                base_model_name = recmodel_name.split("/")[0]

                # Get base forecasts as numpy array (n_series, horizon)
                y_before = (
                    Y_hat_nw[base_model_name].to_numpy().reshape(n_series, -1)
                )
                # Get reconciled forecasts
                y_after = (
                    Y_tilde_nw[recmodel_name].to_numpy().reshape(n_series, -1)
                )

                # Compute coherence residuals for the entire array
                residual_before = _compute_coherence_residual(y_before, S_numpy, idx_bottom_diag)
                residual_after = _compute_coherence_residual(y_after, S_numpy, idx_bottom_diag)

                model_diagnostics: dict[str, dict[str, float]] = {}

                for level_name, level_indices in tags_with_overall.items():
                    level_metrics = _compute_diagnostics_for_level(
                        y_before=y_before,
                        y_after=y_after,
                        residual_before=residual_before,
                        residual_after=residual_after,
                        level_indices=level_indices,
                    )
                    model_diagnostics[level_name] = level_metrics

                # Add overall coherence check metrics
                max_violation = float(np.max(np.abs(residual_after)))
                is_coherent = float(max_violation <= diagnostics_atol)
                model_diagnostics["Overall"]["is_coherent"] = is_coherent
                model_diagnostics["Overall"]["coherence_max_violation"] = max_violation

                diagnostics_per_model[recmodel_name] = model_diagnostics

            self.diagnostics = _aggregate_diagnostics(
                diagnostics_per_model, backend=native_namespace
            )
        else:
            self.diagnostics = None

        Y_tilde_df = Y_tilde_nw.to_native()

        return Y_tilde_df

    def bootstrap_reconcile(
        self,
        Y_hat_df: Frame,
        S_df: Frame,
        tags: dict[str, np.ndarray],
        Y_df: Frame | None = None,
        level: list[int] | None = None,
        intervals_method: str = "normality",
        num_samples: int = -1,
        num_seeds: int = 1,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> FrameT:
        """Bootstraped Hierarchical Reconciliation Method.

        Applies N times, based on different random seeds, the `reconcile` method
        for the different reconciliation techniques instantiated in the `reconcilers` list.

        Args:
            Y_hat_df (Frame): DataFrame, base forecasts with columns ['unique_id', 'ds'] and models to reconcile.
            S_df (Frame): DataFrame with summing matrix of size `(base, bottom)`, see [aggregate method](./utils.html#aggregate).
            tags (dict[str, np.ndarray]): Each key is a level and its value contains tags associated to that level.
            Y_df (Optional[Frame], optional): DataFrame, training set of base time series with columns `['unique_id', 'ds', 'y']`.
                If a class of `self.reconciles` receives `y_hat_insample`, `Y_df` must include them as columns. Default is None.
            level (Optional[list[int]], optional): positive float list [0,100), confidence levels for prediction intervals. Default is None.
            intervals_method (str, optional): method used to calculate prediction intervals, one of `normality`, `bootstrap`, `permbu`. Default is "normality".
            num_samples (int, optional): if positive return that many probabilistic coherent samples. Default is -1.
            num_seeds (int, optional): random seed for numpy generator's replicability. Default is 1.
            id_col (str, optional): column that identifies each serie. Default is "unique_id".
            time_col (str, optional): column that identifies each timestep, its values can be timestamps or integers. Default is "ds".
            target_col (str, optional): column that contains the target. Default is "y".

        Returns:
            (FrameT): DataFrame, with bootstraped reconciled predictions.
        """
        # Bootstrap reconciled predictions
        Y_tilde_list = []
        for seed in range(num_seeds):
            Y_tilde_df = self.reconcile(
                Y_hat_df=Y_hat_df,
                S_df=S_df,
                tags=tags,
                Y_df=Y_df,
                level=level,
                intervals_method=intervals_method,
                num_samples=num_samples,
                seed=seed,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
            )
            Y_tilde_nw = nw.from_native(Y_tilde_df)
            Y_tilde_nw = Y_tilde_nw.with_columns(nw.lit(seed).alias("seed"))

            # TODO: fix broken recmodel_names
            if seed == 0:
                first_columns = Y_tilde_nw.columns
            Y_tilde_nw = Y_tilde_nw.rename(
                {col: first_columns[i] for i, col in enumerate(first_columns)}
            )
            Y_tilde_list.append(Y_tilde_nw)

        Y_bootstrap_nw = nw.concat(Y_tilde_list, how="vertical")
        Y_bootstrap_df = Y_bootstrap_nw.to_native()

        return Y_bootstrap_df

    def reconcile_ratio(
        self,
        numerator_df: Frame,
        denominator_df: Frame | None = None,
        S_df: "Frame | SMatrix" = None,
        tags: dict[str, np.ndarray] = None,
        numerator_col: str | None = None,
        denominator_col: str | None = None,
        ratio_col: str = "rate",
        Y_df: Frame | None = None,
        is_balanced: bool = False,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> FrameT:
        r"""Reconcile a ratio by reconciling its numerator and denominator.

        A ratio such as a conversion rate (orders over visits) cannot be
        reconciled directly: averaging store-level rates lets small stores
        dominate and drifts from the true aggregate rate whenever the number of
        stores or their visit volumes change. Instead, this method reconciles
        the numerator and the denominator separately on the same (series,
        timestamp) pairs, then derives the rate from the two reconciled
        columns.

        The numerator and denominator must describe exactly the same keys
        before reconciliation. Series or timestamps present on only one side
        raise a `ValueError` up front rather than producing a silently mixed
        result.

        Args:
            numerator_df (Frame): Base forecasts of the numerator (e.g. orders),
                with `id_col`, `time_col` and one value column.
            denominator_df (Frame, optional): Base forecasts of the denominator
                (e.g. visits), with the same keys as `numerator_df`. When
                omitted, `numerator_df` must hold both value columns and
                `denominator_col` must be given.
            S_df (Frame | SMatrix, optional): Summing matrix, see
                :meth:`~hierarchicalforecast.core.HierarchicalReconciliation.reconcile`.
            tags (dict[str, np.ndarray]): Each key is a level and its value the
                series belonging to that level.
            numerator_col (str, optional): Value column of `numerator_df`.
                Inferred when the frame has a single non-key column.
            denominator_col (str, optional): Value column of `denominator_df`.
                Inferred when the frame has a single non-key column.
            ratio_col (str, optional): Name of the rate column in the report.
                Default is `"rate"`.
            Y_df (Frame, optional): Training data, required by in-sample
                reconcilers. It must carry both value columns.
            is_balanced (bool, optional): Whether `Y_df` is balanced.
            id_col (str, optional): Series identifier column. Default is
                `"unique_id"`.
            time_col (str, optional): Timestamp column. Default is `"ds"`.
            target_col (str, optional): Target column of `Y_df`. Default `"y"`.

        Returns:
            Frame: Long-format report with one row per (level, id, time, model)
            holding `numerator`, `denominator` and `ratio_col`. The numerator
            and denominator are the reconciled values, so aggregate levels
            report a pooled rate. A zero denominator yields a missing rate.

        Raises:
            ValueError: If the two inputs disagree on their (series, timestamp)
                pairs, contain repeated keys, or a value column is missing.
        """
        numerator_nw = nw.from_native(numerator_df)
        # When both columns live in one frame, reconcile each of them from it.
        denominator_nw = (
            numerator_nw if denominator_df is None else nw.from_native(denominator_df)
        )
        if tags is None or len(tags) == 0:
            raise ValueError(
                "`tags` is required to map each series to its hierarchy level."
            )
        keys = [id_col, time_col]
        for side_name, frame in (
            ("numerator", numerator_nw),
            ("denominator", denominator_nw),
        ):
            missing = [c for c in keys if c not in frame.columns]
            if missing:
                raise ValueError(
                    f"`{side_name}_df` is missing required column(s) "
                    f"{reprlib.repr(missing)}."
                )
        numerator_value = _resolve_ratio_value_col(
            numerator_nw, numerator_col, id_col, time_col, "numerator"
        )
        # A shared frame needs both columns named explicitly: auto-detection
        # would see two numeric columns and could not tell them apart.
        infer_denominator_col = denominator_df is not None
        denominator_value = _resolve_ratio_value_col(
            denominator_nw,
            denominator_col,
            id_col,
            time_col,
            "denominator",
            infer=infer_denominator_col,
        )
        # Fail before reconciliation when the two sides do not line up.
        _validate_ratio_inputs(numerator_nw, denominator_nw, id_col, time_col)

        # Each side is reconciled on its own so an in-sample method uses that
        # side's own history. The sides are renamed to internal column names
        # first so both inputs may legitimately use the same value name
        # (e.g. both called `y`) without colliding in the joined output.
        side_frames = {}
        side_output_col = {}
        for side_name, frame, value_col in (
            ("numerator", numerator_nw, numerator_value),
            ("denominator", denominator_nw, denominator_value),
        ):
            internal_col = f"__ratio_{side_name}"
            side_output_col[side_name] = internal_col
            side_Y = None
            if Y_df is not None:
                Y_nw = nw.from_native(Y_df)
                if value_col not in Y_nw.columns:
                    raise ValueError(
                        f"`Y_df` must contain the {side_name} value column "
                        f"`{value_col}` so its history can be reconciled."
                    )
                # Point the target at this side's history without dropping the
                # value column that interval methods look up.
                side_Y = Y_nw.with_columns(
                    nw.col(value_col).alias(target_col)
                ).to_native()
            side_hat = frame.select(
                nw.col(id_col), nw.col(time_col), nw.col(value_col)
            ).rename({value_col: internal_col})
            reconciled_side = self.reconcile(
                Y_hat_df=side_hat.to_native(),
                S_df=S_df,
                tags=tags,
                Y_df=side_Y,
                is_balanced=is_balanced,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
            )
            side_frames[side_name] = nw.from_native(reconciled_side)

        reconciled_nw = side_frames["numerator"].join(
            side_frames["denominator"], on=keys, how="left"
        )
        reconciled_nw = reconciled_nw.sort(by=keys)
        # `reconcile` names each output `{internal_col}/{method}`; pair the two
        # sides per method so both reconciled columns line up.
        model_specs = [
            (
                _build_fn_name(reconciler),
                f"{side_output_col['numerator']}/{_build_fn_name(reconciler)}",
                f"{side_output_col['denominator']}/{_build_fn_name(reconciler)}",
            )
            for reconciler in self.reconcilers
        ]
        report = _build_ratio_report(
            reconciled_nw=reconciled_nw,
            model_specs=model_specs,
            tags=tags,
            id_col=id_col,
            time_col=time_col,
            ratio_col=ratio_col,
        )
        return report.to_native()


def reconcile_ratio(
    numerator_df: Frame,
    denominator_df: Frame | None = None,
    reconcilers: list[HReconciler] | None = None,
    S_df: "Frame | SMatrix" = None,
    tags: dict[str, np.ndarray] = None,
    numerator_col: str | None = None,
    denominator_col: str | None = None,
    ratio_col: str = "rate",
    **kwargs,
) -> FrameT:
    """Reconcile a ratio from separately reconciled numerator and denominator.

    Convenience wrapper around
    :meth:`~hierarchicalforecast.core.HierarchicalReconciliation.reconcile_ratio`.

    Args:
        numerator_df (Frame): Base forecasts of the numerator (e.g. orders).
        denominator_df (Frame, optional): Base forecasts of the denominator
            (e.g. visits). When omitted, `numerator_df` must carry both value
            columns and `denominator_col` must be given.
        reconcilers (list[HReconciler]): Reconcilers applied to both columns.
        S_df (Frame | SMatrix, optional): Summing matrix.
        tags (dict[str, np.ndarray]): Mapping of level name to its series.
        numerator_col (str, optional): Value column of `numerator_df`.
        denominator_col (str, optional): Value column of `denominator_df`.
        ratio_col (str, optional): Name of the rate column. Default `"rate"`.
        **kwargs: Forwarded to
            :meth:`~hierarchicalforecast.core.HierarchicalReconciliation.reconcile_ratio`.

    Returns:
        Frame: Long-format report with `level`, series id, timestamp, `model`,
        `numerator`, `denominator` and `ratio_col`.
    """
    if reconcilers is None:
        raise ValueError("`reconcilers` is required.")
    hrec = HierarchicalReconciliation(reconcilers=reconcilers)
    return hrec.reconcile_ratio(
        numerator_df=numerator_df,
        denominator_df=denominator_df,
        S_df=S_df,
        tags=tags,
        numerator_col=numerator_col,
        denominator_col=denominator_col,
        ratio_col=ratio_col,
        **kwargs,
    )
