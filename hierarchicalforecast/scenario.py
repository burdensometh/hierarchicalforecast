__all__ = [
    "validate_quantile_curves",
    "validate_scenario_axis",
    "quantile_curves_to_scenarios",
    "aggregate_scenarios",
    "scenarios_to_quantiles",
    "ScenarioReconciler",
]


import numpy as np

# Tolerance used when checking monotonicity of quantile levels and curve values.
# Quantile levels must be strictly increasing (up to this tolerance), while
# quantile values only need to be non-decreasing (flat sections are allowed).
_MONO_TOL = 1e-9


def validate_quantile_curves(
    quantile_levels: np.ndarray,
    quantile_values: np.ndarray,
) -> None:
    r"""Validate quantile forecast curves.

    Checks that the upstream per-series quantile curves are well formed:

    - Quantile levels are a 1D array with at least two points.
    - Quantile levels lie in the open interval ``(0, 1)``.
    - Quantile levels are strictly increasing.
    - ``quantile_values`` has shape ``(n_series, n_horizon, n_quantiles)``.
    - Quantile values are monotonically non-decreasing along the quantile
      axis for every series and horizon, i.e. each curve is a valid
      (generalized) quantile function. Flat sections (e.g. discrete
      distributions) are allowed.

    Args:
        quantile_levels (np.ndarray): 1D array of quantile levels in ``(0, 1)``.
        quantile_values (np.ndarray): Quantile forecasts of shape
            ``(n_series, n_horizon, n_quantiles)``.

    Raises:
        ValueError: If any of the structural or monotonicity checks fail.
    """
    levels = np.asarray(quantile_levels, dtype=np.float64)

    if levels.ndim != 1:
        raise ValueError(
            f"quantile_levels must be a 1D array, got {levels.ndim}D array "
            f"with shape {levels.shape}."
        )
    if levels.size < 2:
        raise ValueError(
            "At least two quantile levels are required to define a curve, "
            f"got {levels.size}."
        )
    if np.any(np.isnan(levels)):
        raise ValueError("quantile_levels must not contain NaN values.")
    if np.any(levels <= 0.0) or np.any(levels >= 1.0):
        raise ValueError(
            "All quantile levels must lie in the open interval (0, 1); "
            f"got levels {levels.tolist()}."
        )
    if np.any(np.diff(levels) <= _MONO_TOL):
        raise ValueError(
            "Quantile levels must be strictly increasing; "
            f"got levels {levels.tolist()}."
        )

    values = np.asarray(quantile_values, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(
            "quantile_values must be a 3D array of shape "
            "(n_series, n_horizon, n_quantiles), got "
            f"{values.ndim}D array with shape {values.shape}."
        )
    if np.any(np.isnan(values)):
        raise ValueError("quantile_values must not contain NaN values.")
    if values.shape[2] != levels.size:
        raise ValueError(
            f"quantile_values last dimension ({values.shape[2]}) must match "
            f"the number of quantile levels ({levels.size})."
        )

    # Non-decreasing values along the quantile axis: every diff must be >= 0.
    value_diffs = np.diff(values, axis=2)
    if np.any(value_diffs < -_MONO_TOL):
        bad_idx = np.argwhere(value_diffs < -_MONO_TOL)
        first = bad_idx[0]
        raise ValueError(
            "Quantile values must be monotonically non-decreasing along the "
            "quantile axis for every series and horizon. First violation at "
            f"(series={first[0]}, horizon={first[1]}, "
            f"between levels {levels[first[2]]:.4f} and "
            f"{levels[first[2] + 1]:.4f})."
        )


def validate_scenario_axis(scenario_probs: np.ndarray) -> np.ndarray:
    r"""Validate the common probability-rank scenario axis.

    The scenario axis is the set of probability ranks at which every store's
    quantile curve is interpolated. It must be:

    - A 1D array with at least one scenario.
    - Free of NaNs and contained in the open interval ``(0, 1)``.
    - Strictly increasing and free of duplicates.

    Args:
        scenario_probs (np.ndarray): 1D array of probability ranks in ``(0, 1)``,
            one per aligned scenario.

    Returns:
        np.ndarray: The validated axis as a float64 array.

    Raises:
        ValueError: If the axis is empty, out of range, non-monotonic or
            contains duplicates.
    """
    probs = np.asarray(scenario_probs, dtype=np.float64)

    if probs.ndim != 1:
        raise ValueError(
            f"scenario_probs must be a 1D array, got {probs.ndim}D array "
            f"with shape {probs.shape}."
        )
    if probs.size == 0:
        raise ValueError(
            "scenario_probs is empty. At least one scenario is required."
        )
    if np.any(np.isnan(probs)):
        raise ValueError("scenario_probs must not contain NaN values.")
    if np.any(probs <= 0.0) or np.any(probs >= 1.0):
        raise ValueError(
            "All scenario probability ranks must lie in the open interval "
            f"(0, 1); got {probs.tolist()}."
        )
    if np.any(np.diff(probs) <= _MONO_TOL):
        raise ValueError(
            "scenario_probs must be strictly increasing and contain no "
            f"duplicates; got {probs.tolist()}."
        )
    return probs


def quantile_curves_to_scenarios(
    quantile_levels: np.ndarray,
    quantile_values: np.ndarray,
    scenario_probs: np.ndarray,
    validate: bool = True,
) -> np.ndarray:
    r"""Interpolate quantile curves onto a common probability-rank axis.

    For every series and horizon the quantile curve
    :math:`\hat{Q}_{i,t}(\tau)` is evaluated at each scenario probability
    rank :math:`\tau_k` via linear interpolation. Ranks outside the range of
    the supplied quantile levels are clamped to the nearest end-point value,
    so no extrapolated values are produced.

    Because every store is evaluated at the *same* rank :math:`\tau_k`, the
    ``k``-th column of the output is an aligned comonotonic scenario that can
    be summed across stores.

    Args:
        quantile_levels (np.ndarray): 1D array of quantile levels in ``(0, 1)``,
            strictly increasing.
        quantile_values (np.ndarray): Quantile forecasts of shape
            ``(n_series, n_horizon, n_quantiles)``, non-decreasing on the last
            axis.
        scenario_probs (np.ndarray): 1D array of probability ranks in ``(0, 1)``
            defining the aligned scenarios.
        validate (bool, optional): If True, validate the curve monotonicity and
            the scenario axis before interpolating. Default is True.

    Returns:
        np.ndarray: Bottom-level scenario values of shape
        ``(n_series, n_horizon, n_scenarios)``. The scenario at index ``k``
        is aligned across all series.
    """
    if validate:
        validate_quantile_curves(quantile_levels, quantile_values)
        probs = validate_scenario_axis(scenario_probs)
    else:
        probs = np.asarray(scenario_probs, dtype=np.float64)
    levels = np.asarray(quantile_levels, dtype=np.float64)
    values = np.asarray(quantile_values, dtype=np.float64)

    # Vectorized linear interpolation along the last axis. np.interp only
    # accepts 1D values, so use searchsorted + manual interpolation. Query
    # ranks outside the supplied levels are clamped to the end-point values.
    n_levels = levels.size
    right_idx = np.searchsorted(levels, probs)
    right_idx = np.clip(right_idx, 1, n_levels - 1)
    left_idx = right_idx - 1

    x0 = levels[left_idx]
    x1 = levels[right_idx]
    f0 = values[..., left_idx]
    f1 = values[..., right_idx]
    weights = (probs - x0) / (x1 - x0)
    scenarios = f0 + (f1 - f0) * weights

    # Clamp ranks below the first / above the last supplied level.
    below = probs <= levels[0]
    above = probs >= levels[-1]
    if np.any(below):
        scenarios[..., below] = values[..., :1]
    if np.any(above):
        scenarios[..., above] = values[..., -1:]
    return scenarios


def aggregate_scenarios(
    bottom_scenarios: np.ndarray,
    S: np.ndarray | None = None,
) -> np.ndarray:
    r"""Aggregate aligned scenarios into coherent regional scenarios.

    Regional values are built by summing the *same* scenario across the
    member stores. This preserves the rank alignment (the joint dependence
    implied by the common probability axis) instead of adding marginal
    quantiles, which would be incoherent.

    With a summing matrix :math:`\mathbf{S}` of shape
    ``(n_base, n_bottom)`` the aggregation is, for every horizon and
    scenario:

    ```math
    \tilde{\mathbf{y}}_{\cdot,t,k} = \mathbf{S}\,
        \hat{\mathbf{y}}_{\cdot,t,k}^{scen}
    ```

    Args:
        bottom_scenarios (np.ndarray): Aligned bottom-level scenario values of
            shape ``(n_bottom, n_horizon, n_scenarios)``.
        S (np.ndarray, optional): Summing matrix of shape
            ``(n_base, n_bottom)``. If None, a single regional total is
            returned (equivalent to a row of ones). Default is None.

    Returns:
        np.ndarray: Coherent scenario values for every node of the hierarchy,
        shape ``(n_base, n_horizon, n_scenarios)``. When ``S`` is None the
        output has shape ``(1, n_horizon, n_scenarios)``.

    Raises:
        ValueError: If the scenario array is not 3D, contains NaNs, or its
            first dimension does not match the number of columns of ``S``.
    """
    scenarios = np.asarray(bottom_scenarios, dtype=np.float64)
    if scenarios.ndim != 3:
        raise ValueError(
            "bottom_scenarios must be a 3D array of shape "
            "(n_bottom, n_horizon, n_scenarios), got "
            f"{scenarios.ndim}D array with shape {scenarios.shape}."
        )
    if np.any(np.isnan(scenarios)):
        raise ValueError("bottom_scenarios must not contain NaN values.")

    if S is None:
        # Single regional total: sum across stores within each scenario.
        return scenarios.sum(axis=0, keepdims=True)

    S_arr = np.asarray(S, dtype=np.float64)
    if S_arr.ndim != 2:
        raise ValueError(
            f"S must be a 2D matrix of shape (n_base, n_bottom), got "
            f"{S_arr.ndim}D array with shape {S_arr.shape}."
        )
    if S_arr.shape[1] != scenarios.shape[0]:
        raise ValueError(
            f"S has {S_arr.shape[1]} columns but bottom_scenarios has "
            f"{scenarios.shape[0]} bottom series; these must match."
        )

    # (n_base, n_bottom) @ (n_bottom, n_horizon * n_scenarios)
    n_bottom, n_horizon, n_scenarios = scenarios.shape
    flat = scenarios.reshape(n_bottom, n_horizon * n_scenarios)
    aggregated = S_arr @ flat
    return aggregated.reshape(S_arr.shape[0], n_horizon, n_scenarios)


def scenarios_to_quantiles(
    scenarios: np.ndarray,
    quantile_levels: np.ndarray,
    axis: int = -1,
    validate: bool = True,
) -> np.ndarray:
    r"""Compute marginal quantiles from aligned scenarios.

    Quantiles are taken across the scenario axis of the (already coherent)
    regional scenario values. This must be run *after* scenario aggregation;
    taking quantiles of each store separately and adding them would ignore
    the joint dependence and give incoherent regional quantiles.

    Args:
        scenarios (np.ndarray): Coherent scenario values, e.g. the output of
            :func:`aggregate_scenarios`, shape
            ``(n_base, n_horizon, n_scenarios)``.
        quantile_levels (np.ndarray): 1D array of quantile levels in ``(0, 1)``
            at which to evaluate the regional quantile curves.
        axis (int, optional): Axis holding the scenarios. Default is -1.
        validate (bool, optional): If True, validate the requested levels and
            check the input for NaNs. Default is True.

    Returns:
        np.ndarray: Regional quantile curves. For a 3D input with
        ``axis=-1`` the output has shape ``(n_base, n_horizon, n_quantiles)``.

    Raises:
        ValueError: If the requested levels are invalid or the scenarios
            contain NaNs.
    """
    levels = np.asarray(quantile_levels, dtype=np.float64)
    if validate:
        if levels.ndim != 1 or levels.size == 0:
            raise ValueError(
                "quantile_levels must be a non-empty 1D array of levels in "
                f"(0, 1); got shape {levels.shape}."
            )
        if np.any(np.isnan(levels)):
            raise ValueError("quantile_levels must not contain NaN values.")
        if np.any(levels <= 0.0) or np.any(levels >= 1.0):
            raise ValueError(
                "All quantile levels must lie in the open interval (0, 1); "
                f"got {levels.tolist()}."
            )
        if np.any(np.isnan(scenarios)):
            raise ValueError("scenarios must not contain NaN values.")

    # np.quantile inserts the quantile axis at position 0; move it to the
    # reduction axis so the output keeps the input's layout with quantiles
    # in place of the scenario axis.
    return np.moveaxis(
        np.quantile(scenarios, levels, axis=axis),
        source=0,
        destination=axis,
    )


class ScenarioReconciler:
    r"""Bottom-up probabilistic reconciliation via aligned rank scenarios.

    The upstream model supplies, for every store and horizon, a marginal
    quantile forecast curve together with a common set of probability-rank
    scenarios. This reconciler:

    1. Validates that every quantile curve is monotonic
       (:func:`validate_quantile_curves`) and that the scenario axis is a
       complete, strictly increasing set of ranks in ``(0, 1)``
       (:func:`validate_scenario_axis`).
    2. Interpolates each store's curve at every scenario rank
       (:func:`quantile_curves_to_scenarios`), producing aligned bottom-level
       scenario values.
    3. Aggregates regionally by summing values that share the *same* scenario
       index through the summing matrix (:func:`aggregate_scenarios`), never
       by adding marginal quantiles.
    4. Derives regional quantile curves from the aggregated scenarios
       (:func:`scenarios_to_quantiles`).

    The common probability-rank axis acts as a comonotonic coupling between
    stores: scenario ``k`` of every store corresponds to the same rank
    :math:`\tau_k` of its marginal distribution.

    Args:
        S (np.ndarray, optional): Summing matrix of shape
            ``(n_base, n_bottom)`` whose last ``n_bottom`` rows form the
            identity (the bottom-level nodes). If None, the reconciler
            produces a single regional total. Default is None.
        scenario_probs (np.ndarray): 1D array of probability ranks in ``(0, 1)``
            defining the aligned scenarios.

    Attributes:
        S (np.ndarray | None): The summing matrix.
        scenario_probs (np.ndarray): The validated scenario axis.
        bottom_scenarios (np.ndarray | None): Bottom-level scenario values from
            the last call, shape ``(n_bottom, n_horizon, n_scenarios)``.
        regional_scenarios (np.ndarray | None): Coherent regional scenario
            values from the last call, shape
            ``(n_base, n_horizon, n_scenarios)``.

    Examples:
        >>> # Two stores, one horizon, three quantile levels
        >>> levels = np.array([0.1, 0.5, 0.9])
        >>> values = np.array([
        ...     [[1.0, 2.0, 3.0]],   # store A
        ...     [[4.0, 5.0, 6.0]],   # store B
        ... ])
        >>> probs = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        >>> rec = ScenarioReconciler(scenario_probs=probs)
        >>> regional_q = rec.reconcile_quantiles(levels, values,
        ...                                       output_levels=levels)
        >>> regional_q.shape
        (1, 1, 3)
    """

    def __init__(
        self,
        scenario_probs: np.ndarray,
        S: np.ndarray | None = None,
    ):
        self.scenario_probs = validate_scenario_axis(scenario_probs)
        if S is not None:
            S = np.asarray(S, dtype=np.float64)
            if S.ndim != 2:
                raise ValueError(
                    "S must be a 2D matrix of shape (n_base, n_bottom), got "
                    f"{S.ndim}D array with shape {S.shape}."
                )
        self.S = S
        self.bottom_scenarios: np.ndarray | None = None
        self.regional_scenarios: np.ndarray | None = None

    def fit_transform_scenarios(
        self,
        quantile_levels: np.ndarray,
        quantile_values: np.ndarray,
    ) -> np.ndarray:
        r"""Build bottom-level scenarios and aggregate them regionally.

        Args:
            quantile_levels (np.ndarray): 1D array of quantile levels in
                ``(0, 1)``, strictly increasing.
            quantile_values (np.ndarray): Bottom-level quantile forecasts of
                shape ``(n_bottom, n_horizon, n_quantiles)``.

        Returns:
            np.ndarray: Coherent regional scenario values of shape
            ``(n_base, n_horizon, n_scenarios)`` (or
            ``(1, n_horizon, n_scenarios)`` when no summing matrix was given).
        """
        self.bottom_scenarios = quantile_curves_to_scenarios(
            quantile_levels=quantile_levels,
            quantile_values=quantile_values,
            scenario_probs=self.scenario_probs,
            validate=True,
        )
        if self.S is not None and self.S.shape[1] != self.bottom_scenarios.shape[0]:
            raise ValueError(
                f"S has {self.S.shape[1]} columns but received "
                f"{self.bottom_scenarios.shape[0]} bottom-level series."
            )
        self.regional_scenarios = aggregate_scenarios(
            bottom_scenarios=self.bottom_scenarios,
            S=self.S,
        )
        return self.regional_scenarios

    def reconcile_quantiles(
        self,
        quantile_levels: np.ndarray,
        quantile_values: np.ndarray,
        output_levels: np.ndarray | None = None,
    ) -> np.ndarray:
        r"""Reconcile bottom-level quantile curves into regional quantiles.

        This is the end-to-end pipeline: interpolate to scenarios, aggregate
        within each scenario, then take quantiles of the aggregated scenarios.

        Args:
            quantile_levels (np.ndarray): 1D array of quantile levels in
                ``(0, 1)`` describing the input curves.
            quantile_values (np.ndarray): Bottom-level quantile forecasts of
                shape ``(n_bottom, n_horizon, n_quantiles)``.
            output_levels (np.ndarray, optional): Quantile levels to evaluate
                on the regional distributions. If None, the input
                ``quantile_levels`` are reused. Default is None.

        Returns:
            np.ndarray: Regional quantile curves of shape
            ``(n_base, n_horizon, n_output_quantiles)``.
        """
        regional_scenarios = self.fit_transform_scenarios(
            quantile_levels=quantile_levels,
            quantile_values=quantile_values,
        )
        if output_levels is None:
            output_levels = quantile_levels
        return scenarios_to_quantiles(
            scenarios=regional_scenarios,
            quantile_levels=output_levels,
            validate=True,
        )
