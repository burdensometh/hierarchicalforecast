"""Asymmetric stockout cost used for forecast / reconciler selection.

The operations team historically selected forecasting methods by average
error (MAE/RMSE/MAPE).  Those losses are *symmetric*: an under-forecast of
one unit costs the same as an over-forecast of one unit.  In inventory
planning the two errors are not equivalent -- an under-forecast drives a
stockout (lost margin, contractual penalties, expediting), while an
over-forecast only creates holding / obsolescence cost.

This module implements the standard asymmetric piecewise-linear cost, which
is exactly the *pinball / quantile loss* whose optimal point forecast is the
service-level quantile.  It also supports per-series unit costs so that key
regions can be weighted through their actual economics instead of an ad-hoc
multiplier on MAE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["AsymmetricStockoutCost"]


@dataclass(frozen=True)
class AsymmetricStockoutCost:
    """Piecewise-linear asymmetric stockout cost.

    For realized demand ``y`` and forecast ``y_hat`` the per-period cost of
    series ``i`` is::

        c_u_i * max(y - y_hat, 0)  +  c_o_i * max(y_hat - y, 0)

    where ``c_u`` is the unit under-forecast (stockout) cost and ``c_o`` is
    the unit over-forecast (holding / obsolescence) cost.  With constant unit
    costs this is the pinball loss at quantile level::

        tau = c_u / (c_u + c_o)

    so a method minimizing this loss is steered toward the service-level
    quantile rather than the conditional mean.  Key regions are handled by
    giving them larger ``c_u`` (real penalty / expediting economics), not by
    changing the loss shape.

    Args:
        c_under: Unit stockout cost. Scalar or array of shape ``(n_series,)``.
        c_over: Unit overage cost. Scalar or array of shape ``(n_series,)``.
        series_weights: Optional non-negative weights of shape
            ``(n_series,)``. Use these for importance that is *not* already
            monetized in ``c_under`` / ``c_over`` (e.g. strategic accounts).
            Defaults to uniform weights.
    """

    c_under: float | np.ndarray
    c_over: float | np.ndarray
    series_weights: np.ndarray | None = None

    def _as_array(self, value: float | np.ndarray, n: int) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            arr = np.full(n, float(arr))
        if arr.shape != (n,):
            raise ValueError(
                f"Expected cost of shape ({n},), got {arr.shape}."
            )
        if np.any(arr < 0):
            raise ValueError("Unit costs must be non-negative.")
        return arr

    def quantile_level(self, n_series: int) -> np.ndarray:
        """Return the implied service-level quantile per series."""
        c_u = self._as_array(self.c_under, n_series)
        c_o = self._as_array(self.c_over, n_series)
        denom = c_u + c_o
        if np.any(denom <= 0):
            raise ValueError(
                "c_under + c_over must be strictly positive to define a quantile."
            )
        return c_u / denom

    def per_cell(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """Cost for every (series, horizon) cell.

        Args:
            y: Realizations, shape ``(n_series, horizon)``.
            y_hat: Forecasts, same shape as ``y``.

        Returns:
            Array of costs with the same shape as ``y``.
        """
        if y.shape != y_hat.shape:
            raise ValueError(
                f"y shape {y.shape} != y_hat shape {y_hat.shape}."
            )
        n = y.shape[0]
        c_u = self._as_array(self.c_under, n)[:, None]
        c_o = self._as_array(self.c_over, n)[:, None]
        under = np.maximum(y - y_hat, 0.0)
        over = np.maximum(y_hat - y, 0.0)
        return c_u * under + c_o * over

    def score(
        self,
        y: np.ndarray,
        y_hat: np.ndarray,
        level_weights: dict[str, float] | None = None,
        tags: dict[str, np.ndarray] | None = None,
    ) -> float:
        """Hierarchically weighted mean per-period cost for one candidate.

        Aggregation convention (see ``docs/methodology``):

        1. Per-cell cost is computed with the series' own unit costs.
        2. Cells are averaged over the holdout horizon, giving a per-series
           mean cost.
        3. Within each hierarchy level, series are combined with
           ``series_weights`` (normalized to sum to 1 inside the level).
        4. Levels are combined with ``level_weights`` (normalized to sum to
           1).  A separate *total* monetary cost can be recovered with
           :meth:`monetary_total` for budget checks.

        Args:
            y: Realizations, shape ``(n_series, horizon)``.
            y_hat: Forecasts, same shape.
            level_weights: Weight of each hierarchy level. When ``None`` the
                levels implied by ``tags`` share weight uniformly.
            tags: Mapping level name -> integer indices of its series in the
                rows of ``y``. When ``None`` everything is one level.

        Returns:
            A single comparable scalar (weighted mean cost per period).
        """
        cell_cost = self.per_cell(y, y_hat)
        n = cell_cost.shape[0]
        w = (
            np.ones(n, dtype=np.float64)
            if self.series_weights is None
            else np.asarray(self.series_weights, dtype=np.float64)
        )
        if w.shape != (n,) or np.any(w < 0):
            raise ValueError("series_weights must be non-negative and match n_series.")

        if tags is None:
            tags = {"Overall": np.arange(n)}
        if level_weights is None:
            level_weights = {lvl: 1.0 for lvl in tags}

        missing = set(tags) - set(level_weights)
        if missing:
            raise ValueError(f"Missing level weights for: {sorted(missing)}")
        extra = set(level_weights) - set(tags)
        if extra:
            raise ValueError(f"Unknown levels in weights: {sorted(extra)}")

        per_series = cell_cost.mean(axis=1)
        level_scores: dict[str, float] = {}
        total_w = float(sum(level_weights.values()))
        if total_w <= 0:
            raise ValueError("level_weights must have a positive sum.")

        for lvl, idx in tags.items():
            idx = np.asarray(idx)
            w_lvl = w[idx]
            w_sum = w_lvl.sum()
            if w_sum <= 0:
                raise ValueError(f"Level '{lvl}' has zero total series weight.")
            level_scores[lvl] = float(np.dot(per_series[idx], w_lvl) / w_sum)

        return float(
            sum(level_scores[lvl] * level_weights[lvl] for lvl in tags) / total_w
        )

    def monetary_total(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """Unweighted total monetary cost over all cells (budget reporting).

        Unlike :meth:`score` this is *not* normalized, so it answers the
        budget question "how many currency units did this candidate burn on
        the holdout?".
        """
        return float(self.per_cell(y, y_hat).sum())

    def stockout_component(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """Total under-forecast (stockout-only) monetary cost.

        This is the quantity the key-region stockout budget is defined on.
        """
        n = y.shape[0]
        c_u = self._as_array(self.c_under, n)[:, None]
        return float((c_u * np.maximum(y - y_hat, 0.0)).sum())
