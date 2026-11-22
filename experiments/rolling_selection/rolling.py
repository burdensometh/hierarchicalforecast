"""Rolling-origin holdout selection of hierarchical reconciliation methods.

The workflow implemented here replaces "pick the method with the best average
error on one train/test split" with an online, leakage-free procedure:

1. Define expanding cutoffs over the history.
2. At every cutoff ``t``, fit *each candidate reconciler* using only data
   available up to ``t`` (the base forecasters are refit on the same prefix,
   so their hyper-parameters never see the holdout either).
3. Forecast the next ``horizon`` periods and score every candidate on the
   realized values with the asymmetric, hierarchy-weighted stockout cost.
4. Select a winner -- either per-cutoff or from the accumulated paired
   evidence across cutoffs -- then refit that winner on all data available
   at ``t`` and publish the next-period forecast.
5. Roll to the next cutoff and repeat.

Candidates may enter the comparison late (new method, data source with a
shorter history).  Their *support* -- the set of (cutoff, horizon-step)
cells they actually produce a forecast for -- is recorded, and pairwise
comparisons are always restricted to the **intersection of supports**.  See
``methodology.md`` for why comparing across different date ranges is invalid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .cost import AsymmetricStockoutCost

__all__ = [
    "Candidate",
    "RollingSelectionResult",
    "rolling_holdout_selection",
    "common_support_score",
]


class ForecastFn(Protocol):
    """Signature of a base-forecast factory used by :class:`Candidate`.

    Given the training data available at a cutoff and the horizon, it returns
    a tuple ``(y_hat_holdout, y_hat_insample)`` where ``y_hat_holdout`` has
    shape ``(n_series, horizon)`` and ``y_hat_insample`` has shape
    ``(n_series, n_train)`` with in-sample fitted values (needed by
    covariance-based reconcilers such as MinTrace ``mint_shrink``).
    """

    def __call__(
        self, train: np.ndarray, horizon: int
    ) -> tuple[np.ndarray, np.ndarray | None]: ...


class ReconcilerFactory(Protocol):
    """Builds a *fresh*, unfitted reconciler instance for one cutoff.

    A new instance per cutoff is mandatory: reconcilers store fitted
    projection matrices (``P``, ``W``), so reusing an instance across cutoffs
    would silently carry old parameters forward.
    """

    def __call__(self) -> Any: ...


@dataclass
class Candidate:
    """One candidate in the selection grid.

    A candidate is a full pipeline: base forecasts + reconciliation.  Two
    candidates that share a reconciler but use different base models are
    different candidates and must carry different names.

    Args:
        name: Unique candidate label.
        make_reconciler: Zero-argument factory returning a fresh reconciler
            from ``hierarchicalforecast.methods`` (or any object exposing the
            same ``fit_predict(S, y_hat, y_insample, y_hat_insample, tags)``
            interface).
        forecast_fn: Base-forecast factory, refit on every cutoff prefix.
        entry_cutoff: First cutoff position (integer index into ``cutoffs``)
            at which this candidate is available. Candidates introduced later
            -- a new method, or one backed by a data source with shorter
            history -- default to entering at the first cutoff. Before entry
            the candidate is skipped entirely, so its missing cells never
            enter any score or comparison.
    """

    name: str
    make_reconciler: ReconcilerFactory
    forecast_fn: ForecastFn
    entry_cutoff: int = 0


@dataclass
class CutoffRecord:
    """Everything that happened at one rolling cutoff."""

    cutoff: Any
    forecasts: dict[str, np.ndarray]
    actuals: np.ndarray
    scores: dict[str, float]
    stockout_cost: dict[str, float]
    monetary_cost: dict[str, float]
    support: dict[str, np.ndarray]  # boolean mask over horizon steps
    cell_cost: dict[str, np.ndarray]  # per-cell costs, NaN off support
    winner: str
    published: np.ndarray


@dataclass
class RollingSelectionResult:
    """Output of :func:`rolling_holdout_selection`."""

    records: list[CutoffRecord]
    level_weights: dict[str, float]
    selection_rule: str
    candidates: list[str]

    def leaderboard(self) -> pd.DataFrame:
        """Per-cutoff weighted cost for every candidate (NaN off support)."""
        frame = pd.DataFrame(
            {
                "cutoff": [r.cutoff for r in self.records],
                **{
                    name: [r.scores.get(name, np.nan) for r in self.records]
                    for name in self.candidates
                },
            }
        )
        return frame

    def winners(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "cutoff": [r.cutoff for r in self.records],
                "winner": [r.winner for r in self.records],
            }
        )

    def cumulative_paired_scores(self) -> pd.DataFrame:
        """Mean cost on common support, re-computed after each cutoff.

        All values at a row use the intersection of supports accumulated up to
        that cutoff, so late-entering candidates are never penalized for
        missing history they could not have produced.
        """
        rows: list[dict[str, Any]] = []
        for k in range(len(self.records)):
            sub = self.records[: k + 1]
            row: dict[str, Any] = {"cutoff": sub[-1].cutoff}
            for name in self.candidates:
                vals = [r.scores[name] for r in sub if name in r.scores]
                row[name] = float(np.mean(vals)) if vals else np.nan
            rows.append(row)
        return pd.DataFrame(rows)


def _fit_reconciler(
    reconciler: Any,
    S: np.ndarray,
    y_hat: np.ndarray,
    y_train: np.ndarray,
    y_hat_insample: np.ndarray | None,
    tags: dict[str, np.ndarray],
) -> np.ndarray:
    """Call a library reconciler behind a uniform interface.

    All library reconcilers accept the same keyword arguments (TopDown takes
    ``tags`` positionally in its signature but accepts the keyword too), so a
    single call shape works for the whole methods module.
    """
    kwargs: dict[str, Any] = dict(
        S=S,
        y_hat=y_hat,
        y_insample=y_train,
        y_hat_insample=y_hat_insample,
        tags=tags,
    )
    out = reconciler.fit_predict(**kwargs)
    # Library methods return {"mean": ..., optional quantiles}
    if isinstance(out, dict):
        return np.asarray(out["mean"], dtype=np.float64)
    return np.asarray(out, dtype=np.float64)


def common_support_score(
    records: list[CutoffRecord],
    candidate_a: str,
    candidate_b: str,
) -> tuple[float, float, int]:
    """Score two candidates on the intersection of their (cutoff, step) cells.

    Returns ``(score_a, score_b, n_cells)``.  A cell is included only when
    both candidates produced a forecast at that cutoff and horizon step.  The
    per-cell costs are averaged without hierarchy re-weighting at this layer
    (hierarchy weights were already applied when the records were scored);
    callers needing weighted comparisons should score directly with
    :class:`AsymmetricStockoutCost` on masked arrays.
    """
    costs_a: list[float] = []
    costs_b: list[float] = []
    n_cells = 0
    for rec in records:
        if candidate_a not in rec.support or candidate_b not in rec.support:
            continue
        mask = rec.support[candidate_a] & rec.support[candidate_b]
        if not mask.any():
            continue
        costs_a.append(np.nanmean(rec.cell_cost[candidate_a][:, mask]))
        costs_b.append(np.nanmean(rec.cell_cost[candidate_b][:, mask]))
        n_cells += int(mask.sum()) * rec.actuals.shape[0]
    if not costs_a:
        return np.nan, np.nan, 0
    return float(np.mean(costs_a)), float(np.mean(costs_b)), n_cells


def rolling_holdout_selection(
    Y: np.ndarray,
    ds: np.ndarray,
    S: np.ndarray,
    tags: dict[str, np.ndarray],
    candidates: list[Candidate],
    cost_fn: AsymmetricStockoutCost,
    cutoffs: list[int],
    horizon: int,
    series_labels: list[str] | np.ndarray | None = None,
    level_weights: dict[str, float] | None = None,
    selection_rule: str = "cumulative",
    stockout_budget: float | None = None,
    min_switch_gain: float = 0.0,
    incumbent: str | None = None,
) -> RollingSelectionResult:
    """Run the rolling-origin holdout selection.

    Args:
        Y: Full realized history, shape ``(n_series, n_time)``. Used only to
            slice training prefixes and evaluate holdouts -- a candidate
            fitted at cutoff index ``t`` never sees ``Y[:, t:]``.
        ds: Time labels of length ``n_time``; ``ds[t]`` labels the cutoff
            (last observed period). Holdout periods are ``ds[t+1 : t+1+h]``.
        S: Hierarchical summing matrix ``(n_series, n_bottom)``.
        tags: Level name -> row indices into ``Y`` (as returned by
            ``hierarchicalforecast.utils.aggregate``).
        candidates: Candidate pipelines to compare.
        cost_fn: Asymmetric stockout cost carrying per-series unit costs.
        cutoffs: Integer cutoff positions. Each must leave at least
            ``horizon`` realized periods after it.
        horizon: Holdout / forecast horizon in periods.
        series_labels: Row-order ``unique_id`` labels of ``Y``. Required only
            when ``tags`` contain labels (as produced by
            ``hierarchicalforecast.utils.aggregate``) rather than integer
            indices; used to map labels to row positions for scoring.
        level_weights: Weights per hierarchy level for cost aggregation.
        selection_rule: Either ``"per_cutoff"`` (winner = lowest cost on this
            cutoff alone) or ``"cumulative"`` (winner = lowest mean cost over
            all common-support cells seen so far; recommended -- less churn,
            more evidence per decision).
        stockout_budget: Optional ceiling on the holdout *stockout-only*
            cost. Candidates violating it are excluded from winning even if
            their aggregate score is lowest; if all violate, the one with the
            smallest stockout cost wins and the breach is flagged.
        min_switch_gain: Minimum fractional cost improvement over the
            incumbent required to switch winners (anti-churn guardrail, e.g.
            0.02 = require 2% improvement). Applied under ``"cumulative"``.
        incumbent: Name of the initial incumbent (e.g. the method currently in
            production). Defaults to the first candidate.

    Returns:
        :class:`RollingSelectionResult` with per-cutoff records, scores,
        supports, winners and the published forecasts.
    """
    if Y.ndim != 2:
        raise ValueError("Y must have shape (n_series, n_time).")
    if len(ds) != Y.shape[1]:
        raise ValueError("ds length must match Y's time dimension.")
    names = [c.name for c in candidates]
    if len(set(names)) != len(names):
        raise ValueError("Candidate names must be unique.")
    if selection_rule not in {"per_cutoff", "cumulative"}:
        raise ValueError("selection_rule must be 'per_cutoff' or 'cumulative'.")
    n_time = Y.shape[1]
    for t in cutoffs:
        if t < 1 or t + horizon >= n_time:
            raise ValueError(
                f"Cutoff {t} needs {horizon} holdout periods within {n_time}."
            )
    for cand in candidates:
        if not 0 <= cand.entry_cutoff < len(cutoffs):
            raise ValueError(
                f"Candidate '{cand.name}' entry_cutoff={cand.entry_cutoff} "
                f"outside [0, {len(cutoffs) - 1}]."
            )

    incumbent = incumbent or candidates[0].name
    if incumbent not in names:
        raise ValueError(f"Incumbent '{incumbent}' is not a candidate.")

    # ``aggregate`` returns tags as arrays of unique_id labels, while the
    # underlying reconcilers (and the cost module) address rows by integer
    # position -- the library's own HierarchicalReconciliation wrapper does
    # exactly this label->index conversion in core.py. Reconciler calls and
    # scoring below therefore both use ``tags_idx``.
    tags_idx: dict[str, np.ndarray] = {}
    for lvl, members in tags.items():
        members = np.asarray(members)
        if np.issubdtype(members.dtype, np.integer):
            tags_idx[lvl] = members.astype(np.int64, copy=False)
        else:
            if series_labels is None:
                raise ValueError(
                    f"Level '{lvl}' uses unique_id labels; pass series_labels "
                    "(the row-order unique_ids of Y) so they can be mapped to indices."
                )
            lookup = {uid: i for i, uid in enumerate(series_labels)}
            try:
                tags_idx[lvl] = np.array(
                    [lookup[m] for m in members], dtype=np.int64
                )
            except KeyError as exc:
                raise ValueError(
                    f"Tag '{lvl}' references unknown unique_id: {exc.args[0]}"
                ) from exc

    records: list[CutoffRecord] = []
    active_incumbent = incumbent

    for cutoff_idx, t in enumerate(cutoffs):
        y_train = Y[:, : t + 1]
        y_holdout = Y[:, t + 1 : t + 1 + horizon]

        forecasts: dict[str, np.ndarray] = {}
        supports: dict[str, np.ndarray] = {}
        cell_costs: dict[str, np.ndarray] = {}
        scores: dict[str, float] = {}
        stockout_only: dict[str, float] = {}
        monetary: dict[str, float] = {}

        for cand in candidates:
            # Candidates with a later entry point are absent, not zero: they
            # get no support mask at this cutoff and cannot win or be compared.
            if cutoff_idx < cand.entry_cutoff:
                continue

            # 1. Refit the BASE forecaster on the cutoff prefix only.
            y_hat_h, y_hat_in = cand.forecast_fn(y_train, horizon)
            if y_hat_h.shape != y_holdout.shape:
                raise ValueError(
                    f"Candidate '{cand.name}' returned holdout shape "
                    f"{y_hat_h.shape}, expected {y_holdout.shape}."
                )

            # 2. Fit a FRESH reconciler on data available at the cutoff.
            reconciler = cand.make_reconciler()
            y_tilde = _fit_reconciler(
                reconciler,
                S=S,
                y_hat=y_hat_h,
                y_train=y_train,
                y_hat_insample=y_hat_in,
                tags=tags_idx,
            )

            # 3. Record support: NaN/inf means the candidate could not cover
            #    that (series, step) cell at this cutoff.
            finite = np.isfinite(y_tilde)
            support = finite.all(axis=0)  # per horizon-step, all series valid
            safe = np.where(finite, y_tilde, np.nan)
            forecasts[cand.name] = safe
            supports[cand.name] = support

            if support.all():
                cell = cost_fn.per_cell(y_holdout, y_tilde)
                cell_costs[cand.name] = cell
                scores[cand.name] = cost_fn.score(
                    y_holdout, y_tilde, level_weights=level_weights, tags=tags_idx
                )
                stockout_only[cand.name] = cost_fn.stockout_component(
                    y_holdout, y_tilde
                )
                monetary[cand.name] = cost_fn.monetary_total(y_holdout, y_tilde)
            else:
                # Partial support: score on the covered steps only and keep
                # the full mask so paired comparisons intersect supports.
                covered = np.where(support)[0]
                cell = np.full_like(y_holdout, np.nan, dtype=np.float64)
                cell[:, covered] = cost_fn.per_cell(
                    y_holdout[:, covered], y_tilde[:, covered]
                )
                cell_costs[cand.name] = cell
                scores[cand.name] = cost_fn.score(
                    y_holdout[:, covered],
                    y_tilde[:, covered],
                    level_weights=level_weights,
                    tags=tags_idx,
                )
                stockout_only[cand.name] = cost_fn.stockout_component(
                    y_holdout[:, covered], y_tilde[:, covered]
                )
                monetary[cand.name] = cost_fn.monetary_total(
                    y_holdout[:, covered], y_tilde[:, covered]
                )

        # 4. Select the winner.
        winner = _select_winner(
            records=records,
            current_scores=scores,
            current_stockout=stockout_only,
            names=names,
            rule=selection_rule,
            budget=stockout_budget,
            incumbent=active_incumbent,
            min_switch_gain=min_switch_gain,
        )

        # 5. Refit the winning pipeline on the full prefix and publish the
        #    next-period forecast (first horizon step). In production this is
        #    the only number that leaves the process; the rest is audit data.
        win_cand = next(c for c in candidates if c.name == winner)
        y_hat_pub, y_hat_pub_in = win_cand.forecast_fn(y_train, horizon)
        pub_reconciler = win_cand.make_reconciler()
        pub_full = _fit_reconciler(
            pub_reconciler,
            S=S,
            y_hat=y_hat_pub,
            y_train=y_train,
            y_hat_insample=y_hat_pub_in,
            tags=tags_idx,
        )

        rec = CutoffRecord(
            cutoff=ds[t],
            forecasts=forecasts,
            actuals=y_holdout,
            scores=scores,
            stockout_cost=stockout_only,
            monetary_cost=monetary,
            support=supports,
            cell_cost=cell_costs,
            winner=winner,
            published=pub_full[:, 0],
        )
        records.append(rec)
        active_incumbent = winner

    return RollingSelectionResult(
        records=records,
        level_weights=level_weights or {lvl: 1.0 for lvl in tags},
        selection_rule=selection_rule,
        candidates=names,
    )


def _select_winner(
    records: list[CutoffRecord],
    current_scores: dict[str, float],
    current_stockout: dict[str, float],
    names: list[str],
    rule: str,
    budget: float | None,
    incumbent: str,
    min_switch_gain: float,
) -> str:
    """Apply the selection rule plus budget and anti-churn guardrails."""
    eligible = [n for n in names if n in current_scores]
    if budget is not None:
        within = [n for n in eligible if current_stockout[n] <= budget]
        # If everything breaches, keep the smallest breach rather than
        # pretending the budget is fine.
        eligible = within if within else eligible

    if rule == "per_cutoff":
        return min(eligible, key=lambda n: current_scores[n])
    if not records:
        # Zero accumulated evidence: keep the incumbent rather than switch
        # on a single noisy cutoff.
        return incumbent if incumbent in eligible else min(
            eligible, key=lambda n: current_scores[n]
        )

    # Cumulative: mean per-cutoff score over cutoffs where the candidate was
    # present (its own support). Note these means can rest on different cutoff
    # sets for late entrants; for a strictly apples-to-apples pairing use
    # common_support_score(), which intersects the supports explicitly.
    cum: dict[str, float] = {}
    for n in eligible:
        history = [r.scores[n] for r in records if n in r.scores]
        cum[n] = float(np.mean(history + [current_scores[n]]))

    best = min(eligible, key=lambda n: cum[n])
    if best == incumbent or incumbent not in cum:
        return best
    gain = (cum[incumbent] - cum[best]) / max(abs(cum[incumbent]), 1e-12)
    return best if gain >= min_switch_gain else incumbent
