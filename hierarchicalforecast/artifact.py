__all__ = ["ReconciliationArtifact", "ArtifactValidationError"]

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import narwhals.stable.v2 as nw
import numpy as np

from .methods import HReconciler
from .utils import SMatrix

if TYPE_CHECKING:
    from narwhals.typing import Frame, FrameT

# Bumped whenever the on-disk layout of the artifact changes.
ARTIFACT_FORMAT_VERSION = 1

# Metadata keys stored alongside the numerical arrays.
_META_KEYS = (
    "format_version",
    "reconciler_name",
    "init_params",
    "id_col",
    "s_fingerprint",
)


class ArtifactValidationError(ValueError):
    """Raised when a saved reconciliation artifact is incompatible with the
    current hierarchy.

    This happens when the series that make up the hierarchy change identity or
    when their summing relations (the ``S`` matrix) change, even if the new
    hierarchy happens to have the same dimensions as the one used at fit time.
    """


def _to_str_array(values) -> np.ndarray:
    """Coerce sequence of identifiers into a 1-D ``<U`` string array."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError("Identifiers must be a one-dimensional sequence.")
    return arr.astype(str, copy=False)


def _fingerprint_S(S: np.ndarray) -> str:
    """Content hash of the summing matrix.

    The hash covers shape, dtype and every entry (including their byte
    representation), so two matrices of identical shape but with different
    summing relations always produce different fingerprints.
    """
    S = np.ascontiguousarray(S, dtype=np.float64)
    hasher = hashlib.sha256()
    hasher.update(str(S.shape).encode("utf-8"))
    hasher.update(S.tobytes())
    return hasher.hexdigest()


def _check_unique_ids(ids: np.ndarray, what: str) -> None:
    unique, counts = np.unique(ids, return_counts=True)
    duplicated = unique[counts > 1]
    if duplicated.size:
        preview = ", ".join(duplicated[:5].tolist())
        raise ValueError(
            f"Duplicate identifiers found in {what}: {preview}. "
            "Each hierarchical series must have a unique identifier."
        )


class ReconciliationArtifact:
    r"""Persisted result of fitting a hierarchical reconciliation method.

    A reconciliation method (e.g. :class:`~hierarchicalforecast.methods.MinTrace`)
    fits a projection matrix :math:`\mathbf{P}` (and, when applicable, a weight
    matrix :math:`\mathbf{W}`) whose rows and columns are positional at fit
    time. This artifact stores those matrices together with the row/column
    identifiers of the training hierarchy and a fingerprint of the summing
    matrix :math:`\mathbf{S}`, allowing the same fitted weights to be saved
    once and reused across several forecasting runs (for example inside a
    scheduled job).

    At inference time the artifact:

    1. Rejects weights whose stored hierarchy differs from the current one,
       either because the member series changed or because their summing
       relations changed. Same-shaped but different hierarchies are *not*
       accepted.
    2. Reorders the incoming forecasts by identifier before applying the
       stored matrices, so callers may pass series in any order.
    3. Returns the reconciled forecasts in the same order as the input.

    Only reconcilers with a static, dense projection matrix are supported:
    :class:`~hierarchicalforecast.methods.BottomUp`,
    :class:`~hierarchicalforecast.methods.TopDown` (``average_proportions`` and
    ``proportion_averages``), dense :class:`~hierarchicalforecast.methods.MinTrace`
    (including ``nonnegative=True``) and
    :class:`~hierarchicalforecast.methods.ERM`. Sparse methods
    (:class:`~hierarchicalforecast.methods.MinTraceSparse`,
    :class:`~hierarchicalforecast.methods.TopDownSparse`,
    :class:`~hierarchicalforecast.methods.MiddleOut`) do not expose a static
    matrix and cannot be persisted.

    Args:
        reconciler_name (str): Class name of the fitted reconciler.
        init_params (dict): Initialization parameters of the reconciler.
        S (np.ndarray): Dense summing matrix of size
            (``n_series``, ``n_bottom``) used at fit time.
        P (np.ndarray): Fitted projection matrix of size
            (``n_bottom``, ``n_series``).
        W (np.ndarray | None): Fitted weight matrix of size
            (``n_series``, ``n_series``), if available.
        series_ids (np.ndarray): Identifiers of every row of :math:`\mathbf{S}`,
            in fit-time order.
        bottom_ids (np.ndarray): Identifiers of every column of
            :math:`\mathbf{S}` (the bottom-level series), in fit-time order.
        id_col (str): Name of the identifier column.

    References:
        - [Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019). "Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization". Journal of the American Statistical Association, 114 , 804-819.](https://doi.org/10.1080/01621459.2018.1448825).
    """

    def __init__(
        self,
        reconciler_name: str,
        init_params: dict,
        S: np.ndarray,
        P: np.ndarray,
        W: np.ndarray | None,
        series_ids: np.ndarray,
        bottom_ids: np.ndarray,
        id_col: str = "unique_id",
    ):
        S = np.asarray(S, dtype=np.float64)
        P = np.asarray(P, dtype=np.float64)
        series_ids = _to_str_array(series_ids)
        bottom_ids = _to_str_array(bottom_ids)

        if S.ndim != 2:
            raise ValueError("`S` must be a two-dimensional matrix.")
        if P.ndim != 2:
            raise ValueError("`P` must be a two-dimensional matrix.")
        n_series, n_bottom = S.shape
        if P.shape != (n_bottom, n_series):
            raise ValueError(
                f"`P` has shape {P.shape}, expected {(n_bottom, n_series)} "
                "matching the dimensions of `S`."
            )
        if series_ids.shape != (n_series,):
            raise ValueError(
                f"`series_ids` has length {series_ids.shape[0]}, expected {n_series}."
            )
        if bottom_ids.shape != (n_bottom,):
            raise ValueError(
                f"`bottom_ids` has length {bottom_ids.shape[0]}, expected {n_bottom}."
            )
        _check_unique_ids(series_ids, "`series_ids`")
        _check_unique_ids(bottom_ids, "`bottom_ids`")
        missing_bottom = np.setdiff1d(bottom_ids, series_ids)
        if missing_bottom.size:
            raise ValueError(
                "Every bottom identifier must be part of `series_ids`. "
                f"Missing: {missing_bottom[:5].tolist()}"
            )
        if W is not None:
            W = np.asarray(W, dtype=np.float64)
            if W.shape != (n_series, n_series):
                raise ValueError(
                    f"`W` has shape {W.shape}, expected {(n_series, n_series)}."
                )

        self.reconciler_name = reconciler_name
        self.init_params = dict(init_params)
        self.S = S
        self.P = P
        self.W = W
        self.series_ids = series_ids
        self.bottom_ids = bottom_ids
        self.id_col = id_col
        self.s_fingerprint = _fingerprint_S(S)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_reconciler(
        cls,
        reconciler: HReconciler,
        S: "np.ndarray | SMatrix",
        series_ids: np.ndarray,
        bottom_ids: np.ndarray | None = None,
        id_col: str = "unique_id",
    ) -> "ReconciliationArtifact":
        """Build an artifact from a fitted reconciler.

        Args:
            reconciler (HReconciler): A fitted reconciler exposing dense
                ``P`` (and optionally ``W``) matrices.
            S (np.ndarray | SMatrix): Dense summing matrix or
                :class:`~hierarchicalforecast.utils.SMatrix` used at fit time.
            series_ids (np.ndarray): Identifiers of every row of :math:`\mathbf{S}`,
                in the same order used at fit time.
            bottom_ids (np.ndarray, optional): Identifiers of every column of
                :math:`\mathbf{S}`. Defaults to the last ``n_bottom`` entries
                of ``series_ids``, matching the library's convention that the
                bottom block of :math:`\mathbf{S}` is an identity matrix.
            id_col (str): Name of the identifier column. Default is
                ``"unique_id"``.

        Returns:
            ReconciliationArtifact: A persistable fitted result.
        """
        if not isinstance(reconciler, HReconciler):
            raise TypeError(
                "`reconciler` must be an instance of a hierarchicalforecast "
                f"HReconciler subclass, got {type(reconciler).__name__}."
            )
        if not getattr(reconciler, "fitted", False):
            raise ValueError(
                f"Reconciler `{type(reconciler).__name__}` is not fitted. "
                "Call `fit` or `fit_predict` before creating an artifact."
            )

        P = getattr(reconciler, "P", None)
        if not isinstance(P, np.ndarray):
            raise TypeError(
                f"Reconciler `{type(reconciler).__name__}` does not expose a static "
                "dense projection matrix (sparse/operator-based and middle-out "
                "methods cannot be persisted). Use a dense reconciler such as "
                "BottomUp, TopDown, MinTrace or ERM."
            )
        W = getattr(reconciler, "W", None)
        if W is not None and not isinstance(W, np.ndarray):
            raise TypeError(
                f"Reconciler `{type(reconciler).__name__}` exposes a non-dense "
                "weight matrix and cannot be persisted."
            )

        if isinstance(S, SMatrix):
            s_matrix = S
            dense_S = s_matrix.to_dense()
            series_ids = _to_str_array(series_ids)
            if not np.array_equal(series_ids, _to_str_array(s_matrix.row_labels)):
                raise ValueError(
                    "`series_ids` must match the row labels of the `SMatrix` "
                    "and be provided in the same order."
                )
            if bottom_ids is None:
                bottom_ids = _to_str_array(s_matrix.col_labels)
            else:
                bottom_ids = _to_str_array(bottom_ids)
                if not np.array_equal(bottom_ids, _to_str_array(s_matrix.col_labels)):
                    raise ValueError(
                        "`bottom_ids` must match the column labels of the `SMatrix`."
                    )
            id_col = s_matrix.id_col
        else:
            dense_S = np.asarray(S, dtype=np.float64)
            series_ids = _to_str_array(series_ids)
            n_bottom = dense_S.shape[1]
            if bottom_ids is None:
                bottom_ids = series_ids[-n_bottom:].copy()
            else:
                bottom_ids = _to_str_array(bottom_ids)

        return cls(
            reconciler_name=type(reconciler).__name__,
            init_params=getattr(reconciler, "_init_params", None) or {},
            S=dense_S,
            P=P,
            W=W,
            series_ids=series_ids,
            bottom_ids=bottom_ids,
            id_col=id_col,
        )

    # ------------------------------------------------------------------
    # Validation against a new hierarchy
    # ------------------------------------------------------------------

    def _validate(
        self,
        current_ids: np.ndarray,
        current_S: np.ndarray | None = None,
        id_col: str | None = None,
    ) -> None:
        if id_col is not None and id_col != self.id_col:
            raise ArtifactValidationError(
                f"Identifier column mismatch: artifact was fitted with "
                f"`{self.id_col}` but received `{id_col}`."
            )

        if current_ids.shape != self.series_ids.shape:
            raise ArtifactValidationError(
                f"Hierarchy membership changed: artifact was fitted on "
                f"{self.series_ids.shape[0]} series but received "
                f"{current_ids.shape[0]}. Refusing to reuse old weights."
            )
        # Membership is order-independent: callers may pass series in any
        # order, and the reconcile methods reposition rows by identifier.
        if not np.array_equal(np.sort(current_ids), np.sort(self.series_ids)):
            missing = np.setdiff1d(self.series_ids, current_ids)
            added = np.setdiff1d(current_ids, self.series_ids)
            details = []
            if missing.size:
                details.append(f"missing {missing[:5].tolist()}")
            if added.size:
                details.append(f"unexpected {added[:5].tolist()}")
            raise ArtifactValidationError(
                "Hierarchy membership changed ("
                + "; ".join(details)
                + "). Refusing to reuse weights fitted on a different set of series."
            )

        if current_S is not None:
            current_S = np.asarray(current_S, dtype=np.float64)
            if current_S.shape != self.S.shape:
                raise ArtifactValidationError(
                    f"Summing relations changed: stored S has shape {self.S.shape} "
                    f"but received S has shape {current_S.shape}. Refusing to "
                    "reuse old weights on a different hierarchy."
                )
            if _fingerprint_S(current_S) != self.s_fingerprint:
                raise ArtifactValidationError(
                    "Summing relations changed: the summing matrix differs from "
                    "the one used at fit time even though the series identifiers "
                    "and dimensions match. Refusing to reuse old weights."
                )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def reconcile_array(
        self,
        y_hat: np.ndarray,
        ids: np.ndarray,
        S: np.ndarray | None = None,
        id_col: str | None = None,
    ) -> np.ndarray:
        """Reconcile forecasts given as a NumPy array.

        Args:
            y_hat (np.ndarray): Base forecasts of size
                (``n_series``, ``horizon``) or (``n_series``,). The rows follow
                the order given by ``ids``, which may differ from the fit-time
                order.
            ids (np.ndarray): Identifier of every row of ``y_hat``.
            S (np.ndarray, optional): Current summing matrix. When provided,
                its content is compared against the stored matrix; pass it to
                detect changed summing relations.
            id_col (str, optional): Name of the identifier column, checked
                against the stored name when provided.

        Returns:
            np.ndarray: Reconciled forecasts with the same shape and row order
            as ``y_hat``.
        """
        ids = _to_str_array(ids)
        y_hat = np.asarray(y_hat, dtype=np.float64)
        squeeze = False
        if y_hat.ndim == 1:
            y_hat = y_hat[:, None]
            squeeze = True
        elif y_hat.ndim != 2:
            raise ValueError(
                "`y_hat` must be a 1-D or 2-D array with one row per series."
            )

        self._validate(current_ids=ids, current_S=S, id_col=id_col)

        # `_validate` guarantees identical membership. Map each caller row to
        # its fit-time position, apply the stored matrices in fit-time order,
        # then write the rows back to the caller's order.
        fit_position = {name: i for i, name in enumerate(self.series_ids.tolist())}
        caller_to_fit = np.array([fit_position[name] for name in ids.tolist()])

        # Scatter caller rows into their fit-time positions.
        y_hat_fit_order = np.empty_like(y_hat)
        y_hat_fit_order[caller_to_fit, :] = y_hat
        reconciled_fit_order = self.S @ (self.P @ y_hat_fit_order)

        # Gather back: the row at fit position i belongs to the caller row
        # whose caller_to_fit entry equals i.
        reconciled = reconciled_fit_order[caller_to_fit, :]

        if squeeze:
            reconciled = reconciled[:, 0]
        return reconciled

    def reconcile(
        self,
        Y_hat_df: "Frame",
        value_col: str,
        S_df: "Frame | SMatrix | None" = None,
        id_col: str | None = None,
        time_col: str = "ds",
    ) -> "FrameT":
        """Reconcile forecasts stored in a DataFrame.

        The frame may contain the series in any order; output rows preserve the
        input row order exactly.

        Args:
            Y_hat_df (Frame): DataFrame containing at least an identifier
                column, a time column and ``value_col``.
            value_col (str): Column holding the base forecasts to reconcile.
            S_df (Frame | SMatrix, optional): Current summing matrix (DataFrame
                with an identifier column plus one column per bottom series, or
                an :class:`~hierarchicalforecast.utils.SMatrix`). When provided,
                its content is verified against the stored summing matrix.
            id_col (str, optional): Identifier column name. Defaults to the
                name stored in the artifact.
            time_col (str): Time column name. Defaults to ``"ds"``.

        Returns:
            FrameT: Copy of ``Y_hat_df`` with ``value_col`` replaced by the
            reconciled forecasts, keeping the original row order and backend
            (pandas or polars).
        """
        id_col = self.id_col if id_col is None else id_col
        Y_hat_nw = nw.from_native(Y_hat_df)
        backend = Y_hat_nw.implementation

        if id_col not in Y_hat_nw.columns:
            raise ValueError(
                f"Column `{id_col}` not found in `Y_hat_df` "
                f"(available columns: {Y_hat_nw.columns})."
            )
        if value_col not in Y_hat_nw.columns:
            raise ValueError(
                f"Column `{value_col}` not found in `Y_hat_df` "
                f"(available columns: {Y_hat_nw.columns})."
            )

        # Extract one row per series, preserving caller order.
        ids_frame = Y_hat_nw.select(nw.col(id_col)).unique(
            subset=id_col, keep="first", maintain_order=True
        )
        caller_ids = ids_frame.to_numpy().reshape(-1).astype(str)

        # If the caller supplies the current summing matrix, verify both the
        # membership (using its row order) and the summing relations first.
        if S_df is not None:
            current_S, s_ids = _extract_s_matrix(S_df, id_col)
            self._validate(current_ids=s_ids, current_S=current_S, id_col=id_col)

        # Membership/identifier validation happens before any reshaping. It
        # is order-independent: rows are repositioned by identifier below.
        self._validate(current_ids=caller_ids, current_S=None, id_col=id_col)

        fit_position = {name: i for i, name in enumerate(self.series_ids.tolist())}

        # Pivot to (n_series, horizon). The library requires balanced panels
        # (every identifier has the same timestamps), and sorting by identifier
        # gives a deterministic wide layout independent of the caller's order.
        wide = (
            Y_hat_nw.select(nw.col(id_col), nw.col(time_col), nw.col(value_col))
            .pivot(on=time_col, index=id_col, values=value_col, sort_columns=True)
            .sort(by=id_col)
        )
        wide_ids = wide[id_col].to_numpy().astype(str)
        value_matrix = wide.select(
            nw.col(c) for c in wide.columns if c != id_col
        ).to_numpy()

        # Reorder to fit-time positions, reconcile, restore the pivoted order.
        wide_fit_pos = np.array(
            [fit_position[name] for name in wide_ids.tolist()]
        )
        y_hat_fit_order = np.empty_like(value_matrix)
        y_hat_fit_order[wide_fit_pos, :] = value_matrix
        reconciled_fit_order = self.S @ (self.P @ y_hat_fit_order)

        # Gather fit-positioned rows back into the sorted-wide row order.
        reconciled_wide = reconciled_fit_order[wide_fit_pos, :]

        # Broadcast the wide result back to the original long frame. Use the
        # series position and the ordinal rank of each timestamp within its
        # identifier group, so arbitrary input row order (including timestamps
        # that are not contiguous within a group) is handled correctly. The
        # ordinal rank matches the sorted time columns produced by `pivot`.
        original_ids = Y_hat_nw[id_col].to_numpy().astype(str)
        horizon = value_matrix.shape[1]
        if len(original_ids) != len(wide_ids) * horizon:
            raise ValueError(
                "`Y_hat_df` must contain the same number of observations for "
                f"every `{id_col}` (a balanced panel): got {len(original_ids)} "
                f"rows for {len(wide_ids)} series and a horizon of {horizon}."
            )
        wide_position = {name: i for i, name in enumerate(wide_ids.tolist())}
        row_series_pos = np.array(
            [wide_position[name] for name in original_ids.tolist()]
        )
        row_time_pos = (
            Y_hat_nw.select(
                nw.col(time_col)
                .rank(method="ordinal")
                .over(id_col)
                .alias("_t_rank")
            )["_t_rank"].to_numpy()
            - 1
        ).astype(np.int64)
        reconciled_values = reconciled_wide[row_series_pos, row_time_pos]

        result = Y_hat_nw.with_columns(
            nw.new_series(value_col, reconciled_values, backend=backend)
        )
        return result.to_native()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the artifact to a single ``.npz`` file.

        Args:
            path (str | Path): Destination path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            S=self.S,
            P=self.P,
            W=(self.W if self.W is not None else np.array([])),
            series_ids=self.series_ids,
            bottom_ids=self.bottom_ids,
            format_version=np.int64(ARTIFACT_FORMAT_VERSION),
            reconciler_name=self.reconciler_name,
            init_params=json.dumps(self.init_params),
            id_col=self.id_col,
            s_fingerprint=self.s_fingerprint,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ReconciliationArtifact":
        """Load an artifact from a ``.npz`` file created by :meth:`save`.

        Args:
            path (str | Path): Path to the saved artifact.

        Returns:
            ReconciliationArtifact: The restored fitted result.

        Raises:
            ArtifactValidationError: If the file format version is unsupported
                or its stored fingerprint does not match its summing matrix.
        """
        path = Path(path)
        # Match numpy's `savez` behavior, which appends `.npz` when the
        # provided path does not already end with that suffix.
        if not path.exists() and not path.name.endswith(".npz"):
            suffixed = path.with_name(path.name + ".npz")
            if suffixed.exists():
                path = suffixed
        if not path.exists():
            raise FileNotFoundError(f"No reconciliation artifact found at `{path}`.")
        with np.load(path, allow_pickle=False) as blob:
            missing = [
                key
                for key in ("S", "P", "W", "series_ids", "bottom_ids", *_META_KEYS)
                if key not in blob.files
            ]
            if missing:
                raise ArtifactValidationError(
                    f"Artifact at `{path}` is missing fields: {missing}."
                )
            format_version = int(blob["format_version"])
            if format_version != ARTIFACT_FORMAT_VERSION:
                raise ArtifactValidationError(
                    f"Unsupported artifact format version {format_version}; "
                    f"this library supports version {ARTIFACT_FORMAT_VERSION}. "
                    "Refit and save the reconciler with the current version."
                )
            S = blob["S"].astype(np.float64, copy=False)
            stored_fingerprint = str(blob["s_fingerprint"])
            if _fingerprint_S(S) != stored_fingerprint:
                raise ArtifactValidationError(
                    "The stored summing matrix does not match its fingerprint. "
                    "The artifact may be corrupted or tampered with."
                )

            W_blob = blob["W"]
            W = W_blob.astype(np.float64, copy=False) if W_blob.size else None
            try:
                init_params = json.loads(str(blob["init_params"]))
            except json.JSONDecodeError as exc:
                raise ArtifactValidationError(
                    "Stored init parameters are not valid JSON."
                ) from exc

            artifact = cls(
                reconciler_name=str(blob["reconciler_name"]),
                init_params=init_params,
                S=S,
                P=blob["P"].astype(np.float64, copy=False),
                W=W,
                series_ids=blob["series_ids"].astype(str, copy=False),
                bottom_ids=blob["bottom_ids"].astype(str, copy=False),
                id_col=str(blob["id_col"]),
            )
        return artifact

    def save_json(self, path: str | Path) -> None:
        """Save the artifact metadata next to a ``.npz`` file.

        Writes :meth:`save` output to ``path`` (replacing any suffix with
        ``.npz``) and a human-readable sidecar ``.json`` with the hierarchy
        identifiers, reconciler settings and summing-matrix fingerprint.
        """
        path = Path(path)
        npz_path = path.with_suffix(".npz")
        self.save(npz_path)
        metadata = {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "reconciler_name": self.reconciler_name,
            "init_params": self.init_params,
            "id_col": self.id_col,
            "s_fingerprint": self.s_fingerprint,
            "n_series": int(self.S.shape[0]),
            "n_bottom": int(self.S.shape[1]),
            "series_ids": self.series_ids.tolist(),
            "bottom_ids": self.bottom_ids.tolist(),
        }
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> "ReconciliationArtifact":
        """Load an artifact from the ``.json`` sidecar written by :meth:`save_json`.

        The accompanying ``.npz`` file must exist next to it.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        npz_path = path.with_suffix(".npz")
        artifact = cls.load(npz_path)
        expected = {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "reconciler_name": artifact.reconciler_name,
            "init_params": artifact.init_params,
            "id_col": artifact.id_col,
            "s_fingerprint": artifact.s_fingerprint,
        }
        for key, expected_value in expected.items():
            if metadata.get(key) != expected_value:
                raise ArtifactValidationError(
                    f"Metadata sidecar `{path}` is inconsistent with `{npz_path}` "
                    f"(field `{key}` differs)."
                )
        if metadata.get("series_ids") != artifact.series_ids.tolist():
            raise ArtifactValidationError(
                f"Metadata sidecar `{path}` is inconsistent with `{npz_path}` "
                "(series_ids differ)."
            )
        if metadata.get("bottom_ids") != artifact.bottom_ids.tolist():
            raise ArtifactValidationError(
                f"Metadata sidecar `{path}` is inconsistent with `{npz_path}` "
                "(bottom_ids differ)."
            )
        return artifact

    def __repr__(self) -> str:
        return (
            f"ReconciliationArtifact(reconciler={self.reconciler_name}, "
            f"n_series={self.S.shape[0]}, n_bottom={self.S.shape[1]}, "
            f"id_col='{self.id_col}')"
        )


def _extract_s_matrix(
    S_df: "Frame | SMatrix", id_col: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return the dense summing matrix and its ordered row identifiers."""
    if isinstance(S_df, SMatrix):
        return S_df.to_dense(), _to_str_array(S_df.row_labels)

    S_nw = nw.from_native(S_df)
    if id_col not in S_nw.columns:
        raise ValueError(
            f"Column `{id_col}` not found in `S_df` "
            f"(available columns: {S_nw.columns})."
        )
    ids = S_nw[id_col].to_numpy().astype(str)
    value_columns = [col for col in S_nw.columns if col != id_col]
    matrix = S_nw.select(nw.col(c) for c in value_columns).to_numpy().astype(
        np.float64, copy=False
    )
    return matrix, ids
