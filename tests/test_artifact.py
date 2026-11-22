import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from hierarchicalforecast.artifact import (
    ARTIFACT_FORMAT_VERSION,
    ArtifactValidationError,
    ReconciliationArtifact,
)
from hierarchicalforecast.methods import (
    ERM,
    BottomUp,
    MinTrace,
    MinTraceSparse,
    TopDown,
)
from hierarchicalforecast.utils import aggregate


# ---------------------------------------------------------------------------
# Offline fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hierarchy():
    """Small 3-level hierarchy: Total / 2 groups / 4 bottom series (S is 7x4)."""
    S = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],  # Total
            [1.0, 1.0, 0.0, 0.0],  # Group 1
            [0.0, 0.0, 1.0, 1.0],  # Group 2
            [1.0, 0.0, 0.0, 0.0],  # Bottom 1
            [0.0, 1.0, 0.0, 0.0],  # Bottom 2
            [0.0, 0.0, 1.0, 0.0],  # Bottom 3
            [0.0, 0.0, 0.0, 1.0],  # Bottom 4
        ]
    )
    series_ids = np.array(["Total", "G1", "G2", "b1", "b2", "b3", "b4"])
    bottom_ids = np.array(["b1", "b2", "b3", "b4"])
    return S, series_ids, bottom_ids


@pytest.fixture
def fit_data(hierarchy):
    S, _, _ = hierarchy
    rng = np.random.default_rng(0)
    n_obs, horizon = 40, 3
    y_bottom = rng.normal(size=(4, n_obs))
    y_insample = S @ y_bottom + rng.normal(scale=0.1, size=(7, n_obs))
    y_hat_insample = y_insample + rng.normal(scale=0.5, size=(7, n_obs))
    y_hat = S @ rng.normal(size=(4, horizon)) + rng.normal(
        scale=0.3, size=(7, horizon)
    )
    return {
        "S": S,
        "y_hat": y_hat,
        "y_insample": y_insample,
        "y_hat_insample": y_hat_insample,
        "horizon": horizon,
    }


@pytest.fixture
def fitted_mint(fit_data):
    rec = MinTrace(method="mint_shrink")
    rec.fit(
        S=fit_data["S"],
        y_hat=fit_data["y_hat"],
        y_insample=fit_data["y_insample"],
        y_hat_insample=fit_data["y_hat_insample"],
    )
    return rec


@pytest.fixture
def artifact(fitted_mint, hierarchy):
    S, series_ids, _ = hierarchy
    return ReconciliationArtifact.from_reconciler(
        fitted_mint, S=S, series_ids=series_ids
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_artifact_stores_identifiers_and_weights(fitted_mint, hierarchy):
    S, series_ids, bottom_ids = hierarchy
    artifact = ReconciliationArtifact.from_reconciler(
        fitted_mint, S=S, series_ids=series_ids
    )
    assert artifact.reconciler_name == "MinTrace"
    assert artifact.init_params == {"method": "mint_shrink", "nonnegative": False,
                                    "mint_shr_ridge": 2e-8}
    assert np.array_equal(artifact.series_ids, series_ids)
    assert np.array_equal(artifact.bottom_ids, bottom_ids)
    assert artifact.P.shape == (4, 7)
    assert artifact.W.shape == (7, 7)
    np.testing.assert_array_equal(artifact.S, S)


def test_artifact_rejects_unfitted_reconciler(fit_data, hierarchy):
    S, series_ids, _ = hierarchy
    with pytest.raises(ValueError, match="not fitted"):
        ReconciliationArtifact.from_reconciler(
            MinTrace(method="ols"), S=S, series_ids=series_ids
        )


def test_artifact_rejects_sparse_reconciler(fit_data, hierarchy):
    S, series_ids, _ = hierarchy
    sparse_rec = MinTraceSparse(method="ols").fit(
        S=sparse.csr_matrix(S), y_hat=fit_data["y_hat"]
    )
    with pytest.raises(TypeError, match="static"):
        ReconciliationArtifact.from_reconciler(
            sparse_rec, S=S, series_ids=series_ids
        )


def test_artifact_validates_matrix_shapes(hierarchy):
    S, series_ids, bottom_ids = hierarchy
    with pytest.raises(ValueError, match="`P` has shape"):
        ReconciliationArtifact(
            reconciler_name="MinTrace",
            init_params={},
            S=S,
            P=np.zeros((7, 4)),  # transposed
            W=None,
            series_ids=series_ids,
            bottom_ids=bottom_ids,
        )


def test_artifact_rejects_duplicate_identifiers(hierarchy):
    S, _, bottom_ids = hierarchy
    duplicated = np.array(["Total", "G1", "G2", "b1", "b2", "b3", "b3"])
    with pytest.raises(ValueError, match="Duplicate identifiers"):
        ReconciliationArtifact(
            reconciler_name="BottomUp",
            init_params={},
            S=S,
            P=np.zeros((4, 7)),
            W=None,
            series_ids=duplicated,
            bottom_ids=bottom_ids,
        )


@pytest.mark.parametrize(
    "reconciler",
    [
        BottomUp(),
        MinTrace(method="ols"),
        MinTrace(method="wls_struct"),
        MinTrace(method="wls_var"),
        MinTrace(method="mint_cov"),
        MinTrace(method="mint_shrink"),
        TopDown(method="average_proportions"),
        TopDown(method="proportion_averages"),
        ERM(method="closed"),
    ],
)
def test_artifact_supports_all_dense_reconcilers(reconciler, fit_data, hierarchy):
    S, series_ids, _ = hierarchy
    reconciler.fit(
        S=S,
        y_hat=fit_data["y_hat"],
        y_insample=fit_data["y_insample"],
        y_hat_insample=fit_data["y_hat_insample"],
    )
    expected = reconciler.predict(S=S, y_hat=fit_data["y_hat"])["mean"]
    artifact = ReconciliationArtifact.from_reconciler(
        reconciler, S=S, series_ids=series_ids
    )
    np.testing.assert_allclose(
        artifact.reconcile_array(fit_data["y_hat"], series_ids), expected
    )


def test_artifact_nonnegative_mint(fit_data, hierarchy):
    S, series_ids, _ = hierarchy
    rec = MinTrace(method="ols", nonnegative=True)
    rec.fit(
        S=S,
        y_hat=np.abs(fit_data["y_hat"]) + 1.0,
        y_insample=fit_data["y_insample"],
        y_hat_insample=fit_data["y_hat_insample"],
    )
    artifact = ReconciliationArtifact.from_reconciler(
        rec, S=S, series_ids=series_ids
    )
    y_hat = np.abs(fit_data["y_hat"]) + 1.0
    np.testing.assert_allclose(
        artifact.reconcile_array(y_hat, series_ids),
        rec.predict(S=S, y_hat=y_hat)["mean"],
    )


# ---------------------------------------------------------------------------
# Inference: reordering and caller-order restoration
# ---------------------------------------------------------------------------


def test_reconcile_array_shuffled_order(artifact, fit_data, hierarchy):
    _, series_ids, _ = hierarchy
    y_hat = fit_data["y_hat"]
    expected = artifact.S @ (artifact.P @ y_hat)

    perm = np.array([3, 0, 5, 1, 6, 2, 4])
    shuffled_ids = series_ids[perm]
    result = artifact.reconcile_array(y_hat[perm, :], shuffled_ids)

    np.testing.assert_allclose(result, expected[perm, :])


def test_reconcile_array_1d(artifact, fit_data, hierarchy):
    _, series_ids, _ = hierarchy
    y_hat = fit_data["y_hat"]
    perm = np.array([6, 2, 0, 4, 1, 5, 3])
    result = artifact.reconcile_array(y_hat[perm, 0], series_ids[perm])
    expected = artifact.S @ (artifact.P @ y_hat)
    assert result.ndim == 1
    np.testing.assert_allclose(result, expected[perm, 0])


def test_reconcile_array_repeated_runs_with_reload(
    artifact, fit_data, hierarchy, tmp_path
):
    """One saved artifact serves several forecasting runs with varying order."""
    S, series_ids, _ = hierarchy
    path = tmp_path / "weights.npz"
    artifact.save(path)
    served = ReconciliationArtifact.load(path)
    rng = np.random.default_rng(123)

    for _ in range(4):
        new_yhat = S @ rng.normal(size=(4, 3)) + rng.normal(scale=0.2, size=(7, 3))
        perm = rng.permutation(7)
        result = served.reconcile_array(new_yhat[perm, :], series_ids[perm], S=S)
        expected = served.S @ (served.P @ new_yhat)
        np.testing.assert_allclose(result, expected[perm, :], atol=1e-12)


def test_reconcile_dataframe_restores_caller_order(artifact, fit_data, hierarchy):
    _, series_ids, _ = hierarchy
    y_hat = fit_data["y_hat"]
    horizon = y_hat.shape[1]
    dates = pd.date_range("2026-01-01", periods=horizon, freq="MS")

    perm = np.array([3, 0, 5, 1, 6, 2, 4])
    rows = []
    for caller_row, fit_row in enumerate(perm):
        sid = series_ids[fit_row]
        for j, ds in enumerate(dates):
            rows.append(
                {"unique_id": sid, "ds": ds, "y_hat": y_hat[fit_row, j]}
            )
    Y_hat_df = pd.DataFrame(rows).sample(frac=1.0, random_state=5).reset_index(
        drop=True
    )
    original = Y_hat_df.copy()

    result = artifact.reconcile(Y_hat_df, value_col="y_hat")

    # Row order (both identifier and timestamp columns) is preserved exactly.
    pd.testing.assert_series_equal(
        result["unique_id"], original["unique_id"], check_names=False
    )
    pd.testing.assert_series_equal(
        result["ds"], original["ds"], check_names=False
    )
    expected = artifact.S @ (artifact.P @ y_hat)
    expected_long = {
        (series_ids[i], dates[j]): expected[i, j]
        for i in range(7)
        for j in range(horizon)
    }
    for _, row in result.iterrows():
        np.testing.assert_allclose(
            row["y_hat"], expected_long[(row["unique_id"], row["ds"])]
        )


def test_reconcile_dataframe_integer_timestamps(artifact, fit_data, hierarchy):
    _, series_ids, _ = hierarchy
    y_hat = fit_data["y_hat"]
    horizon = y_hat.shape[1]
    rows = []
    perm = np.array([6, 1, 4, 0, 3, 5, 2])
    for caller_row, fit_row in enumerate(perm):
        for j in range(horizon):
            rows.append(
                {
                    "unique_id": series_ids[fit_row],
                    "ds": j + 1,
                    "y_hat": y_hat[fit_row, j],
                }
            )
    Y_hat_df = pd.DataFrame(rows).sample(frac=1.0, random_state=9).reset_index(
        drop=True
    )
    result = artifact.reconcile(Y_hat_df, value_col="y_hat")
    expected = artifact.S @ (artifact.P @ y_hat)
    for _, row in result.iterrows():
        i = list(series_ids).index(row["unique_id"])
        np.testing.assert_allclose(row["y_hat"], expected[i, row["ds"] - 1])


def test_reconcile_dataframe_verifies_s_df(artifact, fit_data, hierarchy):
    S, series_ids, _ = hierarchy
    y_hat = fit_data["y_hat"]
    horizon = y_hat.shape[1]
    dates = pd.date_range("2026-01-01", periods=horizon, freq="MS")
    Y_hat_df = pd.DataFrame(
        [
            {"unique_id": sid, "ds": dates[j], "y_hat": y_hat[i, j]}
            for i, sid in enumerate(series_ids)
            for j in range(horizon)
        ]
    )
    S_df = pd.DataFrame(
        {"unique_id": series_ids, **{f"b{k + 1}": S[:, k] for k in range(4)}}
    )
    # Identical S is accepted.
    result = artifact.reconcile(Y_hat_df, value_col="y_hat", S_df=S_df)
    assert len(result) == len(Y_hat_df)

    # Same shape, same identifiers, different summing relations is rejected.
    S_evil = S.copy()
    S_evil[0, :] = [2.0, 0.0, 1.0, 1.0]
    S_evil_df = S_df.copy()
    for k in range(4):
        S_evil_df[f"b{k + 1}"] = S_evil[:, k]
    with pytest.raises(ArtifactValidationError, match="Summing relations changed"):
        artifact.reconcile(Y_hat_df, value_col="y_hat", S_df=S_evil_df)


# ---------------------------------------------------------------------------
# Deployment-change rejection
# ---------------------------------------------------------------------------


def test_reject_membership_change_same_dimensions(artifact, fit_data, hierarchy):
    _, series_ids, _ = hierarchy
    renamed = series_ids.copy()
    renamed[3] = "b1_renamed"
    assert renamed.shape == series_ids.shape
    with pytest.raises(ArtifactValidationError, match="Hierarchy membership changed"):
        artifact.reconcile_array(fit_data["y_hat"], renamed)


def test_reject_different_number_of_series(artifact, fit_data, hierarchy):
    _, series_ids, _ = hierarchy
    with pytest.raises(ArtifactValidationError, match="fitted on 7 series"):
        artifact.reconcile_array(fit_data["y_hat"][:6, :], series_ids[:6])


def test_reject_changed_summing_relations_same_shape(artifact, fit_data, hierarchy):
    S, series_ids, _ = hierarchy
    S_changed = S.copy()
    # Swap the two group rows' contents: shape and ids are unchanged.
    S_changed[1, :], S_changed[2, :] = S[2, :].copy(), S[1, :].copy()
    assert S_changed.shape == S.shape
    with pytest.raises(ArtifactValidationError, match="Summing relations changed"):
        artifact.reconcile_array(fit_data["y_hat"], series_ids, S=S_changed)


def test_reject_s_shape_change(artifact, fit_data, hierarchy):
    _, series_ids, _ = hierarchy
    S_wrong_shape = np.ones((7, 3))
    with pytest.raises(ArtifactValidationError, match="Summing relations changed"):
        artifact.reconcile_array(
            fit_data["y_hat"], series_ids, S=S_wrong_shape
        )


def test_reject_id_column_name_mismatch(artifact, fit_data, hierarchy):
    _, series_ids, _ = hierarchy
    with pytest.raises(ArtifactValidationError, match="Identifier column mismatch"):
        artifact.reconcile_array(
            fit_data["y_hat"], series_ids, id_col="different_id_col"
        )


def test_custom_id_column_roundtrip(fitted_mint, fit_data, tmp_path):
    S = fit_data["S"]
    ids = np.array(["s0", "s1", "s2", "s3", "s4", "s5", "s6"])
    artifact = ReconciliationArtifact.from_reconciler(
        fitted_mint, S=S, series_ids=ids, id_col="series"
    )
    assert artifact.id_col == "series"
    path = tmp_path / "custom.npz"
    artifact.save(path)
    loaded = ReconciliationArtifact.load(path)
    assert loaded.id_col == "series"
    np.testing.assert_allclose(
        loaded.reconcile_array(fit_data["y_hat"], ids, id_col="series"),
        artifact.reconcile_array(fit_data["y_hat"], ids),
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(artifact, fit_data, hierarchy, tmp_path):
    _, series_ids, _ = hierarchy
    path = tmp_path / "artifact.npz"
    artifact.save(path)
    assert path.exists()

    loaded = ReconciliationArtifact.load(path)
    np.testing.assert_array_equal(loaded.S, artifact.S)
    np.testing.assert_array_equal(loaded.P, artifact.P)
    np.testing.assert_array_equal(loaded.W, artifact.W)
    np.testing.assert_array_equal(loaded.series_ids, artifact.series_ids)
    np.testing.assert_array_equal(loaded.bottom_ids, artifact.bottom_ids)
    assert loaded.s_fingerprint == artifact.s_fingerprint

    perm = np.array([2, 6, 1, 4, 0, 5, 3])
    np.testing.assert_allclose(
        loaded.reconcile_array(fit_data["y_hat"][perm, :], series_ids[perm]),
        artifact.reconcile_array(fit_data["y_hat"][perm, :], series_ids[perm]),
    )


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ReconciliationArtifact.load(tmp_path / "does_not_exist.npz")


def test_save_appends_npz_suffix_and_load_resolves_it(artifact, tmp_path):
    # numpy's savez appends `.npz` when the path lacks that suffix; load
    # must resolve the same path transparently.
    bare_path = tmp_path / "weights"
    artifact.save(bare_path)
    assert (tmp_path / "weights.npz").exists()
    loaded = ReconciliationArtifact.load(bare_path)
    np.testing.assert_array_equal(loaded.P, artifact.P)
    # An explicit .npz path still works.
    loaded_explicit = ReconciliationArtifact.load(tmp_path / "weights.npz")
    np.testing.assert_array_equal(loaded_explicit.P, artifact.P)


def test_load_tampered_fingerprint_raises(artifact, tmp_path):
    path = tmp_path / "tampered.npz"
    artifact.save(path)
    blob = dict(np.load(path))
    blob["s_fingerprint"] = np.array("0" * 64)
    tampered_path = tmp_path / "tampered_copy.npz"
    np.savez(tampered_path, **blob)
    with pytest.raises(ArtifactValidationError, match="fingerprint"):
        ReconciliationArtifact.load(tampered_path)


def test_load_unsupported_format_version_raises(artifact, tmp_path):
    path = tmp_path / "future.npz"
    artifact.save(path)
    blob = dict(np.load(path))
    blob["format_version"] = np.int64(ARTIFACT_FORMAT_VERSION + 99)
    future_path = tmp_path / "future_copy.npz"
    np.savez(future_path, **blob)
    with pytest.raises(ArtifactValidationError, match="format version"):
        ReconciliationArtifact.load(future_path)


def test_save_json_sidecar_roundtrip(artifact, fit_data, hierarchy, tmp_path):
    _, series_ids, _ = hierarchy
    stem = tmp_path / "job"
    artifact.save_json(stem)
    assert (tmp_path / "job.npz").exists()
    assert (tmp_path / "job.json").exists()

    with open(tmp_path / "job.json") as handle:
        metadata = json.load(handle)
    assert metadata["reconciler_name"] == "MinTrace"
    assert metadata["series_ids"] == series_ids.tolist()
    assert metadata["s_fingerprint"] == artifact.s_fingerprint

    loaded = ReconciliationArtifact.load_json(tmp_path / "job.json")
    np.testing.assert_allclose(
        loaded.reconcile_array(fit_data["y_hat"], series_ids),
        artifact.reconcile_array(fit_data["y_hat"], series_ids),
    )


def test_bottomup_persists_without_w(fit_data, hierarchy, tmp_path):
    S, series_ids, _ = hierarchy
    rec = BottomUp().fit(S=S, y_hat=fit_data["y_hat"])
    artifact = ReconciliationArtifact.from_reconciler(
        rec, S=S, series_ids=series_ids
    )
    assert artifact.W is None
    path = tmp_path / "bu.npz"
    artifact.save(path)
    loaded = ReconciliationArtifact.load(path)
    assert loaded.W is None
    np.testing.assert_allclose(
        loaded.reconcile_array(fit_data["y_hat"], series_ids),
        rec.predict(S=S, y_hat=fit_data["y_hat"])["mean"],
    )


# ---------------------------------------------------------------------------
# Integration with aggregate / SMatrix
# ---------------------------------------------------------------------------


def test_artifact_from_smatrix(fit_data):
    rng = np.random.default_rng(11)
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    structure = {
        ("AU", "NSW", "Sydney"): 100.0,
        ("AU", "NSW", "Newcastle"): 50.0,
        ("AU", "VIC", "Melbourne"): 80.0,
        ("AU", "VIC", "Geelong"): 40.0,
    }
    rows = []
    for (country, state, region), base in structure.items():
        for ds in dates:
            rows.append(
                {
                    "Country": country,
                    "State": state,
                    "Region": region,
                    "ds": ds,
                    "y": base + rng.normal() * 5,
                }
            )
    spec = [["Country"], ["Country", "State"], ["Country", "State", "Region"]]
    _, S_matrix, _ = aggregate(pd.DataFrame(rows), spec, sparse_s=True)
    S = S_matrix.to_dense()

    rec = MinTrace(method="ols").fit(S=S, y_hat=fit_data["y_hat"][: S.shape[0], :])
    artifact = ReconciliationArtifact.from_reconciler(
        rec, S=S_matrix, series_ids=S_matrix.row_labels
    )
    np.testing.assert_array_equal(artifact.bottom_ids, S_matrix.col_labels)
    np.testing.assert_array_equal(artifact.series_ids, S_matrix.row_labels)

    # Reusing the same SMatrix at inference is accepted.
    y_hat = fit_data["y_hat"][: S.shape[0], :]
    result = artifact.reconcile_array(
        y_hat, S_matrix.row_labels, S=S_matrix.to_dense()
    )
    np.testing.assert_allclose(result, rec.predict(S=S, y_hat=y_hat)["mean"])


def test_smatrix_and_dense_fingerprints_agree(fit_data):
    rng = np.random.default_rng(3)
    dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    rows = []
    for region, base in [("r1", 10.0), ("r2", 20.0), ("r3", 30.0), ("r4", 40.0)]:
        for ds in dates:
            rows.append({"Country": "AU", "Region": region, "ds": ds,
                         "y": base + rng.normal()})
    spec = [["Country"], ["Country", "Region"]]
    _, S_df, _ = aggregate(pd.DataFrame(rows), spec)
    _, S_matrix, _ = aggregate(pd.DataFrame(rows), spec, sparse_s=True)

    ids = S_df["unique_id"].to_numpy()
    S_dense = S_df.drop(columns=["unique_id"]).to_numpy().astype(np.float64)
    a1 = ReconciliationArtifact.from_reconciler(
        BottomUp().fit(S=S_dense, y_hat=fit_data["y_hat"][: len(ids), :]),
        S=S_dense,
        series_ids=ids,
    )
    a2 = ReconciliationArtifact.from_reconciler(
        BottomUp().fit(S=S_matrix.to_dense(),
                       y_hat=fit_data["y_hat"][: len(ids), :]),
        S=S_matrix,
        series_ids=S_matrix.row_labels,
    )
    assert a1.s_fingerprint == a2.s_fingerprint
