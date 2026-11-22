"""Tests for gap-aware moving block bootstrap.

The store-level sales history can skip a reporting week. Resampled residual
blocks must therefore come from continuous windows of the *real* timeline:
dates on either side of a missing report can never be stitched into one
block, horizon-length windows must be selectable, an empty candidate set must
report the span required, and repeated seeds must reproduce identical samples.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from hierarchicalforecast.methods import (
    BottomUp,
    BottomUpSparse,
    TopDown,
    TopDownSparse,
)
from hierarchicalforecast.probabilistic_methods import (
    Bootstrap,
    _continuous_block_starts,
)


@pytest.fixture
def gap_data():
    """Two-series residual history with a missing week in the middle.

    Timeline (weekly): periods 0..3, then a one-week gap, then periods 5..8.
    Residuals themselves are never NaN, so the gap is only detectable through
    the timestamps. This mirrors a store whose reports stop for one week and
    resume afterwards.
    """
    rng = np.random.default_rng(42)
    n_series, h = 2, 3
    residuals = rng.normal(size=(n_series, 9))
    # Weekly cadence with one week missing between index 3 and index 4.
    dates = np.array(
        [
            "2024-01-01",
            "2024-01-08",
            "2024-01-15",
            "2024-01-22",
            # 2024-01-29 is missing (the stopped report week)
            "2024-02-05",
            "2024-02-12",
            "2024-02-19",
            "2024-02-26",
            "2024-03-04",
        ],
        dtype="datetime64[ns]",
    )
    y_hat = np.zeros((n_series, h))
    y_insample = residuals.copy()
    y_hat_insample = np.zeros_like(residuals)
    return {
        "S": np.eye(n_series),
        "P": np.eye(n_series),
        "residuals": residuals,
        "dates": dates,
        "y_hat": y_hat,
        "y_insample": y_insample,
        "y_hat_insample": y_hat_insample,
        "h": h,
    }


class TestContinuousBlockStarts:
    def test_continuous_history_no_gap(self):
        residuals = np.arange(20, dtype=float).reshape(2, 10)
        starts = _continuous_block_starts(residuals, block_length=3)
        np.testing.assert_array_equal(starts, np.arange(8))

    def test_gap_splits_candidate_windows(self, gap_data):
        # Without dates the NaN-free history looks fully continuous.
        naive_starts = _continuous_block_starts(
            gap_data["residuals"], block_length=gap_data["h"]
        )
        np.testing.assert_array_equal(naive_starts, np.arange(7))

        # With real dates the week-long gap removes every window crossing it.
        starts = _continuous_block_starts(
            gap_data["residuals"],
            block_length=gap_data["h"],
            y_dates=gap_data["dates"],
        )
        # Segments [0, 4) and [4, 9); windows of length 3: 0,1 and 4,5,6.
        np.testing.assert_array_equal(starts, np.array([0, 1, 4, 5, 6]))
        for start in starts:
            window_dates = gap_data["dates"][start : start + gap_data["h"]]
            steps = np.diff(window_dates.astype("datetime64[ns]").astype(np.int64))
            assert np.all(steps == steps.min())

    def test_block_equal_to_segment_length_is_allowed(self, gap_data):
        # Each segment has 4 and 5 periods; a block of length 4 yields one
        # window per segment, including a window exactly equal to the segment.
        starts = _continuous_block_starts(
            gap_data["residuals"], block_length=4, y_dates=gap_data["dates"]
        )
        np.testing.assert_array_equal(starts, np.array([0, 4, 5]))

    def test_block_equal_to_full_history_single_window(self):
        # Off-by-one guard: when the full history has exactly h periods there
        # must still be one selectable window, not an empty candidate set.
        residuals = np.ones((2, 3))
        dates = np.array(
            ["2024-01-01", "2024-01-08", "2024-01-15"], dtype="datetime64[ns]"
        )
        starts = _continuous_block_starts(residuals, block_length=3, y_dates=dates)
        np.testing.assert_array_equal(starts, np.array([0]))

    def test_nan_column_breaks_window_without_compressing_axis(self):
        # A missing report for one series creates a NaN residual column. The
        # columns on its two sides must not be joined by index compression.
        residuals = np.ones((2, 6))
        residuals[:, 3] = np.nan
        starts = _continuous_block_starts(residuals, block_length=2)
        # Segments [0,3) -> starts 0,1 and [4,6) -> start 4. Index 3 stays a
        # boundary instead of being silently removed.
        np.testing.assert_array_equal(starts, np.array([0, 1, 4]))

    def test_integer_index_dates_are_treated_as_continuous(self):
        residuals = np.ones((2, 5))
        integer_dates = np.arange(5)
        starts = _continuous_block_starts(
            residuals, block_length=2, y_dates=integer_dates
        )
        np.testing.assert_array_equal(starts, np.arange(4))

    def test_misaligned_dates_raise(self, gap_data):
        with pytest.raises(ValueError, match="aligned"):
            _continuous_block_starts(
                gap_data["residuals"],
                block_length=3,
                y_dates=gap_data["dates"][:-1],
            )


class TestBootstrapSamples:
    def test_samples_never_cross_reporting_gap(self, gap_data):
        sampler = Bootstrap(
            S=gap_data["S"],
            P=gap_data["P"],
            y_hat=gap_data["y_hat"],
            y_insample=gap_data["y_insample"],
            y_hat_insample=gap_data["y_hat_insample"],
            num_samples=500,
            seed=7,
            y_dates=gap_data["dates"],
        )
        samples = sampler.get_samples(num_samples=500)
        # Reconstruct which residual block each sample used (forecast is zero).
        residual_matrix = gap_data["residuals"]
        flat_residuals = residual_matrix.T
        used_starts = set()
        for sample_idx in range(samples.shape[2]):
            drawn = samples[:, :, sample_idx]
            for start in range(residual_matrix.shape[1] - gap_data["h"] + 1):
                block = residual_matrix[:, start : start + gap_data["h"]]
                if np.allclose(drawn, block):
                    used_starts.add(start)
                    break
            else:  # pragma: no cover - every sample must match a real block
                pytest.fail("A bootstrap sample matches no real residual block.")
        assert used_starts == {0, 1, 4, 5, 6}
        # No block may use the boundary pair (index 3 -> index 4).
        assert not {2, 3} & used_starts

    def test_seed_reproduces_identical_coherent_samples(self, gap_data):
        def draw(seed):
            sampler = Bootstrap(
                S=gap_data["S"],
                P=gap_data["P"],
                y_hat=gap_data["y_hat"],
                y_insample=gap_data["y_insample"],
                y_hat_insample=gap_data["y_hat_insample"],
                num_samples=100,
                seed=seed,
                y_dates=gap_data["dates"],
            )
            return sampler.get_samples(num_samples=100)

        np.testing.assert_array_equal(draw(123), draw(123))
        assert not np.allclose(draw(123), draw(124))

    def test_horizon_length_history_is_bootstrappable(self):
        h = 3
        residuals = np.arange(6, dtype=float).reshape(2, h)
        sampler = Bootstrap(
            S=np.eye(2),
            P=np.eye(2),
            y_hat=np.zeros((2, h)),
            y_insample=residuals,
            y_hat_insample=np.zeros_like(residuals),
            num_samples=10,
            seed=0,
        )
        samples = sampler.get_samples(num_samples=10)
        assert samples.shape == (2, h, 10)
        # The only available block is the whole history.
        expected = np.broadcast_to(residuals[:, :, None], (2, h, 10))
        np.testing.assert_allclose(samples, expected)

    def test_no_window_reports_required_span(self, gap_data):
        # The gap splits history into segments of length 4 and 5; asking for a
        # block of length 6 leaves no continuous window anywhere.
        sampler = Bootstrap(
            S=gap_data["S"],
            P=gap_data["P"],
            y_hat=gap_data["y_hat"],
            y_insample=gap_data["y_insample"],
            y_hat_insample=gap_data["y_hat_insample"],
            num_samples=10,
            seed=0,
            y_dates=gap_data["dates"],
        )
        sampler.y_hat = np.zeros((gap_data["y_hat"].shape[0], 6))
        with pytest.raises(ValueError, match="length 6") as exc_info:
            sampler.get_samples(num_samples=10)
        message = str(exc_info.value)
        assert "longest continuous segment has 5" in message
        assert "At least 6 consecutive" in message

    def test_nan_only_history_reports_span(self):
        residuals = np.full((2, 4), np.nan)
        sampler = Bootstrap(
            S=np.eye(2),
            P=np.eye(2),
            y_hat=np.zeros((2, 2)),
            y_insample=residuals,
            y_hat_insample=np.zeros_like(residuals),
            num_samples=10,
            seed=0,
        )
        with pytest.raises(ValueError, match="length 2"):
            sampler.get_samples(num_samples=10)

    def test_backward_compatible_without_dates(self, gap_data):
        # Existing callers pass no y_dates; behavior must remain available.
        sampler = Bootstrap(
            S=gap_data["S"],
            P=gap_data["P"],
            y_hat=gap_data["y_hat"],
            y_insample=gap_data["y_insample"],
            y_hat_insample=gap_data["y_hat_insample"],
            num_samples=50,
            seed=0,
        )
        samples = sampler.get_samples(num_samples=50)
        assert samples.shape == (2, gap_data["h"], 50)


def _hierarchical_frames_with_gap():
    """End-to-end frames: bottom series whose weekly history skips one week."""
    ids = ["total", "g1", "g2", "a", "b", "c", "d"]
    S = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    tags = {
        "level1": np.array(["total"]),
        "level2": np.array(["g1", "g2"]),
        "level3": np.array(["a", "b", "c", "d"]),
    }
    history_dates = pd.to_datetime(
        [
            "2024-01-01",
            "2024-01-08",
            "2024-01-15",
            "2024-01-22",
            # gap: 2024-01-29 missing (the stopped report week)
            "2024-02-05",
            "2024-02-12",
            "2024-02-19",
        ]
    )
    horizon_dates = pd.to_datetime(["2024-02-26", "2024-03-04"])
    rng = np.random.default_rng(1)

    bottom_values = rng.normal(size=(4, 7)) + np.arange(1, 5)[:, None]
    base_values = S @ bottom_values

    rows = []
    for ds_idx, ds in enumerate(history_dates):
        for row_idx, unique_id in enumerate(ids):
            value = float(base_values[row_idx, ds_idx])
            rows.append(
                {"unique_id": unique_id, "ds": ds, "y": value, "model": value}
            )
    Y_df = pd.DataFrame(rows)

    forecast_rows = []
    for row_idx, unique_id in enumerate(ids):
        for ds in horizon_dates:
            forecast_rows.append(
                {"unique_id": unique_id, "ds": ds, "model": float(row_idx + 1.0)}
            )
    Y_hat_df = pd.DataFrame(forecast_rows)
    S_df = pd.DataFrame(
        {
            "unique_id": ids,
            "a": S[:, 0],
            "b": S[:, 1],
            "c": S[:, 2],
            "d": S[:, 3],
        }
    )
    return Y_df, Y_hat_df, S_df, tags


class TestEndToEndReconciliation:
    @pytest.mark.parametrize("reconciler", [BottomUp(), BottomUpSparse()])
    def test_reconcile_respects_timeline_gap(self, reconciler):
        from hierarchicalforecast.core import HierarchicalReconciliation

        Y_df, Y_hat_df, S_df, tags = _hierarchical_frames_with_gap()
        h = 2
        # Segments are length 4 ([0,4)) and 3 ([4,7)); horizon h=2 gives
        # starts 0,1,2 and 4,5. A block crossing the gap (starts 2,3) is banned.
        expected_starts = {0, 1, 2, 4, 5}

        recon = HierarchicalReconciliation(reconcilers=[reconciler])
        result = recon.reconcile(
            Y_hat_df=Y_hat_df,
            S_df=S_df,
            tags=tags,
            Y_df=Y_df,
            level=[80],
            intervals_method="bootstrap",
            num_samples=200,
            seed=11,
        )
        # Interval columns exist and contain no NaNs (blocks were drawable).
        interval_cols = [
            c
            for c in result.columns
            if "-lo-" in c or "-hi-" in c or "-sample-" in c
        ]
        assert interval_cols
        assert result[interval_cols].notna().all().all()

        # Repeating the same seed must reproduce identical coherent samples.
        result_repeat = recon.reconcile(
            Y_hat_df=Y_hat_df,
            S_df=S_df,
            tags=tags,
            Y_df=Y_df,
            level=[80],
            intervals_method="bootstrap",
            num_samples=200,
            seed=11,
        )
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True),
            result_repeat.reset_index(drop=True),
        )

        residuals = (
            reconciler.sampler.y_insample - reconciler.sampler.y_hat_insample
        )
        starts = _continuous_block_starts(
            residuals, block_length=h, y_dates=reconciler.sampler.y_dates
        )
        assert set(starts.tolist()) == expected_starts

    @pytest.mark.parametrize(
        "reconciler",
        [
            TopDown(method="forecast_proportions"),
            TopDownSparse(method="forecast_proportions"),
        ],
    )
    def test_forecast_proportions_respects_timeline_gap(self, reconciler):
        # forecast_proportions builds its blocks through the shared helper;
        # exercise it directly with arrays carrying the same gapped timeline.
        Y_df, Y_hat_df, _, _ = _hierarchical_frames_with_gap()
        ids = ["total", "g1", "g2", "a", "b", "c", "d"]
        S = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        int_tags = {
            "level1": np.array([0]),
            "level2": np.array([1, 2]),
            "level3": np.array([3, 4, 5, 6]),
        }
        pivot_y = Y_df.pivot(index="unique_id", columns="ds", values="y").loc[ids]
        pivot_f = Y_df.pivot(index="unique_id", columns="ds", values="model").loc[ids]
        y_insample = pivot_y.to_numpy()
        y_hat_insample = pivot_f.to_numpy()
        y_dates = pivot_y.columns.to_numpy()
        h = 2
        y_hat = (
            Y_hat_df.pivot(index="unique_id", columns="ds", values="model")
            .loc[ids]
            .to_numpy()
        )

        result = reconciler.fit_predict(
            S=sparse.csr_matrix(S) if reconciler.is_sparse_method else S,
            y_hat=y_hat,
            tags=int_tags,
            y_insample=y_insample,
            y_hat_insample=y_hat_insample,
            level=[80],
            intervals_method="bootstrap",
            num_samples=200,
            seed=11,
            y_dates=y_dates,
        )
        assert result["quantiles"].shape == (len(ids), h, 2)
        assert np.isfinite(result["quantiles"]).all()

        # Same seed -> identical quantiles through the same block draws.
        result_repeat = reconciler.fit_predict(
            S=sparse.csr_matrix(S) if reconciler.is_sparse_method else S,
            y_hat=y_hat,
            tags=int_tags,
            y_insample=y_insample,
            y_hat_insample=y_hat_insample,
            level=[80],
            intervals_method="bootstrap",
            num_samples=200,
            seed=11,
            y_dates=y_dates,
        )
        np.testing.assert_array_equal(
            result["quantiles"], result_repeat["quantiles"]
        )

        starts = _continuous_block_starts(
            y_insample - y_hat_insample, block_length=h, y_dates=y_dates
        )
        assert set(starts.tolist()) == {0, 1, 2, 4, 5}

    def test_reconcile_raises_when_span_too_long(self):
        from hierarchicalforecast.core import HierarchicalReconciliation

        Y_df, Y_hat_df, S_df, tags = _hierarchical_frames_with_gap()
        # Horizon 5 exceeds both segments (lengths 4 and 3): no valid block.
        long_dates = pd.to_datetime(
            ["2024-02-26", "2024-03-04", "2024-03-11", "2024-03-18", "2024-03-25"]
        )
        forecast_rows = []
        for name in ("total", "g1", "g2", "a", "b", "c", "d"):
            for ds in long_dates:
                forecast_rows.append({"unique_id": name, "ds": ds, "model": 1.0})
        Y_hat_long = pd.DataFrame(forecast_rows)

        recon = HierarchicalReconciliation(reconcilers=[BottomUp()])
        with pytest.raises(ValueError, match="length 5"):
            recon.reconcile(
                Y_hat_df=Y_hat_long,
                S_df=S_df,
                tags=tags,
                Y_df=Y_df,
                level=[80],
                intervals_method="bootstrap",
                num_samples=50,
                seed=0,
            )
