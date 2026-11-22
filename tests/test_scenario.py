import numpy as np
import pytest

from hierarchicalforecast.scenario import (
    ScenarioReconciler,
    aggregate_scenarios,
    quantile_curves_to_scenarios,
    scenarios_to_quantiles,
    validate_quantile_curves,
    validate_scenario_axis,
)


@pytest.fixture
def two_store_curves():
    """Two stores, one horizon, three quantile levels.

    Store B is strongly right-skewed so that aligned-scenario aggregation
    produces different regional quantiles than naively adding marginals.
    """
    levels = np.array([0.1, 0.5, 0.9])
    values = np.array(
        [
            [[1.0, 2.0, 3.0]],  # store A
            [[4.0, 10.0, 20.0]],  # store B
        ]
    )
    return levels, values


@pytest.fixture
def hierarchy_summing_matrix():
    """Total -> two regions -> four bottom stores."""
    return np.array(
        [
            [1, 1, 1, 1],  # total
            [1, 1, 0, 0],  # region 1
            [0, 0, 1, 1],  # region 2
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidateQuantileCurves:
    def test_valid_curves(self, two_store_curves):
        levels, values = two_store_curves
        validate_quantile_curves(levels, values)

    def test_flat_sections_allowed(self):
        # Discrete / lumpy distributions can have flat quantile sections.
        levels = np.array([0.1, 0.5, 0.9])
        values = np.array([[[1.0, 1.0, 2.0]], [[0.0, 5.0, 5.0]]])
        validate_quantile_curves(levels, values)

    def test_levels_must_be_1d(self, two_store_curves):
        _, values = two_store_curves
        with pytest.raises(ValueError, match="1D array"):
            validate_quantile_curves(np.array([[0.1], [0.5]]), values)

    def test_at_least_two_levels(self, two_store_curves):
        _, values = two_store_curves
        with pytest.raises(ValueError, match="At least two quantile levels"):
            validate_quantile_curves(np.array([0.5]), values[:, :, :1])

    def test_levels_outside_open_interval(self, two_store_curves):
        _, values = two_store_curves
        with pytest.raises(ValueError, match="open interval"):
            validate_quantile_curves(np.array([0.0, 0.5, 0.9]), values)
        with pytest.raises(ValueError, match="open interval"):
            validate_quantile_curves(np.array([0.1, 0.5, 1.0]), values)

    def test_levels_not_strictly_increasing(self, two_store_curves):
        _, values = two_store_curves
        with pytest.raises(ValueError, match="strictly increasing"):
            validate_quantile_curves(np.array([0.1, 0.1, 0.9]), values)

    def test_levels_unsorted(self, two_store_curves):
        _, values = two_store_curves
        with pytest.raises(ValueError, match="strictly increasing"):
            validate_quantile_curves(np.array([0.5, 0.1, 0.9]), values)

    def test_values_must_be_3d(self, two_store_curves):
        levels, _ = two_store_curves
        with pytest.raises(ValueError, match="3D array"):
            validate_quantile_curves(levels, np.ones((2, 3)))

    def test_values_dimension_mismatch(self, two_store_curves):
        levels, _ = two_store_curves
        with pytest.raises(ValueError, match="must match"):
            validate_quantile_curves(levels, np.ones((2, 1, 4)))

    def test_values_not_monotone(self, two_store_curves):
        levels, values = two_store_curves
        bad = values.copy()
        bad[0, 0, 1] = bad[0, 0, 2] + 1.0
        with pytest.raises(ValueError, match="monotonically non-decreasing"):
            validate_quantile_curves(levels, bad)

    def test_nan_values_rejected(self, two_store_curves):
        levels, values = two_store_curves
        bad = values.copy()
        bad[0, 0, 1] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            validate_quantile_curves(levels, bad)


class TestValidateScenarioAxis:
    def test_valid_axis(self):
        probs = validate_scenario_axis(np.array([0.1, 0.5, 0.9]))
        assert probs.dtype == np.float64
        np.testing.assert_array_equal(probs, [0.1, 0.5, 0.9])

    def test_empty_axis(self):
        with pytest.raises(ValueError, match="empty"):
            validate_scenario_axis(np.array([]))

    def test_duplicates_rejected(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            validate_scenario_axis(np.array([0.2, 0.2, 0.5]))

    def test_boundary_ranks_rejected(self):
        with pytest.raises(ValueError, match="open interval"):
            validate_scenario_axis(np.array([0.0, 0.5]))
        with pytest.raises(ValueError, match="open interval"):
            validate_scenario_axis(np.array([0.5, 1.0]))

    def test_unsorted_rejected(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            validate_scenario_axis(np.array([0.9, 0.1]))

    def test_nan_rejected(self):
        with pytest.raises(ValueError, match="NaN"):
            validate_scenario_axis(np.array([0.1, np.nan, 0.9]))


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


class TestQuantileCurvesToScenarios:
    def test_recover_input_at_same_ranks(self, two_store_curves):
        levels, values = two_store_curves
        probs = np.array([0.1, 0.5, 0.9])
        scenarios = quantile_curves_to_scenarios(levels, values, probs)
        assert scenarios.shape == (2, 1, 3)
        np.testing.assert_allclose(scenarios, values)

    def test_linear_interpolation_between_levels(self, two_store_curves):
        levels, values = two_store_curves
        scenarios = quantile_curves_to_scenarios(
            levels, values, np.array([0.3])
        )
        # A: 1 + 0.5 * 1 = 1.5 ; B: 4 + 0.5 * 6 = 7
        np.testing.assert_allclose(scenarios[:, 0, 0], [1.5, 7.0])

    def test_out_of_range_ranks_are_clamped(self, two_store_curves):
        levels, values = two_store_curves
        probs = np.array([0.01, 0.5, 0.99])
        scenarios = quantile_curves_to_scenarios(levels, values, probs)
        np.testing.assert_allclose(scenarios[:, 0, 0], values[:, 0, 0])
        np.testing.assert_allclose(scenarios[:, 0, -1], values[:, 0, -1])

    def test_output_shape_multi_horizon(self, two_store_curves):
        levels, values = two_store_curves
        values_h = np.broadcast_to(values, (2, 5, 3)).copy()
        probs = np.linspace(0.05, 0.95, 11)
        scenarios = quantile_curves_to_scenarios(levels, values_h, probs)
        assert scenarios.shape == (2, 5, 11)

    def test_invalid_input_raises(self, two_store_curves):
        levels, values = two_store_curves
        with pytest.raises(ValueError):
            quantile_curves_to_scenarios(
                levels, values, np.array([0.0, 0.5])
            )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestAggregateScenarios:
    def test_single_region_sums_same_scenario(self, two_store_curves):
        levels, values = two_store_curves
        probs = np.array([0.1, 0.5, 0.9])
        scenarios = quantile_curves_to_scenarios(levels, values, probs)
        regional = aggregate_scenarios(scenarios)
        assert regional.shape == (1, 1, 3)
        # Each column is a shared-rank scenario summed across stores.
        np.testing.assert_allclose(regional[0, 0], [5.0, 12.0, 23.0])

    def test_summing_matrix_hierarchy(self, two_store_curves, hierarchy_summing_matrix):
        levels, _ = two_store_curves
        values = np.arange(4 * 2 * 3, dtype=float).reshape(4, 2, 3) + 1
        probs = np.array([0.1, 0.5, 0.9])
        scenarios = quantile_curves_to_scenarios(levels, values, probs)
        regional = aggregate_scenarios(scenarios, S=hierarchy_summing_matrix)
        assert regional.shape == (7, 2, 3)
        # Bottom rows pass through unchanged.
        np.testing.assert_allclose(regional[3:], scenarios)
        # Aggregation constraints hold scenario by scenario.
        np.testing.assert_allclose(regional[1], regional[3] + regional[4])
        np.testing.assert_allclose(regional[2], regional[5] + regional[6])
        np.testing.assert_allclose(regional[0], regional[1] + regional[2])

    def test_summing_matrix_column_mismatch(self, two_store_curves):
        levels, values = two_store_curves
        scenarios = quantile_curves_to_scenarios(
            levels, values, np.array([0.1, 0.5, 0.9])
        )
        with pytest.raises(ValueError, match="must match"):
            aggregate_scenarios(scenarios, S=np.ones((2, 5)))

    def test_rejects_non_3d_input(self):
        with pytest.raises(ValueError, match="3D array"):
            aggregate_scenarios(np.ones((4, 3)))


# ---------------------------------------------------------------------------
# Regional quantiles
# ---------------------------------------------------------------------------


class TestScenariosToQuantiles:
    def test_shape_and_values(self, two_store_curves):
        levels, values = two_store_curves
        probs = np.array([0.1, 0.5, 0.9])
        regional = aggregate_scenarios(
            quantile_curves_to_scenarios(levels, values, probs)
        )
        out = scenarios_to_quantiles(regional, levels)
        assert out.shape == (1, 1, 3)
        np.testing.assert_allclose(
            out[0, 0], np.quantile([5.0, 12.0, 23.0], levels)
        )

    def test_levels_outside_interval_rejected(self, two_store_curves):
        levels, values = two_store_curves
        regional = aggregate_scenarios(
            quantile_curves_to_scenarios(
                levels, values, np.array([0.1, 0.5, 0.9])
            )
        )
        with pytest.raises(ValueError, match="open interval"):
            scenarios_to_quantiles(regional, np.array([0.5, 1.0]))


# ---------------------------------------------------------------------------
# End-to-end reconciler
# ---------------------------------------------------------------------------


class TestScenarioReconciler:
    def test_end_to_end_coherence(self, hierarchy_summing_matrix):
        levels = np.array([0.1, 0.5, 0.9])
        values = np.arange(4 * 2 * 3, dtype=float).reshape(4, 2, 3) + 1
        # Scenario axis includes the input quantile levels so the bottom
        # rows are reproduced exactly when the output levels are taken.
        probs = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
        rec = ScenarioReconciler(scenario_probs=probs, S=hierarchy_summing_matrix)
        out = rec.reconcile_quantiles(levels, values, output_levels=levels)

        assert out.shape == (7, 2, 3)
        # Bottom-level quantile curves are preserved.
        np.testing.assert_allclose(out[3:], values)
        # Regional quantiles are coherent with their children.
        np.testing.assert_allclose(out[1], out[3] + out[4])
        np.testing.assert_allclose(out[2], out[5] + out[6])
        np.testing.assert_allclose(out[0], out[1] + out[2])
        # Output quantile curves remain monotonic.
        assert np.all(np.diff(out, axis=2) >= -1e-12)

    def test_default_output_levels_reuse_input_levels(
        self, hierarchy_summing_matrix
    ):
        levels = np.array([0.1, 0.5, 0.9])
        values = np.arange(4 * 1 * 3, dtype=float).reshape(4, 1, 3) + 1
        rec = ScenarioReconciler(
            scenario_probs=np.array([0.1, 0.5, 0.9]),
            S=hierarchy_summing_matrix,
        )
        out = rec.reconcile_quantiles(levels, values)
        assert out.shape == (7, 1, 3)

    def test_stored_scenarios(self, hierarchy_summing_matrix):
        levels = np.array([0.1, 0.5, 0.9])
        values = np.arange(4 * 1 * 3, dtype=float).reshape(4, 1, 3) + 1
        probs = np.array([0.1, 0.5, 0.9])
        rec = ScenarioReconciler(scenario_probs=probs, S=hierarchy_summing_matrix)
        rec.fit_transform_scenarios(levels, values)
        assert rec.bottom_scenarios.shape == (4, 1, 3)
        assert rec.regional_scenarios.shape == (7, 1, 3)

    def test_single_region_without_s_matrix(self, two_store_curves):
        levels, values = two_store_curves
        rec = ScenarioReconciler(scenario_probs=np.array([0.1, 0.5, 0.9]))
        out = rec.reconcile_quantiles(levels, values)
        assert out.shape == (1, 1, 3)
        assert rec.regional_scenarios.shape == (1, 1, 3)

    def test_invalid_scenario_axis_at_construction(self):
        with pytest.raises(ValueError):
            ScenarioReconciler(scenario_probs=np.array([0.0, 0.5]))

    def test_s_matrix_column_mismatch(self, two_store_curves):
        levels, values = two_store_curves
        rec = ScenarioReconciler(
            scenario_probs=np.array([0.1, 0.5, 0.9]),
            S=np.ones((2, 5)),
        )
        with pytest.raises(ValueError, match="columns"):
            rec.reconcile_quantiles(levels, values)

    def test_aligned_scenarios_differ_from_naive_marginal_sums(
        self, two_store_curves
    ):
        """The core guarantee: regional quantiles come from summing shared-rank
        scenarios and re-taking quantiles, never from adding marginal quantiles.
        """
        levels, values = two_store_curves
        dense_probs = np.linspace(0.05, 0.95, 50)
        rec = ScenarioReconciler(scenario_probs=dense_probs)
        regional_q = rec.reconcile_quantiles(levels, values)
        naive_marginal_sum = values.sum(axis=0)
        assert not np.allclose(regional_q, naive_marginal_sum)

    def test_shared_rank_scenario_sum_matches_marginal_values_at_grid(
        self, two_store_curves
    ):
        """At the exact input ranks, the aligned scenario k equals the sum of
        the k-th marginal quantiles (comonotonic coupling at shared ranks).
        """
        levels, values = two_store_curves
        rec = ScenarioReconciler(scenario_probs=levels)
        rec.fit_transform_scenarios(levels, values)
        np.testing.assert_allclose(
            rec.regional_scenarios[0, 0], values.sum(axis=0)[0]
        )
