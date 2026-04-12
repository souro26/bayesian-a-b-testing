import numpy as np
import pytest

from experimentation.decision_rules.metrics import (
    compute_prob_best,
    compute_expected_loss,
    compute_cvar,
    compute_rope,
    compute_lift_hdi,
    compute_all_metrics,
    _validate_inputs,
    _validate_config,
)


class TestComputeProbBest:

    def test_clear_winner(self):
        """Variant B wins all draws → probability 1.0."""
        A = np.array([0.1, 0.2, 0.3])
        B = np.array([0.5, 0.6, 0.7])
        samples = np.column_stack([A, B])

        res = compute_prob_best(samples, ["A", "B"])

        assert res.probabilities["B"] == 1.0
        assert res.probabilities["A"] == 0.0
        assert res.best_variant == "B"

    def test_mixed_outcomes(self):
        """Mixed wins produce fractional probabilities."""
        A = np.array([0.5, 0.1, 0.3])
        B = np.array([0.4, 0.2, 0.6])
        samples = np.column_stack([A, B])

        res = compute_prob_best(samples, ["A", "B"])

        assert 0 < res.probabilities["A"] < 1
        assert 0 < res.probabilities["B"] < 1

    def test_probabilities_sum_to_one(self):
        """Probabilities across variants sum to one."""
        samples = np.random.rand(100, 3)
        res = compute_prob_best(samples, ["A", "B", "C"])

        total = sum(res.probabilities.values())
        assert np.isclose(total, 1.0)

    def test_three_variants(self):
        """Handles argmax correctly across 3 variants."""
        A = np.array([1, 1, 1])
        B = np.array([2, 2, 2])
        C = np.array([0, 0, 0])
        samples = np.column_stack([A, B, C])

        res = compute_prob_best(samples, ["A", "B", "C"])
        assert res.best_variant == "B"

    def test_n_draws_stored(self):
        """n_draws reflects number of posterior samples."""
        A = np.array([0.1, 0.2, 0.3])
        B = np.array([0.4, 0.5, 0.6])
        samples = np.column_stack([A, B])

        res = compute_prob_best(samples, ["A", "B"])

        assert res.n_draws == 3


class TestComputeExpectedLoss:

    def test_zero_loss_for_best(self):
        """Best variant has zero expected loss."""
        A = np.array([1, 1, 1])
        B = np.array([2, 2, 2])
        samples = np.column_stack([A, B])

        res = compute_expected_loss(samples, ["A", "B"], "A")

        assert res.expected_loss["B"] == 0.0

    def test_positive_loss_for_worse(self):
        """Worse variant has positive expected loss."""
        A = np.array([2, 2, 2])
        B = np.array([1, 1, 1])
        samples = np.column_stack([A, B])

        res = compute_expected_loss(samples, ["A", "B"], "A")

        assert res.expected_loss["B"] > 0

    def test_loss_distribution_shape(self):
        """Loss distribution matches number of draws."""
        samples = np.random.rand(50, 2)
        res = compute_expected_loss(samples, ["A", "B"], "A")

        assert len(res.loss_distributions["A"]) == 50


class TestComputeCVAR:

    def test_cvar_greater_than_expected_loss(self):
        """CVaR is always ≥ expected loss."""
        samples = np.random.rand(1000, 2)
        loss = compute_expected_loss(samples, ["A", "B"], "A")
        cvar = compute_cvar(loss, ["A", "B"])

        for v in ["A", "B"]:
            assert cvar.cvar[v] >= loss.expected_loss[v]

    def test_cvar_with_zero_loss(self):
        """Zero loss leads to zero CVaR."""
        A = np.array([2, 2, 2])
        B = np.array([1, 1, 1])
        samples = np.column_stack([A, B])

        loss = compute_expected_loss(samples, ["A", "B"], "A")
        cvar = compute_cvar(loss, ["A", "B"])

        assert cvar.cvar["A"] == 0.0


class TestComputeROPE:

    def test_all_inside_rope(self):
        """All lifts inside ROPE → inside_rope = 1."""
        A = np.array([1, 1, 1])
        B = np.array([1.01, 1.02, 1.01])
        samples = np.column_stack([A, B])

        res = compute_rope(samples, ["A", "B"], "A", -0.05, 0.05)

        assert res.inside_rope["B"] == 1.0

    def test_all_outside_rope(self):
        """All lifts outside ROPE → inside_rope = 0."""
        A = np.array([1, 1, 1])
        B = np.array([2, 2, 2])
        samples = np.column_stack([A, B])

        res = compute_rope(samples, ["A", "B"], "A", -0.1, 0.1)

        assert res.inside_rope["B"] == 0.0

    def test_control_zero_raises(self):
        """All-zero control raises error."""
        A = np.array([0.0, 0.0, 0.0])
        B = np.array([1.0, 1.0, 1.0])
        samples = np.column_stack([A, B])

        with pytest.raises(ValueError):
            compute_rope(samples, ["A", "B"], "A", -0.1, 0.1)


class TestComputeLiftHDI:

    def test_control_has_zero_lift(self):
        """Control always has zero lift."""
        samples = np.random.rand(100, 2)
        res = compute_lift_hdi(samples, ["A", "B"], "A")

        assert res.mean["A"] == 0.0

    def test_hdi_bounds_valid(self):
        """HDI lower bound is ≤ upper bound."""
        samples = np.random.rand(1000, 2)
        res = compute_lift_hdi(samples, ["A", "B"], "A")

        assert res.hdi_low["B"] <= res.hdi_high["B"]


class TestValidation:

    def test_invalid_variant_alignment(self):
        """Mismatch between names and samples raises error."""
        samples = np.random.rand(10, 2)

        with pytest.raises(ValueError):
            _validate_inputs(samples, ["A"], "A")

    def test_invalid_control(self):
        """Missing control raises error."""
        samples = np.random.rand(10, 2)

        with pytest.raises(ValueError):
            _validate_inputs(samples, ["A", "B"], "C")

    def test_invalid_alpha(self):
        """Alpha outside (0,1) raises error."""
        with pytest.raises(ValueError):
            _validate_config(1.5, 0.95, (-0.1, 0.1))

    def test_invalid_rope_bounds(self):
        """Invalid rope bounds raise error."""
        with pytest.raises(ValueError):
            _validate_config(0.95, 0.95, (0.1, -0.1))


class TestComputeAllMetrics:

    def test_full_pipeline_runs(self):
        """End-to-end metric computation returns bundle."""
        samples = np.random.rand(500, 2)

        res = compute_all_metrics(
            samples,
            ["A", "B"],
            control="A",
            rope_bounds=(-0.1, 0.1),
        )

        assert res.prob_best is not None
        assert res.loss is not None
        assert res.cvar is not None
        assert res.rope is not None
        assert res.lift is not None

    def test_warnings_collected(self):
        """Low sample size triggers warnings collection."""
        samples = np.random.rand(10, 2)

        res = compute_all_metrics(
            samples,
            ["A", "B"],
            control="A",
            rope_bounds=(-0.1, 0.1),
        )

        assert len(res.warnings) > 0