import numpy as np
import pytest

from experimentation.decision_rules.guardrails import (
    compute_guardrail,
    compute_all_guardrails,
    _validate_guardrail_inputs,
)

class TestGuardrailValidation:

    def test_invalid_variant_alignment(self):
        """Mismatch between variant_names and samples raises error."""
        samples = np.random.rand(10, 2)

        with pytest.raises(ValueError):
            _validate_guardrail_inputs(samples, ["A"], "A", 0.5)

    def test_invalid_control(self):
        """Missing control in variant_names raises error."""
        samples = np.random.rand(10, 2)

        with pytest.raises(ValueError):
            _validate_guardrail_inputs(samples, ["A", "B"], "C", 0.5)

    def test_invalid_threshold(self):
        """Threshold outside (0,1) raises error."""
        samples = np.random.rand(10, 2)

        with pytest.raises(ValueError):
            _validate_guardrail_inputs(samples, ["A", "B"], "A", 1.5)

    def test_nan_samples_raise(self):
        """NaN in samples raises error."""
        samples = np.array([[1.0, np.nan], [1.0, 2.0]])

        with pytest.raises(ValueError):
            _validate_guardrail_inputs(samples, ["A", "B"], "A", 0.5)

    def test_single_variant_raises(self):
        """Less than two variants raises error."""
        samples = np.random.rand(10, 1)

        with pytest.raises(ValueError):
            _validate_guardrail_inputs(samples, ["A"], "A", 0.5)

class TestComputeGuardrail:

    def test_clear_degradation(self):
        """Variant always worse → prob_degraded = 1."""
        control = np.array([1, 1, 1])
        variant = np.array([2, 2, 2])
        samples = np.column_stack([control, variant])

        res = compute_guardrail(
            samples,
            ["A", "B"],
            control="A",
            metric="latency",
            threshold=0.5,
            lower_is_better=True,
        )

        assert res[0].prob_degraded == 1.0
        assert res[0].passed is False

    def test_no_degradation(self):
        """Variant always better → prob_degraded = 0."""
        control = np.array([2, 2, 2])
        variant = np.array([1, 1, 1])
        samples = np.column_stack([control, variant])

        res = compute_guardrail(
            samples,
            ["A", "B"],
            control="A",
            metric="latency",
            threshold=0.5,
            lower_is_better=True,
        )

        assert res[0].prob_degraded == 0.0
        assert res[0].passed is True

    def test_lower_is_better_false(self):
        """Flip degradation logic when higher is better."""
        control = np.array([1, 1, 1])
        variant = np.array([0, 0, 0])
        samples = np.column_stack([control, variant])

        res = compute_guardrail(
            samples,
            ["A", "B"],
            control="A",
            metric="revenue",
            threshold=0.5,
            lower_is_better=False,
        )

        assert res[0].prob_degraded == 1.0

    def test_expected_degradation_positive(self):
        """Expected degradation computed when degradation exists."""
        control = np.array([1, 1, 1])
        variant = np.array([2, 3, 4])
        samples = np.column_stack([control, variant])

        res = compute_guardrail(
            samples,
            ["A", "B"],
            control="A",
            metric="latency",
            threshold=0.5,
        )

        assert res[0].expected_degradation > 0

    def test_severity_levels(self):
        """Severity classification matches probability thresholds."""
        control = np.array([1, 1, 1, 1])
        variant = np.array([2, 2, 1, 1]) 
        samples = np.column_stack([control, variant])

        res = compute_guardrail(
            samples,
            ["A", "B"],
            control="A",
            metric="latency",
            threshold=0.9,
        )

        assert res[0].severity == "low"

class TestComputeAllGuardrails:

    def test_empty_guardrails(self):
        """No guardrails returns vacuously passing bundle."""
        samples = np.random.rand(10, 2)

        res = compute_all_guardrails(
            guardrail_samples={},
            variant_names=["A", "B"],
            control="A",
            thresholds={},
            primary_passed=True,
        )

        assert res.all_passed is True
        assert len(res.guardrails) == 0
        assert len(res.warnings) > 0

    def test_missing_threshold_raises(self):
        """Missing threshold for metric raises error."""
        samples = np.random.rand(10, 2)

        with pytest.raises(ValueError):
            compute_all_guardrails(
                guardrail_samples={"latency": samples},
                variant_names=["A", "B"],
                control="A",
                thresholds={},
                primary_passed=True,
            )

    def test_conflict_detection(self):
        """Primary pass + guardrail fail creates conflict."""
        control = np.array([1, 1, 1])
        variant = np.array([2, 2, 2])
        samples = np.column_stack([control, variant])

        res = compute_all_guardrails(
            guardrail_samples={"latency": samples},
            variant_names=["A", "B"],
            control="A",
            thresholds={"latency": 0.5},
            primary_passed=True,
        )

        assert len(res.conflicts) == 1

    def test_no_conflict_when_primary_fails(self):
        """No conflict if primary metric did not pass."""
        control = np.array([1, 1, 1])
        variant = np.array([2, 2, 2])
        samples = np.column_stack([control, variant])

        res = compute_all_guardrails(
            guardrail_samples={"latency": samples},
            variant_names=["A", "B"],
            control="A",
            thresholds={"latency": 0.5},
            primary_passed=False,
        )

        assert len(res.conflicts) == 0

    def test_variant_passed_logic(self):
        """variant_passed aggregates across guardrails correctly."""
        control = np.array([1, 1, 1])
        variant = np.array([1, 1, 1])
        samples = np.column_stack([control, variant])

        res = compute_all_guardrails(
            guardrail_samples={"latency": samples},
            variant_names=["A", "B"],
            control="A",
            thresholds={"latency": 0.5},
            primary_passed=True,
        )

        assert res.variant_passed["B"] is True

    def test_lower_is_better_default_warning(self):
        """Missing lower_is_better triggers warning."""
        samples = np.random.rand(10, 2)

        res = compute_all_guardrails(
            guardrail_samples={"latency": samples},
            variant_names=["A", "B"],
            control="A",
            thresholds={"latency": 0.5},
            primary_passed=True,
        )

        assert any("lower_is_better" in w for w in res.warnings)