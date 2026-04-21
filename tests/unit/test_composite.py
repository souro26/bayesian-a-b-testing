import numpy as np
import pytest

from argonx.decision_rules.composite import compute_composite_score


class MockGuardrails:
    """Minimal mock for GuardrailBundle."""
    def __init__(self, variant_passed):
        self.variant_passed = variant_passed


def make_basic_inputs():
    """Create simple valid inputs for composite."""
    primary = np.array([
        [1.0, 1.2],
        [1.0, 1.3],
        [1.0, 1.1],
    ])

    guardrails = {
        "error": np.array([
            [0.1, 0.08],
            [0.1, 0.07],
            [0.1, 0.09],
        ])
    }

    names = ["control", "B"]
    weights = {"primary": 1.0, "error": -1.0}

    gb = MockGuardrails({
        "control": True,
        "B": True
    })

    return primary, guardrails, names, weights, gb


class TestCompositeBasic:

    def test_basic_score_computation(self):
        """Computes positive score when variant improves."""
        primary, guardrails, names, weights, gb = make_basic_inputs()

        res = compute_composite_score(
            primary, guardrails, names, "control", weights, gb
        )

        assert res.score["B"] > 0

    def test_best_variant_selected(self):
        """Selects variant with highest score."""
        primary, guardrails, names, weights, gb = make_basic_inputs()

        res = compute_composite_score(
            primary, guardrails, names, "control", weights, gb
        )

        assert res.best_variant == "B"

    def test_score_distribution_shape(self):
        """Score distribution matches number of draws."""
        primary, guardrails, names, weights, gb = make_basic_inputs()

        res = compute_composite_score(
            primary, guardrails, names, "control", weights, gb
        )

        assert len(res.score_distribution["B"]) == primary.shape[0]

    def test_score_distribution_is_numpy_array(self):
        """Score distribution is numpy array."""
        primary, guardrails, names, weights, gb = make_basic_inputs()

        res = compute_composite_score(
            primary, guardrails, names, "control", weights, gb
        )

        assert isinstance(res.score_distribution["B"], np.ndarray)


class TestThreshold:

    def test_prob_exceeds_threshold(self):
        """Probability above threshold computed correctly."""
        primary, guardrails, names, weights, gb = make_basic_inputs()

        res = compute_composite_score(
            primary, guardrails, names, "control", weights, gb,
            threshold=0.0
        )

        assert 0 <= res.prob_exceeds_threshold["B"] <= 1

    def test_gap_distribution_centered(self):
        """Gap distribution shifts by threshold."""
        primary, guardrails, names, weights, gb = make_basic_inputs()

        res = compute_composite_score(
            primary, guardrails, names, "control", weights, gb,
            threshold=0.1
        )

        gap = res.gap_distribution["B"]
        score = res.score_distribution["B"]

        assert np.allclose(gap, score - 0.1)

    def test_gap_hdi_positive_when_far_above_threshold(self):
        """HDI bounds positive when variant strongly exceeds threshold."""
        primary = np.array([
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
        ])

        guardrails = {}
        names = ["control", "B"]
        weights = {"primary": 1.0}
        gb = MockGuardrails({"control": True, "B": True})

        res = compute_composite_score(
            primary, guardrails, names, "control", weights, gb,
            threshold=0.5
        )

        low, _ = res.gap_hdi["B"]
        assert low > 0


class TestGuardrails:

    def test_penalty_applied_when_failed(self):
        """Penalty reduces score when guardrail fails."""
        primary, guardrails, names, weights, _ = make_basic_inputs()

        gb = MockGuardrails({
            "control": True,
            "B": False
        })

        res = compute_composite_score(
            primary, guardrails, names, "control", weights, gb,
            guardrail_penalty=1.0
        )

        assert res.score["B"] < 0

    def test_no_penalty_when_passed(self):
        """No penalty when guardrail passes."""
        primary, guardrails, names, weights, gb = make_basic_inputs()

        res = compute_composite_score(
            primary, guardrails, names, "control", weights, gb,
            guardrail_penalty=1.0
        )

        assert res.score["B"] > 0

    def test_penalty_zero_when_all_pass(self):
        """Penalty path not triggered when all pass."""
        primary, guardrails, names, weights, gb = make_basic_inputs()

        res = compute_composite_score(
            primary, guardrails, names, "control", weights, gb,
            guardrail_penalty=10.0
        )

        assert res.score["B"] > 0


class TestWeights:

    def test_missing_weights_raises(self):
        """Raises error when weights missing."""
        primary, guardrails, names, _, gb = make_basic_inputs()

        with pytest.raises(ValueError):
            compute_composite_score(
                primary, guardrails, names, "control", {}, gb
            )

    def test_invalid_weight_keys_raises(self):
        """Raises error for unmatched weight keys."""
        primary, guardrails, names, _, gb = make_basic_inputs()

        with pytest.raises(ValueError):
            compute_composite_score(
                primary,
                guardrails,
                names,
                "control",
                {"invalid": 1.0},
                gb
            )

    def test_partial_weights_ignore_other_metrics(self):
        """Metrics not in weights contribute zero."""
        primary, guardrails, names, _, gb = make_basic_inputs()

        weights = {"primary": 1.0}

        res = compute_composite_score(
            primary, guardrails, names, "control", weights, gb
        )

        contrib = res.metric_contributions["B"]

        assert contrib.get("error", 0.0) == 0.0

    def test_asymmetric_deterioration_weights(self):
        """Higher deterioration weight penalizes losses more."""
        primary = np.array([
            [1.0, 0.8],
            [1.0, 0.7],
            [1.0, 0.9],
        ])

        guardrails = {}
        names = ["control", "B"]
        weights = {"primary": 1.0}
        gb = MockGuardrails({"control": True, "B": True})

        res_sym = compute_composite_score(
            primary, guardrails, names, "control", weights, gb
        )

        res_asym = compute_composite_score(
            primary,
            guardrails,
            names,
            "control",
            weights,
            gb,
            deterioration_weights={"primary": 2.0}
        )

        assert res_asym.score["B"] < res_sym.score["B"]


class TestMultiVariant:

    def test_three_variants_ranking(self):
        """Selects correct best among three variants."""
        primary = np.array([
            [1.0, 1.2, 0.9],
            [1.0, 1.3, 0.8],
            [1.0, 1.25, 0.85],
        ])

        guardrails = {}
        names = ["control", "B", "C"]
        weights = {"primary": 1.0}

        gb = MockGuardrails({
            "control": True,
            "B": True,
            "C": True
        })

        res = compute_composite_score(
            primary, guardrails, names, "control", weights, gb
        )

        assert res.best_variant == "B"


class TestEdgeCases:

    def test_zero_effect_results_in_zero_score(self):
        """Zero delta produces zero composite score."""
        primary = np.ones((5, 2))

        guardrails = {}
        names = ["control", "B"]
        weights = {"primary": 1.0}

        gb = MockGuardrails({"control": True, "B": True})

        res = compute_composite_score(
            primary, guardrails, names, "control", weights, gb
        )

        assert np.isclose(res.score["B"], 0.0)

    def test_control_not_in_outputs(self):
        """Control is excluded from outputs."""
        primary, guardrails, names, weights, gb = make_basic_inputs()

        res = compute_composite_score(
            primary, guardrails, names, "control", weights, gb
        )

        assert "control" not in res.score
        assert "control" not in res.score_distribution
        assert "control" not in res.metric_contributions

    def test_all_variants_below_threshold(self):
        """Warns when no variant exceeds threshold."""
        primary, guardrails, names, weights, gb = make_basic_inputs()

        res = compute_composite_score(
            primary, guardrails, names, "control", weights, gb,
            threshold=100.0
        )

        assert "No variant exceeds composite threshold" in res.warnings

    def test_negative_score_warning(self):
        """Warns when variant has negative score."""
        primary, guardrails, names, weights, gb = make_basic_inputs()

        bad_weights = {"primary": -1.0, "error": 1.0}

        res = compute_composite_score(
            primary, guardrails, names, "control", bad_weights, gb
        )

        assert any("negative composite score" in w for w in res.warnings)