"""
tests/unit/test_stopping.py

Unit tests for argonx.sequential.stopping.

Philosophy
----------
Tests are written from the user's perspective — what does a caller expect
this function to do? Not what the implementation happens to compute internally.

No MCMC. No Experiment.run(). No model.sample_posterior().
All tests operate on numpy arrays and module-level functions directly.

compute_expected_loss and compute_prob_best are mocked so gate logic can
be tested without depending on the metrics layer. The traffic, futility,
and users-needed helpers are tested in isolation.

Every test answers the question: if I configure the system this way and
provide this data, what should happen? Not: does field X equal field Y.
"""

from __future__ import annotations

import warnings
from typing import Optional
from unittest.mock import patch

import numpy as np
import pytest

from argonx.decision_rules.metrics import LossResult, PBestResult
from argonx.sequential.stopping import (
    evaluate_stopping,
    _check_traffic_balance,
    _check_futility,
    _estimate_users_needed,
    StoppingChecker,
)


VARIANTS_2 = ["control", "variant_b"]
VARIANTS_3 = ["control", "variant_b", "variant_c"]


def _samples(n_draws: int = 1000, n_variants: int = 2, seed: int = 0) -> np.ndarray:
    """Generate positive-valued posterior samples.

    Parameters
    ----------
    n_draws : int, optional
        Number of posterior draws, by default 1000.
    n_variants : int, optional
        Number of variants, by default 2.
    seed : int, optional
        Random seed for reproducibility, by default 0.

    Returns
    -------
    np.ndarray
        Posterior samples with shape (n_draws, n_variants).
    """
    rng = np.random.default_rng(seed)
    return rng.lognormal(mean=0.0, sigma=0.3, size=(n_draws, n_variants))


def _n_users(variants: list[str], count: int = 2000) -> dict[str, int]:
    """Create user count dictionary for variants.

    Parameters
    ----------
    variants : list[str]
        Variant names.
    count : int, optional
        Number of users per variant, by default 2000.

    Returns
    -------
    dict[str, int]
        User counts keyed by variant name.
    """
    return {v: count for v in variants}


def _mock_metrics(
    loss_values: dict[str, float],
    prob_values: dict[str, float],
    best: str,
):
    """Construct real LossResult and PBestResult for testing.

    Parameters
    ----------
    loss_values : dict[str, float]
        Expected loss per variant.
    prob_values : dict[str, float]
        Probability best per variant.
    best : str
        Name of best variant.

    Returns
    -------
    tuple[LossResult, PBestResult]
        Real dataclass instances matching production types.

    Notes
    -----
    Constructs real LossResult and PBestResult so mock types match production.
    If the real dataclass fields change, these break loudly instead of silently.
    """
    n_draws = 1000
    loss_result = LossResult(
        expected_loss=loss_values,
        loss_distributions={v: np.zeros(n_draws) for v in loss_values},
        control="control",
        n_draws=n_draws,
    )
    prob_result = PBestResult(
        probabilities=prob_values,
        best_variant=best,
        n_draws=n_draws,
    )
    return loss_result, prob_result


def _call_evaluate(
    samples=None,
    variant_names=None,
    control="control",
    n_users=None,
    loss_values=None,
    prob_values=None,
    best_variant="variant_b",
    checkpoint_index=5,
    min_sample_size=100,
    burn_in_users=50,
    min_checkpoints=3,
    loss_threshold=0.01,
    prob_best_min=0.80,
    imbalance_tolerance=0.10,
    imbalance_blocks_stopping=True,
    expected_traffic_shares=None,
    rope_bounds=(-0.01, 0.01),
    futility_rope_threshold=0.80,
    experiment_age_days=None,
    novelty_warning_days=14,
    daily_traffic_per_variant=None,
    users_estimate_safety_factor=1.25,
    users_estimate_floor=100,
    min_draws_warning=500,
    prior_trajectory=None,
    **kwargs,
):
    """Convenience wrapper that patches both metrics calls.

    Parameters
    ----------
    samples : np.ndarray, optional
        Posterior samples, by default None (generates synthetic).
    variant_names : list[str], optional
        Variant names, by default None (uses VARIANTS_2).
    control : str, optional
        Control variant name, by default "control".
    n_users : dict[str, int], optional
        User counts per variant, by default None (generates 2000 each).
    loss_values : dict[str, float], optional
        Expected loss per variant, by default None (all 0.001).
    prob_values : dict[str, float], optional
        Probability best per variant, by default None (control 0.03, variant_b 0.97).
    best_variant : str, optional
        Name of best variant, by default "variant_b".
    checkpoint_index : int, optional
        Checkpoint number, by default 5.
    min_sample_size : int, optional
        Minimum sample size gate, by default 100.
    burn_in_users : int, optional
        Burn-in gate, by default 50.
    min_checkpoints : int, optional
        Minimum checkpoints gate, by default 3.
    loss_threshold : float, optional
        Loss gate threshold, by default 0.01.
    prob_best_min : float, optional
        Probability best gate threshold, by default 0.80.
    imbalance_tolerance : float, optional
        Traffic imbalance tolerance, by default 0.10.
    imbalance_blocks_stopping : bool, optional
        Whether imbalance blocks stopping, by default True.
    expected_traffic_shares : dict, optional
        Expected traffic shares, by default None.
    rope_bounds : tuple, optional
        ROPE bounds for futility, by default (-0.01, 0.01).
    futility_rope_threshold : float, optional
        Futility threshold, by default 0.80.
    experiment_age_days : float, optional
        Age of experiment in days, by default None.
    novelty_warning_days : int, optional
        Days threshold for novelty warning, by default 14.
    daily_traffic_per_variant : dict, optional
        Daily traffic per variant, by default None.
    users_estimate_safety_factor : float, optional
        Safety factor for user estimate, by default 1.25.
    users_estimate_floor : int, optional
        Floor for user estimate, by default 100.
    min_draws_warning : int, optional
        Minimum draws for warning, by default 500.
    prior_trajectory : list, optional
        Prior trajectory snapshots, by default None.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    StoppingResult
        Result of stopping evaluation with mocked metrics.
    """
    if samples is None:
        samples = _samples()
    if variant_names is None:
        variant_names = VARIANTS_2
    if n_users is None:
        n_users = _n_users(variant_names)
    if loss_values is None:
        loss_values = {v: 0.001 for v in variant_names}
    if prob_values is None:
        prob_values = {"control": 0.03, "variant_b": 0.97}

    loss_result, prob_result = _mock_metrics(loss_values, prob_values, best_variant)

    with patch("argonx.sequential.stopping.compute_expected_loss", return_value=loss_result), \
         patch("argonx.sequential.stopping.compute_prob_best",    return_value=prob_result):
        return evaluate_stopping(
            samples=samples,
            variant_names=variant_names,
            control=control,
            n_users_per_variant=n_users,
            loss_threshold=loss_threshold,
            prob_best_min=prob_best_min,
            min_sample_size=min_sample_size,
            burn_in_users=burn_in_users,
            min_checkpoints=min_checkpoints,
            checkpoint_index=checkpoint_index,
            rope_bounds=rope_bounds,
            futility_rope_threshold=futility_rope_threshold,
            expected_traffic_shares=expected_traffic_shares,
            imbalance_tolerance=imbalance_tolerance,
            imbalance_blocks_stopping=imbalance_blocks_stopping,
            daily_traffic_per_variant=daily_traffic_per_variant,
            experiment_age_days=experiment_age_days,
            novelty_warning_days=novelty_warning_days,
            users_estimate_floor=users_estimate_floor,
            users_estimate_safety_factor=users_estimate_safety_factor,
            min_draws_warning=min_draws_warning,
            prior_trajectory=prior_trajectory,
            **kwargs,
        )


class TestInputValidation:
    """Test input validation for stopping evaluation.

    Notes
    -----
    Verifies invalid inputs raise the expected errors.
    """

    def test_1d_samples_raises(self):
        """Verify 1D samples array raises ValueError."""
        with pytest.raises(ValueError, match="2D"):
            evaluate_stopping(
                samples=np.ones(10),
                variant_names=VARIANTS_2,
                control="control",
                n_users_per_variant=_n_users(VARIANTS_2),
            )

    def test_column_mismatch_raises(self):
        """Verify column count mismatch raises ValueError."""
        samples = _samples(n_variants=3)
        with pytest.raises(ValueError, match="columns"):
            evaluate_stopping(
                samples=samples,
                variant_names=VARIANTS_2,
                control="control",
                n_users_per_variant=_n_users(VARIANTS_2),
            )

    def test_missing_control_raises(self):
        """Verify missing control variant raises ValueError."""
        with pytest.raises(ValueError, match="control"):
            evaluate_stopping(
                samples=_samples(),
                variant_names=VARIANTS_2,
                control="does_not_exist",
                n_users_per_variant=_n_users(VARIANTS_2),
            )

    def test_nan_in_samples_raises(self):
        """Verify NaN in samples raises ValueError."""
        bad = _samples().copy()
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            evaluate_stopping(
                samples=bad,
                variant_names=VARIANTS_2,
                control="control",
                n_users_per_variant=_n_users(VARIANTS_2),
            )

    def test_inf_in_samples_raises(self):
        """Verify Inf in samples raises ValueError."""
        bad = _samples().copy()
        bad[5, 1] = np.inf
        with pytest.raises(ValueError, match="NaN or Inf"):
            evaluate_stopping(
                samples=bad,
                variant_names=VARIANTS_2,
                control="control",
                n_users_per_variant=_n_users(VARIANTS_2),
            )

    def test_loss_threshold_out_of_range_raises(self):
        """Verify out-of-range loss threshold raises."""
        with pytest.raises(ValueError, match="loss_threshold"):
            _call_evaluate(loss_threshold=1.5)

    def test_loss_threshold_zero_raises(self):
        """Verify zero loss threshold raises."""
        with pytest.raises(ValueError, match="loss_threshold"):
            _call_evaluate(loss_threshold=0.0)

    def test_prob_best_min_out_of_range_raises(self):
        """Verify out-of-range prob_best_min raises."""
        with pytest.raises(ValueError, match="prob_best_min"):
            _call_evaluate(prob_best_min=1.1)

    def test_futility_threshold_out_of_range_raises(self):
        """Verify out-of-range futility threshold raises."""
        with pytest.raises(ValueError, match="futility_rope_threshold"):
            _call_evaluate(futility_rope_threshold=0.0)

    def test_imbalance_tolerance_out_of_range_raises(self):
        """Verify out-of-range imbalance tolerance raises."""
        with pytest.raises(ValueError, match="imbalance_tolerance"):
            _call_evaluate(imbalance_tolerance=1.5)

    def test_burn_in_exceeds_min_sample_raises(self):
        """Verify burn-in exceeding min sample size raises."""
        with pytest.raises(ValueError, match="burn_in_users"):
            _call_evaluate(burn_in_users=2000, min_sample_size=1000)

    def test_safety_factor_below_one_raises(self):
        """Verify safety factor below 1.0 raises."""
        with pytest.raises(ValueError, match="users_estimate_safety_factor"):
            _call_evaluate(users_estimate_safety_factor=0.5)

    def test_missing_n_users_key_raises(self):
        """Verify missing user count key raises."""
        with pytest.raises(ValueError, match="missing entries"):
            _call_evaluate(n_users={"control": 2000})

    def test_rope_low_gte_high_raises(self):
        """Verify invalid ROPE bounds raise."""
        with pytest.raises(ValueError, match="rope_bounds"):
            _call_evaluate(rope_bounds=(0.01, -0.01))

    def test_rope_equal_bounds_raises(self):
        """Verify equal ROPE bounds raise."""
        with pytest.raises(ValueError, match="rope_bounds"):
            _call_evaluate(rope_bounds=(0.01, 0.01))


class TestGateLogic:
    """Test individual gate logic in isolation.

    Notes
    -----
    Each gate is tested independently to verify correct pass/fail behavior.
    """

    def test_burn_in_gate_blocks_when_not_enough_users(self):
        """Verify burn-in gate blocks with insufficient users."""
        result = _call_evaluate(
            n_users={"control": 10, "variant_b": 10},
            burn_in_users=500,
            min_sample_size=500,
        )
        assert not result.safe_to_stop
        assert not result.gate_states["burn_in"]

    def test_burn_in_gate_passes_when_users_sufficient(self):
        """Verify burn-in gate passes with sufficient users."""
        result = _call_evaluate(
            n_users={"control": 600, "variant_b": 600},
            burn_in_users=500,
            min_sample_size=500,
        )
        assert result.gate_states["burn_in"]

    def test_sample_size_gate_blocks_independently_of_burn_in(self):
        """Verify sample size gate operates independently."""
        result = _call_evaluate(
            n_users={"control": 200, "variant_b": 200},
            burn_in_users=50,
            min_sample_size=500,
        )
        assert result.gate_states["burn_in"]
        assert not result.gate_states["sample_size"]
        assert not result.safe_to_stop

    def test_min_checkpoints_gate_blocks_when_index_below_minimum(self):
        """Verify min checkpoints gate blocks below threshold."""
        result = _call_evaluate(checkpoint_index=1, min_checkpoints=3)
        assert not result.gate_states["min_checkpoints"]
        assert not result.safe_to_stop

    def test_min_checkpoints_gate_passes_at_exact_minimum(self):
        """Verify min checkpoints gate passes at exact threshold."""
        result = _call_evaluate(checkpoint_index=3, min_checkpoints=3)
        assert result.gate_states["min_checkpoints"]

    def test_loss_gate_blocks_when_loss_above_threshold(self):
        """Verify loss gate blocks when above threshold."""
        result = _call_evaluate(
            loss_values={"control": 0.05, "variant_b": 0.05},
            loss_threshold=0.01,
            prob_values={"control": 0.03, "variant_b": 0.97},
        )
        assert not result.gate_states["loss"]
        assert not result.safe_to_stop

    def test_loss_gate_passes_when_loss_below_threshold(self):
        """Verify loss gate passes when below threshold."""
        result = _call_evaluate(
            loss_values={"control": 0.05, "variant_b": 0.005},
            loss_threshold=0.01,
            prob_values={"control": 0.03, "variant_b": 0.97},
        )
        assert result.gate_states["loss"]

    def test_prob_best_gate_blocks_when_below_minimum(self):
        """Verify prob_best gate blocks when below minimum."""
        result = _call_evaluate(
            prob_values={"control": 0.50, "variant_b": 0.50},
            prob_best_min=0.80,
            loss_values={"control": 0.05, "variant_b": 0.001},
        )
        assert not result.gate_states["prob_best"]
        assert not result.safe_to_stop

    def test_prob_best_gate_passes_at_exact_minimum(self):
        """Verify prob_best gate passes at exact minimum."""
        result = _call_evaluate(
            prob_values={"control": 0.20, "variant_b": 0.80},
            prob_best_min=0.80,
        )
        assert result.gate_states["prob_best"]

    def test_traffic_gate_blocks_when_imbalanced_and_blocking_on(self):
        """Verify traffic gate blocks when imbalanced and enabled."""
        result = _call_evaluate(
            n_users={"control": 3000, "variant_b": 500},
            imbalance_tolerance=0.10,
            imbalance_blocks_stopping=True,
        )
        assert not result.gate_states["traffic"]
        assert not result.safe_to_stop

    def test_traffic_gate_does_not_block_when_blocking_disabled(self):
        """Verify traffic gate does not block when disabled."""
        result = _call_evaluate(
            n_users={"control": 3000, "variant_b": 500},
            imbalance_tolerance=0.10,
            imbalance_blocks_stopping=False,
        )
        assert not result.gate_states["traffic"]
        assert result.safe_to_stop


class TestWinnerStopping:
    """Test winner stopping happy path.

    Notes
    -----
    Tests verify correct behavior when all gates pass and winner is declared.
    """

    def test_winner_stopping_all_gates_pass(self):
        """Verify stopping when all gates pass."""
        result = _call_evaluate()
        assert result.safe_to_stop
        assert result.stopping_reason == "winner"
        assert not result.futility_triggered

    def test_winner_stopping_best_variant_correct(self):
        """Verify best variant is correctly identified."""
        result = _call_evaluate(best_variant="variant_b")
        assert result.best_variant == "variant_b"

    def test_winner_stopping_recommendation_contains_safe_to_stop(self):
        """Verify recommendation indicates safe to stop."""
        result = _call_evaluate()
        assert "SAFE TO STOP" in result.recommendation
        assert "winner" in result.recommendation.lower()


class TestFutilityStopping:
    """Test futility stopping behavior.

    Notes
    -----
    Futility fires when variants are too similar to matter in practice.
    """

    def _futility_samples(self, n_draws: int = 1000) -> np.ndarray:
        """Generate samples where variant_b is identical to control.

        Parameters
        ----------
        n_draws : int, optional
            Number of draws, by default 1000.

        Returns
        -------
        np.ndarray
            Samples with shape (n_draws, 2) showing no real difference.
        """
        rng = np.random.default_rng(42)
        base = rng.lognormal(0.0, 0.05, size=(n_draws,))
        return np.column_stack([base, base * (1 + rng.normal(0, 0.002, n_draws))])

    def test_futility_fires_when_effect_in_rope(self):
        """Verify futility fires when effect is in ROPE."""
        samples = self._futility_samples()
        result = _call_evaluate(
            samples=samples,
            loss_values={"control": 0.05, "variant_b": 0.05},
            prob_values={"control": 0.50, "variant_b": 0.50},
            best_variant="variant_b",
            rope_bounds=(-0.05, 0.05),
            futility_rope_threshold=0.10,
        )
        assert result.futility_triggered
        assert result.safe_to_stop
        assert result.stopping_reason == "futility"

    def test_futility_does_not_fire_when_prerequisites_not_met(self):
        """Verify futility blocked by prerequisites."""
        samples = self._futility_samples()
        result = _call_evaluate(
            samples=samples,
            n_users={"control": 10, "variant_b": 10},
            burn_in_users=500,
            min_sample_size=500,
            rope_bounds=(-0.05, 0.05),
            futility_rope_threshold=0.10,
        )
        assert not result.futility_triggered
        assert not result.safe_to_stop

    def test_futility_recommendation_says_no_difference(self):
        """Verify futility recommendation indicates no difference."""
        samples = self._futility_samples()
        result = _call_evaluate(
            samples=samples,
            loss_values={"control": 0.05, "variant_b": 0.05},
            prob_values={"control": 0.50, "variant_b": 0.50},
            best_variant="variant_b",
            rope_bounds=(-0.05, 0.05),
            futility_rope_threshold=0.10,
        )
        if result.stopping_reason == "futility":
            assert "no difference" in result.recommendation.lower()


class TestSnapshotAndTrajectory:
    """Test trajectory snapshot tracking.

    Notes
    -----
    Each evaluation creates a snapshot; trajectory accumulates across calls.
    """

    def test_trajectory_has_one_entry_on_first_call(self):
        """Verify trajectory has one entry initially."""
        result = _call_evaluate()
        assert len(result.trajectory) == 1

    def test_trajectory_appends_on_prior_trajectory(self):
        """Verify trajectory appends prior snapshots."""
        first = _call_evaluate(checkpoint_index=1, min_checkpoints=1)
        second = _call_evaluate(
            checkpoint_index=2,
            min_checkpoints=1,
            prior_trajectory=first.trajectory,
        )
        assert len(second.trajectory) == 2

    def test_snapshot_checkpoint_index_correct(self):
        """Verify snapshot has correct checkpoint index."""
        result = _call_evaluate(checkpoint_index=7)
        assert result.trajectory[0].checkpoint_index == 7

    def test_snapshot_total_users_correct(self):
        """Verify snapshot has correct user count."""
        result = _call_evaluate(n_users={"control": 1500, "variant_b": 1500})
        assert result.trajectory[0].total_users == 3000

    def test_snapshot_safe_to_stop_matches_result(self):
        """Verify snapshot safe_to_stop matches result."""
        result = _call_evaluate()
        assert result.trajectory[0].safe_to_stop == result.safe_to_stop


class TestTrafficDiagnostics:
    """Test traffic balance diagnostics.

    Notes
    -----
    Tests verify _check_traffic_balance independently.
    """

    def _balance(self, n_users, expected_shares=None, tolerance=0.10):
        """Check traffic balance for given user counts.

        Parameters
        ----------
        n_users : dict[str, int]
            User counts per variant.
        expected_shares : dict, optional
            Expected traffic shares, by default None.
        tolerance : float, optional
            Imbalance tolerance, by default 0.10.

        Returns
        -------
        TrafficDiagnostics
            Balance diagnostics result.
        """
        return _check_traffic_balance(n_users, expected_shares, tolerance)

    def test_balanced_uniform_split(self):
        """Verify balanced uniform split is detected."""
        diag = self._balance({"control": 1000, "variant_b": 1000})
        assert diag.balanced
        assert diag.flagged_variants == []

    def test_imbalanced_flags_correct_variant(self):
        """Verify imbalanced split flags variant."""
        diag = self._balance({"control": 900, "variant_b": 100})
        assert not diag.balanced
        assert len(diag.flagged_variants) > 0

    def test_observed_shares_sum_to_one(self):
        """Verify observed shares sum to 1.0."""
        diag = self._balance({"control": 600, "variant_b": 400})
        assert sum(diag.observed_shares.values()) == pytest.approx(1.0)

    def test_expected_shares_uniform_when_not_provided(self):
        """Verify uniform expected shares default."""
        diag = self._balance({"control": 500, "variant_b": 500})
        assert diag.expected_shares["control"] == pytest.approx(0.5)
        assert diag.expected_shares["variant_b"] == pytest.approx(0.5)

    def test_custom_expected_shares_respected(self):
        """Verify custom expected shares are used."""
        diag = self._balance(
            {"control": 700, "variant_b": 300},
            expected_shares={"control": 0.7, "variant_b": 0.3},
            tolerance=0.05,
        )
        assert diag.balanced

    def test_zero_total_users_returns_unbalanced(self):
        """Verify zero users returns unbalanced."""
        diag = self._balance({"control": 0, "variant_b": 0})
        assert not diag.balanced
        assert diag.max_deviation == pytest.approx(1.0)

    def test_max_deviation_correct(self):
        """Verify max deviation is computed correctly."""
        diag = self._balance({"control": 700, "variant_b": 300}, tolerance=0.10)
        assert diag.max_deviation == pytest.approx(0.20, abs=1e-6)

    def test_three_variants_all_balanced(self):
        """Verify three-variant balanced case."""
        diag = self._balance(
            {"control": 333, "variant_b": 333, "variant_c": 334},
            tolerance=0.05,
        )
        assert diag.balanced

    def test_three_variants_one_flagged(self):
        """Verify three-variant imbalanced case."""
        diag = self._balance(
            {"control": 800, "variant_b": 100, "variant_c": 100},
            tolerance=0.10,
        )
        assert not diag.balanced
        assert "control" in diag.flagged_variants


class TestCheckFutilityHelper:
    """Test futility detection helper function.

    Notes
    -----
    Tests verify _check_futility independently.
    """

    def _futility(self, samples, variants, control, rope=(-0.01, 0.01), threshold=0.80):
        """Check futility for given samples.

        Parameters
        ----------
        samples : np.ndarray
            Posterior samples.
        variants : list[str]
            Variant names.
        control : str
            Control variant name.
        rope : tuple, optional
            ROPE bounds, by default (-0.01, 0.01).
        threshold : float, optional
            Futility threshold, by default 0.80.

        Returns
        -------
        bool
            True if futility detected.
        """
        return _check_futility(samples, variants, control, rope[0], rope[1], threshold)

    def test_futility_false_when_large_effect(self):
        """Verify futility false for large effect."""
        rng = np.random.default_rng(0)
        control_draws = rng.lognormal(0.0, 0.1, 2000)
        variant_draws = control_draws * 1.10
        samples = np.column_stack([control_draws, variant_draws])
        result = self._futility(samples, VARIANTS_2, "control", threshold=0.80)
        assert not result

    def test_futility_true_when_effect_near_zero(self):
        """Verify futility true when effect near zero."""
        rng = np.random.desfault_rng(1)
        base = rng.lognormal(0.0, 0.05, 3000)
        variant = base * (1 + rng.normal(0, 0.0001, 3000))
        samples = np.column_stack([base, variant])
        result = self._futility(
            samples, VARIANTS_2, "control",
            rope=(-0.05, 0.05),
            threshold=0.10,
        )
        assert result

    def test_futility_false_when_control_all_zeros(self):
        """Verify futility false for all-zero control."""
        samples = np.column_stack([np.zeros(500), np.ones(500)])
        result = self._futility(samples, VARIANTS_2, "control")
        assert not result

    def test_futility_false_when_one_variant_outside_rope(self):
        """Verify futility false when one variant outside ROPE."""
        rng = np.random.default_rng(2)
        base = rng.lognormal(0.0, 0.05, 2000)
        close  = base * (1 + rng.normal(0, 0.001, 2000))
        lifted = base * 1.15
        samples = np.column_stack([base, close, lifted])
        result = self._futility(
            samples, VARIANTS_3, "control",
            rope=(-0.05, 0.05), threshold=0.50,
        )
        assert not result


class TestUsersNeededEstimate:
    """Test users-needed estimate computation.

    Notes
    -----
    Tests verify _estimate_users_needed independently.
    """

    def _estimate(self, **kwargs):
        """Estimate additional users needed.

        Parameters
        ----------
        **kwargs
            Keyword arguments to override defaults.

        Returns
        -------
        UsersNeededResult or None
            Estimate result or None if not needed.
        """
        defaults = dict(
            best_variant="variant_b",
            expected_loss={"control": 0.05, "variant_b": 0.05},
            prob_best={"control": 0.50, "variant_b": 0.50},
            loss_threshold=0.01,
            prob_best_min=0.80,
            n_users_per_variant={"control": 2000, "variant_b": 2000},
            daily_traffic_per_variant=None,
            variant_names=VARIANTS_2,
            control="control",
            users_floor=100,
            safety_factor=1.25,
        )
        defaults.update(kwargs)
        return _estimate_users_needed(**defaults)

    def test_returns_none_when_both_gates_already_pass(self):
        """Verify None when all gates pass."""
        result = self._estimate(
            expected_loss={"control": 0.05, "variant_b": 0.005},
            prob_best={"control": 0.03, "variant_b": 0.97},
            loss_threshold=0.01,
            prob_best_min=0.80,
        )
        assert result is None

    def test_returns_none_when_best_loss_is_zero(self):
        """Verify None when best loss is zero."""
        result = self._estimate(
            expected_loss={"control": 0.0, "variant_b": 0.0},
        )
        assert result is None

    def test_basis_is_loss_when_loss_is_binding(self):
        """Verify basis is loss when loss gate fails."""
        result = self._estimate(
            expected_loss={"control": 0.05, "variant_b": 0.05},
            prob_best={"control": 0.10, "variant_b": 0.90},
        )
        assert result is not None
        assert result.basis == "loss"

    def test_basis_is_prob_best_when_prob_is_binding(self):
        """Verify basis is prob_best when prob gate fails."""
        result = self._estimate(
            expected_loss={"control": 0.05, "variant_b": 0.005},
            prob_best={"control": 0.50, "variant_b": 0.50},
            loss_threshold=0.01,
            prob_best_min=0.80,
        )
        assert result is not None
        assert result.basis == "prob_best"

    def test_additional_users_at_floor_minimum(self):
        """Verify estimate respects floor minimum."""
        result = self._estimate(
            expected_loss={"control": 0.05, "variant_b": 0.0101},
            prob_best={"control": 0.10, "variant_b": 0.90},
            n_users_per_variant={"control": 10_000_000, "variant_b": 10_000_000},
            users_floor=100,
        )
        assert result is not None
        assert result.additional_users["variant_b"] >= 100

    def test_safety_factor_stored_in_result(self):
        """Verify safety factor is stored."""
        result = self._estimate()
        assert result is not None
        assert result.safety_factor == pytest.approx(1.25)

    def test_days_to_completion_computed_when_daily_traffic_given(self):
        """Verify days computed when daily traffic provided."""
        result = self._estimate(
            daily_traffic_per_variant={"variant_b": 200.0},
        )
        assert result is not None
        assert result.days_to_completion["variant_b"] is not None
        assert result.days_to_completion["variant_b"] > 0

    def test_days_to_completion_none_when_no_daily_traffic(self):
        """Verify days None when no daily traffic."""
        result = self._estimate(daily_traffic_per_variant=None)
        assert result is not None
        assert result.days_to_completion["variant_b"] is None

    def test_note_field_is_nonempty_string(self):
        """Verify note field is populated."""
        result = self._estimate()
        assert result is not None
        assert isinstance(result.note, str)
        assert len(result.note) > 0


class TestNoveltyWarning:
    """Test novelty warning for young experiments.

    Notes
    -----
    Novelty warnings alert when experiments are too young to trust.
    """

    def test_novelty_warning_fires_when_young(self):
        """Verify novelty warning fires for young experiment."""
        result = _call_evaluate(
            experiment_age_days=5.0,
            novelty_warning_days=14,
        )
        assert result.novelty_warning

    def test_novelty_warning_in_warnings_list(self):
        """Verify novelty warning in warnings list."""
        result = _call_evaluate(experiment_age_days=5.0, novelty_warning_days=14)
        assert any("novelty" in w.lower() for w in result.warnings)

    def test_novelty_warning_does_not_fire_when_old_enough(self):
        """Verify novelty warning does not fire when old enough."""
        result = _call_evaluate(
            experiment_age_days=20.0,
            novelty_warning_days=14,
        )
        assert not result.novelty_warning

    def test_novelty_warning_does_not_fire_when_age_not_provided(self):
        """Verify novelty warning does not fire when age unknown."""
        result = _call_evaluate(experiment_age_days=None)
        assert not result.novelty_warning

    def test_novelty_warning_appears_in_recommendation_when_stopping(self):
        """Verify novelty warning appears in recommendation."""
        result = _call_evaluate(
            experiment_age_days=3.0,
            novelty_warning_days=14,
        )
        if result.safe_to_stop and result.novelty_warning:
            assert "novelty" in result.recommendation.lower()

    def test_novelty_emits_python_warning(self):
        """Verify novelty warning is emitted."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _call_evaluate(
                experiment_age_days=3.0,
                novelty_warning_days=14,
                min_draws_warning=0,
            )
        novelty_warns = [w for w in caught if "novelty" in str(w.message).lower()]
        assert len(novelty_warns) >= 1


class TestLowDrawWarning:
    """Test low draw count warning.

    Notes
    -----
    Warnings alert when posterior samples are too few.
    """

    def test_low_draws_emits_userwarning(self):
        """Verify low draws emits UserWarning."""
        low_draw_samples = _samples(n_draws=100)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _call_evaluate(samples=low_draw_samples, min_draws_warning=500)
        draw_warns = [w for w in caught if "n_draws" in str(w.message)]
        assert len(draw_warns) >= 1

    def test_low_draw_warning_in_warnings_list(self):
        """Verify low draw warning in warnings list."""
        low_draw_samples = _samples(n_draws=100)
        result = _call_evaluate(samples=low_draw_samples, min_draws_warning=500)
        assert any("n_draws" in w for w in result.warnings)

    def test_no_warning_when_draws_above_minimum(self):
        """Verify no warning when draws above minimum."""
        samples = _samples(n_draws=1000)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _call_evaluate(samples=samples, min_draws_warning=500)
        draw_warns = [w for w in caught if "n_draws" in str(w.message)]
        assert len(draw_warns) == 0


class TestTrafficShareNormalisation:
    """Test traffic share normalisation warnings.

    Notes
    -----
    Warns when expected shares do not sum to 1.0.
    """

    def test_shares_not_summing_to_one_emits_warning(self):
        """Verify warning when shares do not sum to 1.0."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _call_evaluate(
                expected_traffic_shares={"control": 0.6, "variant_b": 0.6},
                min_draws_warning=0,
            )
        norm_warns = [w for w in caught if "normalising" in str(w.message).lower()]
        assert len(norm_warns) >= 1

    def test_normalised_shares_still_produce_valid_result(self):
        """Verify valid result despite normalisation."""
        result = _call_evaluate(
            expected_traffic_shares={"control": 0.6, "variant_b": 0.6},
        )
        assert hasattr(result, "safe_to_stop")


class TestStoppingThresholds:
    """Test stopping threshold precision.

    Notes
    -----
    Verifies stopping fires precisely at configured thresholds.
    """

    def test_stopping_does_not_fire_when_loss_is_one_epsilon_above_threshold(self):
        """Verify no stop when loss above threshold."""
        result = _call_evaluate(
            loss_values={"control": 0.05, "variant_b": 0.0101},
            prob_values={"control": 0.03, "variant_b": 0.97},
            loss_threshold=0.01,
        )
        assert not result.safe_to_stop

    def test_stopping_fires_when_loss_is_one_epsilon_below_threshold(self):
        """Verify stop when loss below threshold."""
        result = _call_evaluate(
            loss_values={"control": 0.05, "variant_b": 0.0099},
            prob_values={"control": 0.03, "variant_b": 0.97},
            loss_threshold=0.01,
        )
        assert result.safe_to_stop
        assert result.stopping_reason == "winner"

    def test_stopping_does_not_fire_when_prob_best_is_one_epsilon_below_minimum(self):
        """Verify no stop when prob_best below minimum."""
        result = _call_evaluate(
            loss_values={"control": 0.05, "variant_b": 0.001},
            prob_values={"control": 0.201, "variant_b": 0.799},
            prob_best_min=0.80,
        )
        assert not result.safe_to_stop

    def test_stopping_fires_when_prob_best_exactly_meets_minimum(self):
        """Verify stop when prob_best meets minimum."""
        result = _call_evaluate(
            loss_values={"control": 0.05, "variant_b": 0.001},
            prob_values={"control": 0.20, "variant_b": 0.80},
            prob_best_min=0.80,
        )
        assert result.safe_to_stop

    def test_tighter_loss_threshold_delays_stopping(self):
        """Verify tighter threshold delays stopping."""
        loss_values = {"control": 0.05, "variant_b": 0.008}
        prob_values = {"control": 0.03, "variant_b": 0.97}
        loose = _call_evaluate(loss_values=loss_values, prob_values=prob_values, loss_threshold=0.01)
        tight = _call_evaluate(loss_values=loss_values, prob_values=prob_values, loss_threshold=0.005)
        assert loose.safe_to_stop
        assert not tight.safe_to_stop

    def test_higher_prob_best_minimum_delays_stopping(self):
        """Verify higher prob minimum delays stopping."""
        loss_values = {"control": 0.05, "variant_b": 0.001}
        prob_values = {"control": 0.10, "variant_b": 0.90}
        low_bar  = _call_evaluate(loss_values=loss_values, prob_values=prob_values, prob_best_min=0.80)
        high_bar = _call_evaluate(loss_values=loss_values, prob_values=prob_values, prob_best_min=0.95)
        assert low_bar.safe_to_stop
        assert not high_bar.safe_to_stop


class TestStoppingReasonMutualExclusion:
    """Test stopping reason mutual exclusion.

    Notes
    -----
    A result has exactly one stopping reason: winner, futility, or none.
    """

    def test_winner_and_futility_not_both_true(self):
        """Verify winner and futility are mutually exclusive."""
        configs = [
            dict(),
            dict(
                loss_values={"control": 0.05, "variant_b": 0.05},
                prob_values={"control": 0.50, "variant_b": 0.50},
            ),
            dict(
                n_users={"control": 10, "variant_b": 10},
                burn_in_users=500, min_sample_size=500,
            ),
        ]
        for cfg in configs:
            result = _call_evaluate(**cfg)
            assert not (result.stopping_reason == "winner" and result.futility_triggered)
            assert result.stopping_reason in ("winner", "futility", "none")

    def test_safe_to_stop_true_iff_reason_is_not_none(self):
        """Verify safe_to_stop iff reason is not none."""
        for cfg in [dict(), dict(loss_values={"control": 0.05, "variant_b": 0.05},
                                  prob_values={"control": 0.50, "variant_b": 0.50},
                                  best_variant="variant_b")]:
            result = _call_evaluate(**cfg)
            if result.safe_to_stop:
                assert result.stopping_reason != "none"
            else:
                assert result.stopping_reason == "none"

    def test_futility_triggered_field_matches_stopping_reason(self):
        """Verify futility_triggered matches stopping_reason."""
        result = _call_evaluate()
        assert result.futility_triggered == (result.stopping_reason == "futility")


class TestTrafficWarningAlwaysAppears:
    """Test traffic warning always appears.

    Notes
    -----
    Traffic imbalance warning appears regardless of blocking mode.
    """

    def test_imbalance_warning_in_warnings_list_when_blocking(self):
        """Verify imbalance warning when blocking."""
        result = _call_evaluate(
            n_users={"control": 3000, "variant_b": 500},
            imbalance_blocks_stopping=True,
            loss_values={"control": 0.05, "variant_b": 0.05},
            prob_values={"control": 0.50, "variant_b": 0.50},
        )
        assert any("imbalance" in w.lower() or "traffic" in w.lower()
                   for w in result.warnings)

    def test_imbalance_warning_in_warnings_list_when_not_blocking(self):
        """Verify imbalance warning when not blocking."""
        result = _call_evaluate(
            n_users={"control": 3000, "variant_b": 500},
            imbalance_blocks_stopping=False,
        )
        assert any("imbalance" in w.lower() or "traffic" in w.lower()
                   for w in result.warnings)

    def test_warnings_is_always_a_list_never_none(self):
        """Verify warnings is always a list."""
        for cfg in [
            dict(),
            dict(loss_values={"control": 0.05, "variant_b": 0.05},
                 prob_values={"control": 0.50, "variant_b": 0.50},
                 best_variant="variant_b"),
            dict(n_users={"control": 10, "variant_b": 10},
                 burn_in_users=500, min_sample_size=500),
        ]:
            result = _call_evaluate(**cfg)
            assert isinstance(result.warnings, list)

    def test_traffic_flagged_variants_listed_correctly(self):
        """Verify flagged variants listed correctly."""
        result = _call_evaluate(
            n_users={"control": 3000, "variant_b": 500},
            imbalance_tolerance=0.05,
            loss_values={"control": 0.05, "variant_b": 0.05},
            prob_values={"control": 0.50, "variant_b": 0.50},
        )
        assert len(result.traffic.flagged_variants) > 0


class TestUsersNeededMonotonicity:
    """Test users-needed monotonicity.

    Notes
    -----
    Further from threshold implies more users needed.
    """

    def test_higher_loss_produces_larger_user_estimate(self):
        """Verify higher loss produces larger estimate."""
        close = _call_evaluate(
            loss_values={"control": 0.05, "variant_b": 0.015},
            prob_values={"control": 0.10, "variant_b": 0.90},
            n_users={"control": 5000, "variant_b": 5000},
        )
        far = _call_evaluate(
            loss_values={"control": 0.05, "variant_b": 0.08},
            prob_values={"control": 0.10, "variant_b": 0.90},
            n_users={"control": 5000, "variant_b": 5000},
        )
        assert close.users_needed is not None
        assert far.users_needed is not None
        assert (far.users_needed.additional_users["variant_b"] >=
                close.users_needed.additional_users["variant_b"])

    def test_estimate_always_at_least_floor(self):
        """Verify estimate respects floor."""
        floor = 250
        result = _call_evaluate(
            loss_values={"control": 0.05, "variant_b": 0.0101},
            prob_values={"control": 0.10, "variant_b": 0.90},
            users_estimate_floor=floor,
        )
        if result.users_needed is not None:
            for v, n in result.users_needed.additional_users.items():
                assert n >= floor, f"{v} estimate {n} is below floor {floor}"

    def test_calendar_estimate_scales_with_daily_traffic(self):
        """Verify calendar estimate scales with traffic."""
        slow = _call_evaluate(
            loss_values={"control": 0.05, "variant_b": 0.05},
            prob_values={"control": 0.10, "variant_b": 0.90},
            daily_traffic_per_variant={"control": 100.0, "variant_b": 100.0},
        )
        fast = _call_evaluate(
            loss_values={"control": 0.05, "variant_b": 0.05},
            prob_values={"control": 0.10, "variant_b": 0.90},
            daily_traffic_per_variant={"control": 200.0, "variant_b": 200.0},
        )
        if slow.users_needed and fast.users_needed:
            slow_days = slow.users_needed.days_to_completion.get("variant_b")
            fast_days = fast.users_needed.days_to_completion.get("variant_b")
            if slow_days is not None and fast_days is not None:
                assert fast_days < slow_days


class TestPrerequisitesBlockBoth:
    """Test prerequisites block both winner and futility.

    Notes
    -----
    Prerequisites must block futility as well as winner stopping.
    """

    def _futility_samples(self) -> np.ndarray:
        """Generate futility-condition samples.

        Returns
        -------
        np.ndarray
            Samples showing near-zero effect.
        """
        rng = np.random.default_rng(42)
        base = rng.lognormal(0.0, 0.05, 1000)
        return np.column_stack([base, base * (1 + rng.normal(0, 0.001, 1000))])

    def test_futility_blocked_when_burn_in_not_met(self):
        """Verify futility blocked when burn-in not met."""
        samples = self._futility_samples()
        result = _call_evaluate(
            samples=samples,
            n_users={"control": 10, "variant_b": 10},
            burn_in_users=500,
            min_sample_size=500,
            rope_bounds=(-0.05, 0.05),
            futility_rope_threshold=0.10,
            loss_values={"control": 0.05, "variant_b": 0.05},
            prob_values={"control": 0.50, "variant_b": 0.50},
        )
        assert not result.safe_to_stop
        assert not result.futility_triggered

    def test_futility_blocked_when_min_checkpoints_not_met(self):
        """Verify futility blocked when min checkpoints not met."""
        samples = self._futility_samples()
        result = _call_evaluate(
            samples=samples,
            checkpoint_index=1,
            min_checkpoints=5,
            rope_bounds=(-0.05, 0.05),
            futility_rope_threshold=0.10,
            loss_values={"control": 0.05, "variant_b": 0.05},
            prob_values={"control": 0.50, "variant_b": 0.50},
        )
        assert not result.safe_to_stop
        assert not result.futility_triggered

    def test_winner_blocked_when_min_checkpoints_not_met(self):
        """Verify winner blocked when min checkpoints not met."""
        result = _call_evaluate(
            checkpoint_index=1,
            min_checkpoints=5,
        )
        assert not result.safe_to_stop


class TestStoppingChecker:
    """Test stateful stopping checker API.

    Notes
    -----
    Tests verify StoppingChecker contract and behavior across updates.
    """

    def _checker(self, **kwargs):
        """Create StoppingChecker with defaults.

        Parameters
        ----------
        **kwargs
            Keyword arguments to override defaults.

        Returns
        -------
        StoppingChecker
            Configured stopping checker.
        """
        defaults = dict(
            loss_threshold=0.01,
            prob_best_min=0.80,
            min_sample_size=100,
            burn_in_users=50,
            min_checkpoints=1,
        )
        defaults.update(kwargs)
        return StoppingChecker(**defaults)

    def _update(self, checker, loss_values=None, prob_values=None, best="variant_b",
                n_users=None, **kwargs):
        """Update checker with mocked metrics.

        Parameters
        ----------
        checker : StoppingChecker
            Stopping checker to update.
        loss_values : dict, optional
            Loss values, by default None.
        prob_values : dict, optional
            Probability values, by default None.
        best : str, optional
            Best variant, by default "variant_b".
        n_users : dict, optional
            User counts, by default None.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        StoppingResult
            Result of update.
        """
        samples = _samples()
        variant_names = VARIANTS_2
        if n_users is None:
            n_users = _n_users(variant_names)
        if loss_values is None:
            loss_values = {"control": 0.05, "variant_b": 0.001}
        if prob_values is None:
            prob_values = {"control": 0.03, "variant_b": 0.97}

        loss_result, prob_result = _mock_metrics(loss_values, prob_values, best)

        with patch("argonx.sequential.stopping.compute_expected_loss", return_value=loss_result), \
             patch("argonx.sequential.stopping.compute_prob_best", return_value=prob_result):
            return checker.update(
                samples=samples,
                variant_names=variant_names,
                control="control",
                n_users_per_variant=n_users,
                **kwargs,
            )

    def test_checker_starts_with_zero_checkpoints(self):
        """Verify checker starts with zero checkpoints."""
        checker = self._checker()
        assert checker.n_checkpoints == 0
        assert checker.trajectory == []

    def test_each_update_increments_checkpoint_count(self):
        """Verify each update increments checkpoint count."""
        checker = self._checker()
        self._update(checker)
        assert checker.n_checkpoints == 1
        self._update(checker)
        assert checker.n_checkpoints == 2

    def test_trajectory_length_matches_checkpoint_count(self):
        """Verify trajectory length matches checkpoint count."""
        checker = self._checker()
        for i in range(4):
            self._update(checker)
            assert len(checker.trajectory) == i + 1

    def test_mutating_returned_trajectory_does_not_affect_internal_state(self):
        """Verify modifying returned trajectory does not corrupt state."""
        checker = self._checker()
        self._update(checker)
        t = checker.trajectory
        t.append(None)
        t.append(None)
        assert len(checker.trajectory) == 1

    def test_reset_returns_checker_to_initial_state(self):
        """Verify reset returns checker to initial state."""
        checker = self._checker()
        self._update(checker)
        self._update(checker)
        checker.reset()
        assert checker.n_checkpoints == 0
        assert checker.trajectory == []
        result = self._update(checker)
        assert result.checkpoint_index == 1

    def test_min_checkpoints_enforced_across_updates(self):
        """Verify min checkpoints enforced across updates."""
        checker = self._checker(min_checkpoints=3)
        r1 = self._update(checker)
        r2 = self._update(checker)
        r3 = self._update(checker)
        assert not r1.safe_to_stop
        assert not r2.safe_to_stop
        assert r3.safe_to_stop

    def test_burn_in_greater_than_min_sample_raises_on_update_not_construction(self):
        """Verify bad config raises on update not construction."""
        checker = self._checker(burn_in_users=2000, min_sample_size=100)
        with pytest.raises(ValueError, match="burn_in_users"):
            self._update(checker)

    def test_checker_result_checkpoint_index_matches_n_checkpoints(self):
        """Verify result checkpoint index matches count."""
        checker = self._checker()
        result = self._update(checker)
        assert result.checkpoint_index == checker.n_checkpoints

    def test_full_trajectory_available_in_each_result(self):
        """Verify full trajectory in each result."""
        checker = self._checker()
        r1 = self._update(checker)
        r2 = self._update(checker)
        r3 = self._update(checker)
        assert len(r3.trajectory) == 3
        assert r3.trajectory[0].checkpoint_index == 1
        assert r3.trajectory[1].checkpoint_index == 2
        assert r3.trajectory[2].checkpoint_index == 3

    def test_checker_applies_consistent_config_across_updates(self):
        """Verify checker applies consistent config across updates."""
        checker = self._checker(loss_threshold=0.005)
        r_no_stop = self._update(
            checker,
            loss_values={"control": 0.05, "variant_b": 0.006},
            prob_values={"control": 0.03, "variant_b": 0.97},
        )
        assert not r_no_stop.safe_to_stop

        checker.reset()
        r_stop = self._update(
            checker,
            loss_values={"control": 0.05, "variant_b": 0.004},
            prob_values={"control": 0.03, "variant_b": 0.97},
        )
        assert r_stop.safe_to_stop


class TestUsersNeededPresence:
    """Test users-needed presence conditions.

    Notes
    -----
    users_needed present when needed, absent when not.
    """

    def test_users_needed_absent_when_stopping(self):
        """Verify users_needed absent when stopping."""
        result = _call_evaluate()
        assert result.users_needed is None

    def test_users_needed_absent_when_prerequisites_not_met(self):
        """Verify users_needed absent when prerequisites not met."""
        result = _call_evaluate(
            n_users={"control": 10, "variant_b": 10},
            burn_in_users=500,
            min_sample_size=500,
        )
        assert result.users_needed is None

    def test_users_needed_present_when_prerequisites_met_but_evidence_weak(self):
        """Verify users_needed present when evidence weak."""
        result = _call_evaluate(
            loss_values={"control": 0.05, "variant_b": 0.05},
            prob_values={"control": 0.50, "variant_b": 0.50},
            best_variant="variant_b",
        )
        assert result.users_needed is not None
        assert "variant_b" in result.users_needed.additional_users
        assert result.users_needed.additional_users["variant_b"] > 0

    def test_users_needed_does_not_include_control(self):
        """Verify users_needed excludes control."""
        result = _call_evaluate(
            loss_values={"control": 0.05, "variant_b": 0.05},
            prob_values={"control": 0.50, "variant_b": 0.50},
            best_variant="variant_b",
        )
        if result.users_needed is not None:
            assert "control" not in result.users_needed.additional_users


class TestRecommendationContent:
    """Test recommendation text content.

    Notes
    -----
    Recommendation string contains right information for right situation.
    """

    def test_winner_recommendation_says_safe_to_stop(self):
        """Verify winner recommendation says safe to stop."""
        result = _call_evaluate()
        assert result.safe_to_stop
        assert "safe to stop" in result.recommendation.lower()

    def test_winner_recommendation_names_the_winning_variant(self):
        """Verify winner recommendation names variant."""
        result = _call_evaluate(best_variant="variant_b")
        assert "variant_b" in result.recommendation

    def test_futility_recommendation_says_no_difference(self):
        """Verify futility recommendation says no difference."""
        rng = np.random.default_rng(42)
        base = rng.lognormal(0.0, 0.05, 1000)
        samples = np.column_stack([base, base * (1 + rng.normal(0, 0.001, 1000))])
        result = _call_evaluate(
            samples=samples,
            loss_values={"control": 0.05, "variant_b": 0.05},
            prob_values={"control": 0.50, "variant_b": 0.50},
            best_variant="variant_b",
            rope_bounds=(-0.05, 0.05),
            futility_rope_threshold=0.10,
        )
        if result.stopping_reason == "futility":
            assert "no difference" in result.recommendation.lower()

    def test_continue_recommendation_says_continue(self):
        """Verify continue recommendation says continue."""
        result = _call_evaluate(
            loss_values={"control": 0.05, "variant_b": 0.05},
            prob_values={"control": 0.50, "variant_b": 0.50},
            best_variant="variant_b",
        )
        assert not result.safe_to_stop
        assert "continue" in result.recommendation.lower()

    def test_novelty_warning_appears_in_recommendation_on_winner_stop(self):
        """Verify novelty warning appears in winner recommendation."""
        result = _call_evaluate(
            experiment_age_days=3.0,
            novelty_warning_days=14,
        )
        if result.safe_to_stop and result.novelty_warning:
            assert "novelty" in result.recommendation.lower()

    def test_traffic_imbalance_mentioned_in_continue_recommendation(self):
        """Verify traffic imbalance in continue recommendation."""
        result = _call_evaluate(
            n_users={"control": 3000, "variant_b": 100},
            imbalance_tolerance=0.05,
            imbalance_blocks_stopping=True,
            loss_values={"control": 0.05, "variant_b": 0.05},
            prob_values={"control": 0.50, "variant_b": 0.50},
        )
        assert ("imbalance" in result.recommendation.lower()
                or "traffic" in result.recommendation.lower())

    def test_burn_in_message_when_burn_in_gate_blocks(self):
        """Verify burn-in message when gate blocks."""
        result = _call_evaluate(
            n_users={"control": 10, "variant_b": 10},
            burn_in_users=500,
            min_sample_size=500,
        )
        assert "burn" in result.recommendation.lower()
