"""
Bayesian sequential stopping rules for the argonx decision engine.

Bayesian sequential testing is mathematically valid at any point in time.
Frequentist statistics breaks if you peek early — the false positive rate
inflates. Bayesian expected-loss-based stopping does not have this problem.
Stopping occurs when the evidence is strong enough, not when a sample size
target is hit.

Production design notes
-----------------------
This module mirrors what Spotify, Netflix, and Airbnb actually implement in
their experimentation platforms. Key additions over a naive implementation:

1. Burn-in gate — data quality concern. Early experiment data is polluted by
   novelty effects, cache warming, and bot traffic. Inference on day-1 data
   produces unreliable posteriors. The stopping rule refuses to evaluate until
   burn_in_users are reached per variant.

2. Minimum sample size gate — statistical power concern, separate from burn-in.
   Even after burn-in, a stopping signal on 200 users is unreliable. A hard
   floor ensures the posterior has enough observations to contract meaningfully.

3. Futility stopping — symmetric with winner stopping. If enough data has
   accumulated and the effect is consistently inside the ROPE, the correct
   call is to stop and record no difference — not to run indefinitely.

4. Traffic imbalance detection — if variant_b is receiving 30% of traffic
   instead of the intended 50%, inference is biased. The rule flags this and
   can optionally block stopping until balance is restored.

5. Novelty effect warning — behavioral metrics (CTR, session length, revenue)
   inflate in the first 1-2 weeks as users react to change. The rule warns
   when the experiment is younger than a configurable threshold.

6. Calendar-time estimates — "2,400 more users needed" is not actionable alone.
   "At current traffic rate, that is approximately 4 days" is. The rule
   optionally converts user estimates to calendar days per variant.

7. Per-variant estimates — variants often have unequal traffic splits. The
   estimate is computed per variant from its own n and rate, not uniformly.

No values are hardcoded inside logic functions. All thresholds live in
module-level constants (overridable via function arguments) or are passed
explicitly. The safety factor on user estimates is a named constant.

Stopping criteria
-----------------
Safe to stop (winner) when ALL hold:
    - burn_in_users reached per variant
    - min_sample_size reached per variant
    - min_checkpoints evaluations done
    - expected_loss of best variant < loss_threshold
    - P(best) of best variant >= prob_best_min
    - traffic balanced, OR imbalance_blocks_stopping=False

Safe to stop (futility) when ALL hold:
    - same prerequisites as winner
    - P(effect inside ROPE) >= futility_rope_threshold for ALL variants

Typical usage
-------------
    from argonx.sequential.stopping import evaluate_stopping, StoppingChecker

    # Stateless: single checkpoint
    result = evaluate_stopping(
        samples=posterior_samples,
        variant_names=["control", "variant_b"],
        control="control",
        n_users_per_variant={"control": 5000, "variant_b": 5000},
    )
    print(result.safe_to_stop, result.stopping_reason)
    print(result.recommendation)

    # Stateful: trajectory across checkpoints
    checker = StoppingChecker(
        loss_threshold=0.01,
        min_sample_size=1000,
        burn_in_users=500,
        daily_traffic_per_variant={"control": 500, "variant_b": 500},
    )
    for day, (samples, n_users) in enumerate(daily_data, start=1):
        result = checker.update(
            samples, variant_names, control, n_users,
            experiment_age_days=float(day),
        )
        if result.safe_to_stop:
            break
    fig = checker.plot_trajectory()
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from argonx.decision_rules.metrics import compute_expected_loss, compute_prob_best


# ---------------------------------------------------------------------------
# Module-level defaults — all overridable via function/class arguments.
# Nothing is hardcoded inside logic. These are the named defaults only.
# ---------------------------------------------------------------------------

_DEFAULT_LOSS_THRESHOLD       = 0.01   # expected loss below which stopping fires
_DEFAULT_PROB_BEST_MIN        = 0.80   # minimum P(best) before stopping is considered
_DEFAULT_MIN_SAMPLE_SIZE      = 1000   # hard floor per variant — statistical power gate
_DEFAULT_BURN_IN_USERS        = 500    # data quality floor — early data rejected below this
_DEFAULT_MIN_CHECKPOINTS      = 3      # minimum looks before stopping is allowed
_DEFAULT_FUTILITY_THRESHOLD   = 0.80   # P(inside ROPE) for ALL variants triggers futility
_DEFAULT_IMBALANCE_TOLERANCE  = 0.10   # max deviation from expected traffic share
_DEFAULT_NOVELTY_DAYS         = 14     # warn if experiment is younger than this
_DEFAULT_ROPE_BOUNDS          = (-0.01, 0.01)  # ROPE region for futility, ±1% default

# Estimation internals — named so they are auditable, not buried
_USERS_ESTIMATE_FLOOR         = 100    # never return "0 more users needed"
_USERS_ESTIMATE_SAFETY_FACTOR = 1.25   # conservative multiplier on projection
_MIN_DRAWS_WARNING            = 500    # warn when posterior has fewer draws than this


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TrafficDiagnostics:
    """
    Traffic health check at a single checkpoint.

    balanced : bool
        True when all variants are within imbalance_tolerance of their
        expected share. False means inference may be biased.
    observed_shares : dict[str, float]
        Fraction of total traffic each variant actually received.
    expected_shares : dict[str, float]
        Fraction each variant should receive. Uniform (1/n) when not provided.
    max_deviation : float
        Largest absolute deviation between observed and expected share.
    imbalance_tolerance : float
        The configured tolerance threshold.
    flagged_variants : list[str]
        Variants whose observed share deviates beyond tolerance.
    """
    balanced: bool
    observed_shares: dict[str, float]
    expected_shares: dict[str, float]
    max_deviation: float
    imbalance_tolerance: float
    flagged_variants: list[str]


@dataclass
class UsersNeededEstimate:
    """
    Per-variant estimate of additional users needed at current effect size.

    additional_users : dict[str, int]
        Approximate additional users needed per non-control variant.
    days_to_completion : dict[str, float | None]
        Approximate days at current traffic rate per variant.
        None when daily_traffic_per_variant was not provided.
    basis : str
        Which gate is the binding constraint — "loss" or "prob_best".
    safety_factor : float
        The multiplier applied to the raw projection. Exposed so the user
        knows the estimate is conservative, not exact.
    note : str
        Plain-English caveat.
    """
    additional_users: dict[str, int]
    days_to_completion: dict[str, Optional[float]]
    basis: str
    safety_factor: float
    note: str


@dataclass
class CheckpointSnapshot:
    """
    Complete evidence state at a single checkpoint.

    Stored in StoppingResult.trajectory. The full list of these is what
    plot_trajectory() renders. Each field maps directly to a named gate
    so it is always clear exactly which gate blocked or allowed stopping.
    """
    checkpoint_index: int
    n_users_per_variant: dict[str, int]
    total_users: int
    expected_loss: dict[str, float]
    prob_best: dict[str, float]
    best_variant: str

    # Gate states — one bool per gate, named to match gate_states dict
    burn_in_complete: bool
    sample_size_reached: bool
    min_checkpoints_reached: bool
    loss_below_threshold: bool
    prob_best_sufficient: bool
    traffic_balanced: bool

    # Outcome
    safe_to_stop: bool
    stopping_reason: Literal["winner", "futility", "none"]
    futility_triggered: bool


@dataclass
class StoppingResult:
    """
    Full output of a sequential stopping evaluation.

    safe_to_stop : bool
        True when stopping is warranted — winner found or futility declared.
        Check stopping_reason to distinguish.
    stopping_reason : "winner" | "futility" | "none"
        "winner"   — loss < threshold and P(best) sufficient.
        "futility" — effect consistently inside ROPE across all variants.
        "none"     — continue running.
    best_variant : str
        Variant with highest P(best) at this checkpoint.
    expected_loss : dict[str, float]
        Current expected loss per variant.
    prob_best : dict[str, float]
        Current P(best) per variant via simultaneous argmax.
    loss_threshold : float
        The configured stopping threshold. Stored for downstream display.
    gate_states : dict[str, bool]
        Which gates passed and which blocked stopping. Useful for debugging
        why stopping did not fire when expected.
        Keys: burn_in, sample_size, min_checkpoints, loss, prob_best, traffic.
    traffic : TrafficDiagnostics
        Full traffic health check at this checkpoint.
    users_needed : UsersNeededEstimate | None
        Per-variant estimate when not safe to stop. None when stopping.
    novelty_warning : bool
        True when experiment_age_days < novelty_warning_days.
    futility_triggered : bool
        True when futility fired (same as stopping_reason == "futility").
        Kept as a separate field for programmatic access without string check.
    recommendation : str
        Plain-English recommendation with all context included.
    checkpoint_index : int
        Ordinal index of this checkpoint (1-based).
    trajectory : list[CheckpointSnapshot]
        Full history of all evaluated checkpoints including this one.
    warnings : list[str]
        Non-fatal issues flagged during evaluation.
    """
    safe_to_stop: bool
    stopping_reason: Literal["winner", "futility", "none"]
    best_variant: str
    expected_loss: dict[str, float]
    prob_best: dict[str, float]
    loss_threshold: float
    gate_states: dict[str, bool]
    traffic: TrafficDiagnostics
    users_needed: Optional[UsersNeededEstimate]
    novelty_warning: bool
    futility_triggered: bool
    recommendation: str
    checkpoint_index: int
    trajectory: list[CheckpointSnapshot] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Traffic diagnostics
# ---------------------------------------------------------------------------

def _check_traffic_balance(
    n_users_per_variant: dict[str, int],
    expected_shares: Optional[dict[str, float]],
    imbalance_tolerance: float,
) -> TrafficDiagnostics:
    """
    Check whether traffic is distributed as expected across variants.

    Assumes uniform split when expected_shares is None. A variant is flagged
    when its observed share deviates from expected by more than
    imbalance_tolerance. All inputs come from function arguments — no
    module-level defaults are accessed here.
    """
    n_variants = len(n_users_per_variant)
    total = sum(n_users_per_variant.values())

    if total == 0:
        uniform = {v: 1.0 / n_variants for v in n_users_per_variant}
        return TrafficDiagnostics(
            balanced=False,
            observed_shares={v: 0.0 for v in n_users_per_variant},
            expected_shares=expected_shares or uniform,
            max_deviation=1.0,
            imbalance_tolerance=imbalance_tolerance,
            flagged_variants=list(n_users_per_variant.keys()),
        )

    observed = {v: n / total for v, n in n_users_per_variant.items()}
    expected = expected_shares or {v: 1.0 / n_variants for v in n_users_per_variant}

    deviations = {
        v: abs(observed[v] - expected.get(v, 1.0 / n_variants))
        for v in n_users_per_variant
    }
    max_dev = max(deviations.values())
    flagged = [v for v, dev in deviations.items() if dev > imbalance_tolerance]

    return TrafficDiagnostics(
        balanced=len(flagged) == 0,
        observed_shares=observed,
        expected_shares=expected,
        max_deviation=max_dev,
        imbalance_tolerance=imbalance_tolerance,
        flagged_variants=flagged,
    )


# ---------------------------------------------------------------------------
# Futility detection
# ---------------------------------------------------------------------------

def _check_futility(
    samples: np.ndarray,
    variant_names: list[str],
    control: str,
    rope_low: float,
    rope_high: float,
    futility_rope_threshold: float,
) -> bool:
    """
    Return True when the experiment should be stopped for futility.

    Futility fires when ALL non-control variants show P(effect inside ROPE)
    >= futility_rope_threshold. The posterior is concentrated in the region
    of practical equivalence — no meaningful winner expected with more data.

    Lift computed draw-by-draw as (variant - control) / |control|, matching
    the definition in metrics.py. All thresholds come from arguments — no
    module constants accessed here.

    Returns False (do not stop) when control draws are all zero, since
    relative lift is undefined in that case.
    """
    control_idx = variant_names.index(control)
    control_draws = samples[:, control_idx]

    nonzero_mask = control_draws != 0
    if not nonzero_mask.any():
        return False

    for i, v in enumerate(variant_names):
        if v == control:
            continue

        variant_draws = samples[:, i]
        lift = (
            (variant_draws[nonzero_mask] - control_draws[nonzero_mask])
            / np.abs(control_draws[nonzero_mask])
        )
        prob_inside = float(np.mean((lift >= rope_low) & (lift <= rope_high)))

        if prob_inside < futility_rope_threshold:
            return False  # This variant might still win

    return True  # All variants are practically equivalent


# ---------------------------------------------------------------------------
# Users-needed estimate
# ---------------------------------------------------------------------------

def _estimate_users_needed(
    best_variant: str,
    expected_loss: dict[str, float],
    prob_best: dict[str, float],
    loss_threshold: float,
    prob_best_min: float,
    n_users_per_variant: dict[str, int],
    daily_traffic_per_variant: Optional[dict[str, float]],
    variant_names: list[str],
    control: str,
    users_floor: int,
    safety_factor: float,
) -> Optional[UsersNeededEstimate]:
    """
    Compute per-variant estimates of additional users needed.

    Uses posterior contraction scaling: expected loss ~ 1/sqrt(n), which
    means n_target = n_current * (current_loss / threshold)^2. Applies
    safety_factor as a conservative multiplier. Converts to calendar days
    when daily_traffic_per_variant is provided.

    Binding constraint is whichever gate is furthest from passing — loss
    threshold or P(best) minimum. The basis field records which one drove
    the estimate.

    All thresholds and floors come from arguments. users_floor and
    safety_factor are passed in from the caller (which gets them from
    module constants or user overrides) — not accessed directly here.

    Returns None when the effect is zero across all variants.
    """
    best_loss = expected_loss.get(best_variant, float("inf"))
    best_p = prob_best.get(best_variant, 0.0)

    loss_gap = best_loss - loss_threshold   # positive = loss too high
    prob_gap = prob_best_min - best_p       # positive = P(best) too low

    if loss_gap <= 0 and prob_gap <= 0:
        return None  # Both gates already passing

    if best_loss <= 0:
        return None  # Zero loss — cannot project

    additional_per_variant: dict[str, int] = {}
    days_per_variant: dict[str, Optional[float]] = {}
    basis: str

    for v in variant_names:
        if v == control:
            continue

        n_current = n_users_per_variant.get(v, 0)
        if n_current <= 0:
            additional_per_variant[v] = users_floor
            days_per_variant[v] = None
            continue

        if loss_gap > 0:
            # Loss is the binding constraint — use 1/sqrt(n) projection
            ratio = best_loss / loss_threshold
            n_target = n_current * (ratio ** 2)
            additional = max(
                int(np.ceil((n_target - n_current) * safety_factor)),
                users_floor,
            )
            basis = "loss"
        else:
            # P(best) is binding — harder to project, use fractional shortfall
            shortfall_fraction = prob_gap / max(best_p, 0.01)
            additional = max(
                int(np.ceil(n_current * shortfall_fraction * safety_factor)),
                users_floor,
            )
            basis = "prob_best"

        additional_per_variant[v] = additional

        if daily_traffic_per_variant and v in daily_traffic_per_variant:
            rate = daily_traffic_per_variant[v]
            days_per_variant[v] = round(additional / max(rate, 1), 1) if rate > 0 else None
        else:
            days_per_variant[v] = None

    if not additional_per_variant:
        return None

    return UsersNeededEstimate(
        additional_users=additional_per_variant,
        days_to_completion=days_per_variant,
        basis=basis,
        safety_factor=safety_factor,
        note=(
            f"Projection uses 1/sqrt(n) posterior contraction with {safety_factor}x "
            "safety factor. Assumes current effect size holds — treat as "
            "order-of-magnitude guidance only."
        ),
    )


# ---------------------------------------------------------------------------
# Recommendation builder
# ---------------------------------------------------------------------------

def _build_recommendation(
    safe_to_stop: bool,
    stopping_reason: str,
    best_variant: str,
    expected_loss: dict[str, float],
    loss_threshold: float,
    prob_best: dict[str, float],
    prob_best_min: float,
    gate_states: dict[str, bool],
    traffic: TrafficDiagnostics,
    users_needed: Optional[UsersNeededEstimate],
    novelty_warning: bool,
    novelty_warning_days: int,
    experiment_age_days: Optional[float],
) -> str:
    """
    Generate a detailed plain-English recommendation.

    All values come from arguments — this function has no side effects and
    accesses no module-level state. Recommendation is fully determined by
    what is passed in.
    """
    best_loss = expected_loss.get(best_variant, float("inf"))
    best_p = prob_best.get(best_variant, 0.0)
    parts = []

    # --- Stopping outcomes ---
    if safe_to_stop and stopping_reason == "winner":
        parts.append(
            f"SAFE TO STOP — winner found. '{best_variant}' has expected loss "
            f"{best_loss:.4f} (threshold {loss_threshold:.4f}) "
            f"and P(best) = {best_p:.3f}. Evidence is sufficient."
        )
        if novelty_warning:
            age_str = f"{experiment_age_days:.1f}" if experiment_age_days is not None else "unknown"
            parts.append(
                f"NOTE: Novelty warning active (experiment age: {age_str} days, "
                f"threshold: {novelty_warning_days} days). Verify effect persists "
                "beyond novelty period before shipping."
            )
        return " ".join(parts)

    if safe_to_stop and stopping_reason == "futility":
        parts.append(
            "SAFE TO STOP — futility declared. All variants show high probability "
            "of being practically equivalent. No meaningful winner expected "
            "with additional data. Record as no difference."
        )
        return " ".join(parts)

    # --- Not stopping — explain which gate(s) blocked ---
    parts.append(
        f"CONTINUE EXPERIMENT. '{best_variant}' leads "
        f"with P(best) = {best_p:.3f}."
    )

    if not gate_states["burn_in"]:
        parts.append("Burn-in period not yet complete — data quality gate active.")
    elif not gate_states["sample_size"]:
        parts.append("Minimum sample size not yet reached — statistical power gate active.")
    elif not gate_states["min_checkpoints"]:
        parts.append("Minimum checkpoint count not yet reached.")
    else:
        if not gate_states["loss"]:
            parts.append(
                f"Expected loss ({best_loss:.4f}) has not dropped below "
                f"threshold ({loss_threshold:.4f})."
            )
        if not gate_states["prob_best"]:
            parts.append(
                f"P(best) ({best_p:.3f}) has not cleared minimum ({prob_best_min:.3f})."
            )

    if not gate_states["traffic"]:
        flagged_str = ", ".join(traffic.flagged_variants)
        parts.append(
            f"Traffic imbalance on: {flagged_str}. "
            f"Max deviation from expected: {traffic.max_deviation:.1%}. "
            "Inference may be biased — investigate before trusting results."
        )

    if novelty_warning:
        age_str = f"{experiment_age_days:.1f}" if experiment_age_days is not None else "unknown"
        parts.append(
            f"Novelty warning: experiment is {age_str} days old "
            f"(threshold: {novelty_warning_days} days). "
            "Behavioral metrics may not yet reflect steady-state behavior."
        )

    if users_needed is not None:
        estimate_parts = []
        for v, n in sorted(users_needed.additional_users.items()):
            days = users_needed.days_to_completion.get(v)
            if days is not None:
                estimate_parts.append(f"{v}: ~{n:,} users (~{days} days)")
            else:
                estimate_parts.append(f"{v}: ~{n:,} users")
        if estimate_parts:
            parts.append(
                f"Estimated additional users needed ({users_needed.basis}-driven, "
                f"{users_needed.safety_factor}x safety factor): "
                + ", ".join(estimate_parts) + ". "
                + users_needed.note
            )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Core stateless evaluation
# ---------------------------------------------------------------------------

def evaluate_stopping(
    samples: np.ndarray,
    variant_names: list[str],
    control: str,
    n_users_per_variant: dict[str, int],
    loss_threshold: float = _DEFAULT_LOSS_THRESHOLD,
    prob_best_min: float = _DEFAULT_PROB_BEST_MIN,
    min_sample_size: int = _DEFAULT_MIN_SAMPLE_SIZE,
    burn_in_users: int = _DEFAULT_BURN_IN_USERS,
    min_checkpoints: int = _DEFAULT_MIN_CHECKPOINTS,
    checkpoint_index: int = 1,
    rope_bounds: tuple[float, float] = _DEFAULT_ROPE_BOUNDS,
    futility_rope_threshold: float = _DEFAULT_FUTILITY_THRESHOLD,
    expected_traffic_shares: Optional[dict[str, float]] = None,
    imbalance_tolerance: float = _DEFAULT_IMBALANCE_TOLERANCE,
    imbalance_blocks_stopping: bool = True,
    daily_traffic_per_variant: Optional[dict[str, float]] = None,
    experiment_age_days: Optional[float] = None,
    novelty_warning_days: int = _DEFAULT_NOVELTY_DAYS,
    users_estimate_floor: int = _USERS_ESTIMATE_FLOOR,
    users_estimate_safety_factor: float = _USERS_ESTIMATE_SAFETY_FACTOR,
    min_draws_warning: int = _MIN_DRAWS_WARNING,
    prior_trajectory: Optional[list[CheckpointSnapshot]] = None,
) -> StoppingResult:
    """
    Evaluate whether it is safe to stop the experiment at this checkpoint.

    Stateless — evaluates a single checkpoint. For stateful trajectory
    accumulation across multiple checkpoints, use StoppingChecker.

    Parameters
    ----------
    samples : np.ndarray
        Posterior samples. Shape (n_draws, n_variants). Columns must align
        with variant_names in sorted order — same contract as BaseModel.
    variant_names : list[str]
        Sorted variant names. Must match samples column order.
    control : str
        Name of the control variant.
    n_users_per_variant : dict[str, int]
        Current observed user counts per variant. Must include all variants.
    loss_threshold : float
        Stop when best variant's expected loss falls below this. Default 0.01.
    prob_best_min : float
        Minimum P(best) before stopping is considered. Default 0.80.
    min_sample_size : int
        Hard minimum observations per variant before stopping is evaluated.
        Statistical power gate. Default 1000.
    burn_in_users : int
        Data quality floor. Stopping refused until this many users per variant.
        Must be <= min_sample_size. Default 500.
    min_checkpoints : int
        Minimum evaluations before stopping is allowed regardless of evidence.
        Guards against stopping on a lucky first look. Default 3.
    checkpoint_index : int
        1-based ordinal index of this checkpoint. Managed by StoppingChecker
        when using the stateful API. Pass explicitly when calling directly.
    rope_bounds : tuple[float, float]
        ROPE region in relative lift units for futility detection.
        Default (-0.01, 0.01) = ±1%.
    futility_rope_threshold : float
        P(inside ROPE) >= this for ALL variants triggers futility stop.
        Default 0.80.
    expected_traffic_shares : dict[str, float] | None
        Expected fraction of total traffic per variant. Must sum to 1.0.
        Uniform split assumed when None.
    imbalance_tolerance : float
        Maximum acceptable deviation from expected traffic share. Default 0.10.
    imbalance_blocks_stopping : bool
        If True, traffic imbalance blocks safe_to_stop even when evidence
        thresholds are met. Default True.
    daily_traffic_per_variant : dict[str, float] | None
        Average daily users per variant. Converts user estimates to days.
    experiment_age_days : float | None
        Days since experiment launched. Drives novelty warning.
    novelty_warning_days : int
        Experiments younger than this trigger novelty warning. Default 14.
    users_estimate_floor : int
        Minimum value returned for any per-variant user estimate. Default 100.
    users_estimate_safety_factor : float
        Multiplier applied to raw projection. Default 1.25. Exposed so users
        can inspect or override the conservatism of the estimate.
    min_draws_warning : int
        Warn when posterior has fewer draws than this. Default 500.
    prior_trajectory : list[CheckpointSnapshot] | None
        Snapshots from previous checkpoints. Appended to in the returned result.

    Returns
    -------
    StoppingResult

    Raises
    ------
    ValueError
        Invalid samples shape, missing control, mismatched n_users keys,
        invalid threshold values, or burn_in_users > min_sample_size.
    """
    collected_warnings: list[str] = []

    # -----------------------------------------------------------------------
    # Input validation
    # -----------------------------------------------------------------------

    if samples.ndim != 2:
        raise ValueError(
            f"samples must be 2D (n_draws, n_variants), got shape {samples.shape}"
        )

    if samples.shape[1] != len(variant_names):
        raise ValueError(
            f"samples has {samples.shape[1]} columns but "
            f"{len(variant_names)} variant names provided."
        )

    if control not in variant_names:
        raise ValueError(
            f"control '{control}' not found in variant_names: {variant_names}"
        )

    if not np.isfinite(samples).all():
        raise ValueError("samples contain NaN or Inf values. Check model output.")

    if not 0 < loss_threshold < 1:
        raise ValueError(f"loss_threshold must be in (0, 1), got {loss_threshold}")

    if not 0 < prob_best_min < 1:
        raise ValueError(f"prob_best_min must be in (0, 1), got {prob_best_min}")

    if not 0 < futility_rope_threshold < 1:
        raise ValueError(
            f"futility_rope_threshold must be in (0, 1), got {futility_rope_threshold}"
        )

    if not 0 < imbalance_tolerance < 1:
        raise ValueError(
            f"imbalance_tolerance must be in (0, 1), got {imbalance_tolerance}"
        )

    if burn_in_users > min_sample_size:
        raise ValueError(
            f"burn_in_users ({burn_in_users}) must be <= min_sample_size "
            f"({min_sample_size}). Burn-in is a data quality floor; "
            "min_sample_size is the statistical power floor."
        )

    if users_estimate_safety_factor < 1.0:
        raise ValueError(
            f"users_estimate_safety_factor must be >= 1.0, "
            f"got {users_estimate_safety_factor}. "
            "A factor below 1.0 produces optimistic (underestimated) projections."
        )

    missing = set(variant_names) - set(n_users_per_variant.keys())
    if missing:
        raise ValueError(
            f"n_users_per_variant missing entries for: {sorted(missing)}"
        )

    rope_low, rope_high = rope_bounds
    if rope_low >= rope_high:
        raise ValueError(
            f"rope_bounds must satisfy low < high, got ({rope_low}, {rope_high})"
        )

    if samples.shape[0] < min_draws_warning:
        msg = (
            f"n_draws={samples.shape[0]} is below recommended minimum "
            f"({min_draws_warning}). Stopping decisions on thin posteriors "
            "may be unreliable."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        collected_warnings.append(msg)

    if expected_traffic_shares is not None:
        share_sum = sum(expected_traffic_shares.values())
        if abs(share_sum - 1.0) > 0.01:
            msg = (
                f"expected_traffic_shares sum to {share_sum:.3f}, not 1.0. "
                "Normalising automatically."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
            collected_warnings.append(msg)
            expected_traffic_shares = {
                v: s / share_sum for v, s in expected_traffic_shares.items()
            }

    # -----------------------------------------------------------------------
    # Gate 1 — Burn-in (data quality)
    # -----------------------------------------------------------------------

    min_users_seen = min(n_users_per_variant.get(v, 0) for v in variant_names)
    burn_in_complete = min_users_seen >= burn_in_users

    if not burn_in_complete:
        msg = (
            f"Burn-in not complete: {min_users_seen} / {burn_in_users} "
            "users per variant. Posterior metrics computed but stopping is blocked."
        )
        collected_warnings.append(msg)

    # -----------------------------------------------------------------------
    # Gate 2 — Minimum sample size (statistical power)
    # -----------------------------------------------------------------------

    sample_size_reached = min_users_seen >= min_sample_size

    # -----------------------------------------------------------------------
    # Gate 3 — Minimum checkpoints
    # -----------------------------------------------------------------------

    min_checkpoints_reached = checkpoint_index >= min_checkpoints

    # -----------------------------------------------------------------------
    # Traffic diagnostics — always run, used for warnings even when not blocking
    # -----------------------------------------------------------------------

    traffic = _check_traffic_balance(
        n_users_per_variant=n_users_per_variant,
        expected_shares=expected_traffic_shares,
        imbalance_tolerance=imbalance_tolerance,
    )

    if not traffic.balanced:
        flagged_str = ", ".join(traffic.flagged_variants)
        msg = (
            f"Traffic imbalance on: {flagged_str}. "
            f"Max deviation: {traffic.max_deviation:.1%}. "
            "Inference may be biased."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        collected_warnings.append(msg)

    # -----------------------------------------------------------------------
    # Novelty warning
    # -----------------------------------------------------------------------

    novelty_warning = (
        experiment_age_days is not None
        and experiment_age_days < novelty_warning_days
    )

    if novelty_warning:
        msg = (
            f"Novelty warning: experiment is {experiment_age_days:.1f} days old "
            f"(threshold: {novelty_warning_days} days). "
            "Behavioral metrics may be inflated by novelty effects."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        collected_warnings.append(msg)

    # -----------------------------------------------------------------------
    # Core metric computation — reuses metrics layer directly, no reimplementation
    # -----------------------------------------------------------------------

    loss_result = compute_expected_loss(samples, variant_names, control)
    prob_best_result = compute_prob_best(samples, variant_names)

    best_variant = prob_best_result.best_variant
    best_loss = loss_result.expected_loss[best_variant]
    best_p = prob_best_result.probabilities[best_variant]

    # -----------------------------------------------------------------------
    # Gate 4 — Loss threshold
    # -----------------------------------------------------------------------

    loss_below_threshold = best_loss < loss_threshold

    # -----------------------------------------------------------------------
    # Gate 5 — P(best) minimum
    # -----------------------------------------------------------------------

    prob_best_sufficient = best_p >= prob_best_min

    # -----------------------------------------------------------------------
    # Gate 6 — Traffic balance (can be configured to warn-only)
    # -----------------------------------------------------------------------

    traffic_gate_passed = traffic.balanced or not imbalance_blocks_stopping

    # -----------------------------------------------------------------------
    # Prerequisites — gates 1-3 and 6 must all pass before evidence is checked
    # -----------------------------------------------------------------------

    prerequisites_met = (
        burn_in_complete
        and sample_size_reached
        and min_checkpoints_reached
        and traffic_gate_passed
    )

    # -----------------------------------------------------------------------
    # Winner stopping
    # -----------------------------------------------------------------------

    winner_stopping = (
        prerequisites_met
        and loss_below_threshold
        and prob_best_sufficient
    )

    # -----------------------------------------------------------------------
    # Futility stopping — separate signal, same prerequisites
    # -----------------------------------------------------------------------

    futility_triggered = prerequisites_met and _check_futility(
        samples=samples,
        variant_names=variant_names,
        control=control,
        rope_low=rope_low,
        rope_high=rope_high,
        futility_rope_threshold=futility_rope_threshold,
    )

    safe_to_stop = winner_stopping or futility_triggered

    if winner_stopping:
        stopping_reason: Literal["winner", "futility", "none"] = "winner"
    elif futility_triggered:
        stopping_reason = "futility"
    else:
        stopping_reason = "none"

    # -----------------------------------------------------------------------
    # Gate states — one dict for programmatic access and display
    # -----------------------------------------------------------------------

    gate_states = {
        "burn_in":          burn_in_complete,
        "sample_size":      sample_size_reached,
        "min_checkpoints":  min_checkpoints_reached,
        "loss":             loss_below_threshold,
        "prob_best":        prob_best_sufficient,
        "traffic":          traffic.balanced,
    }

    # -----------------------------------------------------------------------
    # Users-needed estimate — only when prerequisites met and not stopping
    # -----------------------------------------------------------------------

    users_needed: Optional[UsersNeededEstimate] = None

    if not safe_to_stop and prerequisites_met:
        users_needed = _estimate_users_needed(
            best_variant=best_variant,
            expected_loss=loss_result.expected_loss,
            prob_best=prob_best_result.probabilities,
            loss_threshold=loss_threshold,
            prob_best_min=prob_best_min,
            n_users_per_variant=n_users_per_variant,
            daily_traffic_per_variant=daily_traffic_per_variant,
            variant_names=variant_names,
            control=control,
            users_floor=users_estimate_floor,
            safety_factor=users_estimate_safety_factor,
        )

    # -----------------------------------------------------------------------
    # Snapshot
    # -----------------------------------------------------------------------

    total_users = sum(n_users_per_variant.values())

    snapshot = CheckpointSnapshot(
        checkpoint_index=checkpoint_index,
        n_users_per_variant=dict(n_users_per_variant),
        total_users=total_users,
        expected_loss=dict(loss_result.expected_loss),
        prob_best=dict(prob_best_result.probabilities),
        best_variant=best_variant,
        burn_in_complete=burn_in_complete,
        sample_size_reached=sample_size_reached,
        min_checkpoints_reached=min_checkpoints_reached,
        loss_below_threshold=loss_below_threshold,
        prob_best_sufficient=prob_best_sufficient,
        traffic_balanced=traffic.balanced,
        safe_to_stop=safe_to_stop,
        stopping_reason=stopping_reason,
        futility_triggered=futility_triggered,
    )

    trajectory = list(prior_trajectory or [])
    trajectory.append(snapshot)

    # -----------------------------------------------------------------------
    # Recommendation
    # -----------------------------------------------------------------------

    recommendation = _build_recommendation(
        safe_to_stop=safe_to_stop,
        stopping_reason=stopping_reason,
        best_variant=best_variant,
        expected_loss=loss_result.expected_loss,
        loss_threshold=loss_threshold,
        prob_best=prob_best_result.probabilities,
        prob_best_min=prob_best_min,
        gate_states=gate_states,
        traffic=traffic,
        users_needed=users_needed,
        novelty_warning=novelty_warning,
        novelty_warning_days=novelty_warning_days,
        experiment_age_days=experiment_age_days,
    )

    return StoppingResult(
        safe_to_stop=safe_to_stop,
        stopping_reason=stopping_reason,
        best_variant=best_variant,
        expected_loss=loss_result.expected_loss,
        prob_best=prob_best_result.probabilities,
        loss_threshold=loss_threshold,
        gate_states=gate_states,
        traffic=traffic,
        users_needed=users_needed,
        novelty_warning=novelty_warning,
        futility_triggered=futility_triggered,
        recommendation=recommendation,
        checkpoint_index=checkpoint_index,
        trajectory=trajectory,
        warnings=collected_warnings,
    )


# ---------------------------------------------------------------------------
# Stateful checker
# ---------------------------------------------------------------------------

class StoppingChecker:
    """
    Stateful wrapper around evaluate_stopping() for multi-checkpoint workflows.

    All configuration is set once at construction and applied consistently
    across every call to update(). Call plot_trajectory() at any point to
    visualise evidence accumulation across all checkpoints so far.

    Parameters
    ----------
    loss_threshold : float
        Stop when best variant's expected loss falls below this. Default 0.01.
    prob_best_min : float
        Minimum P(best) before stopping is considered. Default 0.80.
    min_sample_size : int
        Hard minimum users per variant before stopping fires. Default 1000.
    burn_in_users : int
        Data quality floor per variant. Must be <= min_sample_size. Default 500.
    min_checkpoints : int
        Minimum looks before stopping is allowed. Default 3.
    rope_bounds : tuple[float, float]
        ROPE region for futility detection. Default (-0.01, 0.01).
    futility_rope_threshold : float
        P(inside ROPE) threshold for futility. Default 0.80.
    expected_traffic_shares : dict[str, float] | None
        Expected traffic split. Uniform if None.
    imbalance_tolerance : float
        Max acceptable deviation from expected share. Default 0.10.
    imbalance_blocks_stopping : bool
        Imbalance blocks stopping when True. Default True.
    daily_traffic_per_variant : dict[str, float] | None
        Daily traffic per variant for calendar estimates.
    novelty_warning_days : int
        Warn when experiment is younger than this. Default 14.
    users_estimate_floor : int
        Minimum per-variant user estimate. Default 100.
    users_estimate_safety_factor : float
        Conservatism multiplier on user projections. Default 1.25.
    min_draws_warning : int
        Warn when posterior draws fall below this. Default 500.

    Example
    -------
        checker = StoppingChecker(
            loss_threshold=0.01,
            min_sample_size=1000,
            burn_in_users=500,
            daily_traffic_per_variant={"control": 500, "variant_b": 500},
        )

        for day, (samples, n_users) in enumerate(daily_data, start=1):
            result = checker.update(
                samples=samples,
                variant_names=["control", "variant_b"],
                control="control",
                n_users_per_variant=n_users,
                experiment_age_days=float(day),
            )
            print(result.recommendation)
            if result.safe_to_stop:
                print(f"Stopping: {result.stopping_reason}")
                break

        fig = checker.plot_trajectory()
        fig.savefig("trajectory.png", dpi=150, bbox_inches="tight")
    """

    def __init__(
        self,
        loss_threshold: float = _DEFAULT_LOSS_THRESHOLD,
        prob_best_min: float = _DEFAULT_PROB_BEST_MIN,
        min_sample_size: int = _DEFAULT_MIN_SAMPLE_SIZE,
        burn_in_users: int = _DEFAULT_BURN_IN_USERS,
        min_checkpoints: int = _DEFAULT_MIN_CHECKPOINTS,
        rope_bounds: tuple[float, float] = _DEFAULT_ROPE_BOUNDS,
        futility_rope_threshold: float = _DEFAULT_FUTILITY_THRESHOLD,
        expected_traffic_shares: Optional[dict[str, float]] = None,
        imbalance_tolerance: float = _DEFAULT_IMBALANCE_TOLERANCE,
        imbalance_blocks_stopping: bool = True,
        daily_traffic_per_variant: Optional[dict[str, float]] = None,
        novelty_warning_days: int = _DEFAULT_NOVELTY_DAYS,
        users_estimate_floor: int = _USERS_ESTIMATE_FLOOR,
        users_estimate_safety_factor: float = _USERS_ESTIMATE_SAFETY_FACTOR,
        min_draws_warning: int = _MIN_DRAWS_WARNING,
    ) -> None:
        self.loss_threshold = loss_threshold
        self.prob_best_min = prob_best_min
        self.min_sample_size = min_sample_size
        self.burn_in_users = burn_in_users
        self.min_checkpoints = min_checkpoints
        self.rope_bounds = rope_bounds
        self.futility_rope_threshold = futility_rope_threshold
        self.expected_traffic_shares = expected_traffic_shares
        self.imbalance_tolerance = imbalance_tolerance
        self.imbalance_blocks_stopping = imbalance_blocks_stopping
        self.daily_traffic_per_variant = daily_traffic_per_variant
        self.novelty_warning_days = novelty_warning_days
        self.users_estimate_floor = users_estimate_floor
        self.users_estimate_safety_factor = users_estimate_safety_factor
        self.min_draws_warning = min_draws_warning

        self._trajectory: list[CheckpointSnapshot] = []
        self._checkpoint_index: int = 0

    @property
    def trajectory(self) -> list[CheckpointSnapshot]:
        """Read-only copy of all checkpoints evaluated so far."""
        return list(self._trajectory)

    @property
    def n_checkpoints(self) -> int:
        """Number of checkpoints evaluated so far."""
        return self._checkpoint_index

    def update(
        self,
        samples: np.ndarray,
        variant_names: list[str],
        control: str,
        n_users_per_variant: dict[str, int],
        experiment_age_days: Optional[float] = None,
    ) -> StoppingResult:
        """
        Evaluate the current checkpoint and update the internal trajectory.

        Parameters
        ----------
        samples : np.ndarray
            Current posterior samples. Shape (n_draws, n_variants).
        variant_names : list[str]
            Sorted variant names matching samples column order.
        control : str
            Name of the control variant.
        n_users_per_variant : dict[str, int]
            Current observed user counts per variant.
        experiment_age_days : float | None
            Days since experiment launched. Used for novelty warning.

        Returns
        -------
        StoppingResult
        """
        self._checkpoint_index += 1

        result = evaluate_stopping(
            samples=samples,
            variant_names=variant_names,
            control=control,
            n_users_per_variant=n_users_per_variant,
            loss_threshold=self.loss_threshold,
            prob_best_min=self.prob_best_min,
            min_sample_size=self.min_sample_size,
            burn_in_users=self.burn_in_users,
            min_checkpoints=self.min_checkpoints,
            checkpoint_index=self._checkpoint_index,
            rope_bounds=self.rope_bounds,
            futility_rope_threshold=self.futility_rope_threshold,
            expected_traffic_shares=self.expected_traffic_shares,
            imbalance_tolerance=self.imbalance_tolerance,
            imbalance_blocks_stopping=self.imbalance_blocks_stopping,
            daily_traffic_per_variant=self.daily_traffic_per_variant,
            experiment_age_days=experiment_age_days,
            novelty_warning_days=self.novelty_warning_days,
            users_estimate_floor=self.users_estimate_floor,
            users_estimate_safety_factor=self.users_estimate_safety_factor,
            min_draws_warning=self.min_draws_warning,
            prior_trajectory=self._trajectory,
        )

        self._trajectory = result.trajectory
        return result

    def reset(self) -> None:
        """Clear trajectory and reset checkpoint counter."""
        self._trajectory = []
        self._checkpoint_index = 0

    def plot_trajectory(
        self,
        figsize: tuple[int, int] = (13, 10),
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot evidence accumulation across all evaluated checkpoints.

        Three panels:
            Top    — P(best) per variant. Dashed line at prob_best_min.
            Middle — Expected loss per variant. Red dashed line at
                     loss_threshold. Green band = safe-to-stop zone.
            Bottom — Gate status heatmap. Each gate shown as green (passed)
                     or red (blocked) at each checkpoint. Lets you see exactly
                     which gate blocked stopping and when each gate cleared.

        A vertical green dotted line marks the first checkpoint where
        safe_to_stop was True, labelled with the stopping_reason.

        Returns
        -------
        plt.Figure
        """
        if not self._trajectory:
            raise RuntimeError(
                "No checkpoints recorded. Call update() at least once before plotting."
            )

        _PALETTE = ["#2563EB", "#16A34A", "#DC2626", "#9333EA", "#EA580C", "#0891B2"]
        _PASS_C  = "#BBF7D0"
        _FAIL_C  = "#FECACA"
        _GRID_A  = 0.18

        snapshots = self._trajectory
        total_users = [s.total_users for s in snapshots]
        use_users_x = len(set(total_users)) > 1
        x_vals = total_users if use_users_x else [s.checkpoint_index for s in snapshots]
        x_label = "Total users" if use_users_x else "Checkpoint"

        all_variants = list(snapshots[-1].prob_best.keys())

        fig, (ax_prob, ax_loss, ax_gates) = plt.subplots(
            3, 1, figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [2, 2, 1.4]},
        )
        fig.suptitle(
            suptitle or "Sequential Stopping — Evidence Trajectory",
            fontsize=14, fontweight="bold",
        )

        # --- Panel 1: P(best) ---
        for i, v in enumerate(all_variants):
            colour = _PALETTE[i % len(_PALETTE)]
            y = [s.prob_best.get(v, 0.0) for s in snapshots]
            ax_prob.plot(x_vals, y, marker="o", markersize=4,
                         linewidth=1.8, color=colour, label=v, zorder=3)

        ax_prob.axhline(
            self.prob_best_min, color="#6B7280", linewidth=1.2, linestyle="--",
            label=f"P(best) min = {self.prob_best_min:.2f}", zorder=2,
        )
        ax_prob.set_ylim(-0.02, 1.05)
        ax_prob.set_ylabel("P(variant is best)", fontsize=10)
        ax_prob.set_title("P(best) — simultaneous N-variant argmax", fontsize=10)
        ax_prob.legend(framealpha=0.9, fontsize=9, loc="upper left")
        ax_prob.grid(axis="both", alpha=_GRID_A)
        ax_prob.spines["top"].set_visible(False)
        ax_prob.spines["right"].set_visible(False)

        # --- Panel 2: Expected loss ---
        for i, v in enumerate(all_variants):
            colour = _PALETTE[i % len(_PALETTE)]
            y = [s.expected_loss.get(v, 0.0) for s in snapshots]
            ax_loss.plot(x_vals, y, marker="o", markersize=4,
                         linewidth=1.8, color=colour, label=v, zorder=3)

        ax_loss.axhline(
            self.loss_threshold, color="#DC2626", linewidth=1.4, linestyle="--",
            label=f"Loss threshold = {self.loss_threshold:.3f}", zorder=4,
        )
        ax_loss.axhspan(
            0.0, self.loss_threshold,
            color=_PASS_C, alpha=0.3, zorder=1, label="Safe-to-stop zone",
        )
        ax_loss.set_ylabel("Expected loss", fontsize=10)
        ax_loss.set_title(
            "Expected loss — stop when best variant drops below threshold", fontsize=10,
        )
        ax_loss.legend(framealpha=0.9, fontsize=9, loc="upper right")
        ax_loss.grid(axis="both", alpha=_GRID_A)
        ax_loss.spines["top"].set_visible(False)
        ax_loss.spines["right"].set_visible(False)

        # --- Panel 3: Gate heatmap ---
        gate_keys = [
            "burn_in", "sample_size", "min_checkpoints",
            "loss", "prob_best", "traffic",
        ]
        gate_labels = [
            "Burn-in", "Sample size", "Min checkpoints",
            "Loss < threshold", "P(best) ≥ min", "Traffic balance",
        ]

        gate_matrix = np.zeros((len(gate_keys), len(snapshots)))
        for j, snap in enumerate(snapshots):
            gate_vals = {
                "burn_in":          snap.burn_in_complete,
                "sample_size":      snap.sample_size_reached,
                "min_checkpoints":  snap.min_checkpoints_reached,
                "loss":             snap.loss_below_threshold,
                "prob_best":        snap.prob_best_sufficient,
                "traffic":          snap.traffic_balanced,
            }
            for i, gk in enumerate(gate_keys):
                gate_matrix[i, j] = 1.0 if gate_vals[gk] else 0.0

        cmap = ListedColormap([_FAIL_C, _PASS_C])
        ax_gates.imshow(
            gate_matrix,
            aspect="auto",
            cmap=cmap,
            vmin=0, vmax=1,
            extent=[x_vals[0], x_vals[-1], -0.5, len(gate_keys) - 0.5],
            origin="lower",
            zorder=2,
        )
        ax_gates.set_yticks(range(len(gate_keys)))
        ax_gates.set_yticklabels(gate_labels, fontsize=8)
        ax_gates.set_xlabel(x_label, fontsize=10)
        ax_gates.set_title("Gate status — green = passed, red = blocked", fontsize=10)
        ax_gates.spines["top"].set_visible(False)
        ax_gates.spines["right"].set_visible(False)

        pass_patch = mpatches.Patch(color=_PASS_C, label="Gate passed")
        fail_patch = mpatches.Patch(color=_FAIL_C, label="Gate blocked")
        ax_gates.legend(handles=[pass_patch, fail_patch], fontsize=8,
                        loc="upper left", framealpha=0.9)

        # --- Vertical stop marker ---
        stop_snaps = [s for s in snapshots if s.safe_to_stop]
        if stop_snaps:
            first_stop = stop_snaps[0]
            stop_x = first_stop.total_users if use_users_x else first_stop.checkpoint_index
            for ax in (ax_prob, ax_loss, ax_gates):
                ax.axvline(
                    stop_x, color="#16A34A", linewidth=1.5,
                    linestyle=":", alpha=0.85, zorder=5,
                )
            ax_prob.annotate(
                f"Stop: {first_stop.stopping_reason}",
                xy=(stop_x, 0.5),
                xycoords=("data", "axes fraction"),
                xytext=(6, 0), textcoords="offset points",
                fontsize=8, color="#16A34A", fontweight="bold",
            )

        fig.tight_layout()
        return fig