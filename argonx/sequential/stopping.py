from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from argonx.decision_rules.metrics import compute_expected_loss, compute_prob_best


_DEFAULT_LOSS_THRESHOLD       = 0.01   
_DEFAULT_PROB_BEST_MIN        = 0.80  
_DEFAULT_MIN_SAMPLE_SIZE      = 1000 
_DEFAULT_BURN_IN_USERS        = 500 
_DEFAULT_MIN_CHECKPOINTS      = 3      
_DEFAULT_FUTILITY_THRESHOLD   = 0.80   
_DEFAULT_IMBALANCE_TOLERANCE  = 0.10   
_DEFAULT_NOVELTY_DAYS         = 14     
_DEFAULT_ROPE_BOUNDS          = (-0.01, 0.01)  

_USERS_ESTIMATE_FLOOR         = 100  
_USERS_ESTIMATE_SAFETY_FACTOR = 1.25 
_MIN_DRAWS_WARNING            = 500    

@dataclass
class TrafficDiagnostics:
    """Traffic health check at a single checkpoint."""
    balanced: bool
    observed_shares: dict[str, float]
    expected_shares: dict[str, float]
    max_deviation: float
    imbalance_tolerance: float
    flagged_variants: list[str]


@dataclass
class UsersNeededEstimate:
    """Per-variant estimate of additional users needed at current effect size."""
    additional_users: dict[str, int]
    days_to_completion: dict[str, Optional[float]]
    basis: str
    safety_factor: float
    note: str


@dataclass
class CheckpointSnapshot:
    """Complete evidence state at a single checkpoint."""
    checkpoint_index: int
    n_users_per_variant: dict[str, int]
    total_users: int
    expected_loss: dict[str, float]
    prob_best: dict[str, float]
    best_variant: str

    burn_in_complete: bool
    sample_size_reached: bool
    min_checkpoints_reached: bool
    loss_below_threshold: bool
    prob_best_sufficient: bool
    traffic_balanced: bool

    safe_to_stop: bool
    stopping_reason: Literal["winner", "futility", "none"]
    futility_triggered: bool


@dataclass
class StoppingResult:
    """Full output of a sequential stopping evaluation."""
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

def _check_traffic_balance(
    n_users_per_variant: dict[str, int],
    expected_shares: Optional[dict[str, float]],
    imbalance_tolerance: float,
) -> TrafficDiagnostics:
    """Check whether traffic is distributed as expected across variants."""
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

def _check_futility(
    samples: np.ndarray,
    variant_names: list[str],
    control: str,
    rope_low: float,
    rope_high: float,
    futility_rope_threshold: float,
) -> bool:
    
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
            return False  

    return True 

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
    """Compute per-variant estimates of additional users needed."""
    best_loss = expected_loss.get(best_variant, float("inf"))
    best_p = prob_best.get(best_variant, 0.0)

    loss_gap = best_loss - loss_threshold
    prob_gap = prob_best_min - best_p       

    if loss_gap <= 0 and prob_gap <= 0:
        return None

    if best_loss <= 0:
        return None 

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
            ratio = best_loss / loss_threshold
            n_target = n_current * (ratio ** 2)
            additional = max(
                int(np.ceil((n_target - n_current) * safety_factor)),
                users_floor,
            )
            basis = "loss"
        else:
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

    Parameters
    ----------
    samples : np.ndarray
        Posterior samples array.
    variant_names : list[str]
        Ordered list of variant identifiers.
    control : str
        Control variant identifier.
    n_users_per_variant : dict[str, int]
        Number of users observed for each variant.
    loss_threshold : float, optional
        Threshold for expected loss, by default 0.01.
    prob_best_min : float, optional
        Minimum probability of being best, by default 0.80.
    min_sample_size : int, optional
        Minimum total users required, by default 1000.
    burn_in_users : int, optional
        Users required before tracking, by default 500.
    min_checkpoints : int, optional
        Minimum number of checkpoints required, by default 3.
    checkpoint_index : int, optional
        Current checkpoint sequence number, by default 1.
    rope_bounds : tuple[float, float], optional
        Region of practical equivalence, by default (-0.01, 0.01).
    futility_rope_threshold : float, optional
        Probability threshold for futility, by default 0.80.
    expected_traffic_shares : dict[str, float] | None, optional
        Expected traffic allocation, by default None.
    imbalance_tolerance : float, optional
        Max deviation allowed in traffic shares, by default 0.10.
    imbalance_blocks_stopping : bool, optional
        If True, blocks stopping if traffic is imbalanced, by default True.
    daily_traffic_per_variant : dict[str, float] | None, optional
        Traffic per day per variant, by default None.
    experiment_age_days : float | None, optional
        Age of the experiment in days, by default None.
    novelty_warning_days : int, optional
        Threshold for novelty warning, by default 14.
    users_estimate_floor : int, optional
        Minimum users to suggest, by default 100.
    users_estimate_safety_factor : float, optional
        Safety multiplier for estimates, by default 1.25.
    min_draws_warning : int, optional
        Minimum posterior draws before warning, by default 500.
    prior_trajectory : list[CheckpointSnapshot] | None, optional
        Previous checkpoints for historical context, by default None.

    Returns
    -------
    StoppingResult
        The outcome of the stopping evaluation.
    """
    collected_warnings: list[str] = []

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

    min_users_seen = min(n_users_per_variant.get(v, 0) for v in variant_names)
    burn_in_complete = min_users_seen >= burn_in_users

    if not burn_in_complete:
        msg = (
            f"Burn-in not complete: {min_users_seen} / {burn_in_users} "
            "users per variant. Posterior metrics computed but stopping is blocked."
        )
        collected_warnings.append(msg)

    sample_size_reached = min_users_seen >= min_sample_size
    min_checkpoints_reached = checkpoint_index >= min_checkpoints

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

    loss_result = compute_expected_loss(samples, variant_names, control)
    prob_best_result = compute_prob_best(samples, variant_names)

    best_variant = prob_best_result.best_variant
    best_loss = loss_result.expected_loss[best_variant]
    best_p = prob_best_result.probabilities[best_variant]

    loss_below_threshold = best_loss < loss_threshold
    prob_best_sufficient = best_p >= prob_best_min
    traffic_gate_passed = traffic.balanced or not imbalance_blocks_stopping

    prerequisites_met = (
        burn_in_complete
        and sample_size_reached
        and min_checkpoints_reached
        and traffic_gate_passed
    )

    winner_stopping = (
        prerequisites_met
        and loss_below_threshold
        and prob_best_sufficient
    )

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

    gate_states = {
        "burn_in":          burn_in_complete,
        "sample_size":      sample_size_reached,
        "min_checkpoints":  min_checkpoints_reached,
        "loss":             loss_below_threshold,
        "prob_best":        prob_best_sufficient,
        "traffic":          traffic.balanced,
    }

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

class StoppingChecker:
    """
    Stateful evaluator for sequential Bayesian stopping.

    Maintains a trajectory of checkpoint snapshots and updates
    stopping decisions as new data arrives.

    Parameters
    ----------
    loss_threshold : float, optional
        Threshold for expected loss, by default 0.01.
    prob_best_min : float, optional
        Minimum probability of being best, by default 0.80.
    min_sample_size : int, optional
        Minimum total users required, by default 1000.
    burn_in_users : int, optional
        Users required before tracking, by default 500.
    min_checkpoints : int, optional
        Minimum number of checkpoints required, by default 3.
    rope_bounds : tuple[float, float], optional
        Region of practical equivalence, by default (-0.01, 0.01).
    futility_rope_threshold : float, optional
        Probability threshold for futility, by default 0.80.
    expected_traffic_shares : dict[str, float] | None, optional
        Expected traffic allocation, by default None.
    imbalance_tolerance : float, optional
        Max deviation allowed in traffic shares, by default 0.10.
    imbalance_blocks_stopping : bool, optional
        If True, blocks stopping if traffic is imbalanced, by default True.
    daily_traffic_per_variant : dict[str, float] | None, optional
        Traffic per day per variant, by default None.
    novelty_warning_days : int, optional
        Threshold for novelty warning, by default 14.
    users_estimate_floor : int, optional
        Minimum users to suggest, by default 100.
    users_estimate_safety_factor : float, optional
        Safety multiplier for estimates, by default 1.25.
    min_draws_warning : int, optional
        Minimum posterior draws before warning, by default 500.
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
            Posterior samples for the current checkpoint.
        variant_names : list[str]
            List of variant names.
        control : str
            Control variant name.
        n_users_per_variant : dict[str, int]
            Current user counts per variant.
        experiment_age_days : float | None, optional
            Age of the experiment in days, by default None.

        Returns
        -------
        StoppingResult
            The outcome of the evaluation.
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

        Parameters
        ----------
        figsize : tuple[int, int], optional
            Figure size, by default (13, 10).
        suptitle : str | None, optional
            Figure super title, by default None.

        Returns
        -------
        plt.Figure
            The matplotlib Figure containing the trajectory dashboard.
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