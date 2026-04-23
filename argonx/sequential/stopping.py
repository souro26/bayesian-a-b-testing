"""
Bayesian sequential stopping rules for the argonx decision engine.

Bayesian sequential testing is mathematically valid at any point in time.
Frequentist statistics breaks if you peek early — the false positive rate
inflates. Bayesian expected-loss-based stopping does not have this problem.
Stopping occurs when the evidence is strong enough, not when a sample size
target is hit.

Typical usage
-------------
At each data checkpoint, call evaluate_stopping() with the current posterior
samples. It returns a StoppingResult indicating whether it is safe to stop,
the current evidence trajectory, and — if not safe — an estimate of how many
additional users are needed at the current observed effect size.

    from argonx.sequential.stopping import evaluate_stopping, StoppingChecker

    # Stateless: evaluate a single checkpoint
    result = evaluate_stopping(
        samples=posterior_samples,          # (n_draws, n_variants)
        variant_names=["control", "variant_b"],
        control="control",
        n_users_per_variant={"control": 5000, "variant_b": 5000},
        loss_threshold=0.01,
    )
    print(result.safe_to_stop, result.recommendation)

    # Stateful: accumulate a trajectory across checkpoints
    checker = StoppingChecker(loss_threshold=0.01, min_checkpoints=3)
    result = checker.update(samples, variant_names, control, n_users_per_variant)
    fig = checker.plot_trajectory()
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from argonx.decision_rules.metrics import compute_expected_loss, compute_prob_best

_DEFAULT_LOSS_THRESHOLD = 0.01
_DEFAULT_PROB_BEST_MIN = 0.80 
_MIN_CHECKPOINTS_DEFAULT = 3 
_USERS_ESTIMATE_FLOOR = 100        


@dataclass
class CheckpointSnapshot:
    """Evidence state at a single point in time."""
    checkpoint_index: int
    n_users_per_variant: dict[str, int]
    total_users: int
    expected_loss: dict[str, float]
    prob_best: dict[str, float]
    best_variant: str
    loss_below_threshold: bool
    prob_best_sufficient: bool
    safe_to_stop: bool


@dataclass
class StoppingResult:
    """
    Full output of a sequential stopping evaluation.

    Fields
    ------
    safe_to_stop : bool
        True when expected loss is below threshold AND P(best) clears the
        minimum and the minimum checkpoint count has been reached.
    best_variant : str
        Variant with the highest P(best) at this checkpoint.
    expected_loss : dict[str, float]
        Current expected loss per variant. Key signal — when the best
        variant's loss drops below loss_threshold, stopping is warranted.
    prob_best : dict[str, float]
        Current P(best) per variant via simultaneous argmax.
    loss_threshold : float
        The configured threshold. Stopping fires when best variant's
        expected_loss falls below this value.
    users_needed_estimate : dict[str, int] | None
        Approximate additional users needed per variant at the current
        observed effect size. None when safe_to_stop is True — no guidance
        needed. None when effect size is zero or unestimable.
    recommendation : str
        Plain-English stopping recommendation.
    checkpoint_index : int
        How many checkpoints have been evaluated so far (including this one).
    trajectory : list[CheckpointSnapshot]
        Full history of all checkpoints evaluated. Plottable via
        StoppingChecker.plot_trajectory().
    warnings : list[str]
        Non-fatal issues detected during evaluation.
    """
    safe_to_stop: bool
    best_variant: str
    expected_loss: dict[str, float]
    prob_best: dict[str, float]
    loss_threshold: float
    users_needed_estimate: Optional[dict[str, int]]
    recommendation: str
    checkpoint_index: int
    trajectory: list[CheckpointSnapshot] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

def _estimate_users_needed(
    current_loss: float,
    loss_threshold: float,
    n_users: int,
) -> Optional[int]:
    """
    Estimate additional users needed to bring expected loss below threshold.

    Uses a simple linear scaling heuristic: loss decreases roughly as
    1/sqrt(n) as sample size grows (standard Bayesian posterior contraction
    rate). Inverts this relationship to estimate the n needed.

    Returns None when the current loss is zero (already optimal) or when
    the estimate would be unreliable (effect is too small to estimate from).
    The estimate is deliberately conservative — it rounds up and applies a
    1.2x safety factor to account for posterior variance.

    This is an approximation. Its value is not precision — it is giving the
    product team an order-of-magnitude anchor rather than "keep running"
    with no guidance.
    """
    if current_loss <= 0 or n_users <= 0:
        return None

    ratio = current_loss / loss_threshold
    n_target = n_users * (ratio ** 2)
    additional = max(int(np.ceil((n_target - n_users) * 1.2)), _USERS_ESTIMATE_FLOOR)

    return additional


def _build_recommendation(
    safe_to_stop: bool,
    best_variant: str,
    expected_loss: dict[str, float],
    loss_threshold: float,
    prob_best: dict[str, float],
    prob_best_min: float,
    users_needed: Optional[dict[str, int]],
    min_checkpoints_reached: bool,
) -> str:
    """Generate a plain-English recommendation string."""
    best_loss = expected_loss.get(best_variant, float("inf"))
    best_p = prob_best.get(best_variant, 0.0)

    if not min_checkpoints_reached:
        return (
            "Minimum checkpoint count not yet reached. "
            "Continue collecting data before evaluating stopping."
        )

    if safe_to_stop:
        return (
            f"Safe to stop. Expected loss for '{best_variant}' is {best_loss:.4f}, "
            f"below threshold {loss_threshold:.4f}. "
            f"P(best) = {best_p:.3f}. Evidence is sufficient."
        )

    parts = [f"Continue experiment. '{best_variant}' leads with P(best)={best_p:.3f}."]

    if best_loss >= loss_threshold:
        parts.append(
            f"Expected loss ({best_loss:.4f}) has not dropped below "
            f"threshold ({loss_threshold:.4f})."
        )

    if best_p < prob_best_min:
        parts.append(
            f"P(best) ({best_p:.3f}) has not cleared minimum ({prob_best_min:.3f})."
        )

    if users_needed:
        per_variant_str = ", ".join(
            f"{v}: ~{n:,}" for v, n in sorted(users_needed.items())
        )
        parts.append(
            f"Estimated additional users needed at current effect size — {per_variant_str}."
        )

    return " ".join(parts)


def evaluate_stopping(
    samples: np.ndarray,
    variant_names: list[str],
    control: str,
    n_users_per_variant: dict[str, int],
    loss_threshold: float = _DEFAULT_LOSS_THRESHOLD,
    prob_best_min: float = _DEFAULT_PROB_BEST_MIN,
    checkpoint_index: int = 1,
    min_checkpoints: int = _MIN_CHECKPOINTS_DEFAULT,
    prior_trajectory: Optional[list[CheckpointSnapshot]] = None,
) -> StoppingResult:
    """
    Evaluate whether it is safe to stop the experiment at this checkpoint.

    This is a stateless function — it evaluates a single checkpoint and
    returns a full StoppingResult. For stateful trajectory tracking across
    multiple checkpoints, use StoppingChecker instead.

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
        Current number of users (observations) per variant. Used for the
        "users needed" estimate. Must include all variants.
    loss_threshold : float
        Stop when best variant's expected loss falls below this value.
        Default 0.01 matches DEFAULT_CONFIG in experiment.py.
    prob_best_min : float
        Minimum P(best) required before stopping is considered. Guards
        against stopping on weak signals. Default 0.80.
    checkpoint_index : int
        Ordinal index of this checkpoint (1-based). Used for trajectory
        continuity checks and the minimum checkpoint gate.
    min_checkpoints : int
        Minimum number of checkpoints that must pass before stopping is
        allowed, regardless of the evidence. Guards against stopping on
        the first look. Default 3.
    prior_trajectory : list[CheckpointSnapshot], optional
        Trajectory from previous checkpoints. Appended to with this
        checkpoint's snapshot and returned in StoppingResult.trajectory.

    Returns
    -------
    StoppingResult

    Raises
    ------
    ValueError
        If samples shape is inconsistent with variant_names, control is
        missing, n_users_per_variant keys do not match variant_names,
        or thresholds are out of range.
    """
    collected_warnings: list[str] = []

    if samples.ndim != 2:
        raise ValueError(
            f"samples must be 2D (n_draws, n_variants), got shape {samples.shape}"
        )

    if samples.shape[1] != len(variant_names):
        raise ValueError(
            f"samples has {samples.shape[1]} columns but "
            f"{len(variant_names)} variant names were provided."
        )

    if control not in variant_names:
        raise ValueError(
            f"control '{control}' not found in variant_names: {variant_names}"
        )

    if not np.isfinite(samples).all():
        raise ValueError("samples contain NaN or Inf values.")

    if not 0 < loss_threshold < 1:
        raise ValueError(f"loss_threshold must be in (0, 1), got {loss_threshold}")

    if not 0 < prob_best_min < 1:
        raise ValueError(f"prob_best_min must be in (0, 1), got {prob_best_min}")

    missing_variants = set(variant_names) - set(n_users_per_variant.keys())
    if missing_variants:
        raise ValueError(
            f"n_users_per_variant is missing entries for: {sorted(missing_variants)}"
        )

    if samples.shape[0] < 500:
        msg = (
            f"n_draws={samples.shape[0]} is low. "
            "Stopping decisions based on fewer than 500 draws may be unreliable."
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
    min_checkpoints_reached = checkpoint_index >= min_checkpoints

    safe_to_stop = (
        loss_below_threshold
        and prob_best_sufficient
        and min_checkpoints_reached
    )

    users_needed: Optional[dict[str, int]] = None

    if not safe_to_stop and not loss_below_threshold:
        best_n = n_users_per_variant.get(best_variant, 0)
        estimate = _estimate_users_needed(best_loss, loss_threshold, best_n)
        if estimate is not None:
            users_needed = {
                v: estimate
                for v in variant_names
                if v != control
            }

    total_users = sum(n_users_per_variant.values())

    snapshot = CheckpointSnapshot(
        checkpoint_index=checkpoint_index,
        n_users_per_variant=dict(n_users_per_variant),
        total_users=total_users,
        expected_loss=dict(loss_result.expected_loss),
        prob_best=dict(prob_best_result.probabilities),
        best_variant=best_variant,
        loss_below_threshold=loss_below_threshold,
        prob_best_sufficient=prob_best_sufficient,
        safe_to_stop=safe_to_stop,
    )

    trajectory = list(prior_trajectory or [])
    trajectory.append(snapshot)

    recommendation = _build_recommendation(
        safe_to_stop=safe_to_stop,
        best_variant=best_variant,
        expected_loss=loss_result.expected_loss,
        loss_threshold=loss_threshold,
        prob_best=prob_best_result.probabilities,
        prob_best_min=prob_best_min,
        users_needed=users_needed,
        min_checkpoints_reached=min_checkpoints_reached,
    )

    return StoppingResult(
        safe_to_stop=safe_to_stop,
        best_variant=best_variant,
        expected_loss=loss_result.expected_loss,
        prob_best=prob_best_result.probabilities,
        loss_threshold=loss_threshold,
        users_needed_estimate=users_needed,
        recommendation=recommendation,
        checkpoint_index=checkpoint_index,
        trajectory=trajectory,
        warnings=collected_warnings,
    )

class StoppingChecker:
    """
    Stateful wrapper around evaluate_stopping() for multi-checkpoint workflows.

    Maintains the full evidence trajectory across calls to update(). Call
    update() at each data checkpoint (e.g. daily, or every 1000 users).
    Call plot_trajectory() at any point to visualise evidence accumulation.

    Parameters
    ----------
    loss_threshold : float
        Stop when best variant's expected loss falls below this. Default 0.01.
    prob_best_min : float
        Minimum P(best) required before stopping is considered. Default 0.80.
    min_checkpoints : int
        Minimum checkpoints before stopping is allowed. Default 3.

    Example
    -------
        checker = StoppingChecker(loss_threshold=0.01)

        for day, (samples, n_users) in enumerate(daily_updates, start=1):
            result = checker.update(
                samples=samples,
                variant_names=["control", "variant_b"],
                control="control",
                n_users_per_variant=n_users,
            )
            if result.safe_to_stop:
                break

        fig = checker.plot_trajectory()
        fig.savefig("stopping_trajectory.png", dpi=150)
    """

    def __init__(
        self,
        loss_threshold: float = _DEFAULT_LOSS_THRESHOLD,
        prob_best_min: float = _DEFAULT_PROB_BEST_MIN,
        min_checkpoints: int = _MIN_CHECKPOINTS_DEFAULT,
    ) -> None:
        self.loss_threshold = loss_threshold
        self.prob_best_min = prob_best_min
        self.min_checkpoints = min_checkpoints

        self._trajectory: list[CheckpointSnapshot] = []
        self._checkpoint_index: int = 0

    @property
    def trajectory(self) -> list[CheckpointSnapshot]:
        """Read-only view of all checkpoints evaluated so far."""
        return list(self._trajectory)

    @property
    def n_checkpoints(self) -> int:
        return self._checkpoint_index

    def update(
        self,
        samples: np.ndarray,
        variant_names: list[str],
        control: str,
        n_users_per_variant: dict[str, int],
    ) -> StoppingResult:
        """
        Evaluate the current checkpoint and update the internal trajectory.

        Parameters
        ----------
        samples : np.ndarray
            Current posterior samples. Shape (n_draws, n_variants).
        variant_names : list[str]
            Sorted variant names. Must match samples column order.
        control : str
            Name of the control variant.
        n_users_per_variant : dict[str, int]
            Current number of users per variant at this checkpoint.

        Returns
        -------
        StoppingResult
            Full result including the updated trajectory.
        """
        self._checkpoint_index += 1

        result = evaluate_stopping(
            samples=samples,
            variant_names=variant_names,
            control=control,
            n_users_per_variant=n_users_per_variant,
            loss_threshold=self.loss_threshold,
            prob_best_min=self.prob_best_min,
            checkpoint_index=self._checkpoint_index,
            min_checkpoints=self.min_checkpoints,
            prior_trajectory=self._trajectory,
        )

        self._trajectory = result.trajectory
        return result

    def reset(self) -> None:
        """Clear all trajectory history and reset checkpoint counter."""
        self._trajectory = []
        self._checkpoint_index = 0

    def plot_trajectory(
        self,
        figsize: tuple[int, int] = (12, 8),
        suptitle: Optional[str] = None,
    ) -> Figure:
        """
        Plot evidence accumulation across all evaluated checkpoints.

        Two panels:
            Top — P(best) per variant over checkpoints, with prob_best_min
                  threshold drawn as a dashed line.
            Bottom — Expected loss per variant over checkpoints, with
                  loss_threshold drawn as a dashed red line. Y axis is
                  log-scaled so both large and small losses are visible.
                  Green band marks the safe-to-stop zone below threshold.

        A vertical dashed line marks the first checkpoint where safe_to_stop
        was True, if one exists.

        Returns
        -------
        plt.Figure
        """
        if not self._trajectory:
            raise RuntimeError(
                "No checkpoints recorded. Call update() at least once before plotting."
            )

        _PALETTE = [
            "#2563EB", "#16A34A", "#DC2626",
            "#9333EA", "#EA580C", "#0891B2",
        ]

        snapshots = self._trajectory
        x = [s.checkpoint_index for s in snapshots]
        total_users = [s.total_users for s in snapshots]
        x_label = total_users if len(set(total_users)) > 1 else x

        # Collect all variant names from trajectory
        all_variants = list(snapshots[-1].prob_best.keys())

        fig, (ax_prob, ax_loss) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        fig.suptitle(
            suptitle or "Sequential Stopping — Evidence Trajectory",
            fontsize=14,
            fontweight="bold",
        )

        # --- Panel 1: P(best) ---
        for i, v in enumerate(all_variants):
            colour = _PALETTE[i % len(_PALETTE)]
            y = [s.prob_best.get(v, 0.0) for s in snapshots]
            ax_prob.plot(x_label, y, marker="o", markersize=4,
                         linewidth=1.8, color=colour, label=v, zorder=3)

        ax_prob.axhline(
            self.prob_best_min,
            color="#6B7280", linewidth=1.2, linestyle="--",
            label=f"min P(best) = {self.prob_best_min:.2f}", zorder=2,
        )
        ax_prob.set_ylim(-0.02, 1.05)
        ax_prob.set_ylabel("P(variant is best)", fontsize=11)
        ax_prob.set_title("P(best) — simultaneous N-variant argmax", fontsize=11)
        ax_prob.legend(framealpha=0.9, fontsize=9)
        ax_prob.grid(axis="both", alpha=0.2)
        ax_prob.spines["top"].set_visible(False)
        ax_prob.spines["right"].set_visible(False)

        # --- Panel 2: Expected loss ---
        for i, v in enumerate(all_variants):
            colour = _PALETTE[i % len(_PALETTE)]
            y = [s.expected_loss.get(v, 0.0) for s in snapshots]
            ax_loss.plot(x_label, y, marker="o", markersize=4,
                         linewidth=1.8, color=colour, label=v, zorder=3)

        ax_loss.axhline(
            self.loss_threshold,
            color="#DC2626", linewidth=1.4, linestyle="--",
            label=f"loss threshold = {self.loss_threshold:.3f}", zorder=4,
        )

        # Shade safe zone
        y_min_safe = 0.0
        ax_loss.axhspan(
            y_min_safe, self.loss_threshold,
            color="#BBF7D0", alpha=0.3, zorder=1, label="Safe-to-stop zone",
        )

        # Mark first stop point
        stop_points = [s for s in snapshots if s.safe_to_stop]
        if stop_points:
            first_stop = stop_points[0]
            stop_x = (
                first_stop.total_users
                if len(set(total_users)) > 1
                else first_stop.checkpoint_index
            )
            for ax in (ax_prob, ax_loss):
                ax.axvline(
                    stop_x,
                    color="#16A34A", linewidth=1.4, linestyle=":",
                    alpha=0.8, label="First safe-to-stop point", zorder=5,
                )

        ax_loss.set_ylabel("Expected loss", fontsize=11)
        ax_loss.set_title(
            "Expected loss — stop when best variant drops below threshold",
            fontsize=11
        )
        ax_loss.legend(framealpha=0.9, fontsize=9)
        ax_loss.grid(axis="both", alpha=0.2)
        ax_loss.spines["top"].set_visible(False)
        ax_loss.spines["right"].set_visible(False)

        x_label_text = "Total users" if len(set(total_users)) > 1 else "Checkpoint"
        ax_loss.set_xlabel(x_label_text, fontsize=11)

        fig.tight_layout()
        return fig