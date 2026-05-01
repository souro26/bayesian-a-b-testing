from __future__ import annotations

import warnings
import numpy as np
from dataclasses import dataclass, field


@dataclass
class JointResult:
    """Joint policy satisfaction with correlation diagnostics."""
    joint_prob: dict[str, float]
    condition_probs: dict[str, dict[str, float]]
    independence_benchmark: dict[str, float]
    correlation_gap: dict[str, float]
    best_variant: str
    metrics_joined: list[str]
    warnings: list[str] = field(default_factory=list)


def _validate_joint_inputs(
    primary_samples: np.ndarray,
    guardrail_samples: dict[str, np.ndarray],
    variant_names: list[str],
    control: str,
    metrics_to_join: list[str] | None,
) -> None:
    """Validate shapes, control presence, and metric selection."""
    if len(variant_names) != primary_samples.shape[1]:
        raise ValueError("variant_names mismatch with primary_samples")

    if control not in variant_names:
        raise ValueError(f"control '{control}' not found")

    if not np.isfinite(primary_samples).all():
        raise ValueError("primary_samples contain NaN/Inf")

    for m, s in guardrail_samples.items():
        if s.shape != primary_samples.shape:
            raise ValueError(f"{m} shape mismatch")
        if not np.isfinite(s).all():
            raise ValueError(f"{m} contains NaN/Inf")

    if metrics_to_join is not None:
        if len(metrics_to_join) == 0:
            raise ValueError(
                "metrics_to_join is empty. Pass None for all guardrails."
            )
        unknown = [m for m in metrics_to_join if m not in guardrail_samples]
        if unknown:
            raise ValueError(f"Unknown metrics: {unknown}")


def _primary_condition_mask(
    primary_samples,
    variant_names,
    control,
    lower_is_better,
    threshold,
):
    """Primary condition with optional threshold."""
    c_idx = variant_names.index(control)
    c = primary_samples[:, c_idx]

    masks: dict[str, np.ndarray] = {}
    for i, v in enumerate(variant_names):
        if v == control:
            continue
        x = primary_samples[:, i]
        if lower_is_better:
            masks[v] = x < (c - threshold)
        else:
            masks[v] = x > (c + threshold)
    return masks


def _guardrail_condition_masks(
    guardrail_samples,
    variant_names,
    control,
    metrics_to_join,
    lower_is_better,
    thresholds,
):
    """Guardrail masks with direction and thresholds."""
    c_idx = variant_names.index(control)
    masks: dict[str, dict[str, np.ndarray]] = {}

    for m in metrics_to_join:
        s = guardrail_samples[m]
        c = s[:, c_idx]

        lib = lower_is_better.get(m, True)
        thr = thresholds.get(m, 0.0)

        vm: dict[str, np.ndarray] = {}
        for i, v in enumerate(variant_names):
            if v == control:
                continue
            x = s[:, i]
            if lib:
                vm[v] = x <= (c + thr)
            else:
                vm[v] = x >= (c - thr)
        masks[m] = vm

    return masks


def _compute_joint_mask(primary_masks, guardrail_masks, variant):
    """AND all masks for a variant."""
    joint = primary_masks[variant].copy()
    for gm in guardrail_masks.values():
        joint &= gm[variant]
    return joint


def _compute_independence_benchmark(condition_probs):
    """Product of marginal probabilities."""
    out = 1.0
    for p in condition_probs.values():
        out *= p
    return out


def compute_joint_probability(
    primary_samples: np.ndarray,
    guardrail_samples: dict[str, np.ndarray],
    variant_names: list[str],
    control: str,
    primary_lower_is_better: bool = False,
    primary_threshold: float = 0.0,
    lower_is_better: dict[str, bool] | None = None,
    guardrail_thresholds: dict[str, float] | None = None,
    metrics_to_join: list[str] | None = None,
) -> JointResult:
    """
    Compute joint probability and correlation effects.

    Calculates the empirical fraction of posterior draws where the primary metric 
    clears its threshold AND all selected guardrails simultaneous pass their constraints.
    It evaluates whether correlations strictly penalize or benefit the joint policy 
    compared to considering metrics independent.

    Parameters
    ----------
    primary_samples : np.ndarray
        Posterior samples for the primary target metric.
    guardrail_samples : dict[str, np.ndarray]
        A mapping of individual guardrail names to their respective posterior samples.
    variant_names : list[str]
        Ordered sequence of variant identifiers matching the columns of samples.
    control : str
        The identifier for the baseline variant.
    primary_lower_is_better : bool, optional
        Indicates if reductions in the primary metric represent an improvement. Defaults to False.
    primary_threshold : float, optional
        The minimal lift required over control for the primary metric to pass. Defaults to 0.0.
    lower_is_better : dict[str, bool] | None, optional
        Directional configuration mapping guardrail names to their orientation.
    guardrail_thresholds : dict[str, float] | None, optional
        Thresholds determining maximum allowable degradation for guardrails.
    metrics_to_join : list[str] | None, optional
        A specific subset of guardrail metrics to compute joint probabilities with. 
        Defaults to evaluating all provided guardrails.

    Returns
    -------
    JointResult
        The compound probabilities, measured correlation gaps, and diagnostic warnings.
    """
    collected: list[str] = []

    if lower_is_better is None:
        msg = (
            "lower_is_better not provided. Defaulting all guardrails to True. "
            "This may misclassify direction-sensitive metrics."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        collected.append(msg)
        lower_is_better = {}

    if guardrail_thresholds is None:
        guardrail_thresholds = {}

    if metrics_to_join is None:
        metrics_to_join = list(guardrail_samples.keys())

    _validate_joint_inputs(
        primary_samples,
        guardrail_samples,
        variant_names,
        control,
        metrics_to_join,
    )

    primary_masks = _primary_condition_mask(
        primary_samples,
        variant_names,
        control,
        primary_lower_is_better,
        primary_threshold,
    )

    guardrail_masks = _guardrail_condition_masks(
        guardrail_samples,
        variant_names,
        control,
        metrics_to_join,
        lower_is_better,
        guardrail_thresholds,
    )

    variants = [v for v in variant_names if v != control]

    joint_prob: dict[str, float] = {}
    condition_probs: dict[str, dict[str, float]] = {}
    independence_benchmark: dict[str, float] = {}
    correlation_gap: dict[str, float] = {}

    for v in variants:
        cond: dict[str, float] = {"primary": float(np.mean(primary_masks[v]))}

        for m in metrics_to_join:
            cond[m] = float(np.mean(guardrail_masks[m][v]))

        joint_mask = _compute_joint_mask(primary_masks, guardrail_masks, v)
        jp = float(np.mean(joint_mask))

        ind = _compute_independence_benchmark(cond)
        gap = jp - ind

        joint_prob[v] = jp
        condition_probs[v] = cond
        independence_benchmark[v] = ind
        correlation_gap[v] = gap

        if jp < 0.05:
            msg = f"{v}: joint_prob={jp:.3f} very low"
            warnings.warn(msg, UserWarning, stacklevel=2)
            collected.append(msg)

        if gap < -0.10:
            binding = min(cond, key=lambda k: cond[k])
            msg = f"{v}: correlation hurting joint, binding={binding}"
            warnings.warn(msg, UserWarning, stacklevel=2)
            collected.append(msg)

        if abs(gap) < 0.02:
            collected.append(f"{v}: metrics ~independent")

    best = max(joint_prob, key=lambda k: joint_prob[k])

    sorted_vals = sorted(joint_prob.values(), reverse=True)
    if len(sorted_vals) >= 2 and (sorted_vals[0] - sorted_vals[1]) > 0.2:
        collected.append("One variant strongly dominates joint policy")

    return JointResult(
        joint_prob=joint_prob,
        condition_probs=condition_probs,
        independence_benchmark=independence_benchmark,
        correlation_gap=correlation_gap,
        best_variant=best,
        metrics_joined=["primary"] + metrics_to_join,
        warnings=collected,
    )