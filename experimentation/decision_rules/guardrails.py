from __future__ import annotations

import numpy as np
import warnings

from dataclasses import dataclass, field


def _compute_severity(prob_degraded: float) -> str:
    if prob_degraded >= 0.90:
        return "high"
    elif prob_degraded >= 0.70:
        return "medium"
    else:
        return "low"


@dataclass
class GuardrailResult:
    """Result of a single guardrail check for one variant."""
    metric: str
    prob_degraded: float
    passed: bool
    threshold: float
    variant: str
    severity: str 
    expected_degradation: float    


@dataclass
class ConflictResult:
    """A detected conflict between primary metric and a guardrail."""
    metric: str
    prob_degraded: float
    threshold: float
    message: str
    variant: str
    severity: str
    expected_degradation: float


@dataclass
class GuardrailBundle:
    """All guardrail results and conflicts returned to the engine."""
    all_passed: bool
    variant_passed: dict[str, bool]      # per-variant summary across all guardrails
    guardrails: list[GuardrailResult]
    conflicts: list[ConflictResult]
    warnings: list[str] = field(default_factory=list)


def _validate_guardrail_inputs(
    samples: np.ndarray,
    variant_names: list[str],
    control: str,
    threshold: float,
) -> None:
    """
    Validate guardrail-layer preconditions before computation.

    Checks variant_names aligns with sample columns, control exists,
    samples contain no NaN or Inf, threshold is in (0, 1), and at
    least two variants are present. Does not re-check model-layer guarantees.
    """
    if len(variant_names) != samples.shape[1]:
        raise ValueError(
            f"variant_names length ({len(variant_names)}) does not match "
            f"samples.shape[1] ({samples.shape[1]})"
        )

    if control not in variant_names:
        raise ValueError(
            f"control '{control}' not found in variant_names: {variant_names}"
        )

    if not np.isfinite(samples).all():
        raise ValueError(
            "samples contain NaN or Inf values. Check model output."
        )

    if not 0 < threshold < 1:
        raise ValueError(
            f"threshold must be in (0, 1), got {threshold}"
        )

    if samples.shape[1] < 2:
        raise ValueError(
            "At least two variants required for guardrail check."
        )


def compute_guardrail(
    samples: np.ndarray,
    variant_names: list[str],
    control: str,
    metric: str,
    threshold: float,
    lower_is_better: bool = True,
) -> list[GuardrailResult]:
    """
    Check a single guardrail metric for degradation across all variants.

    Computes P(degraded) for each non-control variant. If lower_is_better
    is True, degradation means the variant went up relative to control.
    If False, degradation means the variant went down. A guardrail passes
    when P(degraded) is below the configured threshold.
    """
    _validate_guardrail_inputs(samples, variant_names, control, threshold)

    control_idx = variant_names.index(control)
    control_draws = samples[:, control_idx]

    results = []

    for i, variant in enumerate(variant_names):
        if variant == control:
            continue

        variant_draws = samples[:, i]
        delta = variant_draws - control_draws

        if lower_is_better:
            degraded_mask = variant_draws > control_draws
        else:
            degraded_mask = variant_draws < control_draws

        prob_degraded = float(np.mean(degraded_mask))
        passed = prob_degraded < threshold

        expected_degradation = (
            float(np.mean(np.abs(delta[degraded_mask])))
            if degraded_mask.any()
            else 0.0
        )

        results.append(GuardrailResult(
            metric=metric,
            prob_degraded=prob_degraded,
            passed=passed,
            threshold=threshold,
            variant=variant,
            severity=_compute_severity(prob_degraded),
            expected_degradation=expected_degradation,
        ))

    return results


def compute_all_guardrails(
    guardrail_samples: dict[str, np.ndarray],
    variant_names: list[str],
    control: str,
    thresholds: dict[str, float],
    primary_passed: bool,
    lower_is_better: dict[str, bool] = None,
) -> GuardrailBundle:
    """
    Run all guardrail checks and detect conflicts with the primary metric.

    Calls compute_guardrail for each metric, collects all results, and
    detects conflicts when the primary metric passed but a guardrail failed.
    Conflict detection is left to human review — the framework surfaces the
    tension clearly rather than resolving it automatically.
    """
    collected_warnings = []

    if len(guardrail_samples) == 0:
        msg = (
            "No guardrail metrics provided. all_passed is vacuously True. "
            "Consider defining guardrails to protect secondary metrics."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        collected_warnings.append(msg)

        return GuardrailBundle(
            all_passed=True,
            variant_passed={v: True for v in variant_names if v != control},
            guardrails=[],
            conflicts=[],
            warnings=collected_warnings,
        )

    if lower_is_better is None:
        msg = (
            "lower_is_better not provided. Defaulting all guardrail metrics to "
            "lower_is_better=True. Pass explicitly to avoid silent misclassification."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        collected_warnings.append(msg)
        lower_is_better = {}

    for metric in guardrail_samples:
        if metric not in thresholds:
            raise ValueError(
                f"No threshold configured for guardrail metric '{metric}'. "
                f"Provide a threshold in the thresholds dict."
            )

    all_results = []

    for metric, samples in guardrail_samples.items():
        results = compute_guardrail(
            samples=samples,
            variant_names=variant_names,
            control=control,
            metric=metric,
            threshold=thresholds[metric],
            lower_is_better=lower_is_better.get(metric, True),
        )
        all_results.extend(results)

    all_passed = all(r.passed for r in all_results)

    non_control_variants = [v for v in variant_names if v != control]
    variant_passed = {
        v: all(r.passed for r in all_results if r.variant == v)
        for v in non_control_variants
    }

    conflicts = []
    for result in all_results:
        if primary_passed and not result.passed:
            msg = (
                f"Conflict detected on '{result.metric}' for variant '{result.variant}': "
                f"primary metric passed but guardrail violated with "
                f"P(degraded)={result.prob_degraded:.3f} > threshold={result.threshold}. "
                f"Human review required."
            )
            conflicts.append(ConflictResult(
                metric=result.metric,
                prob_degraded=result.prob_degraded,
                threshold=result.threshold,
                message=msg,
                variant=result.variant,
                severity=result.severity,
                expected_degradation=result.expected_degradation,
            ))

    return GuardrailBundle(
        all_passed=all_passed,
        variant_passed=variant_passed,
        guardrails=all_results,
        conflicts=conflicts,
        warnings=collected_warnings,
    )