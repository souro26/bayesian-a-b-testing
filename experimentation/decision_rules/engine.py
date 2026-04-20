from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .metrics import MetricsBundle, compute_all_metrics
from .guardrails import GuardrailBundle, compute_all_guardrails
from .joint import JointResult, compute_joint_probability
from .composite import CompositeResult, compute_composite_score


@dataclass
class DecisionResult:
    """Final decision output with structured interpretation."""
    state: str
    recommendation: str
    best_variant: str

    primary_strength: str
    risk_level: str
    practical_significance: str
    guardrail_status: str
    confidence: str

    metrics: MetricsBundle
    guardrails: GuardrailBundle
    joint: JointResult | None
    composite: CompositeResult | None

    reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _evaluate_primary_strength(metrics: MetricsBundle, config: dict) -> str:
    """Compute strength of primary signal."""
    best = metrics.prob_best.best_variant
    p_best = metrics.prob_best.probabilities[best]
    loss = metrics.loss.expected_loss[best]
    practical = metrics.rope.prob_practical.get(best, 0.0)

    if (
        p_best >= config["prob_best_strong"]
        and loss <= config["expected_loss_max"]
        and practical >= config["rope_practical_min"]
    ):
        return "strong"

    if p_best >= config["prob_best_moderate"]:
        return "moderate"

    return "weak"


def _evaluate_risk(metrics: MetricsBundle, config: dict) -> str:
    """Classify risk using expected loss and CVaR."""
    best = metrics.prob_best.best_variant
    el = metrics.loss.expected_loss[best]
    cv = metrics.cvar.cvar[best]

    if el <= config["expected_loss_max"] and (
        el == 0 or cv / max(el, 1e-12) <= config["cvar_ratio_max"]
    ):
        return "low"

    if el <= config["expected_loss_max"] * 5:
        return "medium"

    return "high"


def _evaluate_practical_significance(metrics: MetricsBundle, config: dict) -> str:
    """Classify practical significance using ROPE."""
    best = metrics.prob_best.best_variant
    prob = metrics.rope.prob_practical.get(best, 0.0)

    if prob >= config["rope_practical_min"]:
        return "yes"

    if prob >= 0.5:
        return "uncertain"

    return "no"


def _evaluate_guardrails(guardrails: GuardrailBundle) -> str:
    """Summarize guardrail status."""
    return "pass" if guardrails.all_passed else "fail"


def _evaluate_confidence(metrics: MetricsBundle, config: dict) -> str:
    """Assess confidence using posterior certainty."""
    best = metrics.prob_best.best_variant
    p_best = metrics.prob_best.probabilities[best]

    low = metrics.lift.hdi_low[best]
    high = metrics.lift.hdi_high[best]
    width = abs(high - low)

    if p_best >= 0.95 and width < 0.1:
        return "high"

    if p_best >= 0.8:
        return "medium"

    return "low"


def _determine_state(
    primary_strength: str,
    risk_level: str,
    practical_significance: str,
    guardrails: GuardrailBundle,
) -> str:
    """Determine overall decision state."""
    if guardrails.conflicts:
        return "guardrail conflicts"

    if (
        primary_strength == "strong"
        and risk_level == "low"
        and practical_significance == "yes"
        and guardrails.all_passed
    ):
        return "strong win"

    if (
        primary_strength == "moderate"
        and guardrails.all_passed
        and risk_level == "low"
        and practical_significance == "yes"
    ):
        return "weak win"

    if risk_level == "high":
        return "high risk"

    return "inconclusive"


def _map_recommendation(state: str) -> str:
    """Map decision state to recommendation."""
    return {
        "strong win": "ship variant",
        "weak win": "consider shipping",
        "high risk": "do not ship",
        "guardrail conflicts": "review required",
        "inconclusive": "continue experiment",
    }[state]


def _build_reasons(
    primary_strength: str,
    risk_level: str,
    practical_significance: str,
    guardrail_status: str,
) -> list[str]:
    """Generate human readable reasoning signals."""
    reasons = [
        f"Primary signal is {primary_strength}",
        f"Risk level is {risk_level}",
        f"Practical significance is {practical_significance}",
    ]

    if guardrail_status == "pass":
        reasons.append("All guardrails passed")
    else:
        reasons.append("One or more guardrails failed")

    return reasons


def _collect_notes(metrics: MetricsBundle, guardrails: GuardrailBundle, joint: JointResult, composite: CompositeResult) -> list[str]:
    """Collect warnings and edge case signals from all components."""
    notes = []

    notes.extend(metrics.warnings)
    notes.extend(guardrails.warnings)

    if joint is not None:
        notes.extend(joint.warnings)

    if composite is not None:
        notes.extend(composite.warnings)

    for conflict in guardrails.conflicts:
        if conflict.severity == "high":
            notes.append(
                f"High severity guardrail violation on '{conflict.metric}' "
                f"for variant '{conflict.variant}'"
            )

    for v in metrics.loss.expected_loss:
        el = metrics.loss.expected_loss[v]
        cv = metrics.cvar.cvar[v]
        if el > 0 and cv / el > 5:
            notes.append(f"Variant '{v}' has extreme tail risk")

    return notes


def run_engine(
    samples: np.ndarray,
    variant_names: list[str],
    control: str,
    guardrail_samples: dict[str, np.ndarray],
    config: dict,
) -> DecisionResult:
    """Run full decision pipeline from posterior samples."""

    metrics = compute_all_metrics(
        samples=samples,
        variant_names=variant_names,
        control=control,
        rope_bounds=config["rope_bounds"],
        alpha=config.get("alpha", 0.95),
        hdi_prob=config.get("hdi_prob", 0.95),
    )

    primary_strength = _evaluate_primary_strength(metrics, config)
    risk_level = _evaluate_risk(metrics, config)
    practical_significance = _evaluate_practical_significance(metrics, config)

    primary_passed = (
        primary_strength == "strong"
        and risk_level != "high"
        and practical_significance == "yes"
    )

    guardrails = compute_all_guardrails(
        guardrail_samples=guardrail_samples,
        variant_names=variant_names,
        control=control,
        thresholds=config.get("guardrail_thresholds", {}),
        primary_passed=primary_passed,
        lower_is_better=config.get("lower_is_better", {}),
    )

    guardrail_status = _evaluate_guardrails(guardrails)
    confidence = _evaluate_confidence(metrics, config)

    joint = None
    if guardrail_samples:
        joint = compute_joint_probability(
            primary_samples=samples,
            guardrail_samples=guardrail_samples,
            variant_names=variant_names,
            control=control,
            primary_lower_is_better=config.get("primary_lower_is_better", False),
            lower_is_better=config.get("lower_is_better", {}),
            guardrail_thresholds=config.get("guardrail_thresholds", {}),
            metrics_to_join=config.get("metrics_to_join", None),
        )

    composite = None
    if "composite_weights" in config:
        composite = compute_composite_score(
            primary_samples=samples,
            guardrail_samples=guardrail_samples,
            variant_names=variant_names,
            control=control,
            weights=config["composite_weights"],
            guardrail_bundle=guardrails,
            deterioration_weights=config.get("deterioration_weights", None),
            guardrail_penalty=config.get("guardrail_penalty", 0.0),
            threshold=config.get("composite_threshold", 0.0),
        )

    state = _determine_state(
        primary_strength,
        risk_level,
        practical_significance,
        guardrails,
    )

    recommendation = _map_recommendation(state)

    reasons = _build_reasons(
        primary_strength,
        risk_level,
        practical_significance,
        guardrail_status,
    )

    notes = _collect_notes(metrics, guardrails, joint, composite)

    return DecisionResult(
        state=state,
        recommendation=recommendation,
        best_variant=metrics.prob_best.best_variant,

        primary_strength=primary_strength,
        risk_level=risk_level,
        practical_significance=practical_significance,
        guardrail_status=guardrail_status,
        confidence=confidence,

        metrics=metrics,
        guardrails=guardrails,
        joint=joint,
        composite=composite,

        reasons=reasons,
        notes=notes,
    )