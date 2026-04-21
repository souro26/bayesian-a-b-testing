from __future__ import annotations

import numpy as np 
import warnings
from dataclasses import dataclass, field 

from argonx.decision_rules.guardrails import GuardrailBundle

@dataclass
class CompositeResult:
    """Composite score combining metrics into one business signal."""
    score: dict[str, float]
    score_distribution: dict[str, np.ndarray]
    metric_contributions: dict[str, dict[str, float]]
    prob_exceeds_threshold: dict[str, float]
    gap_distribution: dict[str, np.ndarray]
    gap_hdi: dict[str, tuple[float, float]]
    best_variant: str
    threshold: float
    warnings: list[str] = field(default_factory=list)

def _compute_hdi(samples: np.ndarray, prob: float = 0.94) -> tuple[float, float]:
    """Compute HDI interval for 1 dimensional samples."""
    sorted_samples = np.sort(samples)
    n = len(samples)
    interval_idx = int(np.floor(prob * n))
    widths = sorted_samples[interval_idx:] - sorted_samples[: n - interval_idx]
    min_idx = np.argmin(widths)
    return (
        float(sorted_samples[min_idx]),
        float(sorted_samples[min_idx + interval_idx]),
    )

def compute_composite_score(
    primary_samples: np.ndarray,
    guardrail_samples: dict[str, np.ndarray],
    variant_names: list[str],
    control: str,
    weights: dict[str, float],
    guardrail_bundle: GuardrailBundle,
    deterioration_weights: dict[str, float] | None = None,
    guardrail_penalty: float = 0.0,
    threshold: float = 0.0,
) -> CompositeResult:
    """Compute weighted composite score from posterior samples."""

    collected: list[str] = []

    if not weights:
        raise ValueError("weights must be provided")

    valid_metrics = {"primary"} | set(guardrail_samples.keys())
    matched = [k for k in weights if k in valid_metrics]
    if not matched:
        raise ValueError(
            f"No valid metric keys in weights. Expected one of {valid_metrics}"
        )

    if deterioration_weights is None:
        deterioration_weights = weights.copy()
        warnings.warn(
            "deterioration_weights not provided, using symmetric weights",
            UserWarning,
            stacklevel=2,
        )
        collected.append("Using symmetric deterioration weights")

    if guardrail_penalty == 0:
        warnings.warn(
            "guardrail_penalty is zero, guardrails have no effect",
            UserWarning,
            stacklevel=2,
        )
        collected.append("Guardrail penalty is zero")

    c_idx = variant_names.index(control)
    variants = [v for v in variant_names if v != control]

    deltas = {}
    control_vals = primary_samples[:, c_idx]
    deltas["primary"] = primary_samples - control_vals[:, None]
    
    for m,s in guardrail_samples.items():
        deltas[m] = s - s[:, c_idx][:, None]


    score_per_draw: dict[str, np.ndarray] = {}
    contributions: dict[str, dict[str, float]] = {}

    for i,v in enumerate(variant_names):
        if v == control:
            continue

        total = np.zeros(primary_samples.shape[0])
        contrib = {}

        for m,d in deltas.items():
            w = weights.get(m, 0.0)
            dw = deterioration_weights.get(m, w)

            val = d[:, i]

            pos = np.maximum(val, 0)
            neg = np.minimum(val, 0)

            comp = w * pos + dw * neg
            total += comp
            contrib[m] = float(np.mean(comp))

        variant_failed = not guardrail_bundle.variant_passed.get(v, True)
        penalty = guardrail_penalty * (1 if variant_failed else 0)
        total -= penalty

        score_per_draw[v] = total
        contributions[v] = contrib

    score = {v: float(np.mean(score_per_draw[v])) for v in variants}

    prob_exceeds = {
        v: float(np.mean(score_per_draw[v] > threshold))
        for v in variants
    }

    gap_distribution = {
        v: score_per_draw[v] - threshold for v in variants
    }

    gap_hdi = {
        v: _compute_hdi(gap_distribution[v])
        for v in variants
    }

    best = max(score, key=lambda k: score[k])

    if all(s < threshold for s in score.values()):
        collected.append("No variant exceeds composite threshold")

    for v,s in score.items():
        if s<0:
            collected.append(f"{v} has negative composite score")

    return CompositeResult(
        score = score,
        score_distribution = score_per_draw,
        metric_contributions = contributions,
        prob_exceeds_threshold = prob_exceeds,
        gap_distribution = gap_distribution,
        gap_hdi = gap_hdi,
        best_variant = best,
        threshold = threshold,
        warnings = collected
    )

    