from __future__ import annotations

import numpy as np 
import warnings

from dataclasses import dataclass, field

# CVaR at alpha=0.95 needs enough draws in the tail to be reliable.
# At 1000 draws, the tail has ~50 observations, which is an acceptable minimum.
MIN_DRAWS_CVAR = 1000

# HDI via sorting needs enough mass to find the shortest interval stably.
MIN_DRAWS_HDI = 500


@dataclass
class PBestResult:
    """Probability that each variant is the best across all variants."""
    probabilities: dict[str, float]
    best_variant: str
    n_draws: int


@dataclass
class LossResult:
    """Expected loss and per-draw loss distributions for each variant."""
    expected_loss: dict[str, float]
    loss_distributions: dict[str, np.ndarray]
    control: str
    n_draws: int


@dataclass
class CVaRResult:
    """Tail risk (CVaR) and corresponding VaR thresholds per variant."""
    cvar: dict[str, float]
    var_threshold: dict[str, float]
    alpha: float
    n_draws: int


@dataclass 
class ROPEResult:
    """ROPE analysis for practical significance per variant."""
    inside_rope: dict[str, float]
    outside_rope: dict[str, float]
    prob_practical: dict[str, float]
    rope_low: float
    rope_high: float
    control: str


@dataclass
class LiftResult:
    """Lift mean and HDI bounds per variant relative to control."""
    mean: dict[str, float]
    hdi_low: dict[str, float]
    hdi_high: dict[str, float]
    hdi_prob: float
    control: str
    n_draws: int 


@dataclass
class MetricsBundle:
    """All computed metrics returned to the decision engine."""
    prob_best: PBestResult
    loss: LossResult
    cvar: CVaRResult
    rope: ROPEResult
    lift: LiftResult
    warnings: list[str] = field(default_factory=list)


def _validate_inputs(samples: np.ndarray, variant_names: list[str], control: str) -> None:
    """
    Validate metrics-layer preconditions.

    Checks that variant_names aligns with sample columns, control exists
    in variant_names, and samples contain no NaN or Inf values.
    Does NOT re-check shape or dtype — those are guaranteed by BaseModel.
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
            "samples contain NaN or Inf values. Check model output — "
            "LogNormal exp() transforms can overflow at extreme parameter values."
        )
    

def _validate_config(alpha: float, hdi_prob: float, rope_bounds: tuple[float, float]) -> None:
    """
    Validate user-supplied configuration parameters.

    Checks alpha and hdi_prob are in (0, 1), and rope_bounds is a valid
    2-tuple with rope_low strictly less than rope_high.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    if not 0 < hdi_prob < 1:
        raise ValueError(f"hdi_prob must be in (0, 1), got {hdi_prob}")

    if len(rope_bounds) != 2:
        raise ValueError(
            f"rope_bounds must be a 2-tuple (low, high), got {rope_bounds}"
        )
    
    rope_low, rope_high = rope_bounds
    if rope_low >= rope_high:
        raise ValueError(
            f"rope_bounds must satisfy rope_low < rope_high, "
            f"got ({rope_low}, {rope_high})"
        )


def _check_sample_quality(samples: np.ndarray, alpha: float) -> list[str]:
    """
    Check sample quality and emit warnings for unreliable estimates.

    Does not raise — computation proceeds. Warns if draw count is too low
    for stable CVaR or HDI, or if any variant has a degenerate posterior.
    Returns a list of warning strings passed through to MetricsBundle.
    """
    collected = []
    n_draws = samples.shape[0]

    if n_draws < MIN_DRAWS_CVAR:
        msg = (
            f"n_draws={n_draws} is below recommended minimum ({MIN_DRAWS_CVAR}) "
            f"for stable CVaR at alpha={alpha}. "
            f"Tail has only ~{int(n_draws * (1 - alpha))} observations. "
            f"Increase n_draws for reliable tail risk estimates."
        )
        warnings.warn(msg, UserWarning, stacklevel=3)
        collected.append(msg)
 
    if n_draws < MIN_DRAWS_HDI:
        msg = (
            f"n_draws={n_draws} is below recommended minimum ({MIN_DRAWS_HDI}) "
            f"for stable HDI computation. HDI bounds may be noisy."
        )
        warnings.warn(msg, UserWarning, stacklevel=3)
        collected.append(msg)
 
    for i in range(samples.shape[1]):
        if np.std(samples[:, i]) < 1e-10:
            msg = (
                f"Variant at column {i} has near-zero posterior variance. "
                f"MCMC chain may be degenerate. Check model and data."
            )
            warnings.warn(msg, UserWarning, stacklevel=3)
            collected.append(msg)
 
    return collected


def compute_prob_best(samples: np.ndarray, variant_names: list[str]) -> PBestResult:
    """
    Compute P(variant is best) via simultaneous argmax across all variants.

    At each posterior draw, the variant with the highest sample value wins.
    P(best) is the fraction of draws each variant wins. Values sum to 1.0.
    This is the correct N-variant approach — pairwise comparison is wrong
    for 3+ variants as it double-counts the probability space.

    Parameters
    ----------
    samples : np.ndarray
        Array of posterior draws representing performance of each variant.
    variant_names : list[str]
        Ordered sequence of variant identifiers matching array columns.

    Returns
    -------
    PBestResult
        Probabilities of each variant genuinely being the best configuration.
    """
    n_draws = samples.shape[0]

    winner_per_draw = np.argmax(samples, axis=1)

    probabilities = {
        variant: float(np.mean(winner_per_draw == i))
        for i, variant in enumerate(variant_names)
    }

    best_variant = max(probabilities, key=probabilities.__getitem__)

    return PBestResult(
        probabilities=probabilities,
        best_variant=best_variant,
        n_draws=n_draws,
    )


def compute_expected_loss(samples: np.ndarray, variant_names: list[str], control: str) -> LossResult:
    """
    Compute expected loss for each variant if selected as the winner.

    For each variant, loss at a draw is how much better the best alternative
    is at that draw. Clipped at zero — you cannot lose a negative amount.
    Retains per-draw loss distributions so CVaR can reuse them directly.

    Parameters
    ----------
    samples : np.ndarray
        Array of posterior draws across all variants.
    variant_names : list[str]
        Identifiers corresponding to the columns in `samples`.
    control : str
        The baseline variant identifier.

    Returns
    -------
    LossResult
        Per-variant expected loss and associated posterior distributions.
    """
    n_draws, n_variants = samples.shape
    expected_loss = {}
    loss_distributions = {}
 
    for i, variant in enumerate(variant_names):
        variant_draws = samples[:, i]
 
        other_cols = np.concatenate(
            [samples[:, :i], samples[:, i + 1:]], axis=1
        )
        best_alternative = np.max(other_cols, axis=1)
 
        loss_per_draw = np.maximum(0.0, best_alternative - variant_draws)
 
        loss_distributions[variant] = loss_per_draw
        expected_loss[variant] = float(np.mean(loss_per_draw))
 
    return LossResult(
        expected_loss=expected_loss,
        loss_distributions=loss_distributions,
        control=control,
        n_draws=n_draws,
    )


def compute_cvar(loss_result: LossResult, variant_names: list[str], alpha: float = 0.95) -> CVaRResult:
    """
    Compute Conditional Value at Risk (CVaR) from precomputed loss distributions.

    CVaR is the mean loss in the worst alpha-fraction of draws. Controls tail
    risk when expected loss looks acceptable but extreme outcomes are bad.
    Takes LossResult directly to avoid recomputing loss distributions.
    CVaR >= expected_loss by construction — a large gap signals tail risk.

    Parameters
    ----------
    loss_result : LossResult
        Precomputed loss distributions generated from `compute_expected_loss`.
    variant_names : list[str]
        Identifiers corresponding to the underlying variant loss streams.
    alpha : float, optional
        Confidence level determining tail size cutoff (e.g., 0.95), by default 0.95.

    Returns
    -------
    CVaRResult
        Estimated Conditional Value at Risk mapping per variant alongside VaR bounds.
    """
    cvar = {}
    var_threshold = {}
 
    for variant in variant_names:
        loss_dist = loss_result.loss_distributions[variant]
 
        var = float(np.percentile(loss_dist, alpha * 100))
        tail_losses = loss_dist[loss_dist >= var]
 
        cvar_val = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var
 
        cvar[variant] = cvar_val
        var_threshold[variant] = var
 
    return CVaRResult(
        cvar=cvar,
        var_threshold=var_threshold,
        alpha=alpha,
        n_draws=loss_result.n_draws,
    )


def compute_rope(samples: np.ndarray, variant_names: list[str], control: str, rope_low: float, rope_high: float) -> ROPEResult:
    """
    Compute ROPE (Region of Practical Equivalence) analysis per variant.

    Computes relative lift draw-by-draw and asks what fraction falls inside
    vs outside the ROPE region. Effects inside the ROPE are practically
    irrelevant regardless of statistical significance — do not ship for them.
    rope_bounds have no default — the business defines what is irrelevant.
    """
    control_idx = variant_names.index(control)
    control_draws = samples[:, control_idx]
 
    inside_rope = {}
    outside_rope = {}
    prob_practical = {}
 
    for i, variant in enumerate(variant_names):
        if variant == control:
            inside_rope[variant] = 1.0
            outside_rope[variant] = 0.0
            prob_practical[variant] = 0.0
            continue
 
        variant_draws = samples[:, i]
 
        nonzero_mask = control_draws != 0
        if not nonzero_mask.any():
            raise ValueError(
                f"control '{control}' has all-zero posterior draws. "
                f"Cannot compute relative lift."
            )
 
        lift = (
            (variant_draws[nonzero_mask] - control_draws[nonzero_mask])
            / np.abs(control_draws[nonzero_mask])
        )
 
        inside_rope[variant] = float(
            np.mean((lift >= rope_low) & (lift <= rope_high))
        )
        outside_rope[variant] = float(1.0 - inside_rope[variant])
        prob_practical[variant] = float(np.mean(lift > rope_high))
 
    return ROPEResult(
        inside_rope=inside_rope,
        outside_rope=outside_rope,
        prob_practical=prob_practical,
        rope_low=rope_low,
        rope_high=rope_high,
        control=control,
    )


def _compute_hdi(samples: np.ndarray, hdi_prob: float) -> tuple[float, float]:
    """
    Compute the Highest Density Interval (HDI) for a 1D posterior array.

    HDI is the shortest interval containing hdi_prob of the posterior mass.
    More informative than equal-tailed intervals for skewed distributions.
    Algorithm: sort samples, slide a fixed-width window, find smallest range.
    """
    sorted_samples = np.sort(samples)
    n = len(sorted_samples)
    window = int(np.floor(hdi_prob * n))
 
    if window >= n:
        return float(sorted_samples[0]), float(sorted_samples[-1])
 
    interval_widths = sorted_samples[window:] - sorted_samples[:n - window]
    min_idx = int(np.argmin(interval_widths))
 
    return float(sorted_samples[min_idx]), float(sorted_samples[min_idx + window])


def compute_lift_hdi(samples: np.ndarray, variant_names: list[str], control: str, hdi_prob: float = 0.95) -> LiftResult:
    """
    Compute expected lift with Highest Density Interval (HDI) per variant.

    Lift is computed draw-by-draw as (variant - control) / |control|.
    HDI is not a confidence interval — it is an actual probability statement
    that hdi_prob of the posterior mass lies within the returned bounds.
    """
    control_idx = variant_names.index(control)
    control_draws = samples[:, control_idx]
    n_draws = samples.shape[0]
 
    mean = {}
    hdi_low = {}
    hdi_high = {}
 
    for i, variant in enumerate(variant_names):
        if variant == control:
            mean[variant] = 0.0
            hdi_low[variant] = 0.0
            hdi_high[variant] = 0.0
            continue
 
        variant_draws = samples[:, i]
 
        nonzero_mask = control_draws != 0
        if not nonzero_mask.any():
            raise ValueError(
                f"control '{control}' has all-zero posterior draws. "
                f"Cannot compute relative lift."
            )
 
        lift = (
            (variant_draws[nonzero_mask] - control_draws[nonzero_mask])
            / np.abs(control_draws[nonzero_mask])
        )
 
        mean[variant] = float(np.mean(lift))
        low, high = _compute_hdi(lift, hdi_prob)
        hdi_low[variant] = low
        hdi_high[variant] = high
 
    return LiftResult(
        mean=mean,
        hdi_low=hdi_low,
        hdi_high=hdi_high,
        hdi_prob=hdi_prob,
        control=control,
        n_draws=n_draws,
    )


def compute_all_metrics(
    samples: np.ndarray,
    variant_names: list[str],
    control: str,
    rope_bounds: tuple[float, float],
    alpha: float = 0.95,
    hdi_prob: float = 0.95,
) -> MetricsBundle:
    """
    Compute all Bayesian decision metrics and return a populated MetricsBundle.

    Single entry point for engine.py. Validates inputs once, runs all metric
    computations in dependency order, collects warnings, and returns one clean
    object. CVaR is computed after loss to reuse loss distributions directly.
    rope_bounds is required — no default. The business defines what is irrelevant.
    """
    rope_low, rope_high = rope_bounds
 
    _validate_inputs(samples, variant_names, control)
    _validate_config(alpha, hdi_prob, rope_bounds)
 
    collected_warnings = _check_sample_quality(samples, alpha)
 
    prob_best = compute_prob_best(samples, variant_names)
    loss = compute_expected_loss(samples, variant_names, control)
    cvar = compute_cvar(loss, variant_names, alpha=alpha)
    rope_result = compute_rope(
        samples, variant_names, control,
        rope_low=rope_low, rope_high=rope_high,
    )
    lift = compute_lift_hdi(
        samples, variant_names, control,
        hdi_prob=hdi_prob,
    )
 
    for variant in variant_names:
        el = loss.expected_loss[variant]
        cv = cvar.cvar[variant]
        if el > 0 and cv / el > 5.0:
            msg = (
                f"Variant '{variant}': CVaR ({cv:.4f}) is more than 5x "
                f"expected loss ({el:.4f}). Tail risk is disproportionate — "
                f"review before shipping."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
            collected_warnings.append(msg)
 
    return MetricsBundle(
        prob_best=prob_best,
        loss=loss,
        cvar=cvar,
        rope=rope_result,
        lift=lift,
        warnings=collected_warnings,
    )