"""
argonx/results/plots.py

Decision-facing visualizations for the Bayesian A/B decision engine.

Five plots, all matplotlib:
    plot_posteriors       — overlaid KDE per variant with HDI marked
    plot_lift             — lift distribution with ROPE shaded
    plot_prob_best        — P(best) bar chart per variant
    plot_expected_loss    — expected loss bar chart per variant
    plot_guardrails       — guardrail pass/fail with P(degraded) per metric

Top-level convenience:
    plot_all              — all five in one figure (2x3 grid, last cell empty)

All functions accept posterior samples and result objects directly.
None of these functions are methods on Results — they are called directly
from notebooks. Results.plot() is deliberately not implemented.

Usage:
    from argonx.results.plots import plot_all
    plot_all(samples, result)
"""

from __future__ import annotations

import warnings
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Colour palette — consistent across all plots
_PALETTE = [
    "#2563EB",  # control  — blue
    "#16A34A",  # variant 1 — green
    "#DC2626",  # variant 2 — red
    "#9333EA",  # variant 3 — purple
    "#EA580C",  # variant 4 — orange
    "#0891B2",  # variant 5 — cyan
]

_ROPE_COLOUR   = "#FCA5A5"   # soft red fill for ROPE region
_PASS_COLOUR   = "#BBF7D0"   # soft green for guardrail pass
_FAIL_COLOUR   = "#FECACA"   # soft red for guardrail fail
_GRID_ALPHA    = 0.25
_HDI_ALPHA     = 0.15
_KDE_LINEWIDTH = 2.0


def _colour(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


def _compute_hdi(samples: np.ndarray, hdi_prob: float = 0.95) -> tuple[float, float]:
    """Shortest interval containing hdi_prob of the posterior mass."""
    sorted_samples = np.sort(samples)
    n = len(sorted_samples)
    interval_width = int(np.floor(hdi_prob * n))
    if interval_width >= n:
        return float(sorted_samples[0]), float(sorted_samples[-1])
    n_intervals = n - interval_width
    widths = sorted_samples[interval_width:] - sorted_samples[:n_intervals]
    min_idx = int(np.argmin(widths))
    return float(sorted_samples[min_idx]), float(sorted_samples[min_idx + interval_width])


def _validate_samples(samples: np.ndarray, variant_names: list[str]) -> None:
    if samples.ndim != 2:
        raise ValueError(f"samples must be 2D (n_draws, n_variants), got shape {samples.shape}")
    if samples.shape[1] != len(variant_names):
        raise ValueError(
            f"samples has {samples.shape[1]} columns but {len(variant_names)} variant names"
        )
    if np.any(~np.isfinite(samples)):
        raise ValueError("samples contains NaN or Inf values")


# ---------------------------------------------------------------------------
# 1. Posterior distributions
# ---------------------------------------------------------------------------

def plot_posteriors(
    samples: np.ndarray,
    variant_names: list[str],
    metric_name: str = "metric",
    hdi_prob: float = 0.95,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[int, int] = (9, 5),
) -> plt.Axes:
    """
    Overlaid KDE posterior distributions for all variants.

    HDI is shaded under each curve. Vertical dashed line marks the HDI
    bounds. Control variant is always plotted first.

    Parameters
    ----------
    samples : np.ndarray
        Shape (n_draws, n_variants). Columns must align with variant_names
        in sorted order — same contract as BaseModel.sample_posterior().
    variant_names : list[str]
        Sorted variant names. Must match samples column order.
    metric_name : str
        Used in axis label and title.
    hdi_prob : float
        HDI coverage. Default 0.95.
    ax : plt.Axes, optional
        Existing axes to draw on. Creates new figure if None.
    figsize : tuple
        Figure size when creating new figure.

    Returns
    -------
    plt.Axes
    """
    _validate_samples(samples, variant_names)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for i, name in enumerate(variant_names):
        col = samples[:, i]
        colour = _colour(i)

        # KDE
        kde = gaussian_kde(col, bw_method="scott")
        x_min, x_max = col.min(), col.max()
        padding = (x_max - x_min) * 0.15
        x = np.linspace(x_min - padding, x_max + padding, 400)
        y = kde(x)

        ax.plot(x, y, color=colour, linewidth=_KDE_LINEWIDTH, label=name, zorder=3)

        # HDI shading
        hdi_low, hdi_high = _compute_hdi(col, hdi_prob)
        mask = (x >= hdi_low) & (x <= hdi_high)
        ax.fill_between(x, y, where=mask, color=colour, alpha=_HDI_ALPHA, zorder=2)

        # HDI boundary lines
        for bound in (hdi_low, hdi_high):
            ax.axvline(bound, color=colour, linestyle="--", linewidth=0.8, alpha=0.6, zorder=2)

    ax.set_xlabel(metric_name, fontsize=11)
    ax.set_ylabel("Posterior density", fontsize=11)
    ax.set_title(f"Posterior distributions — {metric_name}", fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.9, fontsize=10)
    ax.grid(axis="y", alpha=_GRID_ALPHA)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    hdi_pct = int(hdi_prob * 100)
    ax.annotate(
        f"Shaded region = {hdi_pct}% HDI",
        xy=(0.98, 0.97),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=8,
        color="grey",
    )

    return ax


# ---------------------------------------------------------------------------
# 2. Lift distribution with ROPE
# ---------------------------------------------------------------------------

def plot_lift(
    samples: np.ndarray,
    variant_names: list[str],
    control: str,
    rope_bounds: tuple[float, float] = (-0.01, 0.01),
    hdi_prob: float = 0.95,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[int, int] = (9, 5),
) -> plt.Axes:
    """
    Lift distribution (variant - control) / |control| for each non-control
    variant, with ROPE region shaded and P(practical effect) annotated.

    Parameters
    ----------
    samples : np.ndarray
        Shape (n_draws, n_variants).
    variant_names : list[str]
        Sorted variant names. Must match samples column order.
    control : str
        Name of the control variant.
    rope_bounds : tuple[float, float]
        (low, high) relative lift thresholds defining the ROPE.
        Default (-0.01, 0.01) = ±1%.
    hdi_prob : float
        HDI coverage for lift interval.
    ax : plt.Axes, optional
    figsize : tuple

    Returns
    -------
    plt.Axes
    """
    _validate_samples(samples, variant_names)

    if control not in variant_names:
        raise ValueError(f"control '{control}' not in variant_names {variant_names}")

    control_idx = variant_names.index(control)
    control_draws = samples[:, control_idx]

    # Avoid division by near-zero control
    denom = np.abs(control_draws)
    if np.mean(denom) < 1e-8:
        raise ValueError(
            "Control posterior mean is near zero — relative lift is undefined. "
            "Use absolute lift for this metric."
        )

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    rope_low, rope_high = rope_bounds

    # Draw ROPE region first (behind everything)
    ax.axvspan(rope_low, rope_high, color=_ROPE_COLOUR, alpha=0.35, zorder=1, label="ROPE")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.4, zorder=2)

    variant_idx = 0
    for i, name in enumerate(variant_names):
        if name == control:
            continue

        variant_draws = samples[:, i]
        lift_draws = (variant_draws - control_draws) / denom
        colour = _colour(variant_idx + 1)  # offset so control blue is skipped
        variant_idx += 1

        # KDE of lift
        kde = gaussian_kde(lift_draws, bw_method="scott")
        x_min, x_max = lift_draws.min(), lift_draws.max()
        padding = (x_max - x_min) * 0.12
        x = np.linspace(x_min - padding, x_max + padding, 400)
        y = kde(x)

        # HDI of lift
        hdi_low_lift, hdi_high_lift = _compute_hdi(lift_draws, hdi_prob)

        ax.plot(x, y, color=colour, linewidth=_KDE_LINEWIDTH, label=name, zorder=3)

        # HDI shading
        mask = (x >= hdi_low_lift) & (x <= hdi_high_lift)
        ax.fill_between(x, y, where=mask, color=colour, alpha=_HDI_ALPHA, zorder=2)

        # P(practical effect) — outside ROPE
        prob_practical = float(np.mean((lift_draws < rope_low) | (lift_draws > rope_high)))
        mean_lift = float(np.mean(lift_draws))

        # Annotate mean lift and prob practical
        y_peak = float(kde(mean_lift))
        ax.annotate(
            f"{name}\nμ={mean_lift:+.1%}\nP(practical)={prob_practical:.2f}",
            xy=(mean_lift, y_peak),
            xytext=(mean_lift, y_peak * 1.12),
            ha="center",
            va="bottom",
            fontsize=8,
            color=colour,
            arrowprops=dict(arrowstyle="-", color=colour, lw=0.6),
        )

    ax.set_xlabel("Relative lift vs control", fontsize=11)
    ax.set_ylabel("Posterior density", fontsize=11)
    ax.set_title("Lift distribution with ROPE", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax.legend(framealpha=0.9, fontsize=10)
    ax.grid(axis="y", alpha=_GRID_ALPHA)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax


# ---------------------------------------------------------------------------
# 3. P(best) bar chart
# ---------------------------------------------------------------------------

def plot_prob_best(
    prob_best: dict[str, float],
    threshold: float = 0.95,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[int, int] = (7, 4),
) -> plt.Axes:
    """
    Horizontal bar chart of P(variant is best) for all variants.

    Threshold line drawn at configured prob_best_strong. Bars above
    threshold are filled solid; bars below are hatched.

    Parameters
    ----------
    prob_best : dict[str, float]
        Output of PBestResult.prob_best — {variant_name: probability}.
    threshold : float
        Decision threshold. Default 0.95 matches DEFAULT_CONFIG.
    ax : plt.Axes, optional
    figsize : tuple

    Returns
    -------
    plt.Axes
    """
    if not prob_best:
        raise ValueError("prob_best is empty")

    variants = sorted(prob_best.keys())
    probs = [prob_best[v] for v in variants]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(
        variants,
        probs,
        color=[_colour(i) for i in range(len(variants))],
        edgecolor="white",
        height=0.55,
        zorder=2,
    )

    # Hatch bars below threshold
    for bar, prob in zip(bars, probs):
        if prob < threshold:
            bar.set_hatch("///")
            bar.set_alpha(0.65)

    # Threshold line
    ax.axvline(
        threshold,
        color="#374151",
        linewidth=1.4,
        linestyle="--",
        zorder=3,
        label=f"Threshold ({threshold:.0%})",
    )

    # Value labels
    for bar, prob in zip(bars, probs):
        ax.text(
            min(prob + 0.01, 0.98),
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.3f}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlim(0, 1.08)
    ax.set_xlabel("P(variant is best)", fontsize=11)
    ax.set_title("Probability of being best variant", fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.9, fontsize=9)
    ax.grid(axis="x", alpha=_GRID_ALPHA)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.annotate(
        "Simultaneous N-variant argmax — not pairwise",
        xy=(0.98, 0.02),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=7,
        color="grey",
    )

    return ax


# ---------------------------------------------------------------------------
# 4. Expected loss bar chart
# ---------------------------------------------------------------------------

def plot_expected_loss(
    expected_loss: dict[str, float],
    cvar_loss: Optional[dict[str, float]] = None,
    loss_threshold: float = 0.01,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[int, int] = (7, 4),
) -> plt.Axes:
    """
    Horizontal bar chart of expected loss per variant, with optional CVaR
    overlay markers and threshold line.

    Parameters
    ----------
    expected_loss : dict[str, float]
        {variant_name: expected_loss_value}. From LossResult.expected_loss.
    cvar_loss : dict[str, float], optional
        {variant_name: cvar_value}. From CVaRResult.cvar. If provided,
        diamond markers are overlaid showing tail risk.
    loss_threshold : float
        Maximum acceptable expected loss. Default 0.01 matches DEFAULT_CONFIG.
    ax : plt.Axes, optional
    figsize : tuple

    Returns
    -------
    plt.Axes
    """
    if not expected_loss:
        raise ValueError("expected_loss is empty")

    variants = sorted(expected_loss.keys())
    losses = [expected_loss[v] for v in variants]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(
        variants,
        losses,
        color=[_colour(i) for i in range(len(variants))],
        edgecolor="white",
        height=0.55,
        zorder=2,
    )

    # CVaR diamond markers
    if cvar_loss is not None:
        for i, name in enumerate(variants):
            if name in cvar_loss:
                ax.plot(
                    cvar_loss[name],
                    i,
                    marker="D",
                    color=_colour(i),
                    markeredgecolor="#1F2937",
                    markeredgewidth=0.8,
                    markersize=8,
                    zorder=4,
                    label="CVaR" if i == 0 else "_nolegend_",
                )

    # Threshold line
    ax.axvline(
        loss_threshold,
        color="#DC2626",
        linewidth=1.4,
        linestyle="--",
        zorder=3,
        label=f"Loss threshold ({loss_threshold:.2f})",
    )

    # Value labels
    max_loss = max(losses) if losses else loss_threshold
    for bar, loss in zip(bars, losses):
        ax.text(
            loss + max_loss * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{loss:.4f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_xlabel("Expected loss if variant is shipped", fontsize=11)
    ax.set_title("Expected loss per variant", fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.9, fontsize=9)
    ax.grid(axis="x", alpha=_GRID_ALPHA)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if cvar_loss is not None:
        ax.annotate(
            "◆ = CVaR (tail risk)",
            xy=(0.98, 0.02),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=7,
            color="grey",
        )

    return ax


# ---------------------------------------------------------------------------
# 5. Guardrail status panel
# ---------------------------------------------------------------------------

def plot_guardrails(
    guardrail_results: list,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[int, int] = (8, 4),
) -> plt.Axes:
    """
    Horizontal bar chart of P(degraded) per guardrail metric per variant.
    Bars are coloured green (pass) or red (fail). Pass/fail threshold line
    is drawn.

    Parameters
    ----------
    guardrail_results : list[GuardrailResult]
        List of GuardrailResult objects from GuardrailBundle.results.
        Each has: .metric, .variant, .prob_degraded, .passed, .threshold.
    ax : plt.Axes, optional
    figsize : tuple

    Returns
    -------
    plt.Axes
        Returns empty axes with explanatory text if guardrail_results is empty.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if not guardrail_results:
        ax.text(
            0.5, 0.5,
            "No guardrail metrics defined",
            ha="center", va="center",
            fontsize=12, color="grey",
            transform=ax.transAxes,
        )
        ax.set_title("Guardrail status", fontsize=13, fontweight="bold")
        ax.axis("off")
        return ax

    # Build labels and values
    labels = []
    probs = []
    colours = []
    thresholds = []

    for gr in guardrail_results:
        label = f"{gr.metric}\n({gr.variant})"
        labels.append(label)
        probs.append(gr.prob_degraded)
        colours.append(_PASS_COLOUR if gr.passed else _FAIL_COLOUR)
        thresholds.append(gr.threshold)

    y_pos = np.arange(len(labels))

    bars = ax.barh(
        y_pos,
        probs,
        color=colours,
        edgecolor="#6B7280",
        linewidth=0.8,
        height=0.55,
        zorder=2,
    )

    # Threshold line — use first threshold (typically all same)
    threshold = thresholds[0] if thresholds else 0.1
    ax.axvline(
        threshold,
        color="#374151",
        linewidth=1.4,
        linestyle="--",
        zorder=3,
        label=f"Degradation threshold ({threshold:.2f})",
    )

    # Pass/fail labels on bars
    for bar, prob, gr in zip(bars, probs, guardrail_results):
        status = "PASS ✓" if gr.passed else "FAIL ✗"
        ax.text(
            prob + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.2f}  {status}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color="#166534" if gr.passed else "#991B1B",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("P(metric degraded)", fontsize=11)
    ax.set_title("Guardrail status", fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.9, fontsize=9)
    ax.grid(axis="x", alpha=_GRID_ALPHA)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend patches
    pass_patch = mpatches.Patch(color=_PASS_COLOUR, label="Pass")
    fail_patch = mpatches.Patch(color=_FAIL_COLOUR, label="Fail")
    existing_handles, existing_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=existing_handles + [pass_patch, fail_patch],
        labels=existing_labels + ["Pass", "Fail"],
        framealpha=0.9,
        fontsize=9,
    )

    return ax


# ---------------------------------------------------------------------------
# Convenience — all five plots in one figure
# ---------------------------------------------------------------------------

def plot_all(
    samples: np.ndarray,
    variant_names: list[str],
    control: str,
    prob_best: dict[str, float],
    expected_loss: dict[str, float],
    guardrail_results: list,
    cvar_loss: Optional[dict[str, float]] = None,
    rope_bounds: tuple[float, float] = (-0.01, 0.01),
    metric_name: str = "metric",
    hdi_prob: float = 0.95,
    prob_best_threshold: float = 0.95,
    loss_threshold: float = 0.01,
    figsize: tuple[int, int] = (18, 11),
    suptitle: Optional[str] = None,
) -> plt.Figure:
    """
    All five decision plots in a single 2x3 figure.

    Layout:
        [posteriors]    [lift + ROPE]    [P(best)]
        [exp loss]      [guardrails]     [empty]

    Parameters
    ----------
    samples : np.ndarray
        Shape (n_draws, n_variants).
    variant_names : list[str]
        Sorted variant names matching samples columns.
    control : str
        Name of control variant.
    prob_best : dict[str, float]
        From PBestResult.prob_best.
    expected_loss : dict[str, float]
        From LossResult.expected_loss.
    guardrail_results : list[GuardrailResult]
        From GuardrailBundle.results. Pass [] if no guardrails.
    cvar_loss : dict[str, float], optional
        From CVaRResult.cvar. Overlaid on loss plot as diamonds.
    rope_bounds : tuple[float, float]
        ROPE bounds for lift plot.
    metric_name : str
        Primary metric name for axis labels.
    hdi_prob : float
        HDI coverage probability.
    prob_best_threshold : float
        Decision threshold for P(best) plot.
    loss_threshold : float
        Decision threshold for expected loss plot.
    figsize : tuple
        Overall figure size.
    suptitle : str, optional
        Figure-level title. Defaults to "Experiment Decision Report".

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(
        suptitle or "Experiment Decision Report",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )

    plot_posteriors(
        samples, variant_names, metric_name=metric_name,
        hdi_prob=hdi_prob, ax=axes[0, 0]
    )
    plot_lift(
        samples, variant_names, control=control,
        rope_bounds=rope_bounds, hdi_prob=hdi_prob, ax=axes[0, 1]
    )
    plot_prob_best(
        prob_best, threshold=prob_best_threshold, ax=axes[0, 2]
    )
    plot_expected_loss(
        expected_loss, cvar_loss=cvar_loss,
        loss_threshold=loss_threshold, ax=axes[1, 0]
    )
    plot_guardrails(guardrail_results, ax=axes[1, 1])

    # Last cell — decision summary text box
    ax_summary = axes[1, 2]
    ax_summary.axis("off")

    best = max(prob_best, key=prob_best.get)
    best_prob = prob_best[best]
    best_loss = expected_loss.get(best, float("nan"))
    n_guardrails = len(guardrail_results)
    n_failed = sum(1 for gr in guardrail_results if not gr.passed)

    summary_lines = [
        "DECISION SUMMARY",
        "",
        f"Best variant:  {best}",
        f"P(best):       {best_prob:.3f}",
        f"Expected loss: {best_loss:.4f}",
        "",
        f"Guardrails:    {n_guardrails - n_failed}/{n_guardrails} passed",
    ]

    if n_failed > 0:
        summary_lines.append("")
        summary_lines.append("⚠  Guardrail violation —")
        summary_lines.append("   human review required")

    summary_text = "\n".join(summary_lines)
    box_colour = "#FEF9C3" if n_failed > 0 else "#F0FDF4"
    border_colour = "#CA8A04" if n_failed > 0 else "#16A34A"

    ax_summary.text(
        0.5, 0.55,
        summary_text,
        transform=ax_summary.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.7",
            facecolor=box_colour,
            edgecolor=border_colour,
            linewidth=1.5,
        ),
    )

    fig.tight_layout()
    return fig