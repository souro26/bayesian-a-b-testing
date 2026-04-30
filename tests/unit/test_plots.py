"""Test suite for argonx.results.plots.

Notes
-----
Plots are visual; tests verify that figures represent the underlying data
honestly and that the visual components reflect expected decision output.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np
import pytest
from dataclasses import dataclass

from argonx.results.plots import (
    _FAIL_COLOUR,
    _PASS_COLOUR,
    plot_all,
    plot_expected_loss,
    plot_guardrails,
    plot_lift,
    plot_posteriors,
    plot_prob_best,
)


@dataclass
class _GuardrailResult:
    """Stub for a guardrail result with fields accessed in plots.py.

    Parameters
    ----------
    metric : str
        Metric name.
    variant : str
        Variant name.
    prob_degraded : float
        Probability of degradation.
    passed : bool
        Whether the guardrail passed.
    threshold : float
        Degradation threshold.
    """
    metric: str
    variant: str
    prob_degraded: float
    passed: bool
    threshold: float


VARIANTS_2 = ["control", "variant_b"]
VARIANTS_3 = ["control", "variant_b", "variant_c"]


def _samples(n_draws: int = 800, n_variants: int = 2, seed: int = 0) -> np.ndarray:
    """Generate lognormal posterior samples for testing.

    Parameters
    ----------
    n_draws : int
        Number of draws.
    n_variants : int
        Number of variants.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Posterior sample array with shape (n_draws, n_variants).
    """
    rng = np.random.default_rng(seed)
    return rng.lognormal(mean=0.0, sigma=0.3, size=(n_draws, n_variants))


def _prob_best(variants: list[str], winner: str) -> dict[str, float]:
    """Create synthetic probability-best dictionary with a dominant winner.

    Parameters
    ----------
    variants : list[str]
        Variant names.
    winner : str
        Name of the winning variant.

    Returns
    -------
    dict[str, float]
        Probability mass per variant, with winner at 0.92.
    """
    remainder = (1.0 - 0.92) / (len(variants) - 1)
    return {v: (0.92 if v == winner else remainder) for v in variants}


def _expected_loss(variants: list[str], best: str) -> dict[str, float]:
    """Create synthetic expected loss dictionary.

    Parameters
    ----------
    variants : list[str]
        Variant names.
    best : str
        Name of the best variant.

    Returns
    -------
    dict[str, float]
        Expected loss per variant.
    """
    return {v: (0.002 if v == best else 0.045) for v in variants}


def _close_figures():
    """Close all matplotlib figures to prevent memory leaks.

    Notes
    -----
    Called in teardown_method of all test classes.
    """
    plt.close("all")


class TestInputValidation:
    """Test input validation for plot functions.

    Notes
    -----
    These tests verify invalid inputs raise the expected errors.
    """

    def teardown_method(self, _):
        _close_figures()

    def test_plot_posteriors_1d_samples_raises(self):
        with pytest.raises(ValueError, match="2D"):
            plot_posteriors(np.ones(10), VARIANTS_2)

    def test_plot_posteriors_column_mismatch_raises(self):
        samples = _samples(n_variants=3)
        with pytest.raises(ValueError):
            plot_posteriors(samples, VARIANTS_2)

    def test_plot_posteriors_nan_raises(self):
        bad = _samples().copy()
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            plot_posteriors(bad, VARIANTS_2)

    def test_plot_posteriors_inf_raises(self):
        bad = _samples().copy()
        bad[5, 1] = np.inf
        with pytest.raises(ValueError):
            plot_posteriors(bad, VARIANTS_2)

    def test_plot_lift_missing_control_raises(self):
        with pytest.raises(ValueError, match="control"):
            plot_lift(_samples(), VARIANTS_2, control="nonexistent")

    def test_plot_lift_near_zero_control_raises(self):
        rng = np.random.default_rng(1)
        samples = np.column_stack([
            rng.uniform(0, 1e-10, 800),
            rng.lognormal(0, 0.3, 800),
        ])
        with pytest.raises(ValueError, match="near zero"):
            plot_lift(samples, VARIANTS_2, control="control")

    def test_plot_prob_best_empty_dict_raises(self):
        with pytest.raises(ValueError):
            plot_prob_best({})

    def test_plot_expected_loss_empty_dict_raises(self):
        with pytest.raises(ValueError):
            plot_expected_loss({})


class TestReturnTypes:
    """Test return types from plot functions.

    Notes
    -----
    These tests confirm each plot helper returns the expected Axes or Figure.
    """

    def teardown_method(self, _):
        _close_figures()

    def test_plot_posteriors_returns_axes(self):
        ax = plot_posteriors(_samples(), VARIANTS_2)
        assert isinstance(ax, plt.Axes)

    def test_plot_lift_returns_axes(self):
        ax = plot_lift(_samples(), VARIANTS_2, control="control")
        assert isinstance(ax, plt.Axes)

    def test_plot_prob_best_returns_axes(self):
        ax = plot_prob_best(_prob_best(VARIANTS_2, "variant_b"))
        assert isinstance(ax, plt.Axes)

    def test_plot_expected_loss_returns_axes(self):
        ax = plot_expected_loss(_expected_loss(VARIANTS_2, "variant_b"))
        assert isinstance(ax, plt.Axes)

    def test_plot_guardrails_returns_axes(self):
        grs = [_GuardrailResult("error_rate", "variant_b", 0.12, False, 0.10)]
        ax = plot_guardrails(grs)
        assert isinstance(ax, plt.Axes)

    def test_plot_guardrails_empty_list_returns_axes(self):
        ax = plot_guardrails([])
        assert isinstance(ax, plt.Axes)

    def test_plot_all_returns_figure(self):
        samples = _samples()
        fig = plot_all(
            samples=samples,
            variant_names=VARIANTS_2,
            control="control",
            prob_best=_prob_best(VARIANTS_2, "variant_b"),
            expected_loss=_expected_loss(VARIANTS_2, "variant_b"),
            guardrail_results=[],
        )
        assert isinstance(fig, plt.Figure)


class TestCurveCount:
    """Test curve counts match variant counts.

    Notes
    -----
    These tests ensure the number of plotted elements equals the number of variants.
    """

    def teardown_method(self, _):
        _close_figures()

    def test_posteriors_one_line_per_variant(self):
        ax = plot_posteriors(_samples(n_variants=2), VARIANTS_2)
        labelled_lines = [l for l in ax.get_lines() if l.get_label() in VARIANTS_2]
        assert len(labelled_lines) == len(VARIANTS_2)

    def test_posteriors_three_variants_three_lines(self):
        ax = plot_posteriors(_samples(n_variants=3), VARIANTS_3)
        labelled_lines = [l for l in ax.get_lines() if l.get_label() in VARIANTS_3]
        assert len(labelled_lines) == 3

    def test_prob_best_one_bar_per_variant(self):
        pb = _prob_best(VARIANTS_3, "variant_b")
        ax = plot_prob_best(pb)
        bars = [p for p in ax.patches if p.get_width() > 0]
        assert len(bars) == len(VARIANTS_3)

    def test_expected_loss_one_bar_per_variant(self):
        el = _expected_loss(VARIANTS_3, "variant_b")
        ax = plot_expected_loss(el)
        bars = [p for p in ax.patches if p.get_width() > 0]
        assert len(bars) == len(VARIANTS_3)

    def test_lift_excludes_control_from_curves(self):
        ax = plot_lift(_samples(n_variants=2), VARIANTS_2, control="control")
        labelled = [l.get_label() for l in ax.get_lines()
                    if not l.get_label().startswith("_")]
        assert "control" not in labelled

    def test_lift_three_variants_two_curves(self):
        ax = plot_lift(_samples(n_variants=3), VARIANTS_3, control="control")
        non_control_labels = [l.get_label() for l in ax.get_lines()
                              if l.get_label() in ("variant_b", "variant_c")]
        assert len(non_control_labels) == 2


class TestPlotAllSummary:
    """Test the summary panel in plot_all.

    Notes
    -----
    The summary is the human-facing decision output and must report
    the correct winner and flag guardrail violations.
    """

    def teardown_method(self, _):
        _close_figures()

    def _make_fig(self, winner: str, variants: list[str],
                  guardrail_results: list | None = None) -> plt.Figure:
        samples = _samples(n_draws=len(variants))
        return plot_all(
            samples=samples,
            variant_names=variants,
            control="control",
            prob_best=_prob_best(variants, winner),
            expected_loss=_expected_loss(variants, winner),
            guardrail_results=guardrail_results or [],
        )

    def _summary_text(self, fig: plt.Figure) -> str:
        """Extract text from the summary axes.

        Parameters
        ----------
        fig : plt.Figure
            Figure from plot_all.

        Returns
        -------
        str
            Concatenated text from the summary panel.
        """
        axes = fig.get_axes()
        summary_ax = axes[-1]
        texts = [t.get_text() for t in summary_ax.texts]
        return "\n".join(texts)

    def test_summary_names_correct_winner(self):
        fig = self._make_fig(winner="variant_b", variants=VARIANTS_2)
        summary = self._summary_text(fig)
        assert "variant_b" in summary

    def test_summary_does_not_name_loser_as_winner(self):
        fig = self._make_fig(winner="variant_b", variants=VARIANTS_2)
        summary = self._summary_text(fig)
        lines = summary.split("\n")
        best_line = next((l for l in lines if "Best variant" in l), "")
        assert "control" not in best_line

    def test_summary_winner_is_argmax_of_prob_best(self):
        fig = self._make_fig(winner="variant_c", variants=VARIANTS_3)
        summary = self._summary_text(fig)
        assert "variant_c" in summary

    def test_summary_flags_guardrail_violation(self):
        failed_gr = _GuardrailResult("error_rate", "variant_b", 0.85, False, 0.10)
        fig = self._make_fig(winner="variant_b", variants=VARIANTS_2,
                             guardrail_results=[failed_gr])
        summary = self._summary_text(fig)
        assert ("guardrail" in summary.lower() or "violation" in summary.lower()
                or "review" in summary.lower() or "⚠" in summary)

    def test_summary_no_warning_when_all_guardrails_pass(self):
        passed_gr = _GuardrailResult("error_rate", "variant_b", 0.03, True, 0.10)
        fig = self._make_fig(winner="variant_b", variants=VARIANTS_2,
                             guardrail_results=[passed_gr])
        summary = self._summary_text(fig)
        assert "⚠" not in summary

    def test_summary_guardrail_count_is_correct(self):
        grs = [
            _GuardrailResult("error_rate", "variant_b", 0.85, False, 0.10),
            _GuardrailResult("latency", "variant_b", 0.04, True, 0.10),
            _GuardrailResult("crash_rate", "variant_b", 0.60, False, 0.10),
        ]
        fig = self._make_fig(winner="variant_b", variants=VARIANTS_2,
                             guardrail_results=grs)
        summary = self._summary_text(fig)
        assert "1/3" in summary


class TestCVaRMarkers:
    """Test CVaR marker placement in expected loss plots.

    Notes
    -----
    CVaR must be >= expected loss by mathematical definition.
    Tests verify the graphical markers respect this constraint.
    """

    def teardown_method(self, _):
        _close_figures()

    def test_cvar_marker_not_left_of_expected_loss_bar(self):
        variants = VARIANTS_2
        expected_loss = {"control": 0.040, "variant_b": 0.003}
        cvar_loss = {"control": 0.065, "variant_b": 0.007}

        ax = plot_expected_loss(expected_loss, cvar_loss=cvar_loss)

        bar_data = {}
        for patch in ax.patches:
            y_center = patch.get_y() + patch.get_height() / 2
            bar_data[round(y_center, 3)] = patch.get_width()

        marker_xs = [line.get_xdata()[0] for line in ax.lines
                     if line.get_marker() not in ("None", None, "")]

        for marker_x in marker_xs:
            closest_bar_width = min(bar_data.values(),
                                    key=lambda w: abs(w - marker_x))
            assert marker_x >= closest_bar_width - 1e-9, (
                f"CVaR marker at x={marker_x:.4f} is left of expected loss "
                f"bar width={closest_bar_width:.4f}. Plot contradicts "
                "CVaR >= expected loss."
            )

    def test_cvar_markers_absent_when_cvar_not_provided(self):
        ax = plot_expected_loss({"control": 0.04, "variant_b": 0.003})
        marker_lines = [l for l in ax.lines
                        if l.get_marker() not in ("None", None, "")]
        assert len(marker_lines) == 0

    def test_cvar_markers_present_when_cvar_provided(self):
        ax = plot_expected_loss(
            {"control": 0.04, "variant_b": 0.003},
            cvar_loss={"control": 0.07, "variant_b": 0.006},
        )
        marker_lines = [l for l in ax.lines
                        if l.get_marker() not in ("None", None, "")]
        assert len(marker_lines) > 0


class TestGuardrailColouring:
    """Test guardrail bar colours in plot_guardrails.

    Notes
    -----
    These tests ensure pass/fail colours are not swapped.
    """

    def teardown_method(self, _):
        _close_figures()

    def _hex_to_rgb(self, hex_colour: str) -> tuple:
        """Convert hex colour to RGB tuple.

        Parameters
        ----------
        hex_colour : str
            Hex colour string.

        Returns
        -------
        tuple
            RGB values in [0, 1].
        """
        h = hex_colour.lstrip("#")
        return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))

    def test_passing_guardrail_bar_is_not_red(self):
        grs = [_GuardrailResult("latency", "variant_b", 0.04, True, 0.10)]
        ax = plot_guardrails(grs)
        fail_rgb = self._hex_to_rgb(_FAIL_COLOUR)
        for patch in ax.patches:
            fc = patch.get_facecolor()[:3]
            assert not np.allclose(fc, fail_rgb, atol=0.01), (
                "Passing guardrail bar is coloured red — pass/fail colours swapped"
            )

    def test_failing_guardrail_bar_is_not_green(self):
        grs = [_GuardrailResult("error_rate", "variant_b", 0.85, False, 0.10)]
        ax = plot_guardrails(grs)
        pass_rgb = self._hex_to_rgb(_PASS_COLOUR)
        for patch in ax.patches:
            fc = patch.get_facecolor()[:3]
            assert not np.allclose(fc, pass_rgb, atol=0.01), (
                "Failing guardrail bar is coloured green — pass/fail colours swapped"
            )

    def test_mixed_guardrails_each_coloured_correctly(self):
        grs = [
            _GuardrailResult("error_rate", "variant_b", 0.85, False, 0.10),
            _GuardrailResult("latency",    "variant_b", 0.04, True,  0.10),
        ]
        ax = plot_guardrails(grs)
        pass_rgb = self._hex_to_rgb(_PASS_COLOUR)
        fail_rgb = self._hex_to_rgb(_FAIL_COLOUR)
        patches = [p for p in ax.patches if p.get_width() > 0]
        colours = [tuple(p.get_facecolor()[:3]) for p in patches]
        has_pass_colour = any(np.allclose(c, pass_rgb, atol=0.01) for c in colours)
        has_fail_colour = any(np.allclose(c, fail_rgb, atol=0.01) for c in colours)
        assert has_pass_colour, "No bar has pass colour despite one passing guardrail"
        assert has_fail_colour, "No bar has fail colour despite one failing guardrail"


class TestProbBestVisualCorrectness:
    """Test prob_best bar visual correctness.

    Notes
    -----
    The longest bar must correspond to the highest probability.
    """

    def teardown_method(self, _):
        _close_figures()

    def test_winner_bar_is_longest(self):
        pb = {"control": 0.05, "variant_b": 0.90, "variant_c": 0.05}
        ax = plot_prob_best(pb)
        bars = {p.get_y(): p.get_width() for p in ax.patches if p.get_width() > 0}
        max_width = max(bars.values())
        winner_bar_width = max(pb.values())
        assert max_width == pytest.approx(winner_bar_width, abs=1e-6)

    def test_all_bars_sum_to_approximately_one(self):
        pb = {"control": 0.03, "variant_b": 0.92, "variant_c": 0.05}
        ax = plot_prob_best(pb)
        bar_widths = [p.get_width() for p in ax.patches if p.get_width() > 0]
        assert sum(bar_widths) == pytest.approx(1.0, abs=0.01)


class TestAxesLabels:
    """Test axes labels and titles for plots.

    Notes
    -----
    These tests ensure all plots have the required labels and titles.
    """

    def teardown_method(self, _):
        _close_figures()

    def test_posteriors_has_xlabel(self):
        ax = plot_posteriors(_samples(), VARIANTS_2, metric_name="revenue")
        assert ax.get_xlabel() != ""

    def test_posteriors_metric_name_in_title(self):
        ax = plot_posteriors(_samples(), VARIANTS_2, metric_name="revenue")
        assert "revenue" in ax.get_title()

    def test_lift_has_xlabel(self):
        ax = plot_lift(_samples(), VARIANTS_2, control="control")
        assert ax.get_xlabel() != ""

    def test_lift_has_title(self):
        ax = plot_lift(_samples(), VARIANTS_2, control="control")
        assert ax.get_title() != ""

    def test_prob_best_has_xlabel(self):
        ax = plot_prob_best(_prob_best(VARIANTS_2, "variant_b"))
        assert ax.get_xlabel() != ""

    def test_expected_loss_has_xlabel(self):
        ax = plot_expected_loss(_expected_loss(VARIANTS_2, "variant_b"))
        assert ax.get_xlabel() != ""

    def test_guardrails_has_title(self):
        ax = plot_guardrails([_GuardrailResult("err", "b", 0.5, False, 0.1)])
        assert ax.get_title() != ""


class TestAxesInjection:
    """Test that plot helpers respect injected Axes objects.

    Notes
    -----
    When an Axes is passed, plot helpers must draw into it without
    creating new figures.
    """

    def teardown_method(self, _):
        _close_figures()

    def test_posteriors_uses_injected_ax(self):
        fig, ax = plt.subplots()
        n_figs_before = len(plt.get_fignums())
        returned_ax = plot_posteriors(_samples(), VARIANTS_2, ax=ax)
        assert len(plt.get_fignums()) == n_figs_before
        assert returned_ax is ax

    def test_lift_uses_injected_ax(self):
        fig, ax = plt.subplots()
        n_figs_before = len(plt.get_fignums())
        returned_ax = plot_lift(_samples(), VARIANTS_2, control="control", ax=ax)
        assert len(plt.get_fignums()) == n_figs_before
        assert returned_ax is ax

    def test_prob_best_uses_injected_ax(self):
        fig, ax = plt.subplots()
        n_figs_before = len(plt.get_fignums())
        returned_ax = plot_prob_best(_prob_best(VARIANTS_2, "variant_b"), ax=ax)
        assert len(plt.get_fignums()) == n_figs_before
        assert returned_ax is ax

    def test_expected_loss_uses_injected_ax(self):
        fig, ax = plt.subplots()
        n_figs_before = len(plt.get_fignums())
        returned_ax = plot_expected_loss(_expected_loss(VARIANTS_2, "variant_b"), ax=ax)
        assert len(plt.get_fignums()) == n_figs_before
        assert returned_ax is ax

    def test_guardrails_uses_injected_ax(self):
        fig, ax = plt.subplots()
        n_figs_before = len(plt.get_fignums())
        grs = [_GuardrailResult("err", "b", 0.5, False, 0.1)]
        returned_ax = plot_guardrails(grs, ax=ax)
        assert len(plt.get_fignums()) == n_figs_before
        assert returned_ax is ax


class TestROPEShading:
    """Test ROPE and HDI shading in lift plots.

    Notes
    -----
    These tests verify the ROPE region and HDI fill regions are present.
    """

    def teardown_method(self, _):
        _close_figures()

    def test_lift_plot_has_rope_shading(self):
        ax = plot_lift(_samples(), VARIANTS_2, control="control",
                       rope_bounds=(-0.05, 0.05))
        filled_regions = [c for c in ax.collections
                          if isinstance(c, mcoll.PolyCollection)]
        assert len(filled_regions) > 0

    def test_lift_hdi_region_is_shaded(self):
        ax = plot_lift(_samples(), VARIANTS_2, control="control")
        filled_regions = [c for c in ax.collections
                          if isinstance(c, mcoll.PolyCollection)]
        assert len(filled_regions) >= 2


class TestPlotAllGrid:
    """Test plot_all grid composition.

    Notes
    -----
    These tests ensure plot_all renders the full grid layout with all panels.
    """

    def teardown_method(self, _):
        _close_figures()

    def _make_plot_all(self, guardrail_results=None):
        return plot_all(
            samples=_samples(),
            variant_names=VARIANTS_2,
            control="control",
            prob_best=_prob_best(VARIANTS_2, "variant_b"),
            expected_loss=_expected_loss(VARIANTS_2, "variant_b"),
            guardrail_results=guardrail_results or [],
        )

    def test_plot_all_has_six_axes(self):
        fig = self._make_plot_all()
        assert len(fig.get_axes()) == 6

    def test_plot_all_five_titled_panels(self):
        fig = self._make_plot_all()
        titled = [ax for ax in fig.get_axes() if ax.get_title() != ""]
        assert len(titled) >= 4

    def test_plot_all_with_guardrail_violations_does_not_crash(self):
        grs = [
            _GuardrailResult("error_rate", "variant_b", 0.82, False, 0.10),
            _GuardrailResult("latency",    "variant_b", 0.04, True,  0.10),
        ]
        fig = self._make_plot_all(guardrail_results=grs)
        assert isinstance(fig, plt.Figure)

    def test_plot_all_three_variants_does_not_crash(self):
        fig = plot_all(
            samples=_samples(n_variants=3),
            variant_names=VARIANTS_3,
            control="control",
            prob_best=_prob_best(VARIANTS_3, "variant_b"),
            expected_loss=_expected_loss(VARIANTS_3, "variant_b"),
            guardrail_results=[],
        )
        assert isinstance(fig, plt.Figure)
