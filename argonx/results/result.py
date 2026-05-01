from __future__ import annotations

import numpy as np
import pandas as pd

from argonx.decision_rules.engine import DecisionResult
from argonx.results.plots import plot_all


class Results:
    """
    Container for the final decision recommendation and associated metrics.

    Provides high-level summaries and visualizations of the Bayesian A/B
    experiment results.

    Parameters
    ----------
    decision : DecisionResult
        The computed decision output from the engine.
    config : dict, optional
        Configuration used for the experiment, by default None.
    segment_results : dict[str, DecisionResult] | None, optional
        Results split by segment for hierarchical models, by default None.
    segment_guardrail_violations : dict[str, list[str]] | None, optional
        Guardrail violations per segment, by default None.
    """

    def __init__(
        self,
        decision: DecisionResult,
        config: dict = None,
        segment_results: dict[str, DecisionResult] | None = None,
        segment_guardrail_violations: dict[str, list[str]] | None = None,
    ) -> None:
        self._d = decision
        self._config = config or {}
        self.segment_results = segment_results
        self.segment_guardrail_violations = segment_guardrail_violations

    def __getattr__(self, name: str):
        try:
            return getattr(self._d, name)
        except AttributeError:
            raise AttributeError(
                f"Results object has no attribute '{name}'. "
                f"Available fields: {list(self._d.__dataclass_fields__.keys())}"
            )

    def __repr__(self) -> str:
        """Clean notebook display showing the key decision at a glance."""
        best = self._d.best_variant
        p_best = self._d.metrics.prob_best.probabilities.get(best, 0.0)
        lift_mean = self._d.metrics.lift.mean.get(best, 0.0)

        base = (
            f"Results(\n"
            f"  state          = {self._d.state!r}\n"
            f"  recommendation = {self._d.recommendation!r}\n"
            f"  best_variant   = {best!r}\n"
            f"  prob_best      = {p_best:.3f}\n"
            f"  lift_mean      = {lift_mean:.3f}\n"
            f"  guardrails     = {'PASS' if self._d.guardrails.all_passed else 'FAIL'}\n"
            f"  notes          = {len(self._d.notes)} flagged\n"
        )
        if self.segment_results is not None:
            base += f"  segments       = {sorted(self.segment_results.keys())}\n"
        base += ")"
        return base

    def summary(self) -> None:
        """
        Print a detailed human-readable summary of the experiment results.

        Provides metrics, risk levels, guardrail status, and final recommendation.
        """
        d = self._d
        best = d.best_variant
        metrics = d.metrics
        guardrails = d.guardrails

        lines = []

        lines.append("=" * 60)
        lines.append("EXPERIMENT RESULTS")
        if self.segment_results is not None:
            lines.append("(Aggregate — population-level)")
        lines.append("=" * 60)

        p_best = metrics.prob_best.probabilities.get(best, 0.0)
        lift_mean = metrics.lift.mean.get(best, 0.0)
        lift_low = metrics.lift.hdi_low.get(best, 0.0)
        lift_high = metrics.lift.hdi_high.get(best, 0.0)
        hdi_prob = int(metrics.lift.hdi_prob * 100)

        lines.append("")
        lines.append("PRIMARY METRIC")
        lines.append("-" * 40)
        lines.append(f"Best Variant: {best}")
        lines.append(
            f"Expected lift:    {lift_mean:+.3%} "
            f"({hdi_prob}% HDI:    {lift_low:+.3%} to {lift_high:+.3%})"
        )
        lines.append(f"P(best) across all variants: {p_best:.3f}")

        el = metrics.loss.expected_loss.get(best, 0.0)
        cv = metrics.cvar.cvar.get(best, 0.0)
        alpha = int(metrics.cvar.alpha * 100)

        lines.append("")
        lines.append("RISK")
        lines.append("-" * 40)
        lines.append(f"Expected loss if wrong:          {el:.4f}")
        lines.append(f"CVaR ({alpha}th percentile loss):    {cv:.4f}")
        lines.append(f"Risk level:                      {d.risk_level}")

        inside = metrics.rope.inside_rope.get(best, 0.0)
        outside = metrics.rope.outside_rope.get(best, 0.0)
        prob_practical = metrics.rope.prob_practical.get(best, 0.0)

        lines.append("")
        lines.append("PRACTICAL SIGNIFICANCE (ROPE)")
        lines.append("-" * 40)

        if d.practical_significance == "yes":
            lines.append("Effect is OUTSIDE ROPE — practically meaningful.")
        elif d.practical_significance == "uncertain":
            lines.append("Effect is UNCERTAIN — partially inside ROPE region.")
        else:
            lines.append("Effect is INSIDE ROPE — not practically meaningful.")

        lines.append(
            f"P(practical effect): {prob_practical:.3f}  |  "
            f"Inside ROPE: {inside:.3f}  |  Outside ROPE: {outside:.3f}"
        )

        lines.append("")
        lines.append("GUARDRAILS")
        lines.append("-" * 40)

        if not guardrails.guardrails:
            lines.append("No guardrail metrics defined.")
        else:
            for gr in guardrails.guardrails:
                status = "PASS" if gr.passed else "FAIL"
                lines.append(
                    f"  {gr.metric:<25} [{status}]  "
                    f"variant={gr.variant}  "
                    f"P(degraded)={gr.prob_degraded:.3f}  "
                    f"threshold={gr.threshold:.3f}  "
                    f"severity={gr.severity}"
                )

        if guardrails.conflicts:
            lines.append("")
            lines.append("GUARDRAIL CONFLICTS DETECTED")
            lines.append("-" * 40)
            for c in guardrails.conflicts:
                lines.append(f"{c.message}")
            lines.append("Framework cannot resolve this tradeoff. Human review required.")

        if self.segment_guardrail_violations:
            lines.append("")
            lines.append("SEGMENT GUARDRAIL VIOLATIONS")
            lines.append("-" * 40)
            lines.append(
                "The following segments have guardrail violations that may not "
                "be visible in the aggregate result above."
            )
            for seg, failed in sorted(self.segment_guardrail_violations.items()):
                lines.append(f"  Segment '{seg}': {', '.join(failed)}")
            lines.append("Run result.segment_summary() for full per-segment detail.")

        if d.joint is not None:
            lines.append("")
            lines.append("JOINT POLICY PROBABILITY")
            lines.append("-" * 40)
            lines.append(f"Metrics in policy: {', '.join(d.joint.metrics_joined)}")

            for v, jp in d.joint.joint_prob.items():
                ind = d.joint.independence_benchmark.get(v, 0.0)
                gap = d.joint.correlation_gap.get(v, 0.0)
                lines.append(
                    f"  {v:<20} joint_prob={jp:.3f}  "
                    f"independence_benchmark={ind:.3f}  "
                    f"correlation_gap={gap:+.3f}"
                )

            for v, gap in d.joint.correlation_gap.items():
                if gap < -0.10:
                    cond = d.joint.condition_probs.get(v, {})
                    if cond:
                        binding = min(cond, key=lambda k: cond[k])
                        lines.append(
                            f"  Binding constraint for {v}: '{binding}' "
                            f"(condition_prob={cond[binding]:.3f})"
                        )

        if d.composite is not None:
            lines.append("")
            lines.append("COMPOSITE DECISION SCORE")
            lines.append("-" * 40)
            lines.append(f"Threshold: {d.composite.threshold:.3f}")
            for v, s in d.composite.score.items():
                p_exc = d.composite.prob_exceeds_threshold.get(v, 0.0)
                low, high = d.composite.gap_hdi.get(v, (0.0, 0.0))
                lines.append(
                    f"  {v:<20} score={s:.4f}  "
                    f"P(exceeds threshold)={p_exc:.3f}  "
                    f"gap HDI=[{low:.3f}, {high:.3f}]"
                )

        lines.append("")
        lines.append("=" * 60)
        lines.append("DECISION")
        lines.append("-" * 40)
        lines.append(f"State:          {d.state}")
        lines.append(f"Recommendation: {d.recommendation.upper()}")
        lines.append(f"Confidence:     {d.confidence}")
        lines.append("")
        lines.append("Reasoning:")
        for r in d.reasons:
            lines.append(f"  - {r}")

        if d.notes:
            lines.append("")
            lines.append("FLAGGED ISSUES")
            lines.append("-" * 40)
            for note in d.notes:
                lines.append(f"  ! {note}")

        if self.segment_results is not None:
            lines.append("")
            lines.append("  Run result.segment_summary() for per-segment decisions.")

        lines.append("=" * 60)

        print("\n".join(lines))

    def segment_summary(self) -> None:
        """
        Print per-segment decisions with cross-segment conflict detection.

        Only available for hierarchical experiments. Summarizes the state,
        recommendation, and guardrail status for each segment, highlighting
        any inconsistent winners or shipping conflicts.

        Raises
        ------
        RuntimeError
            If called on a non-hierarchical experiment result.
        """
        if self.segment_results is None:
            raise RuntimeError(
                "segment_summary() is only available for hierarchical experiments. "
                "Set segment_col in Experiment() to enable segment-level inference."
            )

        lines = []
        lines.append("=" * 60)
        lines.append("SEGMENT RESULTS")
        lines.append("=" * 60)

        for seg in sorted(self.segment_results.keys()):
            d = self.segment_results[seg]
            best = d.best_variant
            p_best = d.metrics.prob_best.probabilities.get(best, 0.0)
            lift_mean = d.metrics.lift.mean.get(best, 0.0)
            guardrail_status = "PASS" if d.guardrails.all_passed else "FAIL"

            lines.append("")
            lines.append(f"Segment: {seg}")
            lines.append("-" * 40)
            lines.append(
                f"  Best variant:   {best}  "
                f"(P(best)={p_best:.3f})"
            )
            lines.append(
                f"  State:          {d.state}"
            )
            lines.append(
                f"  Recommendation: {d.recommendation.upper()}"
            )
            lines.append(
                f"  Expected lift:  {lift_mean:+.3%}"
            )
            lines.append(
                f"  Guardrails:     {guardrail_status}"
            )

            if not d.guardrails.all_passed:
                for gr in d.guardrails.guardrails:
                    if not gr.passed:
                        lines.append(
                            f"    ! {gr.metric}: P(degraded)={gr.prob_degraded:.3f}  "
                            f"severity={gr.severity}"
                        )

            if d.notes:
                for note in d.notes:
                    lines.append(f"    ! {note}")

        lines.append("")
        lines.append("=" * 60)
        lines.append("CROSS-SEGMENT ANALYSIS")
        lines.append("-" * 40)

        self._print_cross_segment_analysis(lines)

        lines.append("=" * 60)
        print("\n".join(lines))

    def _print_cross_segment_analysis(self, lines: list[str]) -> None:
        """Detect and describe cross-segment conflicts."""
        if self.segment_results is None:
            return

        seg_best: dict[str, str] = {}
        seg_rec: dict[str, str] = {}
        seg_state: dict[str, str] = {}

        for seg, d in self.segment_results.items():
            seg_best[seg] = d.best_variant
            seg_rec[seg] = d.recommendation
            seg_state[seg] = d.state

        unique_best = set(seg_best.values())
        if len(unique_best) == 1:
            winner = next(iter(unique_best))
            lines.append(
                f"Consistent winner across all segments: '{winner}'."
            )
        else:
            lines.append(
                "INCONSISTENT WINNER ACROSS SEGMENTS:"
            )
            for seg, best in sorted(seg_best.items()):
                lines.append(f"  {seg:<20} best={best}")

        ship_recs = {"ship variant", "consider shipping"}
        no_ship_recs = {"do not ship", "review required", "continue experiment"}

        ship_segs = [
            seg for seg, rec in seg_rec.items() if rec in ship_recs
        ]
        no_ship_segs = [
            seg for seg, rec in seg_rec.items() if rec in no_ship_recs
        ]
        inconclusive_segs = [
            seg for seg, rec in seg_rec.items()
            if rec not in ship_recs and rec not in no_ship_recs
        ]

        lines.append("")
        if ship_segs and no_ship_segs:
            lines.append(
                "SHIPPING CONFLICT DETECTED — segments disagree on the decision:"
            )
            lines.append(
                f"  Ship:       {', '.join(sorted(ship_segs))}"
            )
            lines.append(
                f"  Do not ship: {', '.join(sorted(no_ship_segs))}"
            )
            if inconclusive_segs:
                lines.append(
                    f"  Inconclusive: {', '.join(sorted(inconclusive_segs))}"
                )
            lines.append(
                "Shipping universally is not recommended. "
                "Consider segment-targeted rollout or investigate the discrepancy."
            )
        elif ship_segs and not no_ship_segs:
            lines.append(
                f"Consistent shipping signal across segments: "
                f"{', '.join(sorted(ship_segs))}."
            )
            if inconclusive_segs:
                lines.append(
                    f"Inconclusive segments (insufficient data): "
                    f"{', '.join(sorted(inconclusive_segs))}. "
                    f"Consider holding these until more data is collected."
                )
        elif no_ship_segs and not ship_segs:
            lines.append(
                f"No segments support shipping. "
                f"Do not ship."
            )
        else:
            lines.append("All segments inconclusive. Continue experiment.")

        if self.segment_guardrail_violations:
            lines.append("")
            lines.append("Guardrail violations by segment:")
            for seg, failed in sorted(self.segment_guardrail_violations.items()):
                lines.append(f"  {seg:<20} failed: {', '.join(failed)}")

    def plot(
        self,
        samples: np.ndarray,
        metric_name: str = "metric",
        rope_bounds: tuple[float, float] | None = None,
        figsize: tuple[int, int] = (18, 11),
        suptitle: str | None = None,
    ):
        """
        Render all five decision plots in a single figure.

        Parameters
        ----------
        samples : np.ndarray
            Posterior samples to plot.
        metric_name : str, optional
            Name of the primary metric, by default "metric".
        rope_bounds : tuple[float, float] | None, optional
            Region of Practical Equivalence bounds, by default None.
        figsize : tuple[int, int], optional
            Total figure size, by default (18, 11).
        suptitle : str | None, optional
            Figure super title, by default None.

        Returns
        -------
        plt.Figure
            The complete dashboard figure.
        """
        if rope_bounds is None:
            rope_bounds = self._config.get("rope_bounds", (-0.01, 0.01))

        d = self._d

        return plot_all(
            samples=samples,
            variant_names=sorted(d.metrics.prob_best.probabilities.keys()),
            control=d.metrics.loss.control,
            prob_best=d.metrics.prob_best.probabilities,
            expected_loss=d.metrics.loss.expected_loss,
            guardrail_results=d.guardrails.guardrails,
            cvar_loss=d.metrics.cvar.cvar,
            rope_bounds=rope_bounds,
            metric_name=metric_name,
            hdi_prob=d.metrics.lift.hdi_prob,
            prob_best_threshold=self._config.get("prob_best_strong", 0.95),
            loss_threshold=self._config.get("expected_loss_max", 0.01),
            figsize=figsize,
            suptitle=suptitle,
        )

    def to_dict(self) -> dict:
        """
        Serialize DecisionResult to a plain Python dict.

        All numpy arrays are converted to lists. All numpy scalars are
        converted to Python floats. Safe for JSON serialization.
        """
        d = self._d

        out = {
            "decision": {
                "state": d.state,
                "recommendation": d.recommendation,
                "best_variant": d.best_variant,
                "confidence": d.confidence,
                "primary_strength": d.primary_strength,
                "risk_level": d.risk_level,
                "practical_significance": d.practical_significance,
                "guardrail_status": d.guardrail_status,
                "reasons": d.reasons,
                "notes": d.notes,
            },
            "metrics": {
                "prob_best": d.metrics.prob_best.probabilities,
                "expected_loss": d.metrics.loss.expected_loss,
                "cvar": d.metrics.cvar.cvar,
                "lift_mean": d.metrics.lift.mean,
                "lift_hdi_low": d.metrics.lift.hdi_low,
                "lift_hdi_high": d.metrics.lift.hdi_high,
                "rope_inside": d.metrics.rope.inside_rope,
                "rope_outside": d.metrics.rope.outside_rope,
                "prob_practical": d.metrics.rope.prob_practical,
            },
            "guardrails": {
                "all_passed": d.guardrails.all_passed,
                "variant_passed": d.guardrails.variant_passed,
                "results": [
                    {
                        "metric": gr.metric,
                        "variant": gr.variant,
                        "passed": gr.passed,
                        "prob_degraded": gr.prob_degraded,
                        "threshold": gr.threshold,
                        "severity": gr.severity,
                        "expected_degradation": gr.expected_degradation,
                    }
                    for gr in d.guardrails.guardrails
                ],
                "conflicts": [
                    {
                        "metric": c.metric,
                        "variant": c.variant,
                        "prob_degraded": c.prob_degraded,
                        "threshold": c.threshold,
                        "severity": c.severity,
                        "message": c.message,
                    }
                    for c in d.guardrails.conflicts
                ],
            },
        }

        if d.joint is not None:
            out["joint"] = {
                "joint_prob": d.joint.joint_prob,
                "condition_probs": d.joint.condition_probs,
                "independence_benchmark": d.joint.independence_benchmark,
                "correlation_gap": d.joint.correlation_gap,
                "best_variant": d.joint.best_variant,
                "metrics_joined": d.joint.metrics_joined,
            }
        else:
            out["joint"] = None

        if d.composite is not None:
            out["composite"] = {
                "score": d.composite.score,
                "prob_exceeds_threshold": d.composite.prob_exceeds_threshold,
                "gap_hdi": {
                    v: list(hdi) for v, hdi in d.composite.gap_hdi.items()
                },
                "metric_contributions": d.composite.metric_contributions,
                "best_variant": d.composite.best_variant,
                "threshold": d.composite.threshold,
            }
        else:
            out["composite"] = None

        if self.segment_results is not None:
            out["segment_results"] = {}
            for seg, seg_d in self.segment_results.items():
                out["segment_results"][seg] = {
                    "state": seg_d.state,
                    "recommendation": seg_d.recommendation,
                    "best_variant": seg_d.best_variant,
                    "prob_best": seg_d.metrics.prob_best.probabilities,
                    "expected_loss": seg_d.metrics.loss.expected_loss,
                    "lift_mean": seg_d.metrics.lift.mean,
                    "guardrail_passed": seg_d.guardrails.variant_passed,
                    "notes": seg_d.notes,
                }
            out["segment_guardrail_violations"] = self.segment_guardrail_violations
        else:
            out["segment_results"] = None
            out["segment_guardrail_violations"] = None

        return out

    def to_dataframe(self) -> pd.DataFrame:
        """Return per-variant metrics as a pandas DataFrame."""
        if self.segment_results is not None:
            return self._to_dataframe_hierarchical()
        return self._to_dataframe_flat()

    def _to_dataframe_flat(self) -> pd.DataFrame:
        """Original flat dataframe logic — unchanged."""
        d = self._d
        metrics = d.metrics

        variants = [
            v for v in metrics.prob_best.probabilities
            if v != metrics.loss.control
        ]

        rows = []
        for v in variants:
            row = {
                "variant": v,
                "prob_best": metrics.prob_best.probabilities.get(v, np.nan),
                "expected_loss": metrics.loss.expected_loss.get(v, np.nan),
                "cvar": metrics.cvar.cvar.get(v, np.nan),
                "lift_mean": metrics.lift.mean.get(v, np.nan),
                "lift_hdi_low": metrics.lift.hdi_low.get(v, np.nan),
                "lift_hdi_high": metrics.lift.hdi_high.get(v, np.nan),
                "prob_practical": metrics.rope.prob_practical.get(v, np.nan),
                "inside_rope": metrics.rope.inside_rope.get(v, np.nan),
                "guardrail_passed": d.guardrails.variant_passed.get(v, np.nan),
                "joint_prob": d.joint.joint_prob.get(v, np.nan) if d.joint else np.nan,
                "correlation_gap": d.joint.correlation_gap.get(v, np.nan) if d.joint else np.nan,
                "composite_score": d.composite.score.get(v, np.nan) if d.composite else np.nan,
                "prob_exceeds_threshold": d.composite.prob_exceeds_threshold.get(v, np.nan) if d.composite else np.nan,
            }
            rows.append(row)

        return pd.DataFrame(rows).set_index("variant")

    def _to_dataframe_hierarchical(self) -> pd.DataFrame:
        rows = []

        def _extract_rows(d: DecisionResult, segment_label: str) -> list[dict]:
            metrics = d.metrics
            variants = [
                v for v in metrics.prob_best.probabilities
                if v != metrics.loss.control
            ]
            result_rows = []
            for v in variants:
                row = {
                    "segment": segment_label,
                    "variant": v,
                    "prob_best": metrics.prob_best.probabilities.get(v, np.nan),
                    "expected_loss": metrics.loss.expected_loss.get(v, np.nan),
                    "cvar": metrics.cvar.cvar.get(v, np.nan),
                    "lift_mean": metrics.lift.mean.get(v, np.nan),
                    "lift_hdi_low": metrics.lift.hdi_low.get(v, np.nan),
                    "lift_hdi_high": metrics.lift.hdi_high.get(v, np.nan),
                    "prob_practical": metrics.rope.prob_practical.get(v, np.nan),
                    "inside_rope": metrics.rope.inside_rope.get(v, np.nan),
                    "guardrail_passed": d.guardrails.variant_passed.get(v, np.nan),
                    "joint_prob": d.joint.joint_prob.get(v, np.nan) if d.joint else np.nan,
                    "correlation_gap": d.joint.correlation_gap.get(v, np.nan) if d.joint else np.nan,
                    "composite_score": d.composite.score.get(v, np.nan) if d.composite else np.nan,
                    "prob_exceeds_threshold": d.composite.prob_exceeds_threshold.get(v, np.nan) if d.composite else np.nan,
                    "state": d.state,
                    "recommendation": d.recommendation,
                }
                result_rows.append(row)
            return result_rows

        rows.extend(_extract_rows(self._d, "aggregate"))

        for seg in sorted(self.segment_results.keys()): 
            rows.extend(_extract_rows(self.segment_results[seg], seg)) 

        df = pd.DataFrame(rows)
        df = df.set_index(["segment", "variant"])
        return df