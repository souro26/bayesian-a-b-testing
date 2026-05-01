from __future__ import annotations

import numpy as np
import pandas as pd

from argonx.decision_rules.engine import DecisionResult


class Results:

    def __init__(self, decision: DecisionResult) -> None:
        self._d = decision

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

        return (
            f"Results(\n"
            f"  state          = {self._d.state!r}\n"
            f"  recommendation = {self._d.recommendation!r}\n"
            f"  best_variant   = {best!r}\n"
            f"  prob_best      = {p_best:.3f}\n"
            f"  lift_mean      = {lift_mean:.3f}\n"
            f"  guardrails     = {'PASS' if self._d.guardrails.all_passed else 'FAIL'}\n"
            f"  notes          = {len(self._d.notes)} flagged\n"
            f")"
        )

    def summary(self) -> None:
        d = self._d
        best = d.best_variant
        metrics = d.metrics
        guardrails = d.guardrails

        lines = []

        lines.append("=" * 60)
        lines.append("EXPERIMENT RESULTS")
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

        lines.append("=" * 60)

        print("\n".join(lines))

    def plot(
        self,
        samples: np.ndarray,
        metric_name: str = "metric",
        rope_bounds: tuple[float, float] = (-0.01, 0.01),
        figsize: tuple[int, int] = (18, 11),
        suptitle: str | None = None,
    ):
        """
        Render all five decision plots in a single figure.

        Parameters
        ----------
        samples : np.ndarray
            Shape (n_draws, n_variants). The posterior samples returned by
            the model — same array passed internally through the engine.
            Access via experiment._last_samples if stored, or pass explicitly.
        metric_name : str
            Primary metric name used in axis labels.
        rope_bounds : tuple[float, float]
            ROPE bounds for the lift plot. Should match what was passed to
            exp.run(rope_bounds=...).
        figsize : tuple
            Overall figure size. Default (18, 11).
        suptitle : str, optional
            Figure-level title. Defaults to "Experiment Decision Report".

        Returns
        -------
        plt.Figure

        Example
        -------
        result = exp.run()
        fig = result.plot(samples, metric_name="revenue")
        fig.savefig("experiment_report.png", dpi=150, bbox_inches="tight")
        """
        from argonx.results.plots import plot_all

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
            prob_best_threshold=0.95,
            loss_threshold=0.01, 
            figsize=figsize,
            suptitle=suptitle,
        )

    def to_dict(self) -> dict:
        """
        Serialize DecisionResult to a plain Python dict.

        All numpy arrays are converted to lists. All numpy scalars are
        converted to Python floats. Safe for JSON serialization.
        This is the internal serialization layer that exporters will consume.
        """
        d = self._d
        best = d.best_variant

        out = {
            "decision": {
                "state": d.state,
                "recommendation": d.recommendation,
                "best_variant": best,
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

        return out

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return per-variant metrics as a pandas DataFrame.

        One row per non-control variant. Joint and composite columns
        are NaN when those components were not computed.
        """
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
            }

            if d.joint is not None:
                row["joint_prob"] = d.joint.joint_prob.get(v, np.nan)
                row["correlation_gap"] = d.joint.correlation_gap.get(v, np.nan)
            else:
                row["joint_prob"] = np.nan
                row["correlation_gap"] = np.nan

            if d.composite is not None:
                row["composite_score"] = d.composite.score.get(v, np.nan)
                row["prob_exceeds_threshold"] = d.composite.prob_exceeds_threshold.get(
                    v, np.nan
                )
            else:
                row["composite_score"] = np.nan
                row["prob_exceeds_threshold"] = np.nan

            rows.append(row)

        return pd.DataFrame(rows).set_index("variant")