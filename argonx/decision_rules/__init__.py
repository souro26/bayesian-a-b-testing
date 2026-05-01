"""
Decision rules and analysis engines for the Bayesian A/B testing framework.

This subpackage implements the full analytical pipeline that transforms raw posterior
samples into actionable experiment outcomes. It covers five distinct concerns:

- **Metrics** (`metrics.py`): Core Bayesian decision metrics — P(best), expected loss,
  CVaR, ROPE analysis, and HDI-bounded lift. All metrics are derived from the same
  posterior draws to ensure coherence.

- **Guardrails** (`guardrails.py`): Safety constraint evaluation. Computes P(degraded)
  per metric per variant and surfaces conflicts when the primary metric clears its bar
  but a secondary metric fails.

- **Joint** (`joint.py`): Joint probability of simultaneously satisfying both the primary
  metric and all selected guardrails, with correlation diagnostics to expose when
  metric co-movement helps or hurts the compound decision.

- **Composite** (`composite.py`): Weighted scoring of multiple metrics into a single
  posterior distribution of business value, with asymmetric deterioration weights and
  optional guardrail penalties.

- **Engine** (`engine.py`): Orchestration layer. Calls all of the above in dependency
  order and assembles a structured `DecisionResult` with a plain-English recommendation.
"""

from .composite import (
    CompositeResult,
    compute_composite_score,
)

from .engine import (
    DecisionResult,
    run_engine,
)

from .guardrails import (
    ConflictResult,
    GuardrailBundle,
    GuardrailResult,
    compute_all_guardrails,
    compute_guardrail,
)

from .joint import (
    JointResult,
    compute_joint_probability,
)

from .metrics import (
    CVaRResult,
    LiftResult,
    LossResult,
    MetricsBundle,
    PBestResult,
    ROPEResult,
    compute_all_metrics,
    compute_cvar,
    compute_expected_loss,
    compute_lift_hdi,
    compute_prob_best,
    compute_rope,
)

__all__ = [
    # Dataclasses — composite
    "CompositeResult",
    # Dataclasses — engine
    "DecisionResult",
    # Dataclasses — guardrails
    "ConflictResult",
    "GuardrailBundle",
    "GuardrailResult",
    # Dataclasses — joint
    "JointResult",
    # Dataclasses — metrics
    "CVaRResult",
    "LiftResult",
    "LossResult",
    "MetricsBundle",
    "PBestResult",
    "ROPEResult",
    # Functions — composite
    "compute_composite_score",
    # Functions — engine
    "run_engine",
    # Functions — guardrails
    "compute_all_guardrails",
    "compute_guardrail",
    # Functions — joint
    "compute_joint_probability",
    # Functions — metrics
    "compute_all_metrics",
    "compute_cvar",
    "compute_expected_loss",
    "compute_lift_hdi",
    "compute_prob_best",
    "compute_rope",
]
