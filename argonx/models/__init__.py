"""
Bayesian generative models for the A/B testing framework.

Each model follows the same interface contract defined by `BaseModel`:
accept a ``dict[str, np.ndarray]`` of per-variant observations, expose a
``fit()`` method, and produce posterior draws via ``sample_posterior()``.
The output is always a 2-D array of shape ``(n_draws, n_variants)``, ordered
by ``variant_names`` so the decision engine can index into it without
additional bookkeeping.

Flat models
-----------
One model per session, all variants fitted independently.

- `BinaryModel` — Beta-Bernoulli conjugate update. Exact posterior, no MCMC.
  Use for conversion rates, click-through rates, any 0/1 outcome.

- `LogNormalModel` — LogNormal likelihood with a Normal prior on the log-mean.
  Returns posterior samples of E[X] = exp(μ + σ²/2). Use for revenue, session
  value, or any positive right-skewed metric.

- `GaussianModel` — Normal likelihood with Normal + HalfNormal priors. Use for
  symmetric continuous metrics (latency in ms, score deltas).

- `StudentTModel` — Student-T likelihood, same priors as GaussianModel but with
  a learned degrees-of-freedom parameter. Use when outliers are expected.

- `PoissonModel` — Poisson likelihood with a Gamma prior on the rate (Gamma–Poisson
  conjugate). Use for count data — errors per session, events per user.

Hierarchical models
-------------------
Accept nested ``dict[str, dict[str, np.ndarray]]`` (segment → variant → values).
Partial pooling via a shared hyperprior reduces variance in thin-data segments without
fully sacrificing segment-level estimates. All return both population-level samples
(via ``sample_posterior()``) and per-segment samples (via ``sample_posterior_by_segment()``).

- `HierarchicalBinaryModel`
- `HierarchicalLogNormalModel`
- `HierarchicalGaussianModel`
- `HierarchicalStudentTModel`
- `HierarchicalPoissonModel`
"""

from .base_model import BaseModel

from .binary_model import (
    BinaryModel,
    HierarchicalBinaryModel,
)

from .lognormal_model import (
    LogNormalModel,
    HierarchicalLogNormalModel,
)

from .gaussian_model import (
    GaussianModel,
    HierarchicalGaussianModel,
    StudentTModel,
    HierarchicalStudentTModel,
)

from .count_model import (
    PoissonModel,
    HierarchicalPoissonModel,
)

__all__ = [
    # Base
    "BaseModel",
    # Binary
    "BinaryModel",
    "HierarchicalBinaryModel",
    # LogNormal
    "LogNormalModel",
    "HierarchicalLogNormalModel",
    # Gaussian / Student-T
    "GaussianModel",
    "HierarchicalGaussianModel",
    "StudentTModel",
    "HierarchicalStudentTModel",
    # Poisson
    "PoissonModel",
    "HierarchicalPoissonModel",
]
