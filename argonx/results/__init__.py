"""
Experimental results and visualization tools.

This subpackage handles the structured storage of decision outcomes and provides
rich matplotlib-based visualizations for interpreting Bayesian posteriors and
risk metrics.
"""

from .result import Results
from .plots import (
    plot_all,
    plot_posteriors,
    plot_lift,
    plot_prob_best,
    plot_expected_loss,
    plot_guardrails,
)

__all__ = [
    "Results",
    "plot_all",
    "plot_posteriors",
    "plot_lift",
    "plot_prob_best",
    "plot_expected_loss",
    "plot_guardrails",
]
