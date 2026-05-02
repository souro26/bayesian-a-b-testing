"""
argonx: Bayesian decision engine for robust A/B testing.

This package provides a comprehensive framework for Bayesian experimentation,
integrating primary metrics with guardrails, sequential stopping rules, and
hierarchical partial pooling.
"""

from .experiment import Experiment

__all__ = ["Experiment"]

