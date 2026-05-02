"""
Sequential testing and dynamic stopping rules.

This subpackage implements the logic for evaluating experiments at intermediate 
checkpoints. It provides tools for monitoring traffic health, estimating 
remaining sample size requirements, and making early-stopping decisions based 
on Bayesian expected loss and probability of being best.
"""

from .stopping import (
    StoppingChecker,
    evaluate_stopping,
    StoppingResult,
    CheckpointSnapshot,
    UsersNeededEstimate,
    TrafficDiagnostics,
)

__all__ = [
    "StoppingChecker",
    "evaluate_stopping",
    "StoppingResult",
    "CheckpointSnapshot",
    "UsersNeededEstimate",
    "TrafficDiagnostics",
]
