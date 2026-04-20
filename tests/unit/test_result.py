import numpy as np
import pandas as pd
import pytest

from types import SimpleNamespace as NS

from experimentation.results.result import Results
from experimentation.decision_rules.engine import DecisionResult


def make_decision(joint=True, composite=True, guardrail_fail=False, notes=None):
    """Create structured DecisionResult for testing."""

    metrics = NS()

    metrics.prob_best = NS(
        probabilities={"control": 0.4, "B": 0.6, "C": 0.0}
    )

    metrics.loss = NS(
        expected_loss={"control": 0.0, "B": 0.01, "C": 0.02},
        control="control",
    )

    metrics.cvar = NS(
        cvar={"control": 0.0, "B": 0.02, "C": 0.03},
        alpha=0.95,
    )

    metrics.lift = NS(
        mean={"control": 0.0, "B": 0.05, "C": -0.02},
        hdi_low={"control": 0.0, "B": 0.02, "C": -0.04},
        hdi_high={"control": 0.0, "B": 0.08, "C": 0.0},
        hdi_prob=0.94,
    )

    metrics.rope = NS(
        inside_rope={"control": 1.0, "B": 0.2, "C": 0.7},
        outside_rope={"control": 0.0, "B": 0.8, "C": 0.3},
        prob_practical={"control": 0.0, "B": 0.8, "C": 0.2},
    )

    guardrails = NS(
        all_passed=not guardrail_fail,
        variant_passed={
            "control": True,
            "B": not guardrail_fail,
            "C": False,
        },
        guardrails=[],
        conflicts=[],
    )

    joint_obj = None
    if joint:
        joint_obj = NS(
            joint_prob={"B": 0.7, "C": 0.2},
            independence_benchmark={"B": 0.75, "C": 0.25},
            correlation_gap={"B": -0.05, "C": -0.03},
            condition_probs={
                "B": {"primary": 0.9, "guardrail": 0.7},
                "C": {"primary": 0.4, "guardrail": 0.3},
            },
            metrics_joined=["primary", "guardrail"],
            best_variant="B",
        )

    composite_obj = None
    if composite:
        composite_obj = NS(
            score={"B": 0.3, "C": -0.1},
            prob_exceeds_threshold={"B": 0.9, "C": 0.1},
            gap_hdi={"B": (0.1, 0.5), "C": (-0.3, 0.0)},
            metric_contributions={
                "B": {"primary": 0.3},
                "C": {"primary": -0.1},
            },
            best_variant="B",
            threshold=0.0,
        )

    decision = DecisionResult(
        state="strong_win",
        recommendation="ship",
        best_variant="B",
        confidence="high",
        primary_strength="strong",
        risk_level="low",
        practical_significance="yes",
        guardrail_status="fail" if guardrail_fail else "pass",
        reasons=["Strong lift"],
        notes=notes or [],
        metrics=metrics,
        guardrails=guardrails,
        joint=joint_obj,
        composite=composite_obj,
    )

    return decision

def test_summary_runs():
    """Summary executes without errors."""
    Results(make_decision()).summary()


def test_repr_contains_key_info():
    """__repr__ contains key fields."""
    r = Results(make_decision())
    out = repr(r)
    assert "strong_win" in out
    assert "B" in out


def test_attribute_passthrough():
    """__getattr__ proxies DecisionResult."""
    r = Results(make_decision())
    assert r.state == "strong_win"


def test_invalid_attribute_raises():
    """Invalid attribute raises AttributeError."""
    r = Results(make_decision())
    with pytest.raises(AttributeError):
        _ = r.fake_attr

def test_to_dict_structure():
    """to_dict returns all main sections."""
    d = Results(make_decision()).to_dict()

    assert set(d.keys()) == {
        "decision",
        "metrics",
        "guardrails",
        "joint",
        "composite",
    }


def test_to_dict_consistency():
    """to_dict matches DecisionResult values."""
    r = Results(make_decision())
    d = r.to_dict()

    assert d["decision"]["best_variant"] == r.best_variant


def test_to_dict_no_numpy():
    """to_dict contains no numpy arrays."""
    d = Results(make_decision()).to_dict()

    def check(x):
        if isinstance(x, dict):
            for v in x.values():
                check(v)
        elif isinstance(x, list):
            for v in x:
                check(v)
        else:
            assert not isinstance(x, np.ndarray)

    check(d)


def test_to_dict_optional_joint():
    """Joint is None when absent."""
    d = Results(make_decision(joint=False)).to_dict()
    assert d["joint"] is None


def test_to_dict_optional_composite():
    """Composite is None when absent."""
    d = Results(make_decision(composite=False)).to_dict()
    assert d["composite"] is None

def test_dataframe_basic():
    """DataFrame has expected columns."""
    df = Results(make_decision()).to_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert "prob_best" in df.columns


def test_dataframe_excludes_control():
    """Control is excluded from DataFrame."""
    df = Results(make_decision()).to_dataframe()
    assert "control" not in df.index


def test_dataframe_multi_variant():
    """Multiple variants appear correctly."""
    df = Results(make_decision())
    df = df.to_dataframe()

    assert "B" in df.index
    assert "C" in df.index


def test_dataframe_optional_joint():
    """Joint columns NaN when missing."""
    df = Results(make_decision(joint=False)).to_dataframe()

    assert df["joint_prob"].isna().all()


def test_dataframe_optional_composite():
    """Composite columns NaN when missing."""
    df = Results(make_decision(composite=False)).to_dataframe()

    assert df["composite_score"].isna().all()

def test_guardrail_failure_propagation():
    """Guardrail failure reflected correctly."""
    r = Results(make_decision(guardrail_fail=True))
    assert not r.guardrails.all_passed


def test_negative_lift_present():
    """Negative lift variant handled correctly."""
    df = Results(make_decision()).to_dataframe()
    assert df.loc["C", "lift_mean"] < 0


def test_notes_propagation():
    """Notes propagate correctly."""
    r = Results(make_decision(notes=["issue"]))
    d = r.to_dict()

    assert d["decision"]["notes"] == ["issue"]

def test_joint_binding_safe():
    """Binding constraint logic does not crash."""
    Results(make_decision()).summary()


def test_composite_best_variant_consistency():
    """Composite best matches max score."""
    r = Results(make_decision())

    best = max(r.composite.score, key=r.composite.score.get)
    assert best == r.composite.best_variant

def test_empty_guardrails_safe():
    """Empty guardrails do not break."""
    d = make_decision()
    d.guardrails.guardrails = []
    Results(d).summary()


def test_empty_conflicts_safe():
    """No conflicts does not break."""
    d = make_decision()
    d.guardrails.conflicts = []
    Results(d).summary()


def test_large_notes_safe():
    """Large notes list handled."""
    notes = ["n"] * 100
    Results(make_decision(notes=notes)).summary()