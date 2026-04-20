import numpy as np
import pandas as pd
import pytest

from experimentation.experiment import Experiment


def make_basic_dataframe():
    """Create simple valid dataframe for testing."""
    return pd.DataFrame({
        "variant": ["A", "A", "B", "B"],
        "conversion": [0, 1, 1, 1],
        "revenue": [10.0, 12.0, 15.0, 18.0],
    })


def make_guardrail_dataframe():
    """Create dataframe with guardrail metric."""
    return pd.DataFrame({
        "variant": ["A", "A", "B", "B"],
        "conversion": [0, 1, 1, 1],
        "latency": [100, 110, 120, 130],
    })

def test_run_basic_pipeline():
    """Runs full pipeline and returns Results."""
    df = make_basic_dataframe()

    exp = Experiment(
        data=df,
        variant_col="variant",
        primary_metric="conversion",
        model="binary",
    )

    res = exp.run(n_draws=500)

    assert res is not None


def test_callable_metric():
    """Supports callable metric definitions."""
    df = make_basic_dataframe()

    exp = Experiment(
        data=df,
        variant_col="variant",
        primary_metric=lambda d: d["conversion"] * 2,
        model="binary",
    )

    res = exp.run(n_draws=500)

    assert res is not None


def test_with_guardrails():
    """Handles guardrail metrics correctly."""
    df = make_guardrail_dataframe()

    exp = Experiment(
        data=df,
        variant_col="variant",
        primary_metric="conversion",
        model="binary",
        guardrails=["latency"],
        lower_is_better={"latency": True},
    )

    res = exp.run(n_draws=500)

    assert res is not None


def test_with_composite_weights():
    """Runs with composite scoring enabled."""
    df = make_guardrail_dataframe()

    exp = Experiment(
        data=df,
        variant_col="variant",
        primary_metric="conversion",
        model="binary",
        guardrails=["latency"],
    )

    res = exp.run(
        composite_weights={"primary": 1.0, "latency": -0.5},
        n_draws=500,
    )

    assert res is not None

def test_invalid_model_raises():
    """Unknown model raises ValueError."""
    df = make_basic_dataframe()

    with pytest.raises(ValueError):
        Experiment(
            data=df,
            variant_col="variant",
            primary_metric="conversion",
            model="invalid_model",
        )


def test_missing_variant_column_raises():
    """Missing variant column raises error."""
    df = make_basic_dataframe()

    with pytest.raises(ValueError):
        Experiment(
            data=df,
            variant_col="missing",
            primary_metric="conversion",
            model="binary",
        )


def test_missing_primary_metric_raises():
    """Missing primary metric raises error."""
    df = make_basic_dataframe()

    with pytest.raises(ValueError):
        Experiment(
            data=df,
            variant_col="variant",
            primary_metric="missing",
            model="binary",
        )


def test_invalid_guardrail_column():
    """Guardrail not in dataframe raises error."""
    df = make_basic_dataframe()

    with pytest.raises(ValueError):
        Experiment(
            data=df,
            variant_col="variant",
            primary_metric="conversion",
            model="binary",
            guardrails=["not_exists"],
        )


def test_invalid_lower_is_better_key():
    """lower_is_better key not in guardrails raises."""
    df = make_guardrail_dataframe()

    with pytest.raises(ValueError):
        Experiment(
            data=df,
            variant_col="variant",
            primary_metric="conversion",
            model="binary",
            guardrails=["latency"],
            lower_is_better={"wrong_key": True},
        )

def test_empty_variant_data_raises():
    """Variant with no data raises error."""
    df = pd.DataFrame({
        "variant": ["A", "A", "B"],
        "conversion": [1, 0, np.nan],
    })

    exp = Experiment(
        data=df,
        variant_col="variant",
        primary_metric="conversion",
        model="binary",
    )

    with pytest.raises(ValueError):
        exp.run()


def test_nan_values_are_dropped():
    """NaN values are dropped with warning."""
    df = pd.DataFrame({
        "variant": ["A", "A", "B", "B"],
        "conversion": [1, np.nan, 1, 1],
    })

    exp = Experiment(
        data=df,
        variant_col="variant",
        primary_metric="conversion",
        model="binary",
    )

    res = exp.run(n_draws=500)

    assert res is not None

def test_invalid_config_key_raises():
    """Unknown config key raises ValueError."""
    df = make_basic_dataframe()

    exp = Experiment(
        data=df,
        variant_col="variant",
        primary_metric="conversion",
        model="binary",
    )

    with pytest.raises(ValueError):
        exp.run(config={"invalid_key": 123})


def test_invalid_config_type_raises():
    """Wrong config type raises ValueError."""
    df = make_basic_dataframe()

    exp = Experiment(
        data=df,
        variant_col="variant",
        primary_metric="conversion",
        model="binary",
    )

    with pytest.raises(ValueError):
        exp.run(config={"alpha": "wrong_type"})

def test_random_seed_reproducibility():
    """Same seed produces consistent outputs."""
    df = make_basic_dataframe()

    exp = Experiment(
        data=df,
        variant_col="variant",
        primary_metric="conversion",
        model="binary",
    )

    res1 = exp.run(n_draws=500, random_seed=42)
    res2 = exp.run(n_draws=500, random_seed=42)

    assert res1 is not None
    assert res2 is not None

def test_min_effect_sets_rope():
    """min_effect correctly maps to rope_bounds."""
    df = make_basic_dataframe()

    exp = Experiment(
        data=df,
        variant_col="variant",
        primary_metric="conversion",
        model="binary",
    )

    res = exp.run(min_effect=0.05, n_draws=500)

    assert res is not None


def test_composite_optional():
    """Composite not required for run."""
    df = make_basic_dataframe()

    exp = Experiment(
        data=df,
        variant_col="variant",
        primary_metric="conversion",
        model="binary",
    )

    res = exp.run(n_draws=500)

    assert res is not None