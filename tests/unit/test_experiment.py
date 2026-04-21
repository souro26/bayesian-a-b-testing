from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Callable

from argonx.models.binary_model import BinaryModel
from argonx.models.lognormal_model import LogNormalModel
from argonx.models.gaussian_model import GaussianModel, StudentTModel
from argonx.models.count_model import PoissonModel
from argonx.decision_rules.engine import run_engine
from argonx.results.result import Results


_MODEL_REGISTRY = {
    "binary": BinaryModel,
    "lognormal": LogNormalModel,
    "gaussian": GaussianModel,
    "studentt": StudentTModel,
    "poisson": PoissonModel,
}


_ALLOWED_CONFIG_KEYS = {
    "prob_best_strong":         (float, (0.0, 1.0)),
    "prob_best_moderate":       (float, (0.0, 1.0)),
    "expected_loss_max":        (float, (0.0, None)),
    "cvar_ratio_max":           (float, (1.0, None)),
    "rope_practical_min":       (float, (0.0, 1.0)),
    "alpha":                    (float, (0.0, 1.0)),
    "hdi_prob":                 (float, (0.0, 1.0)),
    "guardrail_thresholds":     (dict,  None),
    "guardrail_penalty":        (float, (0.0, None)),
    "deterioration_weights":    (dict,  None),
    "metrics_to_join":          (list,  None),
    "composite_threshold":      (float, None),
    "primary_lower_is_better":  (bool,  None),
    "lower_is_better":          (dict,  None),
}

_DEFAULT_CONFIG = {
    "prob_best_strong":     0.95,
    "prob_best_moderate":   0.80,
    "expected_loss_max":    0.01,
    "cvar_ratio_max":       5.0,
    "rope_practical_min":   0.80,
    "alpha":                0.95,
    "hdi_prob":             0.95,
    "guardrail_thresholds": {},
    "guardrail_penalty":    0.0,
}


def _validate_config(config: dict) -> None:
    """Validate config keys and values."""
    for key, value in config.items():
        if key not in _ALLOWED_CONFIG_KEYS:
            raise ValueError(f"Unknown config key: '{key}'")

        expected_type, bounds = _ALLOWED_CONFIG_KEYS[key]

        if not isinstance(value, expected_type):
            raise ValueError(f"{key} expects {expected_type.__name__}")

        if bounds and isinstance(value, float):
            lo, hi = bounds
            if lo is not None and value < lo:
                raise ValueError(f"{key} must be >= {lo}")
            if hi is not None and value > hi:
                raise ValueError(f"{key} must be <= {hi}")

def _validate_inputs(
    data: pd.DataFrame,
    variant_col: str,
    primary_metric,
    model: str,
    guardrails: list[str],
    lower_is_better: dict[str, bool],
    control: str | None,
) -> None:
    """Validate experiment inputs."""
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be DataFrame")

    if variant_col not in data.columns:
        raise ValueError("variant_col missing")

    if isinstance(primary_metric, str):
        if primary_metric not in data.columns:
            raise ValueError("primary_metric missing")
    elif not callable(primary_metric):
        raise ValueError("primary_metric must be str or callable")

    if model not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model}'")

    for g in guardrails:
        if g not in data.columns:
            raise ValueError(f"Guardrail '{g}' missing")

    for g in lower_is_better:
        if g not in guardrails:
            raise ValueError(f"{g} not in guardrails")

    if control:
        if control not in data[variant_col].unique():
            raise ValueError("control not found in variants")

def _resolve_metric(data: pd.DataFrame, metric) -> pd.Series:
    """Return Series from column or callable."""
    return data[metric] if isinstance(metric, str) else metric(data)


def _select_model(model_str: str):
    """Select model class."""
    return _MODEL_REGISTRY[model_str]


def _split_by_variant(
    data: pd.DataFrame,
    variant_col: str,
    metric_series: pd.Series,
    variant_names: list[str],
) -> dict[str, np.ndarray]:
    """Split metric into variant arrays."""
    split = {}

    for v in variant_names:
        mask = data[variant_col] == v
        values = metric_series[mask].dropna().values

        if len(values) == 0:
            raise ValueError(f"Variant '{v}' has no valid data")

        dropped = mask.sum() - len(values)
        if dropped > 0:
            warnings.warn(f"{v}: dropped {dropped} NaNs", stacklevel=4)

        split[v] = values

    return split


def _fit_and_sample(
    model_class,
    variant_data,
    n_draws: int,
    random_seed: int | None,
) -> np.ndarray:
    """Fit model and return posterior samples."""
    model = model_class()

    if hasattr(model, "random_seed") and random_seed is not None:
        model.random_seed = random_seed

    model.fit(variant_data, n_draws=n_draws)
    return model.sample()

class Experiment:
    """User-facing API for A/B experiments."""

    def __init__(
        self,
        data: pd.DataFrame,
        variant_col: str,
        primary_metric: str | Callable,
        model: str,
        guardrails: list[str] = None,
        lower_is_better: dict[str, bool] = None,
        control: str | None = None,
    ):
        self.data = data
        self.variant_col = variant_col
        self.primary_metric = primary_metric
        self.model = model
        self.guardrails = guardrails or []
        self.lower_is_better = lower_is_better or {}
        self.control = control

        _validate_inputs(
            data, variant_col, primary_metric, model,
            self.guardrails, self.lower_is_better, control
        )

        self._variant_names = sorted(data[variant_col].unique())
        self._control = control or self._variant_names[0]

    def run(
        self,
        min_effect: float = 0.01,
        rope_bounds: tuple[float, float] | None = None,
        composite_weights: dict[str, float] | None = None,
        n_draws: int = 2000,
        random_seed: int | None = None,
        config: dict | None = None,
    ) -> Results:
        """Run experiment and return results."""

        config = config or {}
        _validate_config(config)

        full_config = {**_DEFAULT_CONFIG, **config}

        if rope_bounds is None:
            rope_bounds = (-min_effect, min_effect)

        full_config["rope_bounds"] = rope_bounds

        override_keys = set(self.lower_is_better) & set(config.get("lower_is_better", {}))
        if override_keys:
            warnings.warn(
                f"Overriding lower_is_better for: {sorted(override_keys)}",
                stacklevel=2,
            )

        full_config["lower_is_better"] = {
            **self.lower_is_better,
            **config.get("lower_is_better", {}),
        }

        if composite_weights:
            full_config["composite_weights"] = composite_weights

        model_class = _select_model(self.model)

        primary_series = _resolve_metric(self.data, self.primary_metric)
        primary_data = _split_by_variant(
            self.data, self.variant_col, primary_series, self._variant_names
        )
        primary_samples = _fit_and_sample(
            model_class, primary_data, n_draws, random_seed
        )

        guardrail_samples = {}
        for g in self.guardrails:
            s = _resolve_metric(self.data, g)
            d = _split_by_variant(self.data, self.variant_col, s, self._variant_names)
            guardrail_samples[g] = _fit_and_sample(
                model_class, d, n_draws, random_seed
            )

        decision = run_engine(
            samples=primary_samples,
            variant_names=self._variant_names,
            control=self._control,
            guardrail_samples=guardrail_samples,
            config=full_config,
        )

        return Results(decision)

    def __repr__(self):
        return (
            f"Experiment(model={self.model}, "
            f"variants={self._variant_names}, "
            f"control={self._control})"
        )