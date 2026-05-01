from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Callable

from argonx.models.binary_model import BinaryModel, HierarchicalBinaryModel
from argonx.models.lognormal_model import LogNormalModel, HierarchicalLogNormalModel
from argonx.models.gaussian_model import GaussianModel, HierarchicalGaussianModel, StudentTModel, HierarchicalStudentTModel
from argonx.models.count_model import PoissonModel, HierarchicalPoissonModel
from argonx.decision_rules.engine import run_engine
from argonx.results.result import Results
from argonx.decision_rules.engine import run_engine, DecisionResult


_MODEL_REGISTRY = {
    "binary":    (BinaryModel,    HierarchicalBinaryModel),
    "lognormal": (LogNormalModel, HierarchicalLogNormalModel),
    "gaussian":  (GaussianModel,  HierarchicalGaussianModel),
    "studentt":  (StudentTModel,  HierarchicalStudentTModel),
    "poisson":   (PoissonModel,   HierarchicalPoissonModel),
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
    """
    Validate config keys, types, and bounds.

    Parameters
    ----------
    config : dict
        Configuration dictionary to validate.

    Raises
    ------
    ValueError
        If an unknown config key is provided, or if types/bounds are violated.
    """
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
    guardrails: list[str] | None,
    lower_is_better: dict[str, bool] | None,
    segment_col: str | None,
    control: str | None,
) -> None:
    """
    Validate experiment inputs.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset containing observations.
    variant_col : str
        Column name identifying the variants.
    primary_metric : str | Callable
        The primary metric column name or callable.
    model : str
        Model identifier from the registry.
    guardrails : list[str] | None
        List of guardrail metric column names.
    lower_is_better : dict[str, bool] | None
        Dictionary mapping metrics to whether lower values are better.
    segment_col : str | None
        Column name for segmenting data.
    control : str | None
        Name of the control variant.

    Raises
    ------
    ValueError
        If any validation check fails.
    """
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

    if segment_col and segment_col not in data.columns:
        raise ValueError("segment_col missing")

    if control:
        if control not in data[variant_col].unique():
            raise ValueError("control not found in variants")


def _resolve_metric(data: pd.DataFrame, metric) -> pd.Series:
    """
    Return Series from column or callable.

    Parameters
    ----------
    data : pd.DataFrame
        The data containing the metric.
    metric : str | Callable
        The metric column name or callable that extracts the series.

    Returns
    -------
    pd.Series
        The resolved metric values.
    """
    return data[metric] if isinstance(metric, str) else metric(data)


def _select_model(model_str: str, hierarchical: bool):
    """
    Select flat or hierarchical model class.

    Parameters
    ----------
    model_str : str
        The registered model name.
    hierarchical : bool
        Whether to select the hierarchical version.

    Returns
    -------
    type
        The selected model class.
    """
    flat, hier = _MODEL_REGISTRY[model_str]
    return hier if hierarchical else flat

def _split_by_variant(
    data: pd.DataFrame,
    variant_col: str,
    metric_series: pd.Series,
    variant_names: list[str],
) -> dict[str, np.ndarray]:
    """
    Split metric into per-variant arrays. Used by flat models.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe containing observations.
    variant_col : str
        The column indicating variants.
    metric_series : pd.Series
        The specific metric values to split.
    variant_names : list[str]
        List of expected variants.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from variant name to its observation array.
    """
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
    variant_data: dict[str, np.ndarray],
    n_draws: int,
    random_seed: int | None,
) -> np.ndarray:
    """
    Fit flat model and return posterior samples.

    Parameters
    ----------
    model_class : type
        The model class to instantiate.
    variant_data : dict[str, np.ndarray]
        Data split by variant.
    n_draws : int
        Number of posterior draws to take.
    random_seed : int | None
        Seed for the random number generator.

    Returns
    -------
    np.ndarray
        Array of posterior samples of shape (n_draws, n_variants).
    """
    model = model_class()

    if hasattr(model, "random_seed") and random_seed is not None:
        model.random_seed = random_seed

    model.fit(variant_data)
    return model.sample_posterior(n_draws)

def _split_by_segment_and_variant(
    data: pd.DataFrame,
    variant_col: str,
    segment_col: str | None,
    metric_series: pd.Series,
    segment_names: list[str],
    variant_names: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """
    Split metric into nested structure.

    Used exclusively by hierarchical models. NaN values are dropped per
    cell with a warning. Empty cells after NaN removal raise ValueError.

    Parameters
    ----------
    data : pd.DataFrame
        The full dataset.
    variant_col : str
        Column denoting the variant.
    segment_col : str | None
        Column denoting the segment.
    metric_series : pd.Series
        The metric values.
    segment_names : list[str]
        List of available segments.
    variant_names : list[str]
        List of available variants.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Nested dictionary mapping segment to variant to data array.
    """
    assert segment_col is not None
    nested: dict[str, dict[str, np.ndarray]] = {}

    for seg in segment_names:
        seg_mask = data[segment_col] == seg
        nested[seg] = {}

        for v in variant_names:
            var_mask = data[variant_col] == v
            cell_mask = seg_mask & var_mask
            values = metric_series[cell_mask].dropna().values

            if len(values) == 0:
                raise ValueError(
                    f"Segment '{seg}', variant '{v}' has no valid data. "
                    f"Every segment must have observations for every variant."
                )

            dropped = int(cell_mask.sum()) - len(values)
            if dropped > 0:
                warnings.warn(
                    f"Segment '{seg}', variant '{v}': dropped {dropped} NaNs.",
                    UserWarning,
                    stacklevel=4,
                )

            nested[seg][v] = values

    return nested


def _fit_and_sample_hierarchical(
    model_class,
    segment_data: dict[str, dict[str, np.ndarray]],
    n_draws: int,
    priors: dict | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Fit hierarchical model and return samples and warnings.

    Parameters
    ----------
    model_class : type
        The hierarchical model class to fit.
    segment_data : dict[str, dict[str, np.ndarray]]
        Data nested by segment then variant.
    n_draws : int
        Number of posterior draws to take.
    priors : dict | None
        Optional dictionary of model prior overrides.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict]
        A tuple of (population_samples, segment_samples, warnings_dict).
    """
    kwargs = {}
    if priors:
        kwargs["priors"] = priors

    model = model_class(**kwargs)
    model.fit(segment_data)
    population_samples = model.sample_posterior(n_draws)
    segment_samples = model.sample_posterior_by_segment(n_draws)
    warnings_dict = model.all_warnings
    return population_samples, segment_samples, warnings_dict


def _run_engine_per_segment(
    segment_samples: np.ndarray,
    segment_names: list[str],
    variant_names: list[str],
    control: str,
    guardrail_segment_samples: dict[str, np.ndarray],
    full_config: dict,
) -> dict[str, DecisionResult]:
    """
    Loop the decision engine over segments.

    Parameters
    ----------
    segment_samples : np.ndarray
        Array of segmented primary metric samples.
    segment_names : list[str]
        List of segment identifiers.
    variant_names : list[str]
        List of variant names.
    control : str
        Control variant name.
    guardrail_segment_samples : dict[str, np.ndarray]
        Dict of segmented guardrail samples.
    full_config : dict
        Engine configuration dictionary.

    Returns
    -------
    dict[str, DecisionResult]
        Dictionary mapping segment names to their DecisionResults.
    """
    segment_results = {}

    for i, seg in enumerate(segment_names):
        seg_primary = segment_samples[:, i, :]

        seg_guardrail = {}
        for g_name, g_arr in guardrail_segment_samples.items():
            seg_guardrail[g_name] = g_arr[:, i, :]

        seg_decision = run_engine(
            samples=seg_primary,
            variant_names=variant_names,
            control=control,
            guardrail_samples=seg_guardrail,
            config=full_config,
        )

        segment_results[seg] = seg_decision

    return segment_results


def _collect_segment_guardrail_violations(
    segment_results: dict[str, object],
) -> dict[str, list[str]]:
    """
    Collect which guardrails failed in which segments.

    Only includes segments with at least one failure.
    Used to surface segment-level violations in the aggregate summary
    so users reading only the top-level result cannot miss them.

    Parameters
    ----------
    segment_results : dict[str, object]
        Mapping from segment to DecisionResult.

    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping segment name to list of failed guardrail names.
    """
    violations: dict[str, list[str]] = {}

    for seg, decision in segment_results.items():
        failed = [
            g.metric
            for g in decision.guardrails.guardrails
            if not g.passed
        ]
        if failed:
            violations[seg] = failed

    return violations


def _route_model_warnings(
    warnings_dict: dict,
    segment_results: dict[str, object],
) -> None:
    """
    Inject model-level warnings into the appropriate DecisionResult notes.

    Parameters
    ----------
    warnings_dict : dict
        Warnings grouped by health or segment.
    segment_results : dict[str, object]
        Mapping of segments to their corresponding DecisionResult.
    """
    health_warnings = warnings_dict.get("health", [])
    seg_warnings = warnings_dict.get("segments", {})

    for seg, decision in segment_results.items():
        for msg in health_warnings:
            if msg not in decision.notes:
                decision.notes.append(f"[MCMC] {msg}")

        for msg in seg_warnings.get(seg, []):
            if msg not in decision.notes:
                decision.notes.append(f"[Segment] {msg}")


class Experiment:
    """
    User-facing API for Bayesian A/B experiments.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing the observations.
    variant_col : str
        Column denoting the variant assignment.
    primary_metric : str | Callable
        The primary metric column name or extraction callable.
    model : str
        The registered model identifier (e.g., 'binary', 'lognormal').
    guardrails : list[str], optional
        List of guardrail metric column names, by default None.
    lower_is_better : dict[str, bool], optional
        Mapping indicating if lower is better for metrics, by default None.
    segment_col : str | None, optional
        Column to segment by for hierarchical models, by default None.
    control : str | None, optional
        Name of the control variant, by default None.
    priors : dict | None, optional
        Override for model priors, by default None.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        variant_col: str,
        primary_metric: str | Callable,
        model: str,
        guardrails: list[str] = None,
        lower_is_better: dict[str, bool] = None,
        segment_col: str | None = None,
        control: str | None = None,
        priors: dict | None = None,
    ):
        self.data = data
        self.variant_col = variant_col
        self.primary_metric = primary_metric
        self.model = model
        self.guardrails = guardrails or []
        self.lower_is_better = lower_is_better or {}
        self.segment_col = segment_col
        self.control = control
        self.priors = priors

        _validate_inputs(
            data, variant_col, primary_metric, model,
            self.guardrails, self.lower_is_better,
            segment_col, control,
        )

        self._variant_names = sorted(data[variant_col].unique().tolist())
        self._control = control or self._variant_names[0]

        if segment_col is not None:
            self._segment_names = sorted(data[segment_col].unique().tolist())
        else:
            self._segment_names = []

    def run(
        self,
        min_effect: float = 0.01,
        rope_bounds: tuple[float, float] | None = None,
        composite_weights: dict[str, float] | None = None,
        n_draws: int = 2000,
        random_seed: int | None = None,
        config: dict | None = None,
    ) -> Results:
        """
        Run the experiment and return a Results object.

        Parameters
        ----------
        min_effect : float, optional
            Minimum practically significant effect, by default 0.01.
        rope_bounds : tuple[float, float] | None, optional
            Override for ROPE bounds, by default None.
        composite_weights : dict[str, float] | None, optional
            Weights for composite score calculation, by default None.
        n_draws : int, optional
            Number of posterior draws, by default 2000.
        random_seed : int | None, optional
            Random seed for reproducibility, by default None.
        config : dict | None, optional
            Dictionary overriding default engine parameters, by default None.

        Returns
        -------
        Results
            Container holding experiment outcome and engine diagnostic.
        """
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

        hierarchical = self.segment_col is not None
        model_class = _select_model(self.model, hierarchical)

        if hierarchical:
            return self._run_hierarchical(
                model_class=model_class,
                full_config=full_config,
                n_draws=n_draws,
            )
        else:
            return self._run_flat(
                model_class=model_class,
                full_config=full_config,
                n_draws=n_draws,
                random_seed=random_seed,
            )

    def _run_flat(
        self,
        model_class,
        full_config: dict,
        n_draws: int,
        random_seed: int | None,
    ) -> Results:
        """Original flat experiment path. No changes from existing behaviour."""
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
            d = _split_by_variant(
                self.data, self.variant_col, s, self._variant_names
            )
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

        return Results(decision, config=full_config)

    def _run_hierarchical(
        self,
        model_class,
        full_config: dict,
        n_draws: int,
    ) -> Results:
        primary_series = _resolve_metric(self.data, self.primary_metric)
        primary_seg_data = _split_by_segment_and_variant(
            self.data, self.variant_col, self.segment_col,
            primary_series, self._segment_names, self._variant_names,
        )

        primary_pop_samples, primary_seg_samples, primary_warnings = (
            _fit_and_sample_hierarchical(
                model_class, primary_seg_data, n_draws, self.priors
            )
        )

        guardrail_pop_samples: dict[str, np.ndarray] = {}
        guardrail_seg_samples: dict[str, np.ndarray] = {}
        guardrail_warnings: dict[str, dict] = {}

        for g in self.guardrails:
            g_series = _resolve_metric(self.data, g)
            g_seg_data = _split_by_segment_and_variant(
                self.data, self.variant_col, self.segment_col,
                g_series, self._segment_names, self._variant_names,
            )
            g_pop, g_seg, g_warn = _fit_and_sample_hierarchical(
                model_class, g_seg_data, n_draws, self.priors
            )
            guardrail_pop_samples[g] = g_pop
            guardrail_seg_samples[g] = g_seg
            guardrail_warnings[g] = g_warn

        aggregate_decision = run_engine(
            samples=primary_pop_samples,
            variant_names=self._variant_names,
            control=self._control,
            guardrail_samples=guardrail_pop_samples,
            config=full_config,
        )

        segment_results = _run_engine_per_segment(
            segment_samples=primary_seg_samples,
            segment_names=self._segment_names,
            variant_names=self._variant_names,
            control=self._control,
            guardrail_segment_samples=guardrail_seg_samples,
            full_config=full_config,
        )

        _route_model_warnings(primary_warnings, segment_results)
        for msg in primary_warnings.get("health", []):
            note = f"[MCMC] {msg}"
            if note not in aggregate_decision.notes:
                aggregate_decision.notes.append(note)

        for seg, msgs in primary_warnings.get("segments", {}).items():
            for msg in msgs:
                note = f"[Segment '{seg}'] {msg}"
                if note not in aggregate_decision.notes:
                    aggregate_decision.notes.append(note)

        seg_guardrail_violations = _collect_segment_guardrail_violations(
            segment_results
        )

        for seg, failed_guardrails in seg_guardrail_violations.items():
            note = (
                f"[Segment '{seg}'] Guardrail violation(s): "
                f"{', '.join(failed_guardrails)}. "
                f"See result.segment_summary() for details."
            )
            if note not in aggregate_decision.notes:
                aggregate_decision.notes.append(note)

        return Results(
            aggregate_decision,
            config=full_config,
            segment_results=segment_results,
            segment_guardrail_violations=seg_guardrail_violations if seg_guardrail_violations else None,
        )

    def __repr__(self):
        base = (
            f"Experiment(model={self.model}, "
            f"variants={self._variant_names}, "
            f"control={self._control}"
        )
        if self.segment_col:
            base += f", segments={self._segment_names}"
        return base + ")"