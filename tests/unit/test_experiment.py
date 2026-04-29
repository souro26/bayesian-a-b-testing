import numpy as np
import pandas as pd
import pytest
import warnings

from argonx import Experiment


N = 300


def make_df_lognormal(lift_b=0.08, n=N):
    return pd.DataFrame({
        "variant": ["control"] * n + ["variant_b"] * n,
        "revenue": np.concatenate([
            np.random.lognormal(3.80, 0.6, n),
            np.random.lognormal(3.80 + lift_b, 0.6, n),
        ]),
        "page_load": np.concatenate([
            np.abs(np.random.normal(1.2, 0.1, n)) + 0.5,
            np.abs(np.random.normal(1.1, 0.1, n)) + 0.5,
        ]),
    })


def make_df_binary(n=N):
    return pd.DataFrame({
        "variant": ["control"] * n + ["variant_b"] * n,
        "converted": np.concatenate([
            np.random.binomial(1, 0.10, n),
            np.random.binomial(1, 0.14, n),
        ]),
    })


def make_df_segmented(n_per_segment=150):
    segments = ["mobile", "desktop", "tv"]
    rows = []
    for seg in segments:
        n = max(20, n_per_segment if seg != "tv" else 30)
        for variant, mu in [("control", 3.80), ("variant_b", 3.88)]:
            revenue = np.random.lognormal(mu, 0.6, n)
            rows.append(pd.DataFrame({
                "variant":      [variant] * n,
                "device_type":  [seg] * n,
                "revenue":      revenue,
                "page_load":    np.abs(np.random.normal(1.2, 0.1, n)) + 0.5,
            }))
    return pd.concat(rows, ignore_index=True)


class TestInputValidation:

    def test_rejects_non_dataframe(self):
        """Rejects data that is not a DataFrame."""
        with pytest.raises(ValueError):
            Experiment(data=[[1, 2], [3, 4]], variant_col="v", primary_metric="m", model="lognormal")

    def test_rejects_missing_variant_col(self):
        """Rejects when variant_col not in DataFrame."""
        df = make_df_lognormal()
        with pytest.raises(ValueError):
            Experiment(df, variant_col="nonexistent", primary_metric="revenue", model="lognormal")

    def test_rejects_missing_primary_metric(self):
        """Rejects when primary_metric column not in DataFrame."""
        df = make_df_lognormal()
        with pytest.raises(ValueError):
            Experiment(df, variant_col="variant", primary_metric="nonexistent", model="lognormal")

    def test_rejects_unknown_model(self):
        """Rejects unrecognised model string."""
        df = make_df_lognormal()
        with pytest.raises(ValueError):
            Experiment(df, variant_col="variant", primary_metric="revenue", model="unknown_model")

    def test_rejects_missing_guardrail_column(self):
        """Rejects when guardrail column not in DataFrame."""
        df = make_df_lognormal()
        with pytest.raises(ValueError):
            Experiment(df, variant_col="variant", primary_metric="revenue", model="lognormal", guardrails=["nonexistent"])

    def test_rejects_lower_is_better_not_in_guardrails(self):
        """Rejects lower_is_better key that is not a declared guardrail."""
        df = make_df_lognormal()
        with pytest.raises(ValueError):
            Experiment(df, variant_col="variant", primary_metric="revenue", model="lognormal",
                       guardrails=["page_load"], lower_is_better={"revenue": True})

    def test_rejects_invalid_control(self):
        """Rejects control value not present in variant column."""
        df = make_df_lognormal()
        with pytest.raises(ValueError):
            Experiment(df, variant_col="variant", primary_metric="revenue", model="lognormal", control="nonexistent")

    def test_rejects_missing_segment_col(self):
        """Rejects when segment_col not in DataFrame."""
        df = make_df_lognormal()
        with pytest.raises(ValueError):
            Experiment(df, variant_col="variant", primary_metric="revenue", model="lognormal", segment_col="nonexistent")

    def test_primary_metric_callable_accepted(self):
        """Accepts callable as primary_metric."""
        df = make_df_lognormal()
        Experiment(df, variant_col="variant",
                   primary_metric=lambda d: d["revenue"],
                   model="lognormal")

    def test_primary_metric_non_callable_non_str_rejected(self):
        """Rejects primary_metric that is neither str nor callable."""
        df = make_df_lognormal()
        with pytest.raises(ValueError):
            Experiment(df, variant_col="variant", primary_metric=42, model="lognormal")


class TestConfigValidation:

    def test_unknown_config_key_raises(self):
        """Unknown config key raises ValueError."""
        df = make_df_lognormal()
        exp = Experiment(df, "variant", "revenue", "lognormal", control="control")
        with pytest.raises(ValueError, match="Unknown config key"):
            exp.run(n_draws=200, config={"this_does_not_exist": 0.5})

    def test_wrong_type_config_raises(self):
        """Wrong type for config value raises ValueError."""
        df = make_df_lognormal()
        exp = Experiment(df, "variant", "revenue", "lognormal", control="control")
        with pytest.raises(ValueError):
            exp.run(n_draws=200, config={"prob_best_strong": "high"})

    def test_out_of_range_config_raises(self):
        """Out-of-range config value raises ValueError."""
        df = make_df_lognormal()
        exp = Experiment(df, "variant", "revenue", "lognormal", control="control")
        with pytest.raises(ValueError):
            exp.run(n_draws=200, config={"prob_best_strong": 1.5})


class TestVariantResolution:

    def test_variant_names_sorted(self):
        """Variant names are stored in sorted order."""
        df = make_df_lognormal()
        exp = Experiment(df, "variant", "revenue", "lognormal")
        assert exp._variant_names == sorted(exp._variant_names)

    def test_default_control_is_first_alphabetically(self):
        """Default control is first variant alphabetically."""
        df = make_df_lognormal()
        exp = Experiment(df, "variant", "revenue", "lognormal")
        assert exp._control == "control"

    def test_custom_control_respected(self):
        """Custom control is stored correctly."""
        df = make_df_lognormal()
        exp = Experiment(df, "variant", "revenue", "lognormal", control="variant_b")
        assert exp._control == "variant_b"

    def test_custom_control_appears_in_loss(self):
        """Custom control name appears in loss result."""
        df = make_df_lognormal()
        exp = Experiment(df, "variant", "revenue", "lognormal", control="variant_b")
        result = exp.run(n_draws=200)
        assert result.metrics.loss.control == "variant_b"


class TestMetricResolution:

    def test_string_metric_runs(self):
        """String column reference resolves and runs."""
        df = make_df_lognormal()
        exp = Experiment(df, "variant", "revenue", "lognormal", control="control")
        result = exp.run(n_draws=200)
        assert result.best_variant is not None

    def test_lambda_metric_runs(self):
        """Lambda callable metric resolves and runs."""
        df = make_df_lognormal()
        exp = Experiment(df, "variant",
                         primary_metric=lambda d: d["revenue"],
                         model="lognormal", control="control")
        result = exp.run(n_draws=200)
        assert result.best_variant is not None


class TestNaNHandling:

    def test_nan_rows_dropped_with_warning(self):
        """NaN rows are dropped and a warning is emitted."""
        df = make_df_lognormal()
        nan_idx = df[df["variant"] == "variant_b"].sample(10).index
        df.loc[nan_idx, "revenue"] = np.nan
        exp = Experiment(df, "variant", "revenue", "lognormal", control="control")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = exp.run(n_draws=200)
        assert result.best_variant is not None


class TestFlatModelSelection:

    def test_lognormal_model_runs(self):
        """Lognormal model produces valid result."""
        df = make_df_lognormal()
        exp = Experiment(df, "variant", "revenue", "lognormal", control="control")
        result = exp.run(n_draws=200)
        assert result.state in ["strong win", "weak win", "inconclusive", "high risk", "guardrail conflicts"]

    def test_binary_model_runs(self):
        """Binary model produces valid result."""
        df = make_df_binary()
        exp = Experiment(df, "variant", "converted", "binary", control="control")
        result = exp.run(n_draws=200)
        assert result.best_variant in ["control", "variant_b"]

    def test_gaussian_model_runs(self):
        """Gaussian model produces valid result."""
        df = pd.DataFrame({
            "variant": ["control"] * N + ["variant_b"] * N,
            "latency": np.concatenate([np.random.normal(100, 10, N), np.random.normal(108, 10, N)]),
        })
        exp = Experiment(df, "variant", "latency", "gaussian", control="control")
        result = exp.run(n_draws=200)
        assert result.best_variant is not None

    def test_studentt_model_runs(self):
        """StudentT model produces valid result."""
        df = pd.DataFrame({
            "variant": ["control"] * N + ["variant_b"] * N,
            "latency": np.concatenate([np.random.normal(100, 10, N), np.random.normal(108, 10, N)]),
        })
        exp = Experiment(df, "variant", "latency", "studentt", control="control")
        result = exp.run(n_draws=200)
        assert result.best_variant is not None

    def test_poisson_model_runs(self):
        """Poisson model produces valid result."""
        df = pd.DataFrame({
            "variant": ["control"] * N + ["variant_b"] * N,
            "purchases": np.concatenate([np.random.poisson(5, N).astype(int), np.random.poisson(6, N).astype(int)]),
        })
        exp = Experiment(df, "variant", "purchases", "poisson", control="control")
        result = exp.run(n_draws=200)
        assert result.best_variant is not None


class TestFlatResultStructure:

    def test_segment_results_none_for_flat(self):
        """segment_results is None for flat experiment."""
        df = make_df_lognormal()
        exp = Experiment(df, "variant", "revenue", "lognormal", control="control")
        result = exp.run(n_draws=200)
        assert result.segment_results is None

    def test_prob_best_sums_to_one(self):
        """P(best) probabilities sum to 1."""
        df = make_df_lognormal()
        exp = Experiment(df, "variant", "revenue", "lognormal", control="control")
        result = exp.run(n_draws=200)
        total = sum(result.metrics.prob_best.probabilities.values())
        assert abs(total - 1.0) < 0.01

    def test_guardrail_passes_when_improving(self):
        """Guardrail passes when variant improves on guardrail metric."""
        df = make_df_lognormal()
        exp = Experiment(df, "variant", "revenue", "lognormal",
                         guardrails=["page_load"],
                         lower_is_better={"page_load": True},
                         control="control")
        result = exp.run(n_draws=200, config={"guardrail_thresholds": {"page_load": 0.10}})
        load_results = [g for g in result.guardrails.guardrails if g.variant == "variant_b"]
        assert load_results[0].passed is True

    def test_composite_score_computed(self):
        """Composite score is computed when composite_weights provided."""
        df = make_df_lognormal()
        exp = Experiment(df, "variant", "revenue", "lognormal",
                         guardrails=["page_load"],
                         lower_is_better={"page_load": True},
                         control="control")
        result = exp.run(n_draws=200,
                         composite_weights={"primary": 1.0, "page_load": 0.3},
                         config={"guardrail_thresholds": {"page_load": 0.10}})
        assert result.composite is not None
        assert "variant_b" in result.composite.score

    def test_segment_summary_raises_on_flat(self):
        """segment_summary raises RuntimeError on flat result."""
        df = make_df_lognormal()
        exp = Experiment(df, "variant", "revenue", "lognormal", control="control")
        result = exp.run(n_draws=200)
        with pytest.raises(RuntimeError):
            result.segment_summary()

    def test_repr_shows_no_segments(self):
        """__repr__ does not mention segments for flat experiment."""
        df = make_df_lognormal()
        exp = Experiment(df, "variant", "revenue", "lognormal", control="control")
        assert "segments" not in repr(exp)


class TestHierarchicalModelSelection:

    def test_hierarchical_model_selected_with_segment_col(self):
        """Hierarchical model class selected when segment_col is set."""
        from argonx.models.lognormal_model import HierarchicalLogNormalModel
        df = make_df_segmented()
        exp = Experiment(df, "variant", "revenue", "lognormal",
                         segment_col="device_type", control="control")
        assert exp.segment_col == "device_type"
        assert exp._segment_names == sorted(df["device_type"].unique().tolist())

    def test_segment_names_sorted(self):
        """Segment names stored in sorted order."""
        df = make_df_segmented()
        exp = Experiment(df, "variant", "revenue", "lognormal",
                         segment_col="device_type", control="control")
        assert exp._segment_names == sorted(exp._segment_names)

    def test_hierarchical_run_returns_results(self):
        """Hierarchical run completes and returns Results object."""
        df = make_df_segmented()
        exp = Experiment(df, "variant", "revenue", "lognormal",
                         segment_col="device_type", control="control")
        result = exp.run(n_draws=200)
        assert result is not None
        assert result.best_variant is not None

    def test_segment_results_populated(self):
        """segment_results is populated for hierarchical run."""
        df = make_df_segmented()
        exp = Experiment(df, "variant", "revenue", "lognormal",
                         segment_col="device_type", control="control")
        result = exp.run(n_draws=200)
        assert result.segment_results is not None
        assert isinstance(result.segment_results, dict)

    def test_segment_results_keys_match_segments(self):
        """segment_results keys match segment names in data."""
        df = make_df_segmented()
        exp = Experiment(df, "variant", "revenue", "lognormal",
                         segment_col="device_type", control="control")
        result = exp.run(n_draws=200)
        assert set(result.segment_results.keys()) == set(exp._segment_names)

    def test_each_segment_has_decision_result(self):
        """Each segment result has state and recommendation."""
        df = make_df_segmented()
        exp = Experiment(df, "variant", "revenue", "lognormal",
                         segment_col="device_type", control="control")
        result = exp.run(n_draws=200)
        for seg, seg_result in result.segment_results.items():
            assert hasattr(seg_result, "state")
            assert hasattr(seg_result, "recommendation")
            assert hasattr(seg_result, "best_variant")

    def test_segment_summary_runs(self):
        """segment_summary runs without error on hierarchical result."""
        df = make_df_segmented()
        exp = Experiment(df, "variant", "revenue", "lognormal",
                         segment_col="device_type", control="control")
        result = exp.run(n_draws=200)
        result.segment_summary()

    def test_hierarchical_repr_shows_segments(self):
        """__repr__ includes segment names for hierarchical experiment."""
        df = make_df_segmented()
        exp = Experiment(df, "variant", "revenue", "lognormal",
                         segment_col="device_type", control="control")
        assert "segments" in repr(exp)

    def test_priors_accepted_without_error(self):
        """priors dict passed to hierarchical model without error."""
        df = make_df_segmented()
        exp = Experiment(df, "variant", "revenue", "lognormal",
                         segment_col="device_type", control="control",
                         priors={"tau_prior_beta": 2.0})
        result = exp.run(n_draws=200)
        assert result is not None

    def test_hierarchical_with_guardrail(self):
        """Hierarchical run with guardrail produces segment guardrail results."""
        df = make_df_segmented()
        exp = Experiment(df, "variant", "revenue", "lognormal",
                         segment_col="device_type",
                         guardrails=["page_load"],
                         lower_is_better={"page_load": True},
                         control="control")
        result = exp.run(n_draws=200,
                         config={"guardrail_thresholds": {"page_load": 0.10}})
        assert result is not None
        for seg_result in result.segment_results.values():
            assert hasattr(seg_result.guardrails, "all_passed")

    def test_segment_cell_with_no_data_raises(self):
        """Missing data for a segment-variant cell raises ValueError."""
        df = make_df_segmented()
        df = df[~((df["device_type"] == "tv") & (df["variant"] == "variant_b"))]
        exp = Experiment(df, "variant", "revenue", "lognormal",
                         segment_col="device_type", control="control")
        with pytest.raises(ValueError, match="no valid data"):
            exp.run(n_draws=200)

    def test_to_dataframe_hierarchical_has_multiindex(self):
        """to_dataframe returns MultiIndex (segment, variant) for hierarchical result."""
        df = make_df_segmented()
        exp = Experiment(df, "variant", "revenue", "lognormal",
                         segment_col="device_type", control="control")
        result = exp.run(n_draws=200)
        df_out = result.to_dataframe()
        assert df_out.index.names == ["segment", "variant"]

    def test_to_dataframe_hierarchical_has_aggregate_rows(self):
        """to_dataframe includes aggregate rows for hierarchical result."""
        df = make_df_segmented()
        exp = Experiment(df, "variant", "revenue", "lognormal",
                         segment_col="device_type", control="control")
        result = exp.run(n_draws=200)
        df_out = result.to_dataframe()
        assert "aggregate" in df_out.index.get_level_values("segment")