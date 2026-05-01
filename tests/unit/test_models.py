import numpy as np
import pytest
import warnings
from argonx.models.binary_model import BinaryModel, HierarchicalBinaryModel
from argonx.models.gaussian_model import GaussianModel, StudentTModel, HierarchicalGaussianModel, HierarchicalStudentTModel
from argonx.models.lognormal_model import LogNormalModel, HierarchicalLogNormalModel
from argonx.models.count_model import PoissonModel, HierarchicalPoissonModel

def make_hier_binary(n_mobile=200, n_desktop=500, n_tv=30):
    return {'mobile': {'control': np.random.binomial(1, 0.1, n_mobile), 'variant_b': np.random.binomial(1, 0.14, n_mobile)}, 'desktop': {'control': np.random.binomial(1, 0.1, n_desktop), 'variant_b': np.random.binomial(1, 0.13, n_desktop)}, 'tv': {'control': np.random.binomial(1, 0.1, n_tv), 'variant_b': np.random.binomial(1, 0.12, n_tv)}}

def make_hier_lognormal(n_mobile=200, n_desktop=500, n_tv=30):
    return {'mobile': {'control': np.random.lognormal(3.8, 0.6, n_mobile), 'variant_b': np.random.lognormal(3.9, 0.6, n_mobile)}, 'desktop': {'control': np.random.lognormal(3.8, 0.6, n_desktop), 'variant_b': np.random.lognormal(3.85, 0.6, n_desktop)}, 'tv': {'control': np.random.lognormal(3.8, 0.6, n_tv), 'variant_b': np.random.lognormal(3.82, 0.6, n_tv)}}

def make_hier_gaussian(n_mobile=200, n_desktop=500, n_tv=30):
    return {'mobile': {'control': np.random.normal(100, 10, n_mobile), 'variant_b': np.random.normal(108, 10, n_mobile)}, 'desktop': {'control': np.random.normal(100, 10, n_desktop), 'variant_b': np.random.normal(105, 10, n_desktop)}, 'tv': {'control': np.random.normal(100, 10, n_tv), 'variant_b': np.random.normal(102, 10, n_tv)}}

def make_hier_poisson(n_mobile=200, n_desktop=500, n_tv=30):
    return {'mobile': {'control': np.random.poisson(5, n_mobile).astype(int), 'variant_b': np.random.poisson(6, n_mobile).astype(int)}, 'desktop': {'control': np.random.poisson(5, n_desktop).astype(int), 'variant_b': np.random.poisson(6, n_desktop).astype(int)}, 'tv': {'control': np.random.poisson(5, n_tv).astype(int), 'variant_b': np.random.poisson(5, n_tv).astype(int)}}

class TestBaseModel:

    def test_rejects_non_dict_input(self):
        """Rejects input that is not a dictionary."""
        model = BinaryModel()
        with pytest.raises(TypeError):
            model.fit([1, 2, 3])

    def test_rejects_single_variant(self):
        """Rejects datasets with fewer than two variants."""
        model = BinaryModel()
        with pytest.raises(ValueError):
            model.fit({'A': np.array([1, 0])})

    def test_rejects_non_numpy_values(self):
        """Rejects values that are not numpy arrays."""
        model = BinaryModel()
        data = {'A': [1, 0, 1], 'B': np.array([1, 0, 1])}
        with pytest.raises(TypeError):
            model.fit(data)

    def test_rejects_empty_arrays(self):
        """Rejects empty arrays in input data."""
        model = BinaryModel()
        data = {'A': np.array([]), 'B': np.array([1, 0])}
        with pytest.raises(ValueError):
            model.fit(data)

    def test_rejects_nan_values(self):
        """Rejects arrays containing NaN values."""
        model = BinaryModel()
        data = {'A': np.array([1, np.nan]), 'B': np.array([0, 1])}
        with pytest.raises(ValueError):
            model.fit(data)

    def test_rejects_multidimensional_arrays(self):
        """Rejects non-1D arrays in input data."""
        model = BinaryModel()
        data = {'A': np.array([[1, 0], [0, 1]]), 'B': np.array([1, 0])}
        with pytest.raises(ValueError):
            model.fit(data)

    def test_stores_sorted_variant_names(self):
        """Stores variant names in deterministic sorted order."""
        model = BinaryModel()
        data = {'Z': np.array([1, 0]), 'A': np.array([0, 1]), 'M': np.array([1, 1])}
        model.fit(data)
        assert model.variant_names == ['A', 'M', 'Z']

    def test_does_not_mutate_input_data(self):
        """Ensures original input arrays are not modified."""
        model = BinaryModel()
        original = {'A': np.array([1, 0]), 'B': np.array([0, 1])}
        copy = {k: v.copy() for k, v in original.items()}
        model.fit(original)
        for k in original:
            assert np.array_equal(original[k], copy[k])

    def test_internal_data_is_not_view(self):
        """Ensures model stores copies not views of input arrays."""
        model = BinaryModel()
        data = {'A': np.array([1, 0]), 'B': np.array([0, 1])}
        model.fit(data)
        model.data['A'][0] = 999
        assert data['A'][0] == 1

    def test_variant_names_match_keys(self):
        """Variant names must match input keys."""
        model = BinaryModel()
        data = {'A': np.array([1]), 'B': np.array([0])}
        model.fit(data)
        assert set(model.variant_names) == set(data.keys())

    def test_fit_overwrites_previous_data(self):
        """Calling fit twice should overwrite previous state."""
        model = BinaryModel()
        data1 = {'A': np.array([1]), 'B': np.array([0])}
        data2 = {'A': np.array([0]), 'B': np.array([1])}
        model.fit(data1)
        model.fit(data2)
        assert np.array_equal(model.data['A'], data2['A'])

    def test_requires_fit_before_sampling(self):
        """Prevents sampling before model is fitted."""
        model = BinaryModel()
        with pytest.raises(ValueError):
            model.sample_posterior()

    def test_sampling_with_zero_draws(self):
        """Handles zero draws correctly."""
        model = BinaryModel()
        data = {'A': np.array([1, 0]), 'B': np.array([0, 1])}
        model.fit(data)
        samples = model.sample_posterior(0)
        assert samples.shape == (0, 2)

    def test_sampling_with_negative_draws(self):
        """Rejects negative number of draws."""
        model = BinaryModel()
        data = {'A': np.array([1, 0]), 'B': np.array([0, 1])}
        model.fit(data)
        with pytest.raises(ValueError):
            model.sample_posterior(-5)

    def test_handles_large_input(self):
        """Handles large datasets without breaking."""
        model = BinaryModel()
        data = {'A': np.random.randint(0, 2, size=100000), 'B': np.random.randint(0, 2, size=100000)}
        model.fit(data)
        assert len(model.data['A']) == 100000

    def test_handles_all_zero_data(self):
        """Handles all-zero datasets."""
        model = BinaryModel()
        data = {'A': np.zeros(10), 'B': np.zeros(10)}
        model.fit(data)
        assert np.sum(model.data['A']) == 0

    def test_handles_all_one_data(self):
        """Handles all-one datasets."""
        model = BinaryModel()
        data = {'A': np.ones(10), 'B': np.ones(10)}
        model.fit(data)
        assert np.sum(model.data['A']) == 10

class TestBinaryModel:

    def test_rejects_non_binary_values(self):
        """Rejects values outside 0/1 domain."""
        model = BinaryModel()
        data = {'A': np.array([0, 1, 2]), 'B': np.array([1, 0, 1])}
        with pytest.raises(ValueError):
            model.fit(data)

    def test_rejects_float_binary_values(self):
        """Accepts float representations of binary values."""
        model = BinaryModel()
        data = {'A': np.array([0.0, 1.0]), 'B': np.array([1.0, 0.0])}
        model.fit(data)

    def test_accepts_valid_binary_input(self):
        """Accepts properly formatted binary input data."""
        model = BinaryModel()
        data = {'A': np.array([0, 1, 1]), 'B': np.array([1, 0, 1])}
        model.fit(data)

    def test_output_shape(self):
        """Ensures output matches expected (n_draws, n_variants)."""
        model = BinaryModel()
        data = {'A': np.array([1, 0, 1]), 'B': np.array([0, 1, 1])}
        model.fit(data)
        samples = model.sample_posterior(1500)
        assert samples.shape == (1500, 2)

    def test_output_range_bounds(self):
        """Ensures posterior samples stay within [0,1]."""
        model = BinaryModel()
        data = {'A': np.array([1, 0, 1]), 'B': np.array([0, 1, 1])}
        model.fit(data)
        samples = model.sample_posterior(1000)
        assert (samples >= 0).all() and (samples <= 1).all()

    def test_requires_fit_before_sampling(self):
        """Prevents sampling before fitting."""
        model = BinaryModel()
        with pytest.raises(ValueError):
            model.sample_posterior()

    def test_sampling_zero_draws(self):
        """Handles zero draws correctly."""
        model = BinaryModel()
        data = {'A': np.array([1, 0]), 'B': np.array([0, 1])}
        model.fit(data)
        assert model.sample_posterior(0).shape == (0, 2)

    def test_sampling_negative_draws(self):
        """Rejects negative number of draws."""
        model = BinaryModel()
        data = {'A': np.array([1, 0]), 'B': np.array([0, 1])}
        model.fit(data)
        with pytest.raises(ValueError):
            model.sample_posterior(-10)

    def test_detects_clear_winner(self):
        """Assigns high probability to clearly better variant."""
        np.random.seed(42)
        model = BinaryModel()
        data = {'A': np.array([0, 0, 0, 0, 0]), 'B': np.array([1, 1, 1, 1, 1])}
        model.fit(data)
        samples = model.sample_posterior(2000)
        assert np.mean(samples[:, 1] > samples[:, 0]) > 0.95

    def test_equal_variants_behavior(self):
        """Produces near 0.5 probability for identical variants."""
        np.random.seed(42)
        model = BinaryModel()
        data = {'A': np.array([1, 0, 1, 0]), 'B': np.array([1, 0, 1, 0])}
        model.fit(data)
        samples = model.sample_posterior(2000)
        assert 0.4 < np.mean(samples[:, 1] > samples[:, 0]) < 0.6

    def test_small_sample_uncertainty(self):
        """Maintains uncertainty with very small datasets."""
        np.random.seed(42)
        model = BinaryModel()
        data = {'A': np.array([1]), 'B': np.array([0])}
        model.fit(data)
        samples = model.sample_posterior(2000)
        assert 0.15 < np.mean(samples[:, 1] > samples[:, 0]) < 0.85

    def test_variant_order_alignment(self):
        """Ensures correct column mapping after sorting variants."""
        np.random.seed(42)
        model = BinaryModel()
        data = {'B': np.array([1, 1, 1]), 'A': np.array([0, 0, 0])}
        model.fit(data)
        samples = model.sample_posterior(1000)
        assert np.mean(samples[:, 1] > samples[:, 0]) > 0.95

    def test_all_success_edge_case(self):
        """Handles all-success data without instability."""
        model = BinaryModel()
        data = {'A': np.ones(100), 'B': np.ones(100)}
        model.fit(data)
        assert np.isfinite(model.sample_posterior(1000)).all()

    def test_all_failure_edge_case(self):
        """Handles all-failure data without instability."""
        model = BinaryModel()
        data = {'A': np.zeros(100), 'B': np.zeros(100)}
        model.fit(data)
        assert np.isfinite(model.sample_posterior(1000)).all()

    def test_large_scale_input(self):
        """Handles large datasets efficiently."""
        model = BinaryModel()
        data = {'A': np.random.randint(0, 2, size=100000), 'B': np.random.randint(0, 2, size=100000)}
        model.fit(data)
        assert model.sample_posterior(1000).shape == (1000, 2)

    def test_reproducibility_with_seed(self):
        """Ensures deterministic output with fixed seed."""
        model = BinaryModel()
        data = {'A': np.array([1, 0, 1]), 'B': np.array([0, 1, 1])}
        model.fit(data)
        np.random.seed(42)
        s1 = model.sample_posterior(500)
        np.random.seed(42)
        s2 = model.sample_posterior(500)
        assert np.allclose(s1, s2)

class TestPoissonModel:

    def test_rejects_negative_values(self):
        """Rejects negative count values."""
        model = PoissonModel()
        data = {'A': np.array([1, 2, -1]), 'B': np.array([0, 1, 2])}
        with pytest.raises(ValueError):
            model.fit(data)

    def test_rejects_non_integer_values(self):
        """Rejects non-integer data."""
        model = PoissonModel()
        data = {'A': np.array([1.5, 2.0]), 'B': np.array([1, 2])}
        with pytest.raises(ValueError):
            model.fit(data)

    def test_accepts_valid_count_data(self):
        """Accepts valid non-negative integer data."""
        model = PoissonModel()
        data = {'A': np.array([0, 1, 2]), 'B': np.array([3, 4, 5])}
        model.fit(data)

    def test_output_shape(self):
        """Ensures output matches expected (n_draws, n_variants)."""
        model = PoissonModel()
        data = {'A': np.array([0, 1, 2]), 'B': np.array([2, 3, 4])}
        model.fit(data)
        assert model.sample_posterior(500).shape == (500, 2)

    def test_no_nan_or_inf(self):
        """Ensures posterior samples contain finite values only."""
        model = PoissonModel()
        data = {'A': np.array([0, 1, 2, 3]), 'B': np.array([2, 3, 4, 5])}
        model.fit(data)
        samples = model.sample_posterior(500)
        assert np.isfinite(samples).all()

    def test_requires_fit_before_sampling(self):
        """Prevents sampling before fitting."""
        model = PoissonModel()
        with pytest.raises(ValueError):
            model.sample_posterior()

    def test_sampling_zero_draws(self):
        """Handles zero draws correctly."""
        model = PoissonModel()
        data = {'A': np.array([1, 2]), 'B': np.array([2, 3])}
        model.fit(data)
        assert model.sample_posterior(0).shape == (0, 2)

    def test_sampling_negative_draws(self):
        """Rejects negative number of draws."""
        model = PoissonModel()
        data = {'A': np.array([1, 2]), 'B': np.array([2, 3])}
        model.fit(data)
        with pytest.raises(ValueError):
            model.sample_posterior(-10)

    def test_detects_higher_rate(self):
        """Assigns higher posterior rate to higher count variant."""
        model = PoissonModel()
        data = {'A': np.array([0, 1, 1, 0, 1]), 'B': np.array([5, 6, 7, 6, 5])}
        model.fit(data)
        samples = model.sample_posterior(1000)
        assert np.mean(samples[:, 1] > samples[:, 0]) > 0.9

    def test_zero_only_data(self):
        """Handles all-zero data without failure."""
        model = PoissonModel()
        data = {'A': np.array([0, 0, 0]), 'B': np.array([0, 0, 0])}
        model.fit(data)
        assert np.isfinite(model.sample_posterior(500)).all()

    def test_large_counts(self):
        """Handles large count values without instability."""
        model = PoissonModel()
        data = {'A': np.array([1000, 1200, 1100]), 'B': np.array([2000, 2100, 2200])}
        model.fit(data)
        assert np.isfinite(model.sample_posterior(500)).all()

    def test_unbalanced_sample_sizes(self):
        """Handles variants with very different sample sizes."""
        model = PoissonModel()
        data = {'A': np.random.poisson(1, size=10), 'B': np.random.poisson(5, size=1000)}
        model.fit(data)
        assert model.sample_posterior(500).shape == (500, 2)

class TestGaussianModel:

    def test_accepts_real_values(self):
        """Accepts arbitrary real-valued input data."""
        model = GaussianModel(0, 10, 5)
        data = {'A': np.array([-1.5, 0.0, 2.3]), 'B': np.array([1.2, -0.7, 3.1])}
        model.fit(data)

    def test_output_shape(self):
        """Ensures output matches expected (n_draws, n_variants)."""
        model = GaussianModel(0, 10, 5)
        data = {'A': np.random.normal(0, 1, 100), 'B': np.random.normal(1, 1, 100)}
        model.fit(data)
        assert model.sample_posterior(1200).shape == (1200, 2)

    def test_no_nan_or_inf(self):
        """Ensures posterior samples contain finite values only."""
        model = GaussianModel(0, 10, 5)
        data = {'A': np.random.normal(0, 1, 100), 'B': np.random.normal(1, 1, 100)}
        model.fit(data)
        assert np.isfinite(model.sample_posterior(1000)).all()

    def test_requires_fit_before_sampling(self):
        """Prevents sampling before fitting."""
        model = GaussianModel()
        with pytest.raises(ValueError):
            model.sample_posterior()

    def test_sampling_zero_draws(self):
        """Handles zero draws correctly."""
        model = GaussianModel()
        data = {'A': np.array([0.0, 1.0]), 'B': np.array([1.0, 2.0])}
        model.fit(data)
        assert model.sample_posterior(0).shape == (0, 2)

    def test_sampling_negative_draws(self):
        """Rejects negative number of draws."""
        model = GaussianModel()
        data = {'A': np.array([0.0, 1.0]), 'B': np.array([1.0, 2.0])}
        model.fit(data)
        with pytest.raises(ValueError):
            model.sample_posterior(-10)

    def test_detects_clear_mean_difference(self):
        """Assigns high probability to clearly higher mean variant."""
        model = GaussianModel(0, 10, 5)
        np.random.seed(42)
        data = {'A': np.random.normal(0, 1, 200), 'B': np.random.normal(3, 1, 200)}
        model.fit(data)
        samples = model.sample_posterior(2000)
        assert np.mean(samples[:, 1] > samples[:, 0]) > 0.95

    def test_small_sample_behavior(self):
        """Maintains uncertainty with very small datasets."""
        model = GaussianModel(0, 10, 5)
        data = {'A': np.array([0.0, 0.5]), 'B': np.array([1.0, 1.5])}
        model.fit(data)
        samples = model.sample_posterior(2000)
        assert 0.2 < np.mean(samples[:, 1] > samples[:, 0]) < 0.8

    def test_zero_variance_data(self):
        """Rejects constant data with zero variance."""
        model = GaussianModel(0, 10, 5)
        data = {'A': np.array([5, 5, 5, 5]), 'B': np.array([5, 5, 5, 5])}
        with pytest.raises(ValueError):
            model.fit(data)

class TestStudentTModel:

    def test_accepts_real_values(self):
        """Accepts arbitrary real-valued input including negatives."""
        model = StudentTModel()
        data = {'A': np.array([-10, 0, 5]), 'B': np.array([1, -3, 2])}
        model.fit(data)

    def test_output_shape(self):
        """Ensures output matches expected (n_draws, n_variants)."""
        model = StudentTModel()
        data = {'A': np.random.standard_t(5, size=100), 'B': np.random.standard_t(5, size=100) + 1}
        model.fit(data)
        assert model.sample_posterior(1500).shape == (1500, 2)

    def test_no_nan_or_inf(self):
        """Ensures posterior samples contain finite values only."""
        model = StudentTModel()
        data = {'A': np.random.standard_t(5, size=100), 'B': np.random.standard_t(5, size=100)}
        model.fit(data)
        assert np.isfinite(model.sample_posterior(1000)).all()

    def test_requires_fit_before_sampling(self):
        """Prevents sampling before fitting."""
        model = StudentTModel()
        with pytest.raises(ValueError):
            model.sample_posterior()

    def test_sampling_zero_draws(self):
        """Handles zero draws correctly."""
        model = StudentTModel()
        data = {'A': np.array([0.0, 1.0]), 'B': np.array([1.0, 2.0])}
        model.fit(data)
        assert model.sample_posterior(0).shape == (0, 2)

    def test_sampling_negative_draws(self):
        """Rejects negative number of draws."""
        model = StudentTModel()
        data = {'A': np.array([0.0, 1.0]), 'B': np.array([1.0, 2.0])}
        model.fit(data)
        with pytest.raises(ValueError):
            model.sample_posterior(-10)

    def test_detects_mean_shift(self):
        """Detects higher mean despite heavy-tailed noise."""
        model = StudentTModel()
        np.random.seed(42)
        data = {'A': np.random.standard_t(5, size=200), 'B': np.random.standard_t(5, size=200) + 1.5}
        model.fit(data)
        samples = model.sample_posterior(2000)
        assert np.mean(samples[:, 1] > samples[:, 0]) > 0.7

    def test_equal_distributions(self):
        """Produces near 0.5 probability for identical distributions."""
        model = StudentTModel()
        np.random.seed(42)
        data = {'A': np.random.standard_t(5, size=200), 'B': np.random.standard_t(5, size=200)}
        model.fit(data)
        samples = model.sample_posterior(2000)
        assert 0.4 < np.mean(samples[:, 1] > samples[:, 0]) < 0.6

    def test_outlier_robustness(self):
        """Remains stable when extreme outliers are present."""
        model = StudentTModel()
        data = {'A': np.concatenate([np.random.normal(0, 1, 100), [1000]]), 'B': np.random.normal(1, 1, 100)}
        model.fit(data)
        samples = model.sample_posterior(2000)
        assert np.mean(samples[:, 1] > samples[:, 0]) > 0.6

    def test_extreme_outlier_does_not_dominate(self):
        """Single extreme value should not fully dominate posterior."""
        model = StudentTModel()
        data = {'A': np.array([0, 0, 0, 0, 10000]), 'B': np.array([1, 1, 1, 1, 1])}
        model.fit(data)
        samples = model.sample_posterior(2000)
        assert np.mean(samples[:, 1] > samples[:, 0]) > 0.5

    def test_unbalanced_sample_sizes(self):
        """Handles variants with highly unequal sample sizes."""
        model = StudentTModel()
        data = {'A': np.random.standard_t(5, size=10), 'B': np.random.standard_t(5, size=1000)}
        model.fit(data)
        assert model.sample_posterior(1000).shape == (1000, 2)

class TestLogNormalModel:

    def test_rejects_non_positive_values(self):
        """Rejects zero or negative input values."""
        model = LogNormalModel()
        data = {'A': np.array([1.0, 2.0, 3.0]), 'B': np.array([0.0, 1.0, 2.0])}
        with pytest.raises(ValueError):
            model.fit(data)

    def test_accepts_positive_values(self):
        """Accepts strictly positive real values."""
        model = LogNormalModel()
        data = {'A': np.array([1.0, 2.0, 3.0]), 'B': np.array([2.0, 3.0, 4.0])}
        model.fit(data)

    def test_output_shape(self):
        """Ensures output matches expected (n_draws, n_variants)."""
        model = LogNormalModel()
        data = {'A': np.random.lognormal(0, 1, 100), 'B': np.random.lognormal(1, 1, 100)}
        model.fit(data)
        assert model.sample_posterior(1200).shape == (1200, 2)

    def test_no_nan_or_inf(self):
        """Ensures posterior samples contain finite values only."""
        model = LogNormalModel()
        data = {'A': np.random.lognormal(0, 1, 100), 'B': np.random.lognormal(1, 1, 100)}
        model.fit(data)
        assert np.isfinite(model.sample_posterior(1000)).all()

    def test_requires_fit_before_sampling(self):
        """Prevents sampling before fitting."""
        model = LogNormalModel()
        with pytest.raises(ValueError):
            model.sample_posterior()

    def test_sampling_zero_draws(self):
        """Handles zero draws correctly."""
        model = LogNormalModel()
        data = {'A': np.array([1.0, 2.0]), 'B': np.array([2.0, 3.0])}
        model.fit(data)
        assert model.sample_posterior(0).shape == (0, 2)

    def test_sampling_negative_draws(self):
        """Rejects negative number of draws."""
        model = LogNormalModel()
        data = {'A': np.array([1.0, 2.0]), 'B': np.array([2.0, 3.0])}
        model.fit(data)
        with pytest.raises(ValueError):
            model.sample_posterior(-10)

    def test_detects_scale_difference(self):
        """Assigns higher expected value to higher-scale variant."""
        model = LogNormalModel()
        np.random.seed(42)
        data = {'A': np.random.lognormal(0, 1, 200), 'B': np.random.lognormal(1, 1, 200)}
        model.fit(data)
        samples = model.sample_posterior(2000)
        assert np.mean(samples[:, 1] > samples[:, 0]) > 0.9

    def test_equal_distributions(self):
        """Produces near 0.5 probability for identical distributions."""
        model = LogNormalModel()
        np.random.seed(42)
        data = {'A': np.random.lognormal(0, 1, 200), 'B': np.random.lognormal(0, 1, 200)}
        model.fit(data)
        samples = model.sample_posterior(2000)
        assert 0.4 < np.mean(samples[:, 1] > samples[:, 0]) < 0.6

    def test_expected_value_transformation(self):
        """Ensures output reflects exp(mu + sigma^2 / 2) transformation."""
        model = LogNormalModel()
        data = {'A': np.array([1.0, 2.0, 3.0]), 'B': np.array([2.0, 3.0, 4.0])}
        model.fit(data)
        assert (model.sample_posterior(500) > 0).all()

    def test_reproducibility(self):
        """Ensures deterministic sampling via PyMC seed."""
        model = LogNormalModel()
        data = {'A': np.array([1.0, 2.0, 3.0]), 'B': np.array([2.0, 3.0, 4.0])}
        model.fit(data)
        s1 = model.sample_posterior(300)
        s2 = model.sample_posterior(300)
        assert np.allclose(s1, s2)

class TestHierarchicalBinaryModel:

    def test_accepts_valid_nested_input(self):
        """Accepts correctly structured nested dict input."""
        model = HierarchicalBinaryModel()
        model.fit(make_hier_binary())

    def test_rejects_flat_dict(self):
        """Rejects flat dict instead of nested segment dict."""
        model = HierarchicalBinaryModel()
        with pytest.raises(TypeError):
            model.fit({'control': np.array([0, 1]), 'variant_b': np.array([1, 0])})

    def test_rejects_single_segment(self):
        """Rejects input with fewer than two segments."""
        model = HierarchicalBinaryModel()
        with pytest.raises(ValueError):
            model.fit({'mobile': {'control': np.array([0, 1]), 'variant_b': np.array([1, 0])}})

    def test_rejects_inconsistent_variants(self):
        """Rejects segments with different variant sets."""
        model = HierarchicalBinaryModel()
        data = {'mobile': {'control': np.array([0, 1]), 'variant_b': np.array([1, 0])}, 'desktop': {'control': np.array([0, 1]), 'variant_c': np.array([1, 0])}}
        with pytest.raises(ValueError):
            model.fit(data)

    def test_rejects_non_binary_values(self):
        """Rejects non-binary values in segment data."""
        model = HierarchicalBinaryModel()
        data = {'mobile': {'control': np.array([0, 1, 2]), 'variant_b': np.array([1, 0, 1])}, 'desktop': {'control': np.array([0, 1]), 'variant_b': np.array([1, 0])}}
        with pytest.raises(ValueError):
            model.fit(data)

    def test_rejects_nan_in_segment(self):
        """Rejects NaN values in any segment cell."""
        model = HierarchicalBinaryModel()
        data = {'mobile': {'control': np.array([0, np.nan]), 'variant_b': np.array([1, 0])}, 'desktop': {'control': np.array([0, 1]), 'variant_b': np.array([1, 0])}}
        with pytest.raises(ValueError):
            model.fit(data)

    def test_population_sample_shape(self):
        """sample_posterior returns (n_draws, n_variants)."""
        model = HierarchicalBinaryModel()
        model.fit(make_hier_binary())
        samples = model.sample_posterior(n_draws=200)
        assert samples.shape == (200, 2)

    def test_segment_sample_shape(self):
        """sample_posterior_by_segment returns (n_draws, n_segments, n_variants)."""
        model = HierarchicalBinaryModel()
        model.fit(make_hier_binary())
        model.sample_posterior(n_draws=200)
        seg_samples = model.sample_posterior_by_segment(n_draws=200)
        assert seg_samples.shape == (200, 3, 2)

    def test_population_samples_in_unit_interval(self):
        """Population-level samples stay within [0, 1]."""
        model = HierarchicalBinaryModel()
        model.fit(make_hier_binary())
        samples = model.sample_posterior(n_draws=200)
        assert (samples >= 0).all() and (samples <= 1).all()

    def test_segment_samples_in_unit_interval(self):
        """Per-segment samples stay within [0, 1]."""
        model = HierarchicalBinaryModel()
        model.fit(make_hier_binary())
        model.sample_posterior(n_draws=200)
        seg = model.sample_posterior_by_segment(n_draws=200)
        assert (seg >= 0).all() and (seg <= 1).all()

    def test_sorted_segment_names(self):
        """Segment names stored in sorted order."""
        model = HierarchicalBinaryModel()
        model.fit(make_hier_binary())
        assert model.segment_names == sorted(model.segment_names)

    def test_sorted_variant_names(self):
        """Variant names stored in sorted order."""
        model = HierarchicalBinaryModel()
        model.fit(make_hier_binary())
        assert model.variant_names == sorted(model.variant_names)

    def test_thin_segment_warns(self):
        """Thin segment emits UserWarning before sampling."""
        model = HierarchicalBinaryModel()
        data = make_hier_binary(n_tv=5)
        with pytest.warns(UserWarning, match='tv'):
            model.fit(data)

    def test_all_warnings_structure(self):
        """all_warnings has health and segments keys."""
        model = HierarchicalBinaryModel()
        model.fit(make_hier_binary())
        model.sample_posterior(n_draws=200)
        w = model.all_warnings
        assert 'health' in w
        assert 'segments' in w

    def test_reuses_trace_on_second_call(self):
        """sample_posterior_by_segment reuses trace without refitting."""
        model = HierarchicalBinaryModel()
        model.fit(make_hier_binary())
        model.sample_posterior(n_draws=200)
        trace_before = model._trace
        model.sample_posterior_by_segment(n_draws=200)
        assert model._trace is trace_before

    def test_invalid_priors_raise(self):
        """Unknown prior key raises ValueError."""
        with pytest.raises(ValueError):
            HierarchicalBinaryModel(priors={'bad_key': 1.0})

class TestHierarchicalLogNormalModel:

    def test_accepts_valid_nested_input(self):
        """Accepts correctly structured nested dict input."""
        model = HierarchicalLogNormalModel()
        model.fit(make_hier_lognormal())

    def test_rejects_flat_dict(self):
        """Rejects flat dict instead of nested segment dict."""
        model = HierarchicalLogNormalModel()
        with pytest.raises(TypeError):
            model.fit({'control': np.random.lognormal(3.8, 0.6, 100), 'variant_b': np.random.lognormal(3.9, 0.6, 100)})

    def test_rejects_non_positive_values(self):
        """Rejects non-positive values in segment data."""
        model = HierarchicalLogNormalModel()
        data = {'mobile': {'control': np.array([1.0, -1.0]), 'variant_b': np.array([1.0, 2.0])}, 'desktop': {'control': np.array([1.0, 2.0]), 'variant_b': np.array([2.0, 3.0])}}
        with pytest.raises(ValueError):
            model.fit(data)

    def test_rejects_inconsistent_variants(self):
        """Rejects segments with different variant sets."""
        model = HierarchicalLogNormalModel()
        data = {'mobile': {'control': np.array([1.0, 2.0]), 'variant_b': np.array([2.0, 3.0])}, 'desktop': {'control': np.array([1.0, 2.0]), 'variant_c': np.array([2.0, 3.0])}}
        with pytest.raises(ValueError):
            model.fit(data)

    def test_population_sample_shape(self):
        """sample_posterior returns (n_draws, n_variants)."""
        model = HierarchicalLogNormalModel()
        model.fit(make_hier_lognormal())
        assert model.sample_posterior(n_draws=200).shape == (200, 2)

    def test_segment_sample_shape(self):
        """sample_posterior_by_segment returns (n_draws, n_segments, n_variants)."""
        model = HierarchicalLogNormalModel()
        model.fit(make_hier_lognormal())
        model.sample_posterior(n_draws=200)
        assert model.sample_posterior_by_segment(n_draws=200).shape == (200, 3, 2)

    def test_population_samples_positive(self):
        """Population-level samples are strictly positive."""
        model = HierarchicalLogNormalModel()
        model.fit(make_hier_lognormal())
        samples = model.sample_posterior(n_draws=200)
        assert (samples > 0).all()

    def test_segment_samples_positive(self):
        """Per-segment samples are strictly positive."""
        model = HierarchicalLogNormalModel()
        model.fit(make_hier_lognormal())
        model.sample_posterior(n_draws=200)
        assert (model.sample_posterior_by_segment(n_draws=200) > 0).all()

    def test_sorted_segment_names(self):
        """Segment names stored in sorted order."""
        model = HierarchicalLogNormalModel()
        model.fit(make_hier_lognormal())
        assert model.segment_names == sorted(model.segment_names)

    def test_thin_segment_warns(self):
        """Thin segment emits UserWarning before sampling."""
        model = HierarchicalLogNormalModel()
        data = make_hier_lognormal(n_tv=5)
        with pytest.warns(UserWarning, match='tv'):
            model.fit(data)

    def test_all_warnings_structure(self):
        """all_warnings has health and segments keys."""
        model = HierarchicalLogNormalModel()
        model.fit(make_hier_lognormal())
        model.sample_posterior(n_draws=200)
        w = model.all_warnings
        assert 'health' in w and 'segments' in w

    def test_invalid_priors_raise(self):
        """Unknown prior key raises ValueError."""
        with pytest.raises(ValueError):
            HierarchicalLogNormalModel(priors={'bad_key': 1.0})

    def test_no_nan_in_population_samples(self):
        """Population-level samples contain no NaN or Inf."""
        model = HierarchicalLogNormalModel()
        model.fit(make_hier_lognormal())
        assert np.isfinite(model.sample_posterior(n_draws=200)).all()

class TestHierarchicalGaussianModel:

    def test_accepts_valid_nested_input(self):
        """Accepts correctly structured nested dict input."""
        model = HierarchicalGaussianModel()
        model.fit(make_hier_gaussian())

    def test_rejects_flat_dict(self):
        """Rejects flat dict instead of nested segment dict."""
        model = HierarchicalGaussianModel()
        with pytest.raises(TypeError):
            model.fit({'control': np.random.normal(100, 10, 100), 'variant_b': np.random.normal(108, 10, 100)})

    def test_rejects_inconsistent_variants(self):
        """Rejects segments with different variant sets."""
        model = HierarchicalGaussianModel()
        data = {'mobile': {'control': np.array([1.0, 2.0]), 'variant_b': np.array([2.0, 3.0])}, 'desktop': {'control': np.array([1.0, 2.0]), 'variant_c': np.array([2.0, 3.0])}}
        with pytest.raises(ValueError):
            model.fit(data)

    def test_population_sample_shape(self):
        """sample_posterior returns (n_draws, n_variants)."""
        model = HierarchicalGaussianModel()
        model.fit(make_hier_gaussian())
        assert model.sample_posterior(n_draws=200).shape == (200, 2)

    def test_segment_sample_shape(self):
        """sample_posterior_by_segment returns (n_draws, n_segments, n_variants)."""
        model = HierarchicalGaussianModel()
        model.fit(make_hier_gaussian())
        model.sample_posterior(n_draws=200)
        assert model.sample_posterior_by_segment(n_draws=200).shape == (200, 3, 2)

    def test_no_nan_in_population_samples(self):
        """Population-level samples contain no NaN or Inf."""
        model = HierarchicalGaussianModel()
        model.fit(make_hier_gaussian())
        assert np.isfinite(model.sample_posterior(n_draws=200)).all()

    def test_sorted_segment_names(self):
        """Segment names stored in sorted order."""
        model = HierarchicalGaussianModel()
        model.fit(make_hier_gaussian())
        assert model.segment_names == sorted(model.segment_names)

    def test_thin_segment_warns(self):
        """Thin segment emits UserWarning before sampling."""
        model = HierarchicalGaussianModel()
        data = make_hier_gaussian(n_tv=5)
        with pytest.warns(UserWarning, match='tv'):
            model.fit(data)

    def test_all_warnings_structure(self):
        """all_warnings has health and segments keys."""
        model = HierarchicalGaussianModel()
        model.fit(make_hier_gaussian())
        model.sample_posterior(n_draws=200)
        w = model.all_warnings
        assert 'health' in w and 'segments' in w

    def test_invalid_priors_raise(self):
        """Unknown prior key raises ValueError."""
        with pytest.raises(ValueError):
            HierarchicalGaussianModel(priors={'bad_key': 1.0})

class TestHierarchicalStudentTModel:

    def test_accepts_valid_nested_input(self):
        """Accepts correctly structured nested dict input."""
        model = HierarchicalStudentTModel()
        model.fit(make_hier_gaussian())

    def test_rejects_flat_dict(self):
        """Rejects flat dict instead of nested segment dict."""
        model = HierarchicalStudentTModel()
        with pytest.raises(TypeError):
            model.fit({'control': np.random.normal(100, 10, 100), 'variant_b': np.random.normal(108, 10, 100)})

    def test_population_sample_shape(self):
        """sample_posterior returns (n_draws, n_variants)."""
        model = HierarchicalStudentTModel()
        model.fit(make_hier_gaussian())
        assert model.sample_posterior(n_draws=200).shape == (200, 2)

    def test_segment_sample_shape(self):
        """sample_posterior_by_segment returns (n_draws, n_segments, n_variants)."""
        model = HierarchicalStudentTModel()
        model.fit(make_hier_gaussian())
        model.sample_posterior(n_draws=200)
        assert model.sample_posterior_by_segment(n_draws=200).shape == (200, 3, 2)

    def test_no_nan_in_population_samples(self):
        """Population-level samples contain no NaN or Inf."""
        model = HierarchicalStudentTModel()
        model.fit(make_hier_gaussian())
        assert np.isfinite(model.sample_posterior(n_draws=200)).all()

    def test_thin_segment_warns(self):
        """Thin segment emits UserWarning before sampling."""
        model = HierarchicalStudentTModel()
        data = make_hier_gaussian(n_tv=5)
        with pytest.warns(UserWarning, match='tv'):
            model.fit(data)

    def test_outlier_robustness(self):
        """Remains stable with extreme outliers present."""
        model = HierarchicalStudentTModel()
        data = make_hier_gaussian()
        data['mobile']['control'] = np.append(data['mobile']['control'], 100000.0)
        model.fit(data)
        assert np.isfinite(model.sample_posterior(n_draws=200)).all()

    def test_invalid_priors_raise(self):
        """Unknown prior key raises ValueError."""
        with pytest.raises(ValueError):
            HierarchicalStudentTModel(priors={'bad_key': 1.0})

class TestHierarchicalPoissonModel:

    def test_accepts_valid_nested_input(self):
        """Accepts correctly structured nested dict input."""
        model = HierarchicalPoissonModel()
        model.fit(make_hier_poisson())

    def test_rejects_flat_dict(self):
        """Rejects flat dict instead of nested segment dict."""
        model = HierarchicalPoissonModel()
        with pytest.raises(TypeError):
            model.fit({'control': np.array([1, 2, 3]), 'variant_b': np.array([2, 3, 4])})

    def test_rejects_negative_values(self):
        """Rejects negative count values in segment data."""
        model = HierarchicalPoissonModel()
        data = {'mobile': {'control': np.array([-1, 2], dtype=int), 'variant_b': np.array([1, 2], dtype=int)}, 'desktop': {'control': np.array([1, 2], dtype=int), 'variant_b': np.array([2, 3], dtype=int)}}
        with pytest.raises(ValueError):
            model.fit(data)

    def test_rejects_non_integer_values(self):
        """Rejects float count values in segment data."""
        model = HierarchicalPoissonModel()
        data = {'mobile': {'control': np.array([1.5, 2.0]), 'variant_b': np.array([1.0, 2.0])}, 'desktop': {'control': np.array([1, 2], dtype=int), 'variant_b': np.array([2, 3], dtype=int)}}
        with pytest.raises(ValueError):
            model.fit(data)

    def test_rejects_inconsistent_variants(self):
        """Rejects segments with different variant sets."""
        model = HierarchicalPoissonModel()
        data = {'mobile': {'control': np.array([1, 2], dtype=int), 'variant_b': np.array([2, 3], dtype=int)}, 'desktop': {'control': np.array([1, 2], dtype=int), 'variant_c': np.array([2, 3], dtype=int)}}
        with pytest.raises(ValueError):
            model.fit(data)

    def test_population_sample_shape(self):
        """sample_posterior returns (n_draws, n_variants)."""
        model = HierarchicalPoissonModel()
        model.fit(make_hier_poisson())
        assert model.sample_posterior(n_draws=200).shape == (200, 2)

    def test_segment_sample_shape(self):
        """sample_posterior_by_segment returns (n_draws, n_segments, n_variants)."""
        model = HierarchicalPoissonModel()
        model.fit(make_hier_poisson())
        model.sample_posterior(n_draws=200)
        assert model.sample_posterior_by_segment(n_draws=200).shape == (200, 3, 2)

    def test_population_samples_positive(self):
        """Population-level rate samples are strictly positive."""
        model = HierarchicalPoissonModel()
        model.fit(make_hier_poisson())
        assert (model.sample_posterior(n_draws=200) > 0).all()

    def test_no_nan_in_population_samples(self):
        """Population-level samples contain no NaN or Inf."""
        model = HierarchicalPoissonModel()
        model.fit(make_hier_poisson())
        assert np.isfinite(model.sample_posterior(n_draws=200)).all()

    def test_sorted_segment_names(self):
        """Segment names stored in sorted order."""
        model = HierarchicalPoissonModel()
        model.fit(make_hier_poisson())
        assert model.segment_names == sorted(model.segment_names)

    def test_thin_segment_warns(self):
        """Thin segment emits UserWarning before sampling."""
        model = HierarchicalPoissonModel()
        data = make_hier_poisson(n_tv=5)
        with pytest.warns(UserWarning, match='tv'):
            model.fit(data)

    def test_all_warnings_structure(self):
        """all_warnings has health and segments keys."""
        model = HierarchicalPoissonModel()
        model.fit(make_hier_poisson())
        model.sample_posterior(n_draws=200)
        w = model.all_warnings
        assert 'health' in w and 'segments' in w

    def test_invalid_priors_raise(self):
        """Unknown prior key raises ValueError."""
        with pytest.raises(ValueError):
            HierarchicalPoissonModel(priors={'bad_key': 1.0})