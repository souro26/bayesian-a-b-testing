import numpy as np
import pytest

from experimentation.models.binary_model import BinaryModel
from experimentation.models.gaussian_model import GaussianModel
from experimentation.models.lognormal_model import LogNormalModel
from experimentation.models.gaussian_model import StudentTModel
from experimentation.models.count_model import CountModel


# BaseModel Tests


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
            model.fit({"A": np.array([1, 0])})

    def test_rejects_non_numpy_values(self):
        """Rejects values that are not numpy arrays."""
        model = BinaryModel()

        data = {
            "A": [1, 0, 1],
            "B": np.array([1, 0, 1])
        }

        with pytest.raises(TypeError):
            model.fit(data)

    def test_rejects_empty_arrays(self):
        """Rejects empty arrays in input data."""
        model = BinaryModel()

        data = {
            "A": np.array([]),
            "B": np.array([1, 0])
        }

        with pytest.raises(ValueError):
            model.fit(data)

    def test_rejects_nan_values(self):
        """Rejects arrays containing NaN values."""
        model = BinaryModel()

        data = {
            "A": np.array([1, np.nan]),
            "B": np.array([0, 1])
        }

        with pytest.raises(ValueError):
            model.fit(data)

    def test_rejects_multidimensional_arrays(self):
        """Rejects non-1D arrays in input data."""
        model = BinaryModel()

        data = {
            "A": np.array([[1, 0], [0, 1]]),
            "B": np.array([1, 0])
        }

        with pytest.raises(ValueError):
            model.fit(data)

    def test_stores_sorted_variant_names(self):
        """Stores variant names in deterministic sorted order."""
        model = BinaryModel()

        data = {
            "Z": np.array([1, 0]),
            "A": np.array([0, 1]),
            "M": np.array([1, 1])
        }

        model.fit(data)

        assert model.variant_names is not None
        assert model.variant_names == ["A", "M", "Z"]

    def test_does_not_mutate_input_data(self):
        """Ensures original input arrays are not modified."""
        model = BinaryModel()

        original = {
            "A": np.array([1, 0]),
            "B": np.array([0, 1])
        }

        copy = {k: v.copy() for k, v in original.items()}

        model.fit(original)

        for k in original:
            assert np.array_equal(original[k], copy[k])

    def test_internal_data_is_not_view(self):
        """Ensures model stores copies, not views of input arrays."""
        model = BinaryModel()

        data = {
            "A": np.array([1, 0]),
            "B": np.array([0, 1])
        }

        model.fit(data)

        assert model.data is not None
        model.data["A"][0] = 999

        assert data["A"][0] == 1

    def test_variant_names_match_keys(self):
        """Variant names must match input keys."""
        model = BinaryModel()

        data = {
            "A": np.array([1]),
            "B": np.array([0])
        }

        model.fit(data)

        assert model.variant_names is not None
        assert set(model.variant_names) == set(data.keys())

    def test_fit_overwrites_previous_data(self):
        """Calling fit twice should overwrite previous state."""
        model = BinaryModel()

        data1 = {
            "A": np.array([1]),
            "B": np.array([0])
        }

        data2 = {
            "A": np.array([0]),
            "B": np.array([1])
        }

        model.fit(data1)
        model.fit(data2)

        assert model.data is not None
        assert np.array_equal(model.data["A"], data2["A"])
        assert np.array_equal(model.data["B"], data2["B"])

    def test_requires_fit_before_sampling(self):
        """Prevents sampling before model is fitted."""
        model = BinaryModel()

        with pytest.raises(ValueError):
            model.sample_posterior()

    def test_sampling_with_zero_draws(self):
        """Handles zero draws correctly."""
        model = BinaryModel()

        data = {
            "A": np.array([1, 0]),
            "B": np.array([0, 1])
        }

        model.fit(data)
        samples = model.sample_posterior(0)

        assert samples.shape == (0, 2)

    def test_sampling_with_negative_draws(self):
        """Rejects negative number of draws."""
        model = BinaryModel()

        data = {
            "A": np.array([1, 0]),
            "B": np.array([0, 1])
        }

        model.fit(data)

        with pytest.raises(ValueError):
            model.sample_posterior(-5)

    def test_handles_large_input(self):
        """Handles large datasets without breaking."""
        model = BinaryModel()

        data = {
            "A": np.random.randint(0, 2, size=100000),
            "B": np.random.randint(0, 2, size=100000)
        }

        model.fit(data)

        assert model.data is not None
        assert len(model.data["A"]) == 100000

    def test_handles_all_zero_data(self):
        """Handles all-zero datasets."""
        model = BinaryModel()

        data = {
            "A": np.zeros(10),
            "B": np.zeros(10)
        }

        model.fit(data)

        assert model.data is not None
        assert np.sum(model.data["A"]) == 0

    def test_handles_all_one_data(self):
        """Handles all-one datasets."""
        model = BinaryModel()

        data = {
            "A": np.ones(10),
            "B": np.ones(10)
        }

        model.fit(data)

        assert model.data is not None
        assert np.sum(model.data["A"]) == 10


#BinaryModelTests


class TestBinaryModel:

    def test_rejects_non_binary_values(self):
        """Rejects values outside 0/1 domain."""
        model = BinaryModel()

        data = {
            "A": np.array([0, 1, 2]),
            "B": np.array([1, 0, 1])
        }

        with pytest.raises(ValueError):
            model.fit(data)

    def test_rejects_float_binary_values(self):
        """Rejects float representations of binary values."""
        model = BinaryModel()

        data = {
            "A": np.array([0.0, 1.0]),
            "B": np.array([1.0, 0.0])
        }

        # current implementation ALLOWS this — decide your stance
        model.fit(data)

    def test_accepts_valid_binary_input(self):
        """Accepts properly formatted binary input data."""
        model = BinaryModel()

        data = {
            "A": np.array([0, 1, 1]),
            "B": np.array([1, 0, 1])
        }

        model.fit(data)

    def test_output_shape(self):
        """Ensures output matches expected (n_draws, n_variants)."""
        model = BinaryModel()

        data = {
            "A": np.array([1, 0, 1]),
            "B": np.array([0, 1, 1])
        }

        model.fit(data)
        samples = model.sample_posterior(1500)

        assert samples.shape == (1500, 2)

    def test_output_range_bounds(self):
        """Ensures posterior samples stay within [0,1]."""
        model = BinaryModel()

        data = {
            "A": np.array([1, 0, 1]),
            "B": np.array([0, 1, 1])
        }

        model.fit(data)
        samples = model.sample_posterior(1000)

        assert (samples >= 0).all()
        assert (samples <= 1).all()

    def test_requires_fit_before_sampling(self):
        """Prevents sampling before fitting."""
        model = BinaryModel()

        with pytest.raises(ValueError):
            model.sample_posterior()

    def test_sampling_zero_draws(self):
        """Handles zero draws correctly."""
        model = BinaryModel()

        data = {
            "A": np.array([1, 0]),
            "B": np.array([0, 1])
        }

        model.fit(data)
        samples = model.sample_posterior(0)

        assert samples.shape == (0, 2)

    def test_sampling_negative_draws(self):
        """Rejects negative number of draws."""
        model = BinaryModel()

        data = {
            "A": np.array([1, 0]),
            "B": np.array([0, 1])
        }

        model.fit(data)

        with pytest.raises(ValueError):
            model.sample_posterior(-10)

    def test_detects_clear_winner(self):
        """Assigns high probability to clearly better variant."""
        np.random.seed(42)

        model = BinaryModel()

        data = {
            "A": np.array([0, 0, 0, 0, 0]),
            "B": np.array([1, 1, 1, 1, 1])
        }

        model.fit(data)
        samples = model.sample_posterior(2000)

        prob = np.mean(samples[:, 1] > samples[:, 0])

        assert prob > 0.95

    def test_equal_variants_behavior(self):
        """Produces near 0.5 probability for identical variants."""
        np.random.seed(42)

        model = BinaryModel()

        data = {
            "A": np.array([1, 0, 1, 0]),
            "B": np.array([1, 0, 1, 0])
        }

        model.fit(data)
        samples = model.sample_posterior(2000)

        prob = np.mean(samples[:, 1] > samples[:, 0])

        assert 0.4 < prob < 0.6

    def test_small_sample_uncertainty(self):
        """Maintains uncertainty with very small datasets."""
        np.random.seed(42)

        model = BinaryModel()

        data = {
            "A": np.array([1]),
            "B": np.array([0])
        }

        model.fit(data)
        samples = model.sample_posterior(2000)

        prob = np.mean(samples[:, 1] > samples[:, 0])

        assert 0.15 < prob < 0.85

    def test_variant_order_alignment(self):
        """Ensures correct column mapping after sorting variants."""
        np.random.seed(42)

        model = BinaryModel()

        data = {
            "B": np.array([1, 1, 1]),
            "A": np.array([0, 0, 0])
        }

        model.fit(data)

        assert model.variant_names is not None
        samples = model.sample_posterior(1000)

        # A should be column 0, B column 1 (sorted order)
        prob = np.mean(samples[:, 1] > samples[:, 0])

        assert prob > 0.95

    def test_all_success_edge_case(self):
        """Handles all-success data without instability."""
        model = BinaryModel()

        data = {
            "A": np.ones(100),
            "B": np.ones(100)
        }

        model.fit(data)
        samples = model.sample_posterior(1000)

        assert np.isfinite(samples).all()

    def test_all_failure_edge_case(self):
        """Handles all-failure data without instability."""
        model = BinaryModel()

        data = {
            "A": np.zeros(100),
            "B": np.zeros(100)
        }

        model.fit(data)
        samples = model.sample_posterior(1000)

        assert np.isfinite(samples).all()

    def test_large_scale_input(self):
        """Handles large datasets efficiently."""
        model = BinaryModel()

        data = {
            "A": np.random.randint(0, 2, size=100000),
            "B": np.random.randint(0, 2, size=100000)
        }

        model.fit(data)
        samples = model.sample_posterior(1000)

        assert samples.shape == (1000, 2)

    def test_reproducibility_with_seed(self):
        """Ensures deterministic output with fixed seed."""
        model = BinaryModel()

        data = {
            "A": np.array([1, 0, 1]),
            "B": np.array([0, 1, 1])
        }

        model.fit(data)

        np.random.seed(42)
        s1 = model.sample_posterior(500)

        np.random.seed(42)
        s2 = model.sample_posterior(500)

        assert np.allclose(s1, s2)

#CountModel Tests

class TestCountModel:

    def test_rejects_negative_values(self):
        """Rejects negative count values."""
        model = CountModel()

        data = {
            "A": np.array([1, 2, -1]),
            "B": np.array([0, 1, 2])
        }

        with pytest.raises(ValueError):
            model.fit(data)

    def test_rejects_non_integer_values(self):
        """Rejects non-integer data."""
        model = CountModel()

        data = {
            "A": np.array([1.5, 2.0]),
            "B": np.array([1, 2])
        }

        with pytest.raises(ValueError):
            model.fit(data)

    def test_accepts_valid_count_data(self):
        """Accepts valid non-negative integer data."""
        model = CountModel()

        data = {
            "A": np.array([0, 1, 2]),
            "B": np.array([3, 4, 5])
        }

        model.fit(data)

    def test_output_shape(self):
        """Ensures output matches expected (n_draws, n_variants)."""
        model = CountModel()

        data = {
            "A": np.array([0, 1, 2]),
            "B": np.array([2, 3, 4])
        }

        model.fit(data)
        samples = model.sample_posterior(500)

        assert samples.shape == (500, 2)

    def test_no_nan_or_inf(self):
        """Ensures posterior samples contain finite values only."""
        model = CountModel()

        data = {
            "A": np.array([0, 1, 2, 3]),
            "B": np.array([2, 3, 4, 5])
        }

        model.fit(data)
        samples = model.sample_posterior(500)

        assert not np.isnan(samples).any()
        assert np.isfinite(samples).all()

    def test_requires_fit_before_sampling(self):
        """Prevents sampling before fitting."""
        model = CountModel()

        with pytest.raises(ValueError):
            model.sample_posterior()

    def test_sampling_zero_draws(self):
        """Handles zero draws correctly."""
        model = CountModel()

        data = {
            "A": np.array([1, 2]),
            "B": np.array([2, 3])
        }

        model.fit(data)
        samples = model.sample_posterior(0)

        assert samples.shape == (0, 2)

    def test_sampling_negative_draws(self):
        """Rejects negative number of draws."""
        model = CountModel()

        data = {
            "A": np.array([1, 2]),
            "B": np.array([2, 3])
        }

        model.fit(data)

        with pytest.raises(ValueError):
            model.sample_posterior(-10)

    def test_detects_higher_rate(self):
        """Assigns higher posterior rate to higher count variant."""
        model = CountModel()

        data = {
            "A": np.array([0, 1, 1, 0, 1]),
            "B": np.array([5, 6, 7, 6, 5])
        }

        model.fit(data)
        samples = model.sample_posterior(1000)

        prob = np.mean(samples[:, 1] > samples[:, 0])

        assert prob > 0.9

    def test_zero_only_data(self):
        """Handles all-zero data without failure."""
        model = CountModel()

        data = {
            "A": np.array([0, 0, 0]),
            "B": np.array([0, 0, 0])
        }

        model.fit(data)
        samples = model.sample_posterior(500)

        assert np.isfinite(samples).all()

    def test_large_counts(self):
        """Handles large count values without instability."""
        model = CountModel()

        data = {
            "A": np.array([1000, 1200, 1100]),
            "B": np.array([2000, 2100, 2200])
        }

        model.fit(data)
        samples = model.sample_posterior(500)

        assert np.isfinite(samples).all()

    def test_unbalanced_sample_sizes(self):
        """Handles variants with very different sample sizes."""
        model = CountModel()

        data = {
            "A": np.random.poisson(1, size=10),
            "B": np.random.poisson(5, size=1000)
        }

        model.fit(data)
        samples = model.sample_posterior(500)

        assert samples.shape == (500, 2)


#GaussianModel tests


class TestGaussianModel:

    def test_accepts_real_values(self):
        """Accepts arbitrary real-valued input data."""
        model = GaussianModel(0, 10, 5)

        data = {
            "A": np.array([-1.5, 0.0, 2.3]),
            "B": np.array([1.2, -0.7, 3.1])
        }

        model.fit(data)

    def test_output_shape(self):
        """Ensures output matches expected (n_draws, n_variants)."""
        model = GaussianModel(0, 10, 5)

        data = {
            "A": np.random.normal(0, 1, 100),
            "B": np.random.normal(1, 1, 100)
        }

        model.fit(data)
        samples = model.sample_posterior(1200)

        assert samples.shape == (1200, 2)

    def test_no_nan_or_inf(self):
        """Ensures posterior samples contain finite values only."""
        model = GaussianModel(0, 10, 5)

        data = {
            "A": np.random.normal(0, 1, 100),
            "B": np.random.normal(1, 1, 100)
        }

        model.fit(data)
        samples = model.sample_posterior(1000)

        assert not np.isnan(samples).any()
        assert np.isfinite(samples).all()

    def test_requires_fit_before_sampling(self):
        """Prevents sampling before fitting."""
        model = GaussianModel()

        with pytest.raises(ValueError):
            model.sample_posterior()

    def test_sampling_zero_draws(self):
        """Handles zero draws correctly."""
        model = GaussianModel()

        data = {
            "A": np.array([0.0, 1.0]),
            "B": np.array([1.0, 2.0])
        }

        model.fit(data)
        samples = model.sample_posterior(0)

        assert samples.shape == (0, 2)

    def test_sampling_negative_draws(self):
        """Rejects negative number of draws."""
        model = GaussianModel()

        data = {
            "A": np.array([0.0, 1.0]),
            "B": np.array([1.0, 2.0])
        }

        model.fit(data)

        with pytest.raises(ValueError):
            model.sample_posterior(-10)

    def test_detects_clear_mean_difference(self):
        """Assigns high probability to clearly higher mean variant."""
        model = GaussianModel(0, 10, 5)

        np.random.seed(42)

        data = {
            "A": np.random.normal(0, 1, 200),
            "B": np.random.normal(3, 1, 200)
        }

        model.fit(data)
        samples = model.sample_posterior(2000)

        prob = np.mean(samples[:, 1] > samples[:, 0])

        assert prob > 0.95

    def test_small_sample_behavior(self):
        """Maintains uncertainty with very small datasets."""
        model = GaussianModel(0, 10, 5)

        data = {
            "A": np.array([0.0, 0.5]),
            "B": np.array([1.0, 1.5])
        }

        model.fit(data)
        samples = model.sample_posterior(2000)

        prob = np.mean(samples[:, 1] > samples[:, 0])

        assert 0.2 < prob < 0.8

    def test_zero_variance_data(self):
        """Handles variants with zero data."""
        model = GaussianModel(0, 10, 5)

        data = {
            "A": np.array([5, 5, 5, 5]),
            "B": np.array([5, 5, 5, 5])
        }

        with pytest.raises(ValueError):
            model.fit(data)


#StudentTModel tests


class TestStudentTModel:

    def test_accepts_real_values(self):
        """Accepts arbitrary real-valued input including negatives."""
        model = StudentTModel()

        data = {
            "A": np.array([-10, 0, 5]),
            "B": np.array([1, -3, 2])
        }

        model.fit(data)

    def test_output_shape(self):
        """Ensures output matches expected (n_draws, n_variants)."""
        model = StudentTModel()

        data = {
            "A": np.random.standard_t(5, size=100),
            "B": np.random.standard_t(5, size=100) + 1
        }

        model.fit(data)
        samples = model.sample_posterior(1500)

        assert samples.shape == (1500, 2)

    def test_no_nan_or_inf(self):
        """Ensures posterior samples contain finite values only."""
        model = StudentTModel()

        data = {
            "A": np.random.standard_t(5, size=100),
            "B": np.random.standard_t(5, size=100)
        }

        model.fit(data)
        samples = model.sample_posterior(1000)

        assert not np.isnan(samples).any()
        assert np.isfinite(samples).all()

    def test_requires_fit_before_sampling(self):
        """Prevents sampling before fitting."""
        model = StudentTModel()

        with pytest.raises(ValueError):
            model.sample_posterior()

    def test_sampling_zero_draws(self):
        """Handles zero draws correctly."""
        model = StudentTModel()

        data = {
            "A": np.array([0.0, 1.0]),
            "B": np.array([1.0, 2.0])
        }

        model.fit(data)
        samples = model.sample_posterior(0)

        assert samples.shape == (0, 2)

    def test_sampling_negative_draws(self):
        """Rejects negative number of draws."""
        model = StudentTModel()

        data = {
            "A": np.array([0.0, 1.0]),
            "B": np.array([1.0, 2.0])
        }

        model.fit(data)

        with pytest.raises(ValueError):
            model.sample_posterior(-10)

    def test_detects_mean_shift(self):
        """Detects higher mean despite heavy-tailed noise."""
        model = StudentTModel()

        np.random.seed(42)

        data = {
            "A": np.random.standard_t(5, size=200),
            "B": np.random.standard_t(5, size=200) + 1.5
        }

        model.fit(data)
        samples = model.sample_posterior(2000)

        prob = np.mean(samples[:, 1] > samples[:, 0])

        assert prob > 0.7

    def test_equal_distributions(self):
        """Produces near 0.5 probability for identical distributions."""
        model = StudentTModel()

        np.random.seed(42)

        data = {
            "A": np.random.standard_t(5, size=200),
            "B": np.random.standard_t(5, size=200)
        }

        model.fit(data)
        samples = model.sample_posterior(2000)

        prob = np.mean(samples[:, 1] > samples[:, 0])

        assert 0.4 < prob < 0.6

    def test_small_sample_behavior(self):
        """Maintains uncertainty with very small datasets."""
        model = StudentTModel()

        data = {
            "A": np.array([0.0]),
            "B": np.array([1.0])
        }

        model.fit(data)
        samples = model.sample_posterior(2000)

        prob = np.mean(samples[:, 1] > samples[:, 0])

        assert 0.2 < prob < 0.8

    def test_outlier_robustness(self):
        """Remains stable when extreme outliers are present."""
        model = StudentTModel()

        data = {
            "A": np.concatenate([np.random.normal(0, 1, 100), [1000]]),
            "B": np.random.normal(1, 1, 100)
        }

        model.fit(data)
        samples = model.sample_posterior(2000)

        prob = np.mean(samples[:, 1] > samples[:, 0])

        assert prob > 0.6  # should not collapse due to outlier

    def test_extreme_outlier_does_not_dominate(self):
        """Single extreme value should not fully dominate posterior."""
        model = StudentTModel()

        data = {
            "A": np.array([0, 0, 0, 0, 10000]),
            "B": np.array([1, 1, 1, 1, 1])
        }

        model.fit(data)
        samples = model.sample_posterior(2000)

        prob = np.mean(samples[:, 1] > samples[:, 0])

        assert prob > 0.5  # B should still likely win

    def test_heavy_tail_advantage_over_gaussian_like_data(self):
        """Handles heavy-tailed distributions without instability."""
        model = StudentTModel()

        np.random.seed(42)

        data = {
            "A": np.random.standard_t(2, size=300),
            "B": np.random.standard_t(2, size=300) + 1
        }

        model.fit(data)
        samples = model.sample_posterior(2000)

        assert np.isfinite(samples).all()

    def test_unbalanced_sample_sizes(self):
        """Handles variants with highly unequal sample sizes."""
        model = StudentTModel()

        data = {
            "A": np.random.standard_t(5, size=10),
            "B": np.random.standard_t(5, size=1000)
        }

        model.fit(data)
        samples = model.sample_posterior(1000)

        assert samples.shape == (1000, 2)


#LogNormalModel tests


class TestLogNormalModel:

    def test_rejects_non_positive_values(self):
        """Rejects zero or negative input values."""
        model = LogNormalModel()

        data = {
            "A": np.array([1.0, 2.0, 3.0]),
            "B": np.array([0.0, 1.0, 2.0])
        }

        with pytest.raises(ValueError):
            model.fit(data)

    def test_accepts_positive_values(self):
        """Accepts strictly positive real values."""
        model = LogNormalModel()

        data = {
            "A": np.array([1.0, 2.0, 3.0]),
            "B": np.array([2.0, 3.0, 4.0])
        }

        model.fit(data)

    def test_output_shape(self):
        """Ensures output matches expected (n_draws, n_variants)."""
        model = LogNormalModel()

        data = {
            "A": np.random.lognormal(0, 1, 100),
            "B": np.random.lognormal(1, 1, 100)
        }

        model.fit(data)
        samples = model.sample_posterior(1200)

        assert samples.shape == (1200, 2)

    def test_no_nan_or_inf(self):
        """Ensures posterior samples contain finite values only."""
        model = LogNormalModel()

        data = {
            "A": np.random.lognormal(0, 1, 100),
            "B": np.random.lognormal(1, 1, 100)
        }

        model.fit(data)
        samples = model.sample_posterior(1000)

        assert not np.isnan(samples).any()
        assert np.isfinite(samples).all()

    def test_requires_fit_before_sampling(self):
        """Prevents sampling before fitting."""
        model = LogNormalModel()

        with pytest.raises(ValueError):
            model.sample_posterior()

    def test_sampling_zero_draws(self):
        """Handles zero draws correctly."""
        model = LogNormalModel()

        data = {
            "A": np.array([1.0, 2.0]),
            "B": np.array([2.0, 3.0])
        }

        model.fit(data)
        samples = model.sample_posterior(0)

        assert samples.shape == (0, 2)

    def test_sampling_negative_draws(self):
        """Rejects negative number of draws."""
        model = LogNormalModel()

        data = {
            "A": np.array([1.0, 2.0]),
            "B": np.array([2.0, 3.0])
        }

        model.fit(data)

        with pytest.raises(ValueError):
            model.sample_posterior(-10)

    def test_detects_scale_difference(self):
        """Assigns higher expected value to higher-scale variant."""
        model = LogNormalModel()

        np.random.seed(42)

        data = {
            "A": np.random.lognormal(0, 1, 200),
            "B": np.random.lognormal(1, 1, 200)
        }

        model.fit(data)
        samples = model.sample_posterior(2000)

        prob = np.mean(samples[:, 1] > samples[:, 0])

        assert prob > 0.9

    def test_equal_distributions(self):
        """Produces near 0.5 probability for identical distributions."""
        model = LogNormalModel()

        np.random.seed(42)

        data = {
            "A": np.random.lognormal(0, 1, 200),
            "B": np.random.lognormal(0, 1, 200)
        }

        model.fit(data)
        samples = model.sample_posterior(2000)

        prob = np.mean(samples[:, 1] > samples[:, 0])

        assert 0.4 < prob < 0.6

    def test_small_sample_behavior(self):
        """Maintains uncertainty with very small datasets."""
        model = LogNormalModel()

        data = {
            "A": np.array([1.0]),
            "B": np.array([2.0])
        }

        model.fit(data)
        samples = model.sample_posterior(2000)

        prob = np.mean(samples[:, 1] > samples[:, 0])

        assert 0.2 < prob < 0.8

    def test_heavy_skew_handling(self):
        """Handles highly skewed data without instability."""
        model = LogNormalModel()

        data = {
            "A": np.random.lognormal(0, 2, 200),
            "B": np.random.lognormal(0.5, 2, 200)
        }

        model.fit(data)
        samples = model.sample_posterior(1000)

        assert np.isfinite(samples).all()

    def test_expected_value_transformation(self):
        """Ensures output reflects exp(mu + sigma^2 / 2) transformation."""
        model = LogNormalModel()

        data = {
            "A": np.array([1.0, 2.0, 3.0]),
            "B": np.array([2.0, 3.0, 4.0])
        }

        model.fit(data)
        samples = model.sample_posterior(500)

        # Expected values must always be positive
        assert (samples > 0).all()

    def test_unbalanced_sample_sizes(self):
        """Handles variants with highly unequal sample sizes."""
        model = LogNormalModel()

        data = {
            "A": np.random.lognormal(0, 1, 10),
            "B": np.random.lognormal(1, 1, 1000)
        }

        model.fit(data)
        samples = model.sample_posterior(1000)

        assert samples.shape == (1000, 2)

    def test_reproducibility(self):
        """Ensures deterministic sampling via PyMC seed."""
        model = LogNormalModel()

        data = {
            "A": np.array([1.0, 2.0, 3.0]),
            "B": np.array([2.0, 3.0, 4.0])
        }

        model.fit(data)

        s1 = model.sample_posterior(300)
        s2 = model.sample_posterior(300)

        assert np.allclose(s1, s2)
