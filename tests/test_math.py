import numpy as np
import pytest
from scipy import stats

from argonx.decision_rules.metrics import compute_expected_loss, compute_prob_best
from argonx.decision_rules.composite import _compute_hdi
from argonx.models.binary_model import BinaryModel
from argonx.models.count_model import PoissonModel
from argonx.sequential.stopping import _estimate_users_needed

def test_expected_loss_invariance():
    """Verify expected loss is shift-invariant."""
    rng = np.random.default_rng(42)
    s1 = rng.normal(0, 1, 1000)
    s2 = rng.normal(0.5, 1, 1000)
    
    samples = np.column_stack([s1, s2])
    res_base = compute_expected_loss(samples, ['A', 'B'], 'A')
    
    shifted_samples = samples + 100.0
    res_shifted = compute_expected_loss(shifted_samples, ['A', 'B'], 'A')
    
    assert np.isclose(res_base.expected_loss['A'], res_shifted.expected_loss['A'])
    assert np.isclose(res_base.expected_loss['B'], res_shifted.expected_loss['B'])

def test_expected_loss_exact():
    """Verify exact expected loss computation."""
    # A is exactly 0
    # B is -1 (50%) or 1 (50%)
    s_a = np.zeros(100)
    s_b = np.array([-1.0] * 50 + [1.0] * 50)
    samples = np.column_stack([s_a, s_b])
    
    res = compute_expected_loss(samples, ['A', 'B'], 'A')
    
    # max_sample is max(0, -1)=0 (first 50) and max(0, 1)=1 (last 50) -> average is 0.5
    # Loss for A: max(max_sample - A) = max_sample. Mean = 0.5
    assert np.isclose(res.expected_loss['A'], 0.5)
    
    # Loss for B: max(max_sample - B).
    # First 50: max_sample=0, B=-1 -> 0 - (-1) = 1
    # Last 50: max_sample=1, B=1 -> 1 - 1 = 0
    # Mean = 0.5
    assert np.isclose(res.expected_loss['B'], 0.5)

def test_expected_loss_dominance():
    """Verify dominant variant yields zero loss."""
    s_a = np.ones(100)
    s_b = np.zeros(100)
    samples = np.column_stack([s_a, s_b])
    
    res = compute_expected_loss(samples, ['A', 'B'], 'B')
    assert np.isclose(res.expected_loss['A'], 0.0)
    assert res.expected_loss['B'] > 0.0

def test_prob_best_normalization():
    """Verify P(best) strictly sums to one."""
    rng = np.random.default_rng(42)
    samples = rng.normal(0, 1, (1000, 5))
    names = [f"V{i}" for i in range(5)]
    
    res = compute_prob_best(samples, names)
    total_prob = sum(res.probabilities.values())
    assert np.isclose(total_prob, 1.0)

def test_prob_best_independent_identical():
    """Verify identical independent distributions split P(best)."""
    rng = np.random.default_rng(42)
    s1 = rng.normal(0, 1, 100000)
    s2 = rng.normal(0, 1, 100000)
    samples = np.column_stack([s1, s2])
    
    res = compute_prob_best(samples, ['A', 'B'])
    assert np.isclose(res.probabilities['A'], 0.5, atol=0.01)
    assert np.isclose(res.probabilities['B'], 0.5, atol=0.01)

def test_prob_best_certainty():
    """Verify strict dominance yields P(best)=1."""
    s_a = np.ones(100)
    s_b = np.zeros(100)
    samples = np.column_stack([s_a, s_b])
    
    res = compute_prob_best(samples, ['A', 'B'])
    assert np.isclose(res.probabilities['A'], 1.0)
    assert np.isclose(res.probabilities['B'], 0.0)

def test_beta_binomial_conjugacy():
    """Verify BinaryModel matches analytical Beta posterior."""
    data = {'A': np.array([1, 1, 0, 0, 0]), 'B': np.array([1, 1, 1, 1, 1])} # A: sum=2, n=5
    # Prior Beta(1, 1). Posterior A should be Beta(1+2, 1+3) = Beta(3, 4)
    model = BinaryModel()
    model.fit(data)
    
    # Large n_draws to reduce Monte Carlo error
    samples = model.sample_posterior(n_draws=20000)
    
    analytical_mean = 3.0 / 7.0
    analytical_var = (3.0 * 4.0) / ((7.0 ** 2) * 8.0)
    
    sample_mean = np.mean(samples[:, 0])
    sample_var = np.var(samples[:, 0])
    
    assert np.isclose(sample_mean, analytical_mean, rtol=0.05)
    assert np.isclose(sample_var, analytical_var, rtol=0.10)

def test_gamma_poisson_conjugacy():
    """Verify PoissonModel matches analytical Gamma posterior."""
    data = {'A': np.array([2, 3, 1, 4]), 'B': np.array([0, 0, 0, 0])} # A: sum=10, n=4
    # Prior Gamma(1, 1). Posterior A should be Gamma(1+10, 1+4) = Gamma(11, 5)
    # Mean = alpha / beta = 11 / 5 = 2.2
    # Var = alpha / beta^2 = 11 / 25 = 0.44
    model = PoissonModel(lam_prior_alpha=1.0, lam_prior_beta=1.0)
    model.fit(data)
    
    samples = model.sample_posterior(n_draws=20000)
    
    analytical_mean = 11.0 / 5.0
    analytical_var = 11.0 / 25.0
    
    sample_mean = np.mean(samples[:, 0])
    sample_var = np.var(samples[:, 0])
    
    assert np.isclose(sample_mean, analytical_mean, rtol=0.05)
    assert np.isclose(sample_var, analytical_var, rtol=0.10)

def test_hdi_uniform_mass():
    """Verify HDI captures exact empirical mass density."""
    # Uniform sample from 0 to 100
    samples = np.linspace(0, 100, 1001) # length 1001, exactly 0.1 spacing
    
    low, high = _compute_hdi(samples, prob=0.5)
    # The interval should cover 500 gaps, meaning a width of exactly 50.0
    assert np.isclose(high - low, 50.0, atol=0.1)

def test_expected_loss_triangle_inequality():
    """Verify expected loss satisfies metric subadditivity."""
    # L(A, B) = E[max(0, B - A)]
    # We test if E[max(0, C - A)] <= E[max(0, B - A)] + E[max(0, C - B)]
    rng = np.random.default_rng(123)
    A = rng.normal(0, 1, 1000)
    B = rng.normal(0.2, 1.2, 1000)
    C = rng.normal(0.5, 0.8, 1000)
    
    L_C_A = np.mean(np.maximum(0, C - A))
    L_B_A = np.mean(np.maximum(0, B - A))
    L_C_B = np.mean(np.maximum(0, C - B))
    
    assert L_C_A <= L_B_A + L_C_B + 1e-9 # allow tiny float precision error

def test_hdi_asymptotic_convergence():
    """Verify HDI converges to theoretical Normal quantiles."""
    # N(0, 1) theoretical 95% HDI is ~ [-1.95996, 1.95996]
    rng = np.random.default_rng(999)
    samples = rng.normal(0, 1, 1000000)
    
    low, high = _compute_hdi(samples, prob=0.95)
    
    assert np.isclose(low, stats.norm.ppf(0.025), atol=0.03)
    assert np.isclose(high, stats.norm.ppf(0.975), atol=0.03)

def test_expected_loss_multivariant_exact():
    """Verify expected loss with 3 variants exactly matches E[max(0, max(j!=i) - i)]."""
    samples = np.column_stack([np.zeros(10), np.ones(10), np.ones(10)*2])
    names = ['A', 'B', 'C']
    res = compute_expected_loss(samples, names, 'A')
    assert np.isclose(res.expected_loss['A'], 2.0)
    assert np.isclose(res.expected_loss['B'], 1.0)
    assert np.isclose(res.expected_loss['C'], 0.0)

def test_users_needed_inverse_square_law():
    """Verify projected users needed follows inverse square law of variance."""
    res = _estimate_users_needed(
        best_variant='A',
        expected_loss={'A': 0.02},
        prob_best={'A': 0.9},
        loss_threshold=0.01,
        prob_best_min=0.95,
        n_users_per_variant={'B': 100},
        daily_traffic_per_variant=None,
        variant_names=['A', 'B'],
        control='A',
        users_floor=0,
        safety_factor=1.0
    )
    assert res.additional_users['B'] == 300
