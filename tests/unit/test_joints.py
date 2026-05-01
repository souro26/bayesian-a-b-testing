import numpy as np
import pytest
from argonx.decision_rules.joint import compute_joint_probability

def make_basic_inputs():
    """Create simple 2-variant valid inputs."""
    control = np.array([1.0, 1.0, 1.0, 1.0])
    variant = np.array([2.0, 2.0, 2.0, 2.0])
    primary = np.column_stack([control, variant])
    guardrail = {'error': np.column_stack([np.array([1.0, 1.0, 1.0, 1.0]), np.array([0.5, 0.5, 0.5, 0.5])])}
    return (primary, guardrail, ['control', 'B'], 'control')

class TestJointCore:

    def test_joint_prob_all_true(self):
        """All conditions satisfied gives joint_prob=1."""
        primary, guardrail, names, control = make_basic_inputs()
        res = compute_joint_probability(primary, guardrail, names, control)
        assert res.joint_prob['B'] == 1.0

    def test_joint_prob_all_false(self):
        """All conditions false gives joint_prob=0."""
        primary, guardrail, names, control = make_basic_inputs()
        primary[:, 1] = 0.0
        res = compute_joint_probability(primary, guardrail, names, control)
        assert res.joint_prob['B'] == 0.0

    def test_best_variant_selection(self):
        """Best variant is selected by joint probability."""
        primary, guardrail, names, control = make_basic_inputs()
        res = compute_joint_probability(primary, guardrail, names, control)
        assert res.best_variant == 'B'

class TestJointThresholds:

    def test_primary_threshold_blocks(self):
        """Primary threshold blocks weak improvements."""
        primary, guardrail, names, control = make_basic_inputs()
        primary[:, 1] = 1.01
        res = compute_joint_probability(primary, guardrail, names, control, primary_threshold=0.05)
        assert res.joint_prob['B'] == 0.0

    def test_guardrail_threshold_allows(self):
        """Guardrail threshold allows small degradation."""
        primary, guardrail, names, control = make_basic_inputs()
        guardrail['error'][:, 1] = 1.05
        res = compute_joint_probability(primary, guardrail, names, control, guardrail_thresholds={'error': 0.1})
        assert res.joint_prob['B'] == 1.0

class TestJointCorrelation:

    def test_independence_gap_zero(self):
        """Independent metrics give near-zero correlation gap."""
        primary, guardrail, names, control = make_basic_inputs()
        primary[:, 1] = [1, 2, 1, 2]
        guardrail['error'][:, 1] = [1, 0, 1, 0]
        res = compute_joint_probability(primary, guardrail, names, control)
        gap = res.correlation_gap['B']
        assert abs(gap) < 0.05

    def test_negative_correlation_gap(self):
        """Opposing conditions produce negative correlation gap."""
        primary, guardrail, names, control = make_basic_inputs()
        primary[:, 1] = [2, 2, 0, 0]
        guardrail['error'][:, 1] = [2, 2, 0, 0]
        res = compute_joint_probability(primary, guardrail, names, control, lower_is_better={'error': True})
        assert res.correlation_gap['B'] < 0

class TestJointValidation:

    def test_invalid_shape(self):
        """Mismatched shapes raise error."""
        primary, guardrail, names, control = make_basic_inputs()
        guardrail['error'] = guardrail['error'][:, :1]
        with pytest.raises(ValueError):
            compute_joint_probability(primary, guardrail, names, control)

    def test_invalid_control(self):
        """Invalid control raises error."""
        primary, guardrail, names, _ = make_basic_inputs()
        with pytest.raises(ValueError):
            compute_joint_probability(primary, guardrail, names, 'X')

    def test_empty_metrics_to_join(self):
        """Empty metrics_to_join raises error."""
        primary, guardrail, names, control = make_basic_inputs()
        with pytest.raises(ValueError):
            compute_joint_probability(primary, guardrail, names, control, metrics_to_join=[])

    def test_unknown_metric(self):
        """Unknown metric in metrics_to_join raises error."""
        primary, guardrail, names, control = make_basic_inputs()
        with pytest.raises(ValueError):
            compute_joint_probability(primary, guardrail, names, control, metrics_to_join=['fake'])

class TestJointWarnings:

    def test_low_joint_warning(self):
        """Low joint probability triggers warning."""
        primary, guardrail, names, control = make_basic_inputs()
        primary[:, 1] = 0.0
        with pytest.warns(UserWarning):
            compute_joint_probability(primary, guardrail, names, control)

    def test_lower_is_better_warning(self):
        """Missing lower_is_better emits warning."""
        primary, guardrail, names, control = make_basic_inputs()
        with pytest.warns(UserWarning):
            compute_joint_probability(primary, guardrail, names, control, lower_is_better=None)

class TestJointMetricSubset:

    def test_metrics_to_join_subset(self):
        """Only selected metrics affect joint computation."""
        control = np.array([1, 1, 1, 1])
        variant = np.array([2, 2, 2, 2])
        primary = np.column_stack([control, variant])
        guardrail = {'good_metric': np.column_stack([np.array([1, 1, 1, 1]), np.array([0.5, 0.5, 0.5, 0.5])]), 'bad_metric': np.column_stack([np.array([1, 1, 1, 1]), np.array([2, 2, 2, 2])])}
        names = ['control', 'B']
        res = compute_joint_probability(primary, guardrail, names, 'control', metrics_to_join=['good_metric'])
        assert res.joint_prob['B'] == 1.0

class TestJointConditionStructure:

    def test_condition_probs_structure(self):
        """Condition probabilities include all metrics and are bounded."""
        primary, guardrail, names, control = make_basic_inputs()
        res = compute_joint_probability(primary, guardrail, names, control)
        cond = res.condition_probs['B']
        assert 'primary' in cond
        assert 'error' in cond
        assert all((0.0 <= v <= 1.0 for v in cond.values()))

class TestJointMultiVariant:

    def test_three_variants_joint(self):
        """Joint works correctly with three variants."""
        control = np.array([1, 1, 1, 1])
        A = np.array([2, 2, 2, 2])
        B = np.array([0, 0, 0, 0])
        primary = np.column_stack([control, A, B])
        guardrail = {'error': np.column_stack([np.array([1, 1, 1, 1]), np.array([0.5, 0.5, 0.5, 0.5]), np.array([2, 2, 2, 2])])}
        names = ['control', 'A', 'B']
        res = compute_joint_probability(primary, guardrail, names, 'control')
        assert res.joint_prob['A'] == 1.0
        assert res.joint_prob['B'] == 0.0

class TestJointDominance:

    def test_dominant_variant_flag(self):
        """Large gap between variants triggers dominance note."""
        control = np.array([1, 1, 1, 1])
        A = np.array([2, 2, 2, 2])
        B = np.array([0, 0, 0, 0])
        primary = np.column_stack([control, A, B])
        guardrail = {'error': np.column_stack([np.array([1, 1, 1, 1]), np.array([0.5, 0.5, 0.5, 0.5]), np.array([2, 2, 2, 2])])}
        names = ['control', 'A', 'B']
        res = compute_joint_probability(primary, guardrail, names, 'control')
        assert any(('dominates' in w for w in res.warnings))