import numpy as np
import pytest
from argonx.decision_rules.engine import run_engine, _evaluate_primary_strength, _evaluate_risk, _evaluate_practical_significance, _evaluate_confidence, _determine_state
CONFIG = {'prob_best_strong': 0.95, 'prob_best_moderate': 0.8, 'expected_loss_max': 0.05, 'rope_practical_min': 0.8, 'cvar_ratio_max': 5.0, 'rope_bounds': (-0.01, 0.01), 'guardrail_thresholds': {}, 'lower_is_better': {}}

class TestEngine:

    def test_strong_win(self):
        """Strong signal leads to strong win."""
        control = np.array([0.4, 0.5, 0.45])
        variant = np.array([0.8, 0.9, 0.85])
        samples = np.column_stack([control, variant])
        result = run_engine(samples, ['control', 'variant_b'], 'control', {}, CONFIG)
        assert result.state == 'strong win'
        assert result.recommendation == 'ship variant'

    def test_guardrail_conflict_overrides(self):
        """Guardrail conflict overrides primary signal."""
        control = np.array([0.4, 0.5, 0.45])
        variant = np.array([0.8, 0.9, 0.85])
        samples = np.column_stack([control, variant])
        guardrail_samples = {'error': np.column_stack([np.array([0.1, 0.1, 0.1]), np.array([0.9, 0.9, 0.9])])}
        config = CONFIG.copy()
        config['guardrail_thresholds'] = {'error': 0.6}
        config['lower_is_better'] = {'error': True}
        result = run_engine(samples, ['control', 'variant_b'], 'control', guardrail_samples, config)
        assert result.state == 'guardrail conflicts'
        assert result.recommendation == 'review required'

    def test_inconclusive(self):
        """Similar variants produce inconclusive result."""
        control = np.array([0.5, 0.5, 0.5])
        variant = np.array([0.51, 0.49, 0.5])
        samples = np.column_stack([control, variant])
        result = run_engine(samples, ['control', 'variant_b'], 'control', {}, CONFIG)
        assert result.state == 'inconclusive'

    def test_single_draw(self):
        """Single draw should not crash engine."""
        samples = np.array([[0.5, 0.6]])
        result = run_engine(samples, ['control', 'variant_b'], 'control', {}, CONFIG)
        assert result.best_variant in ['control', 'variant_b']

    def test_missing_config(self):
        """Missing config raises exception."""
        samples = np.column_stack([np.array([0.4, 0.5, 0.45]), np.array([0.8, 0.9, 0.85])])
        with pytest.raises(Exception):
            run_engine(samples, ['control', 'variant_b'], 'control', {}, {})

class MockMetrics:

    def __init__(self):
        self.prob_best = type('obj', (), {})()
        self.loss = type('obj', (), {})()
        self.rope = type('obj', (), {})()
        self.cvar = type('obj', (), {})()
        self.lift = type('obj', (), {})()

class MockGuardrails:

    def __init__(self, all_passed: bool, conflicts: list):
        self.all_passed = all_passed
        self.conflicts = conflicts
        self.warnings = []

class TestEngineInternals:

    def test_primary_strength_strong(self):
        """High probability and low loss yields strong."""
        metrics = MockMetrics()
        metrics.prob_best.best_variant = 'B'
        metrics.prob_best.probabilities = {'B': 0.97}
        metrics.loss.expected_loss = {'B': 0.001}
        metrics.rope.prob_practical = {'B': 0.9}
        result = _evaluate_primary_strength(metrics, CONFIG)
        assert result == 'strong'

    def test_primary_strength_weak(self):
        """Low probability yields weak signal."""
        metrics = MockMetrics()
        metrics.prob_best.best_variant = 'B'
        metrics.prob_best.probabilities = {'B': 0.6}
        metrics.loss.expected_loss = {'B': 0.2}
        metrics.rope.prob_practical = {'B': 0.2}
        result = _evaluate_primary_strength(metrics, CONFIG)
        assert result == 'weak'

    def test_risk_high(self):
        """Large expected loss triggers high risk."""
        metrics = MockMetrics()
        metrics.prob_best.best_variant = 'B'
        metrics.loss.expected_loss = {'B': 0.5}
        metrics.cvar.cvar = {'B': 10.0}
        result = _evaluate_risk(metrics, CONFIG)
        assert result == 'high'

    def test_risk_low(self):
        """Low loss and ratio yields low risk."""
        metrics = MockMetrics()
        metrics.prob_best.best_variant = 'B'
        metrics.loss.expected_loss = {'B': 0.01}
        metrics.cvar.cvar = {'B': 0.02}
        result = _evaluate_risk(metrics, CONFIG)
        assert result == 'low'

    def test_practical_significance_yes(self):
        """High ROPE probability yields yes."""
        metrics = MockMetrics()
        metrics.prob_best.best_variant = 'B'
        metrics.rope.prob_practical = {'B': 0.9}
        result = _evaluate_practical_significance(metrics, CONFIG)
        assert result == 'yes'

    def test_practical_significance_no(self):
        """Low ROPE probability yields no."""
        metrics = MockMetrics()
        metrics.prob_best.best_variant = 'B'
        metrics.rope.prob_practical = {'B': 0.2}
        result = _evaluate_practical_significance(metrics, CONFIG)
        assert result == 'no'

    def test_confidence_high(self):
        """High probability and tight interval yields high confidence."""
        metrics = MockMetrics()
        metrics.prob_best.best_variant = 'B'
        metrics.prob_best.probabilities = {'B': 0.97}
        metrics.lift.hdi_low = {'B': 0.1}
        metrics.lift.hdi_high = {'B': 0.15}
        result = _evaluate_confidence(metrics, CONFIG)
        assert result == 'high'

    def test_determine_state_conflict(self):
        """Guardrail conflicts override all."""
        guardrails = MockGuardrails(all_passed=False, conflicts=['x'])
        state = _determine_state('strong', 'low', 'yes', guardrails)
        assert state == 'guardrail conflicts'

    def test_determine_state_strong_win(self):
        """Strong + low risk yields strong win."""
        guardrails = MockGuardrails(all_passed=True, conflicts=[])
        state = _determine_state('strong', 'low', 'yes', guardrails)
        assert state == 'strong win'

    def test_determine_state_inconclusive(self):
        """Weak signals lead to inconclusive."""
        guardrails = MockGuardrails(all_passed=True, conflicts=[])
        state = _determine_state('weak', 'low', 'no', guardrails)
        assert state == 'inconclusive'