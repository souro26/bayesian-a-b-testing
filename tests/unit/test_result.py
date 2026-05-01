import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace as NS
from argonx.results.result import Results
from argonx.decision_rules.engine import DecisionResult

def make_decision(joint=True, composite=True, guardrail_fail=False, notes=None):
    """Create structured DecisionResult for testing."""
    metrics = NS()
    metrics.prob_best = NS(probabilities={'control': 0.4, 'B': 0.6, 'C': 0.0})
    metrics.loss = NS(expected_loss={'control': 0.0, 'B': 0.01, 'C': 0.02}, control='control')
    metrics.cvar = NS(cvar={'control': 0.0, 'B': 0.02, 'C': 0.03}, alpha=0.95)
    metrics.lift = NS(mean={'control': 0.0, 'B': 0.05, 'C': -0.02}, hdi_low={'control': 0.0, 'B': 0.02, 'C': -0.04}, hdi_high={'control': 0.0, 'B': 0.08, 'C': 0.0}, hdi_prob=0.94)
    metrics.rope = NS(inside_rope={'control': 1.0, 'B': 0.2, 'C': 0.7}, outside_rope={'control': 0.0, 'B': 0.8, 'C': 0.3}, prob_practical={'control': 0.0, 'B': 0.8, 'C': 0.2})
    guardrails = NS(all_passed=not guardrail_fail, variant_passed={'control': True, 'B': not guardrail_fail, 'C': False}, guardrails=[], conflicts=[])
    joint_obj = None
    if joint:
        joint_obj = NS(joint_prob={'B': 0.7, 'C': 0.2}, independence_benchmark={'B': 0.75, 'C': 0.25}, correlation_gap={'B': -0.05, 'C': -0.03}, condition_probs={'B': {'primary': 0.9, 'guardrail': 0.7}, 'C': {'primary': 0.4, 'guardrail': 0.3}}, metrics_joined=['primary', 'guardrail'], best_variant='B')
    composite_obj = None
    if composite:
        composite_obj = NS(score={'B': 0.3, 'C': -0.1}, prob_exceeds_threshold={'B': 0.9, 'C': 0.1}, gap_hdi={'B': (0.1, 0.5), 'C': (-0.3, 0.0)}, metric_contributions={'B': {'primary': 0.3}, 'C': {'primary': -0.1}}, best_variant='B', threshold=0.0)
    decision = DecisionResult(state='strong_win', recommendation='ship', best_variant='B', confidence='high', primary_strength='strong', risk_level='low', practical_significance='yes', guardrail_status='fail' if guardrail_fail else 'pass', reasons=['Strong lift'], notes=notes or [], metrics=metrics, guardrails=guardrails, joint=joint_obj, composite=composite_obj)
    return decision

def make_segment_decision(best='B', state='strong win', recommendation='ship variant', guardrail_pass=True):
    """Create a minimal DecisionResult for a single segment."""
    metrics = NS()
    metrics.prob_best = NS(probabilities={'control': 0.3, 'B': 0.7})
    metrics.loss = NS(expected_loss={'control': 0.0, 'B': 0.01}, control='control')
    metrics.cvar = NS(cvar={'control': 0.0, 'B': 0.02}, alpha=0.95)
    metrics.lift = NS(mean={'control': 0.0, 'B': 0.05}, hdi_low={'control': 0.0, 'B': 0.02}, hdi_high={'control': 0.0, 'B': 0.08}, hdi_prob=0.94)
    metrics.rope = NS(inside_rope={'control': 1.0, 'B': 0.2}, outside_rope={'control': 0.0, 'B': 0.8}, prob_practical={'control': 0.0, 'B': 0.8})
    failed_guardrail = NS(metric='page_load', variant='B', passed=guardrail_pass, prob_degraded=0.0 if guardrail_pass else 0.92, threshold=0.1, severity='low' if guardrail_pass else 'high', expected_degradation=0.0)
    guardrails = NS(all_passed=guardrail_pass, variant_passed={'control': True, 'B': guardrail_pass}, guardrails=[failed_guardrail], conflicts=[])
    return DecisionResult(state=state, recommendation=recommendation, best_variant=best, confidence='high', primary_strength='strong', risk_level='low', practical_significance='yes', guardrail_status='pass' if guardrail_pass else 'fail', reasons=['Signal present'], notes=[], metrics=metrics, guardrails=guardrails, joint=None, composite=None)

def make_hierarchical_results(violations=False):
    """Create a Results object simulating a hierarchical experiment."""
    aggregate = make_decision(joint=False, composite=False)
    segment_results = {'mobile': make_segment_decision(state='strong win', recommendation='ship variant'), 'desktop': make_segment_decision(state='weak win', recommendation='consider shipping'), 'tv': make_segment_decision(state='inconclusive', recommendation='continue experiment', guardrail_pass=not violations)}
    seg_violations = {'tv': ['page_load']} if violations else None
    return Results(aggregate, config={'rope_bounds': (-0.01, 0.01)}, segment_results=segment_results, segment_guardrail_violations=seg_violations)

def test_summary_runs():
    """Summary executes without errors."""
    Results(make_decision()).summary()

def test_repr_contains_key_info():
    """__repr__ contains key fields."""
    r = Results(make_decision())
    out = repr(r)
    assert 'strong_win' in out
    assert 'B' in out

def test_attribute_passthrough():
    """__getattr__ proxies DecisionResult."""
    r = Results(make_decision())
    assert r.state == 'strong_win'

def test_invalid_attribute_raises():
    """Invalid attribute raises AttributeError."""
    r = Results(make_decision())
    with pytest.raises(AttributeError):
        _ = r.fake_attr

def test_to_dict_structure():
    """to_dict returns all main sections."""
    d = Results(make_decision()).to_dict()
    assert set(d.keys()) >= {'decision', 'metrics', 'guardrails', 'joint', 'composite'}

def test_to_dict_consistency():
    """to_dict matches DecisionResult values."""
    r = Results(make_decision())
    d = r.to_dict()
    assert d['decision']['best_variant'] == r.best_variant

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
    assert Results(make_decision(joint=False)).to_dict()['joint'] is None

def test_to_dict_optional_composite():
    """Composite is None when absent."""
    assert Results(make_decision(composite=False)).to_dict()['composite'] is None

def test_dataframe_basic():
    """DataFrame has expected columns."""
    df = Results(make_decision()).to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert 'prob_best' in df.columns

def test_dataframe_excludes_control():
    """Control is excluded from DataFrame."""
    df = Results(make_decision()).to_dataframe()
    assert 'control' not in df.index

def test_dataframe_multi_variant():
    """Multiple variants appear correctly."""
    df = Results(make_decision()).to_dataframe()
    assert 'B' in df.index and 'C' in df.index

def test_dataframe_optional_joint():
    """Joint columns NaN when missing."""
    df = Results(make_decision(joint=False)).to_dataframe()
    assert df['joint_prob'].isna().all()

def test_dataframe_optional_composite():
    """Composite columns NaN when missing."""
    df = Results(make_decision(composite=False)).to_dataframe()
    assert df['composite_score'].isna().all()

def test_guardrail_failure_propagation():
    """Guardrail failure reflected correctly."""
    r = Results(make_decision(guardrail_fail=True))
    assert not r.guardrails.all_passed

def test_negative_lift_present():
    """Negative lift variant handled correctly."""
    df = Results(make_decision()).to_dataframe()
    assert df.loc['C', 'lift_mean'] < 0

def test_notes_propagation():
    """Notes propagate correctly."""
    r = Results(make_decision(notes=['issue']))
    assert r.to_dict()['decision']['notes'] == ['issue']

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
    Results(make_decision(notes=['n'] * 100)).summary()

class TestFlatResults:

    def test_segment_results_none_for_flat(self):
        """segment_results is None for flat result."""
        r = Results(make_decision())
        assert r.segment_results is None

    def test_segment_guardrail_violations_none_for_flat(self):
        """segment_guardrail_violations is None for flat result."""
        r = Results(make_decision())
        assert r.segment_guardrail_violations is None

    def test_segment_summary_raises_on_flat(self):
        """segment_summary raises RuntimeError on flat result."""
        r = Results(make_decision())
        with pytest.raises(RuntimeError):
            r.segment_summary()

    def test_to_dict_segment_results_none_for_flat(self):
        """to_dict has segment_results=None for flat result."""
        d = Results(make_decision()).to_dict()
        assert d['segment_results'] is None

    def test_repr_no_segments_for_flat(self):
        """__repr__ does not mention segments for flat result."""
        r = Results(make_decision())
        assert 'segments' not in repr(r)

    def test_to_dataframe_flat_has_simple_index(self):
        """to_dataframe returns simple variant index for flat result."""
        df = Results(make_decision()).to_dataframe()
        assert df.index.name == 'variant'

class TestHierarchicalResults:

    def test_segment_results_populated(self):
        """segment_results is a dict when hierarchical."""
        r = make_hierarchical_results()
        assert isinstance(r.segment_results, dict)

    def test_segment_results_keys(self):
        """segment_results has correct segment keys."""
        r = make_hierarchical_results()
        assert set(r.segment_results.keys()) == {'mobile', 'desktop', 'tv'}

    def test_segment_summary_runs(self):
        """segment_summary runs without error."""
        r = make_hierarchical_results()
        r.segment_summary()

    def test_segment_summary_raises_on_flat(self):
        """segment_summary raises RuntimeError on flat result."""
        r = Results(make_decision())
        with pytest.raises(RuntimeError):
            r.segment_summary()

    def test_repr_shows_segments(self):
        """__repr__ includes segment names for hierarchical result."""
        r = make_hierarchical_results()
        assert 'segments' in repr(r)

    def test_aggregate_summary_shows_segment_violation_warning(self):
        """summary prints segment guardrail violation when present."""
        import io, sys
        r = make_hierarchical_results(violations=True)
        buf = io.StringIO()
        sys.stdout = buf
        r.summary()
        sys.stdout = sys.__stdout__
        output = buf.getvalue()
        assert 'tv' in output
        assert 'page_load' in output

    def test_aggregate_summary_shows_segment_summary_hint(self):
        """summary prints hint to call segment_summary."""
        import io, sys
        r = make_hierarchical_results()
        buf = io.StringIO()
        sys.stdout = buf
        r.summary()
        sys.stdout = sys.__stdout__
        assert 'segment_summary' in buf.getvalue()

    def test_segment_summary_prints_all_segments(self):
        """segment_summary output contains all segment names."""
        import io, sys
        r = make_hierarchical_results()
        buf = io.StringIO()
        sys.stdout = buf
        r.segment_summary()
        sys.stdout = sys.__stdout__
        output = buf.getvalue()
        assert 'mobile' in output
        assert 'desktop' in output
        assert 'tv' in output

    def test_segment_summary_detects_conflict(self):
        """segment_summary flags shipping conflict when segments disagree."""
        import io, sys
        aggregate = make_decision(joint=False, composite=False)
        segment_results = {'mobile': make_segment_decision(state='strong win', recommendation='ship variant'), 'desktop': make_segment_decision(state='inconclusive', recommendation='do not ship')}
        r = Results(aggregate, config={}, segment_results=segment_results)
        buf = io.StringIO()
        sys.stdout = buf
        r.segment_summary()
        sys.stdout = sys.__stdout__
        assert 'CONFLICT' in buf.getvalue() or 'conflict' in buf.getvalue().lower()

    def test_segment_summary_consistent_winner_message(self):
        """segment_summary reports consistent winner when all agree."""
        import io, sys
        r = make_hierarchical_results()
        buf = io.StringIO()
        sys.stdout = buf
        r.segment_summary()
        sys.stdout = sys.__stdout__
        output = buf.getvalue()
        assert 'Consistent' in output or 'consistent' in output

    def test_to_dict_includes_segment_results(self):
        """to_dict has segment_results key when hierarchical."""
        d = make_hierarchical_results().to_dict()
        assert 'segment_results' in d
        assert d['segment_results'] is not None

    def test_to_dict_segment_results_keys(self):
        """to_dict segment_results has correct segment keys."""
        d = make_hierarchical_results().to_dict()
        assert set(d['segment_results'].keys()) == {'mobile', 'desktop', 'tv'}

    def test_to_dict_segment_results_none_for_flat(self):
        """to_dict has segment_results=None for flat result."""
        d = Results(make_decision()).to_dict()
        assert d['segment_results'] is None

    def test_to_dataframe_hierarchical_has_multiindex(self):
        """to_dataframe returns MultiIndex (segment, variant) for hierarchical."""
        df = make_hierarchical_results().to_dataframe()
        assert df.index.names == ['segment', 'variant']

    def test_to_dataframe_hierarchical_has_aggregate_rows(self):
        """to_dataframe includes aggregate rows."""
        df = make_hierarchical_results().to_dataframe()
        assert 'aggregate' in df.index.get_level_values('segment')

    def test_to_dataframe_hierarchical_has_segment_rows(self):
        """to_dataframe includes per-segment rows."""
        df = make_hierarchical_results().to_dataframe()
        segs = df.index.get_level_values('segment').unique().tolist()
        assert 'mobile' in segs and 'desktop' in segs and ('tv' in segs)

    def test_to_dataframe_hierarchical_has_prob_best_column(self):
        """to_dataframe hierarchical output has prob_best column."""
        df = make_hierarchical_results().to_dataframe()
        assert 'prob_best' in df.columns

    def test_to_dataframe_hierarchical_has_state_column(self):
        """to_dataframe hierarchical output has state column."""
        df = make_hierarchical_results().to_dataframe()
        assert 'state' in df.columns

    def test_segment_guardrail_violations_none_when_no_violations(self):
        """segment_guardrail_violations is None when all segments pass."""
        r = make_hierarchical_results(violations=False)
        assert r.segment_guardrail_violations is None

    def test_segment_guardrail_violations_populated_when_failing(self):
        """segment_guardrail_violations populated when segment fails guardrail."""
        r = make_hierarchical_results(violations=True)
        assert r.segment_guardrail_violations is not None
        assert 'tv' in r.segment_guardrail_violations

    def test_each_segment_result_has_notes(self):
        """Each segment DecisionResult has a notes attribute."""
        r = make_hierarchical_results()
        for seg, seg_result in r.segment_results.items():
            assert hasattr(seg_result, 'notes')

    def test_segment_results_not_affected_by_flat_path(self):
        """Flat Results has no segment_results side effects."""
        flat = Results(make_decision())
        hier = make_hierarchical_results()
        assert flat.segment_results is None
        assert hier.segment_results is not None