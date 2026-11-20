import math

import pytest
from scipy.stats import fisher_exact

from foe.core.models import AlternativeHypothesis, ExperimentInput
from foe.frequentist.boschloo import BoschlooEngine
from foe.frequentist.operations import FrequentistEngine, apply_sidak

GREATER = AlternativeHypothesis.GREATER
LESS = AlternativeHypothesis.LESS
TWO_SIDED = AlternativeHypothesis.TWO_SIDED


@pytest.fixture
def engine():
    return BoschlooEngine()


def _run(engine, visitors, conversions, **kw):
    data = ExperimentInput(
        visitors=visitors,
        conversions=conversions,
        labels=[chr(65 + i) for i in range(len(visitors))],
        **kw,
    )
    return engine.run(data)


def test_direction_mapping_matches_alternative(engine):
    """GREATER means challenger converts better; scipy's column order must not flip it."""
    p_greater = engine.exact_p_value(5, 200, 20, 200, GREATER)
    p_less = engine.exact_p_value(5, 200, 20, 200, LESS)
    assert p_greater < 0.01
    assert p_less > 0.99


def test_zero_conversion_control_reads_positive(engine):
    """ctrl 0/300, chal 15/300: significant, positive, uplift undefined."""
    r = _run(engine, [300, 300], [0, 15])[0]
    assert r.is_significant
    assert r.absolute_diff > 0
    assert r.uplift is None
    assert r.conclusion.startswith("Significant Positive Impact")
    assert "undefined relative impact" in r.conclusion
    assert r.ci_diff[0] > 0


@pytest.mark.parametrize("conversions", [[0, 0], [200, 200]])
def test_identical_degenerate_arms_are_not_significant(engine, conversions):
    """Both arms at 0% or 100%: p = 1, and the CI is not a zero-width point."""
    r = _run(engine, [200, 200], conversions)[0]
    assert r.p_value == pytest.approx(1.0)
    assert not r.is_significant
    assert r.ci_diff[0] < 0 < r.ci_diff[1]


def test_opposite_degenerate_arms_are_significant(engine):
    """0% vs 100% is the strongest possible evidence, not 'no difference'."""
    r = _run(engine, [100, 100], [0, 100])[0]
    assert r.is_significant
    assert r.p_value < 1e-10


def test_at_least_as_powerful_as_fisher(engine):
    """Boschloo's p-value is never larger than Fisher's (one-sided)."""
    for ctrl, chal in [(3, 10), (1, 6), (8, 15), (0, 4)]:
        fisher_p = fisher_exact(
            [[chal, 120 - chal], [ctrl, 120 - ctrl]], alternative="greater")[1]
        assert engine.exact_p_value(ctrl, 120, chal, 120, GREATER) <= fisher_p + 1e-12


def test_flat_result_is_inconclusive(engine):
    r = _run(engine, [150, 150], [10, 11])[0]
    assert not r.is_significant
    assert r.conclusion.startswith("Inconclusive")
    assert r.uplift == pytest.approx(0.1)


def test_negative_result(engine):
    r = _run(engine, [200, 200], [25, 5])[0]
    assert r.is_significant
    assert r.absolute_diff < 0
    assert r.conclusion.startswith("Significant Negative Impact")


def test_sidak_applied_for_multiple_variants(engine):
    results = _run(engine, [100, 100, 100], [5, 6, 7])
    assert len(results) == 2
    expected = apply_sidak(0.05, 3)
    assert all(r.alpha == pytest.approx(expected) for r in results)
    assert [r.variant_label for r in results] == ["B", "C"]


@pytest.mark.parametrize("alternative,side", [(GREATER, 1), (LESS, 0)])
def test_one_sided_interval_is_unbounded_on_one_side(engine, alternative, side):
    r = _run(engine, [200, 200], [10, 20], alternative=alternative)[0]
    assert math.isinf(r.ci_diff[side])
    assert not math.isinf(r.ci_diff[1 - side])


def test_interval_within_unit_range(engine):
    r = _run(engine, [20, 20], [0, 20])[0]
    assert -1.0 <= r.ci_diff[0] <= r.ci_diff[1] <= 1.0


def test_agrees_with_ztest_at_moderate_counts(engine):
    """With healthy counts the exact and asymptotic tests should be close."""
    data = ExperimentInput(visitors=[400, 400], conversions=[40, 60])
    exact = engine.run(data)[0].p_value
    z = FrequentistEngine().run_synthesis(data)[0].p_value
    assert exact == pytest.approx(z, abs=0.02)


def test_rejects_reduction_factor(engine):
    with pytest.raises(ValueError, match="reduction_factor"):
        _run(engine, [100, 100], [5, 10], reduction_factor=1.5)


def test_rejects_large_arms_by_default(engine):
    with pytest.raises(ValueError, match="max_visitors_per_arm"):
        _run(engine, [5000, 5000], [50, 60])


def test_size_limit_is_configurable():
    r = _run(BoschlooEngine(max_visitors_per_arm=50), [50, 50], [1, 8])[0]
    assert r.test_method == "boschloo_exact"
    with pytest.raises(ValueError):
        _run(BoschlooEngine(max_visitors_per_arm=50), [51, 50], [1, 8])
