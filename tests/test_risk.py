"""
foe.frequentist.risk: FPR, FNDR and FPR at the observed p-value.

Expected values for alpha = 0.05, power = 0.80 come from hexkit's former
page-local implementation (hexkit PR #115); hexkit's Experiment Analysis page
must show identical numbers after switching to FOE.
"""

import math

import pytest
from pydantic import ValidationError
from scipy.stats import norm

from foe.core.models import AlternativeHypothesis, DecisionRiskResult
from foe.frequentist.operations import FrequentistEngine
from foe.frequentist.risk import (
    DEFAULT_PRIOR,
    _fpr_at_z,
    assess_risk,
    false_negative_discovery_rate,
    false_positive_risk,
    fpr_at_p_value,
    resolve_prior,
)

GREATER = AlternativeHypothesis.GREATER
LESS = AlternativeHypothesis.LESS
TWO_SIDED = AlternativeHypothesis.TWO_SIDED
ALPHA, POWER = 0.05, 0.80


# --------------------------------------------------------------------- #
#  Priors
# --------------------------------------------------------------------- #


def test_default_prior_is_skeptical():
    assert resolve_prior() == 0.10
    assert DEFAULT_PRIOR == 0.10
    assert assess_risk(0.01).prior == 0.10
    assert FrequentistEngine.resolve_prior_probability() == 0.10


@pytest.mark.parametrize("mode, expected", [
    ("skeptical", 0.10), ("neutral", 0.50), ("optimistic", 0.90),
])
def test_prior_presets(mode, expected):
    assert resolve_prior(mode) == expected


def test_custom_prior():
    assert resolve_prior("custom", 0.23) == 0.23
    assert resolve_prior("custom", 0.0) == 0.0
    assert resolve_prior("custom", 1.0) == 1.0


@pytest.mark.parametrize("args, match", [
    (("custom",), "must be provided"),
    (("custom", -0.01), "between 0 and 1"),
    (("custom", 23.0), "between 0 and 1"),  # percentage passed by mistake
    (("wishful",), "Unknown sensitivity_mode"),
])
def test_invalid_prior_raises(args, match):
    with pytest.raises(ValueError, match=match):
        resolve_prior(*args)


# --------------------------------------------------------------------- #
#  Standard FPR / FNDR
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("prior, fpr, fndr", [
    (0.5, 0.0588, 0.1739),
    (0.1, 0.3600, 0.0229),
])
def test_standard_fpr_and_fndr(prior, fpr, fndr):
    assert false_positive_risk(ALPHA, POWER, prior) == pytest.approx(fpr, abs=1e-4)
    assert false_negative_discovery_rate(ALPHA, POWER, prior) == pytest.approx(fndr, abs=1e-4)


def test_standard_fpr_and_fndr_at_prior_bounds():
    assert false_positive_risk(ALPHA, POWER, 0.0) == 1.0
    assert false_positive_risk(ALPHA, POWER, 1.0) == 0.0
    assert false_negative_discovery_rate(ALPHA, POWER, 0.0) == 0.0
    assert false_negative_discovery_rate(ALPHA, POWER, 1.0) == 1.0


# --------------------------------------------------------------------- #
#  FPR at the observed p-value
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("p, alternative, prior, expected", [
    (0.05, GREATER, 0.5, 0.269),
    (0.05, TWO_SIDED, 0.5, 0.295),
    (0.001, TWO_SIDED, 0.5, 0.010),
    (0.049, GREATER, 0.1, 0.764),
    (0.01, GREATER, 0.1, 0.379),
    (0.001, GREATER, 0.1, 0.084),
    (0.0001, GREATER, 0.1, 0.019),
    (0.049, TWO_SIDED, 0.1, 0.786),
    (0.01, TWO_SIDED, 0.1, 0.401),
    (0.001, TWO_SIDED, 0.1, 0.083),
    (0.0001, TWO_SIDED, 0.1, 0.017),
])
def test_fpr_at_p_expected_values(p, alternative, prior, expected):
    assert fpr_at_p_value(p, ALPHA, POWER, prior, alternative) == pytest.approx(
        expected, abs=1e-3
    )


@pytest.mark.parametrize("p", [0.049, 0.01, 0.001, 1e-8, 0.3, 0.9])
def test_greater_and_less_identical_for_same_one_sided_p(p):
    assert fpr_at_p_value(p, ALPHA, POWER, 0.1, GREATER) == fpr_at_p_value(
        p, ALPHA, POWER, 0.1, LESS
    )


def test_fpr_at_p_matches_z_based_reference():
    """Rebuilding z from p gives the same answer as the z-based formula."""
    z = 2.4
    assert fpr_at_p_value(norm.sf(z), ALPHA, POWER, 0.2, GREATER) == pytest.approx(
        _fpr_at_z(z, ALPHA, POWER, 0.2, GREATER), rel=1e-9
    )
    assert fpr_at_p_value(2 * norm.sf(z), ALPHA, POWER, 0.2, TWO_SIDED) == pytest.approx(
        _fpr_at_z(-z, ALPHA, POWER, 0.2, TWO_SIDED), rel=1e-9
    )


def test_fpr_at_p_falls_as_evidence_grows():
    values = [fpr_at_p_value(p, ALPHA, POWER, 0.1, TWO_SIDED)
              for p in (0.049, 0.01, 0.001, 0.0001)]
    assert values == sorted(values, reverse=True)


def test_fpr_at_p_exceeds_standard_fpr_near_alpha():
    """Just under alpha, the evidence is weaker than 'p <= alpha' implies."""
    assert fpr_at_p_value(0.049, ALPHA, POWER, 0.1, TWO_SIDED) > false_positive_risk(
        ALPHA, POWER, 0.1
    )


@pytest.mark.parametrize("alternative", [GREATER, TWO_SIDED])
def test_fpr_at_p_prior_bounds(alternative):
    assert fpr_at_p_value(0.001, ALPHA, POWER, 0.0, alternative) == 1.0
    assert fpr_at_p_value(0.001, ALPHA, POWER, 1.0, alternative) == 0.0


@pytest.mark.parametrize("alternative", [GREATER, LESS, TWO_SIDED])
def test_extreme_evidence_is_finite(alternative):
    for z in (60.0, -60.0, 1e6):
        value = _fpr_at_z(z, ALPHA, POWER, 0.1, alternative)
        assert math.isfinite(value) and 0.0 <= value <= 1.0
    assert _fpr_at_z(60.0, ALPHA, POWER, 0.1, alternative) == pytest.approx(0.0, abs=1e-12)
    # p-values at the edge of float range, and exactly 0 or 1
    for p in (1e-300, 0.0, 1.0):
        value = fpr_at_p_value(p, ALPHA, POWER, 0.1, alternative)
        assert math.isfinite(value) and 0.0 <= value <= 1.0
    assert fpr_at_p_value(0.0, ALPHA, POWER, 0.1, alternative) == pytest.approx(0.0, abs=1e-12)


def test_one_sided_p_of_one_is_certain_false_positive():
    assert fpr_at_p_value(1.0, ALPHA, POWER, 0.1, GREATER) == pytest.approx(1.0)


def test_accepts_alternative_as_string():
    assert fpr_at_p_value(0.01, ALPHA, POWER, 0.1, "two-sided") == fpr_at_p_value(
        0.01, ALPHA, POWER, 0.1, TWO_SIDED
    )


# --------------------------------------------------------------------- #
#  Validation
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("alpha, power, prior, match", [
    (0.0, POWER, 0.1, "alpha"),
    (1.0, POWER, 0.1, "alpha"),
    (ALPHA, 0.0, 0.1, "power"),
    (ALPHA, 1.0, 0.1, "power"),
    (ALPHA, 80.0, 0.1, "power"),  # percentage passed by mistake
    (ALPHA, POWER, -0.1, "prior"),
    (ALPHA, POWER, 1.1, "prior"),
])
def test_invalid_inputs_raise(alpha, power, prior, match):
    for call in (
        lambda: false_positive_risk(alpha, power, prior),
        lambda: false_negative_discovery_rate(alpha, power, prior),
        lambda: fpr_at_p_value(0.01, alpha, power, prior),
        lambda: assess_risk(0.01, TWO_SIDED, alpha, power, prior),
    ):
        with pytest.raises(ValueError, match=match):
            call()


@pytest.mark.parametrize("p", [-0.01, 1.01, float("nan")])
def test_invalid_p_value_raises(p):
    with pytest.raises(ValueError, match="p_value"):
        fpr_at_p_value(p, ALPHA, POWER, 0.1)
    with pytest.raises(ValueError, match="p_value"):
        assess_risk(p)


# --------------------------------------------------------------------- #
#  assess_risk: one call returns everything
# --------------------------------------------------------------------- #


def test_assess_risk_significant():
    r = assess_risk(0.01, TWO_SIDED, ALPHA, POWER, 0.1)
    assert isinstance(r, DecisionRiskResult)
    assert r.is_significant
    assert r.fpr == pytest.approx(0.36, abs=1e-4)
    assert r.fndr == pytest.approx(0.0229, abs=1e-4)
    assert r.fpr_at_p == pytest.approx(0.401, abs=1e-3)
    assert (r.alpha, r.power, r.prior, r.alternative) == (ALPHA, POWER, 0.1, TWO_SIDED)


def test_assess_risk_not_significant():
    r = assess_risk(0.20, GREATER, ALPHA, POWER, 0.1)
    assert not r.is_significant
    assert r.fpr_at_p is None
    assert r.fndr == pytest.approx(0.0229, abs=1e-4)


def test_assess_risk_significance_is_strict():
    """p == alpha is not significant, as in FOE's engines (p < alpha)."""
    assert not assess_risk(ALPHA, TWO_SIDED, ALPHA).is_significant


def test_assess_risk_uses_given_alpha_for_decision_and_risk():
    """A Šidák-adjusted alpha drives both the decision and the risk numbers."""
    sidak = 1 - (1 - 0.05) ** 0.5
    r = assess_risk(0.03, TWO_SIDED, sidak, POWER, 0.1)
    assert not r.is_significant
    assert r.fpr == pytest.approx(false_positive_risk(sidak, POWER, 0.1))


def test_assess_risk_serializes_for_exports():
    dumped = assess_risk(0.01, LESS).model_dump(mode="json")
    assert dumped["alternative"] == "less"
    assert set(dumped) == {
        "p_value", "alternative", "alpha", "power", "prior",
        "is_significant", "fpr", "fndr", "fpr_at_p",
    }


def test_result_is_frozen():
    r = assess_risk(0.01)
    with pytest.raises(ValidationError):
        r.fpr = 0.0
