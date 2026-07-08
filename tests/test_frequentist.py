import math

import pytest
from pydantic import ValidationError

from foe.frequentist.operations import FrequentistEngine
from foe.frequentist.confidence import compute_non_inferiority, compute_interval_difference
from foe.core.models import ExperimentInput, AlternativeHypothesis


@pytest.fixture
def engine():
    return FrequentistEngine()


def test_identical_variants_yield_p_value_one(engine):
    """
    If Control and Challenger are identical, the p-value should be 1.0,
    uplift should be 0.0, and the result should not be significant.
    """
    data = ExperimentInput(
        visitors=[1000, 1000],
        conversions=[100, 100],
        labels=["Control", "Challenger"],
    )
    results = engine.run_synthesis(data)
    challenger = results[0]

    assert challenger.p_value == pytest.approx(1.0)
    assert challenger.uplift == 0.0
    assert challenger.is_significant is False


def test_significant_winner_detection(engine):
    """
    A 10% vs 15% conversion rate with n=2000 per variant should be detected
    as a statistically significant positive result with the correct uplift.
    """
    data = ExperimentInput(
        visitors=[2000, 2000],
        conversions=[200, 300],
        labels=["Control", "Winner"],
    )
    results = engine.run_synthesis(data)
    winner = results[0]

    assert winner.conversion_rate == pytest.approx(0.15)
    assert winner.uplift == pytest.approx(0.5)  # (15% - 10%) / 10%
    assert winner.p_value < 0.01
    assert winner.is_significant is True
    assert "Significant Positive Impact" in winner.conclusion


def test_one_sided_hypothesis(engine):
    """
    A GREATER alternative should produce a p-value exactly half that of
    TWO_SIDED when the observed difference is in the winning direction.

    Also verifies that GREATER returns a one-sided CI with an upper bound
    of +inf, while TWO_SIDED returns a finite symmetric interval.
    """
    visitors = [1000, 1000]
    conversions = [100, 120]

    two_sided = engine.run_synthesis(
        ExperimentInput(
            visitors=visitors,
            conversions=conversions,
            alternative=AlternativeHypothesis.TWO_SIDED,
        )
    )[0]

    one_sided = engine.run_synthesis(
        ExperimentInput(
            visitors=visitors,
            conversions=conversions,
            alternative=AlternativeHypothesis.GREATER,
        )
    )[0]

    # In a winning scenario, one-sided p-value is exactly half of two-sided.
    assert one_sided.p_value == pytest.approx(two_sided.p_value / 2)

    # GREATER produces a lower-bound-only CI; upper bound is +inf.
    assert one_sided.ci_diff[1] == float("inf")

    # TWO_SIDED produces a finite symmetric interval.
    assert two_sided.ci_diff[0] != float("-inf")
    assert two_sided.ci_diff[1] != float("inf")


def test_input_validation_integration():
    """
    Ensures that validators.py logic surfaces correctly as ValidationError
    when accessed through ExperimentInput.

    Pydantic's model_validator wraps ValueError from validate_experiment_data
    into a ValidationError — bare ValueError is never raised at the model boundary.
    """
    # Impossible data: conversions exceed visitors for variant at index 0.
    # Our updated validators.py produces a lowercase message:
    # "Variant at index 0: conversions (150) exceed visitors (100)."
    with pytest.raises(ValidationError, match="conversions.*exceed visitors"):
        ExperimentInput(visitors=[100, 100], conversions=[150, 50])

    # Single-variant input violates min_length=2 on the visitors field.
    # Pydantic v2 raises ValidationError before the model_validator runs.
    with pytest.raises(ValidationError, match="at least 2 items"):
        ExperimentInput(visitors=[100], conversions=[10])


def test_confidence_level_zero_rejected():
    """
    confidence_level=0.0 is now rejected (gt=0.0). Previously ge=0.0 allowed
    it, producing alpha=1.0 where every test was always significant.
    """
    with pytest.raises(ValidationError):
        ExperimentInput(
            visitors=[1000, 1000],
            conversions=[100, 110],
            confidence_level=0.0,
        )


def test_sidak_correction_applied_for_multiple_variants(engine):
    """
    With three variants, apply_sidak tightens alpha below the raw 0.05.
    A marginal effect that would be significant in a simple A/B test (alpha=0.05)
    should become insignificant once the per-comparison alpha is corrected.

    apply_sidak(0.05, 3) ≈ 0.0253. The control vs Challenger A comparison is
    constructed to land between 0.025 and 0.05 so it flips from significant to
    not significant only when the correction is in effect.
    """
    # Challenger A: 563/5000 = 11.26% vs control 10%.
    # SE ≈ 0.00616, z ≈ 2.04, p ≈ 0.041 — between Sidak threshold (~0.025) and 0.05.
    data = ExperimentInput(
        visitors=[5000, 5000, 5000],
        conversions=[500, 563, 900],
        labels=["Control", "Challenger A", "Challenger B"],
        confidence_level=0.95,
    )
    results = engine.run_synthesis(data)
    challenger_a = results[0]

    # p_value is between the Sidak-adjusted threshold (~0.025) and raw alpha (0.05).
    assert 0.025 < challenger_a.p_value < 0.05
    # With Sidak applied, this should NOT be significant.
    assert challenger_a.is_significant is False


def test_bootstrap_power_one_sided_greater_than_two_sided(engine):
    """
    For a winning challenger, one-sided (GREATER) bootstrap power should be
    higher than two-sided power, since the one-sided test concentrates all
    rejection power in the direction of the observed effect.
    """
    power_two_sided = FrequentistEngine.run_vectorized_bootstrap_power(
        ctrl_conv=100, ctrl_n=1000,
        chal_conv=130, chal_n=1000,
        alpha=0.05,
        n_bootstraps=5000,
        alternative=AlternativeHypothesis.TWO_SIDED,
    )
    power_one_sided = FrequentistEngine.run_vectorized_bootstrap_power(
        ctrl_conv=100, ctrl_n=1000,
        chal_conv=130, chal_n=1000,
        alpha=0.05,
        n_bootstraps=5000,
        alternative=AlternativeHypothesis.GREATER,
    )
    assert power_one_sided > power_two_sided


def test_less_alternative_produces_lower_bound_only_ci(engine):
    """
    LESS alternative should return a CI with -inf lower bound and finite upper bound.
    Also verifies the p-value equals half the two-sided p-value for an in-direction effect.
    """
    visitors = [1000, 1000]
    conversions = [120, 100]  # challenger is worse — LESS is the correct direction

    less = engine.run_synthesis(
        ExperimentInput(
            visitors=visitors,
            conversions=conversions,
            alternative=AlternativeHypothesis.LESS,
        )
    )[0]

    two_sided = engine.run_synthesis(
        ExperimentInput(
            visitors=visitors,
            conversions=conversions,
            alternative=AlternativeHypothesis.TWO_SIDED,
        )
    )[0]

    assert less.ci_diff[0] == float("-inf")
    assert less.ci_diff[1] != float("inf")
    assert less.p_value == pytest.approx(two_sided.p_value / 2)


def test_run_ztest_zero_se_returns_one():
    """Zero SE must return p_value=1.0 rather than divide-by-zero."""
    p = FrequentistEngine.run_ztest(
        diff=0.01, se_diff=0.0, alternative=AlternativeHypothesis.TWO_SIDED
    )
    assert p == 1.0


def test_analytical_power_high_effect_near_one():
    """A large effect relative to SE should yield power close to 1.0."""
    power = FrequentistEngine.calculate_analytical_power(
        diff=0.10,
        se_diff=0.005,
        alpha=0.05,
        alternative=AlternativeHypothesis.TWO_SIDED,
    )
    assert power > 0.99


def test_analytical_power_zero_se_returns_zero():
    """Zero SE is an invalid input; power should be 0.0 rather than crash."""
    power = FrequentistEngine.calculate_analytical_power(
        diff=0.01,
        se_diff=0.0,
        alpha=0.05,
        alternative=AlternativeHypothesis.TWO_SIDED,
    )
    assert power == 0.0


def test_compute_non_inferiority_passing():
    """Challenger within the NI margin should be declared non-inferior."""
    result = compute_non_inferiority(
        p_ctrl=0.10, p_chal=0.095, se_diff=0.005, margin=0.02, alpha=0.05
    )
    assert result["is_non_inferior"] is True
    assert result["p_value"] < 0.05
    assert "Non-inferiority established" in result["conclusion"]


def test_compute_non_inferiority_failing():
    """Challenger far below control should fail the NI test."""
    result = compute_non_inferiority(
        p_ctrl=0.10, p_chal=0.05, se_diff=0.005, margin=0.02, alpha=0.05
    )
    assert result["is_non_inferior"] is False
    assert "Too Risky" in result["conclusion"]


def test_compute_non_inferiority_invalid_margin():
    """margin outside (0, 1) must raise ValueError."""
    with pytest.raises(ValueError, match="proportion"):
        compute_non_inferiority(
            p_ctrl=0.10, p_chal=0.09, se_diff=0.005, margin=1.5, alpha=0.05
        )


def test_variance_reduction_factor_tightens_intervals(engine):
    """
    Providing a reduction_factor < 1.0 should reduce the standard error
    and produce narrower confidence intervals compared to no adjustment.

    The reduction_factor can originate from CUPED, Lin's adjustment, or the
    aggregate timeseries dispersion method — this test verifies the mechanical
    effect on SE and CI width regardless of source.

    With our corrected SE formula (phi scales variance, not SE directly):
        se_adjusted = sqrt(phi * p*(1-p)/n)  =  se_base * sqrt(phi)
    A phi of 0.8 reduces SE by ~10.6% (sqrt(0.8) ≈ 0.894).
    """
    visitors = [5000, 5000]
    conversions = [500, 550]

    standard = engine.run_synthesis(
        ExperimentInput(visitors=visitors, conversions=conversions, reduction_factor=1.0)
    )[0]

    adjusted = engine.run_synthesis(
        ExperimentInput(visitors=visitors, conversions=conversions, reduction_factor=0.8)
    )[0]

    # SE should be reduced by sqrt(0.8) ≈ 10.6%.
    assert adjusted.standard_error < standard.standard_error
    assert adjusted.standard_error == pytest.approx(standard.standard_error * 0.8 ** 0.5)

    # TWO_SIDED CI (default): both bounds are finite, width comparison is valid.
    std_width = standard.ci_diff[1] - standard.ci_diff[0]
    adj_width = adjusted.ci_diff[1] - adjusted.ci_diff[0]
    assert adj_width < std_width


def test_monetary_impact_point_estimate_scales_linearly(engine):
    """
    revenue(d) = d * daily_visitors * aov * projection_period. Doubling any
    one factor should double the point estimate and both CI bounds.
    """
    result = engine.estimate_monetary_impact(
        diff=0.02,
        ci_diff=(0.01, 0.03),
        daily_visitors=1000,
        aov=50,
        projection_period=183,
    )
    assert result["point_estimate"] == pytest.approx(0.02 * 1000 * 50 * 183)
    assert result["ci_low"] == pytest.approx(0.01 * 1000 * 50 * 183)
    assert result["ci_high"] == pytest.approx(0.03 * 1000 * 50 * 183)

    doubled = engine.estimate_monetary_impact(
        diff=0.02,
        ci_diff=(0.01, 0.03),
        daily_visitors=2000,
        aov=50,
        projection_period=183,
    )
    assert doubled["point_estimate"] == pytest.approx(result["point_estimate"] * 2)
    assert doubled["ci_low"] == pytest.approx(result["ci_low"] * 2)
    assert doubled["ci_high"] == pytest.approx(result["ci_high"] * 2)


def test_monetary_impact_propagates_infinite_ci_bound(engine):
    """
    A one-sided test's infinite CI bound (see test_one_sided_hypothesis)
    should propagate as +/-inf, not be silently clipped to a finite number.
    """
    result = engine.estimate_monetary_impact(
        diff=0.02,
        ci_diff=(0.01, float("inf")),
        daily_visitors=1000,
        aov=50,
        projection_period=183,
    )
    assert result["ci_high"] == float("inf")
    assert math.isfinite(result["ci_low"])
    assert math.isfinite(result["point_estimate"])


def test_monetary_conclusion_formats_infinite_bound_as_unbounded(engine):
    result = engine.estimate_monetary_impact(
        diff=0.02,
        ci_diff=(0.01, float("inf")),
        daily_visitors=1000,
        aov=50,
        projection_period=183,
    )
    conclusion = engine.generate_monetary_conclusion("Variant B", result, is_significant=True)
    assert "unbounded" in conclusion
    assert "inf" not in conclusion.replace("unbounded", "")


def test_monetary_conclusion_negative_point_estimate_reads_as_a_cost(engine):
    result = engine.estimate_monetary_impact(
        diff=-0.02,
        ci_diff=(-0.03, -0.01),
        daily_visitors=1000,
        aov=50,
        projection_period=183,
    )
    conclusion = engine.generate_monetary_conclusion("Variant B", result, is_significant=True)
    assert "cost" in conclusion
    assert "-" not in conclusion.split("cost")[1].split("over")[0]  # magnitude, not a signed number


def test_monetary_conclusion_non_significant_is_illustrative_only(engine):
    result = engine.estimate_monetary_impact(
        diff=0.005,
        ci_diff=(-0.01, 0.02),
        daily_visitors=1000,
        aov=50,
        projection_period=183,
    )
    conclusion = engine.generate_monetary_conclusion("Variant B", result, is_significant=False)
    assert "not statistically significant" in conclusion
    assert "illustrative" in conclusion


def test_per_variant_aov_reduces_to_shared_aov_when_equal(engine):
    """
    When aov_ctrl == aov_chal, estimate_monetary_impact_per_variant should
    match estimate_monetary_impact given the same underlying diff/se_diff,
    since X = aov*(p_chal - p_ctrl) = aov*diff and
    SE(X) = aov*sqrt(se_chal^2 + se_ctrl^2) = aov*se_diff in that case.
    """
    p_ctrl, se_ctrl = 0.10, 0.004
    p_chal, se_chal = 0.12, 0.0043
    aov = 50.0
    daily_visitors = 1000.0

    diff = p_chal - p_ctrl
    se_diff = math.sqrt(se_chal ** 2 + se_ctrl ** 2)
    ci_diff = compute_interval_difference(diff, se_diff, alpha=0.05)

    shared = engine.estimate_monetary_impact(
        diff=diff, ci_diff=ci_diff, daily_visitors=daily_visitors, aov=aov,
    )
    per_variant = engine.estimate_monetary_impact_per_variant(
        p_ctrl=p_ctrl, se_ctrl=se_ctrl, aov_ctrl=aov,
        p_chal=p_chal, se_chal=se_chal, aov_chal=aov,
        daily_visitors=daily_visitors,
    )

    assert per_variant["point_estimate"] == pytest.approx(shared["point_estimate"])
    assert per_variant["ci_low"] == pytest.approx(shared["ci_low"])
    assert per_variant["ci_high"] == pytest.approx(shared["ci_high"])


def test_per_variant_aov_diverges_when_aov_differs(engine):
    """
    A pricing test where the challenger has a materially higher AOV should
    produce a bigger point estimate than treating both variants as if they
    shared the challenger's (or control's) AOV -- the value-weighted
    difference is not a simple rate-difference scaling once AOV differs.
    """
    p_ctrl, se_ctrl = 0.10, 0.004
    p_chal, se_chal = 0.10, 0.004  # identical conversion rate
    daily_visitors = 1000.0

    same_aov = engine.estimate_monetary_impact_per_variant(
        p_ctrl=p_ctrl, se_ctrl=se_ctrl, aov_ctrl=50.0,
        p_chal=p_chal, se_chal=se_chal, aov_chal=50.0,
        daily_visitors=daily_visitors,
    )
    higher_chal_aov = engine.estimate_monetary_impact_per_variant(
        p_ctrl=p_ctrl, se_ctrl=se_ctrl, aov_ctrl=50.0,
        p_chal=p_chal, se_chal=se_chal, aov_chal=80.0,
        daily_visitors=daily_visitors,
    )

    # Equal conversion rates but a materially higher challenger AOV should
    # show a positive monetary uplift purely from the AOV difference, even
    # though estimate_monetary_impact (single shared AOV) would show zero.
    assert same_aov["point_estimate"] == pytest.approx(0.0, abs=1e-9)
    assert higher_chal_aov["point_estimate"] > 0


def test_per_variant_aov_propagates_infinite_ci_bound(engine):
    result = engine.estimate_monetary_impact_per_variant(
        p_ctrl=0.10, se_ctrl=0.004, aov_ctrl=50.0,
        p_chal=0.12, se_chal=0.0043, aov_chal=65.0,
        daily_visitors=1000.0,
        alternative=AlternativeHypothesis.GREATER,
    )
    assert result["ci_high"] == float("inf")
    assert math.isfinite(result["ci_low"])
    assert math.isfinite(result["point_estimate"])


def test_per_variant_aov_conclusion_is_reusable(engine):
    """generate_monetary_conclusion should work unchanged on the per-variant result shape."""
    result = engine.estimate_monetary_impact_per_variant(
        p_ctrl=0.10, se_ctrl=0.004, aov_ctrl=50.0,
        p_chal=0.12, se_chal=0.0043, aov_chal=65.0,
        daily_visitors=1000.0,
    )
    conclusion = engine.generate_monetary_conclusion("Variant B", result, is_significant=True)
    assert "Variant B" in conclusion
    assert "uplift" in conclusion


def test_aov_uncertainty_defaults_to_zero_unchanged(engine):
    """
    Omitting se_aov_ctrl/se_aov_chal should reproduce the exact prior
    behavior (AOV treated as a fixed known constant) -- backward compatible.
    """
    kwargs = dict(
        p_ctrl=0.052367, se_ctrl=0.00057, aov_ctrl=125.0,
        p_chal=0.053262, se_chal=0.00058, aov_chal=127.0,
        daily_visitors=10721.43, alpha=0.05,
        alternative=AlternativeHypothesis.GREATER,
    )
    explicit_zero = engine.estimate_monetary_impact_per_variant(
        se_aov_ctrl=0.0, se_aov_chal=0.0, **kwargs
    )
    omitted = engine.estimate_monetary_impact_per_variant(**kwargs)
    assert explicit_zero == omitted


def test_aov_uncertainty_widens_ci_and_can_flip_lower_bound_negative(engine):
    """
    Reproduces the reported case: conversion rates are not significantly
    different (roughly 5.24% vs 5.33% at n~150k), but Control/Challenger AOV
    differ (125 vs 127). Treating that AOV gap as certain can leave the
    one-sided lower bound positive even though the rate difference alone
    isn't significant. Propagating AOV's own sampling uncertainty (as if AOV
    were estimated from a comparable-sized order sample) should widen the
    interval, lowering (or flipping the sign of) the lower bound.
    """
    p_ctrl, se_ctrl = 7855 / 150000, math.sqrt((7855 / 150000) * (1 - 7855 / 150000) / 150000)
    p_chal, se_chal = 8000 / 150200, math.sqrt((8000 / 150200) * (1 - 8000 / 150200) / 150200)
    daily_visitors = (150000 + 150200) / 28

    no_aov_uncertainty = engine.estimate_monetary_impact_per_variant(
        p_ctrl=p_ctrl, se_ctrl=se_ctrl, aov_ctrl=125.0,
        p_chal=p_chal, se_chal=se_chal, aov_chal=127.0,
        daily_visitors=daily_visitors, alpha=0.05,
        alternative=AlternativeHypothesis.GREATER,
    )

    # AOV estimated from order-count-sized samples with plausible spread
    # (CV=0.5, a common e-commerce assumption) -- se_aov = aov*cv/sqrt(n_orders).
    se_aov_ctrl = 125.0 * 0.5 / math.sqrt(7855)
    se_aov_chal = 127.0 * 0.5 / math.sqrt(8000)
    with_aov_uncertainty = engine.estimate_monetary_impact_per_variant(
        p_ctrl=p_ctrl, se_ctrl=se_ctrl, aov_ctrl=125.0,
        p_chal=p_chal, se_chal=se_chal, aov_chal=127.0,
        daily_visitors=daily_visitors, alpha=0.05,
        alternative=AlternativeHypothesis.GREATER,
        se_aov_ctrl=se_aov_ctrl, se_aov_chal=se_aov_chal,
    )

    # Point estimate (the value-weighted diff itself) is unaffected by SE.
    assert with_aov_uncertainty["point_estimate"] == pytest.approx(no_aov_uncertainty["point_estimate"])
    # But propagating AOV's own uncertainty must push the one-sided lower
    # bound down (a wider, more honest interval), not up.
    assert with_aov_uncertainty["ci_low"] < no_aov_uncertainty["ci_low"]


def test_aov_uncertainty_included_in_two_sided_width(engine):
    """Adding AOV uncertainty should widen a Two-sided interval too, not just shift the one-sided bound."""
    kwargs = dict(
        p_ctrl=0.10, se_ctrl=0.004, aov_ctrl=50.0,
        p_chal=0.12, se_chal=0.0043, aov_chal=55.0,
        daily_visitors=1000.0, alpha=0.05,
    )
    narrow = engine.estimate_monetary_impact_per_variant(**kwargs)
    wide = engine.estimate_monetary_impact_per_variant(
        se_aov_ctrl=5.0, se_aov_chal=5.0, **kwargs
    )
    narrow_width = narrow["ci_high"] - narrow["ci_low"]
    wide_width = wide["ci_high"] - wide["ci_low"]
    assert wide_width > narrow_width
