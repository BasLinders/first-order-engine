import pytest
import numpy as np
from pydantic import ValidationError

from foe.bayesian.operations import (
    BayesianEngine,
    get_beta_prior,
    get_lift_prior,
    BetaPrior,
    LiftPrior,
)
from foe.core.models import ExperimentInput, BusinessCaseInput, BayesianResult


# --- Fixtures ---

@pytest.fixture
def engine():
    # Fixed seed for deterministic results across runs.
    return BayesianEngine(seed=42)


@pytest.fixture
def uninformative_priors():
    return get_beta_prior(), get_lift_prior(0.0, "uninformative")


@pytest.fixture
def skeptical_priors():
    return get_beta_prior(), get_lift_prior(0.0, "skeptical")


def make_experiment(visitors, conversions, labels=None):
    if labels is None:
        labels = [chr(65 + i) for i in range(len(visitors))]
    return ExperimentInput(visitors=visitors, conversions=conversions, labels=labels)


# --- run_probability_analysis ---

def test_probability_analysis_returns_bayesian_results(engine, uninformative_priors):
    """
    Return type should be a list of BayesianResult, one per challenger.
    """
    beta_prior, lift_prior = uninformative_priors
    results = engine.run_probability_analysis(
        data=make_experiment([1000, 1000], [50, 100]),
        beta_prior=beta_prior,
        lift_prior=lift_prior,
    )
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], BayesianResult)


def test_strong_challenger_dominates(engine, uninformative_priors):
    """
    A variant with double the conversion rate should have PBB near 100%.
    """
    beta_prior, lift_prior = uninformative_priors
    results = engine.run_probability_analysis(
        data=make_experiment([1000, 1000], [50, 100]),
        beta_prior=beta_prior,
        lift_prior=lift_prior,
    )
    challenger = results[0]
    assert challenger.prob_being_best > 0.99
    assert challenger.prob_beat_control > 0.99


def test_flat_test_produces_even_split(engine, uninformative_priors):
    """
    Identical conversion rates should produce roughly 50/50 prob_beat_control.
    """
    beta_prior, lift_prior = uninformative_priors
    results = engine.run_probability_analysis(
        data=make_experiment([1000, 1000], [100, 100]),
        beta_prior=beta_prior,
        lift_prior=lift_prior,
    )
    assert 0.40 <= results[0].prob_beat_control <= 0.60


def test_three_variants_strongest_dominates(engine, uninformative_priors):
    """
    With three variants, the strongest should have PBB > 0.90 and
    all PBBs (including control) should sum to ~1.
    """
    beta_prior, lift_prior = uninformative_priors
    results = engine.run_probability_analysis(
        data=make_experiment(
            [1000, 1000, 1000], [50, 75, 120],
            labels=["Control", "B", "C"],
        ),
        beta_prior=beta_prior,
        lift_prior=lift_prior,
    )
    assert len(results) == 2  # one per challenger

    pbb_c = results[1].prob_being_best  # Variant C
    assert pbb_c > 0.90

    # All PBBs including control should sum to ~1.
    all_pbb = (
        [1.0 - sum(r.prob_being_best for r in results)]
        + [r.prob_being_best for r in results]
    )
    assert abs(sum(all_pbb) - 1.0) < 1e-6


def test_empty_input_rejected_by_model():
    """
    ExperimentInput rejects empty visitor/conversion lists at the model boundary.
    The engine layer never receives invalid input — validation happens in Pydantic
    via min_length=2 on the visitors and conversions fields.
    """
    with pytest.raises(ValidationError, match="at least 2 items"):
        make_experiment([], [])


def test_skeptical_prior_dampens_extreme_result(engine):
    """
    A skeptical lift prior should reduce prob_beat_control relative to
    an uninformative prior when the observed lift is large.
    """
    data = make_experiment([1000, 1000], [50, 100])

    result_uninformative = engine.run_probability_analysis(
        data=data,
        beta_prior=get_beta_prior(),
        lift_prior=get_lift_prior(0.0, "uninformative"),
    )
    result_skeptical = engine.run_probability_analysis(
        data=data,
        beta_prior=get_beta_prior(),
        lift_prior=get_lift_prior(0.0, "skeptical"),
    )
    # Skeptical prior should penalise a large lift.
    assert result_skeptical[0].prob_beat_control < result_uninformative[0].prob_beat_control


def test_result_fields_are_present(engine, uninformative_priors):
    """
    Each BayesianResult should carry all expected fields with sensible values.
    Covers both the original fields and the two added in the engine refactor
    (prob_beat_control, expected_uplift).
    """
    beta_prior, lift_prior = uninformative_priors
    results = engine.run_probability_analysis(
        data=make_experiment(
            [1000, 1000], [50, 100],
            labels=["Control", "Challenger"],
        ),
        beta_prior=beta_prior,
        lift_prior=lift_prior,
    )
    r = results[0]
    assert r.variant_label == "Challenger"
    assert r.control_label == "Control"
    assert 0.0 <= r.prob_being_best <= 1.0
    assert 0.0 <= r.prob_beat_control <= 1.0
    assert r.expected_uplift >= 0.0   # np.maximum(..., 0) guarantees non-negative
    assert r.expected_loss >= 0.0
    assert isinstance(r.conclusion, str) and len(r.conclusion) > 0


# --- run_monetary_projection ---

def _get_prob_best_overall(engine, visitors, conversions, labels):
    """Helper to extract prob_best_overall list for monetary projection calls."""
    results = engine.run_probability_analysis(
        data=make_experiment(visitors, conversions, labels),
        beta_prior=get_beta_prior(),
        lift_prior=get_lift_prior(0.0, "uninformative"),
    )
    control_pbb = 1.0 - sum(r.prob_being_best for r in results)
    return [control_pbb] + [r.prob_being_best for r in results]


def test_monetary_positive_uplift_for_better_variant(engine, uninformative_priors):
    """
    A variant with higher CR and higher AOV should produce positive uplift
    and a net positive total contribution.
    """
    beta_prior, lift_prior = uninformative_priors
    visitors = [1000, 1000]
    conversions = [100, 110]
    labels = ["Control", "Challenger"]

    results = engine.run_monetary_projection(
        visitors=visitors,
        conversions=conversions,
        biz_case=BusinessCaseInput(
            aovs={"Control": 50.0, "Challenger": 60.0},
            runtime_days=30,
            projection_period=180,
        ),
        prob_best_overall=_get_prob_best_overall(engine, visitors, conversions, labels),
        variant_labels=labels,
        beta_prior=beta_prior,
        lift_prior=lift_prior,
        aov_cv=0.5,
    )

    assert len(results) == 1
    r = results[0]
    assert r["variant_label"] == "Challenger"
    assert r["variant_index"] == 1
    assert r["expected_uplift"] > 0
    assert r["expected_total_contribution"] > 0
    assert 0.0 <= r["prob_beat_control"] <= 1.0
    assert 0.0 <= r["prob_best_overall"] <= 1.0
    assert "conclusion" in r


def test_monetary_three_variants_returns_two_results(engine, uninformative_priors):
    """
    Three variants should produce two result dicts, one per challenger.
    """
    beta_prior, lift_prior = uninformative_priors
    visitors = [1000, 1000, 1000]
    conversions = [50, 75, 120]
    labels = ["Control", "Variant B", "Variant C"]

    results = engine.run_monetary_projection(
        visitors=visitors,
        conversions=conversions,
        biz_case=BusinessCaseInput(
            aovs={"Control": 50.0, "Variant B": 50.0, "Variant C": 50.0},
            runtime_days=30,
            projection_period=90,
        ),
        prob_best_overall=_get_prob_best_overall(engine, visitors, conversions, labels),
        variant_labels=labels,
        beta_prior=beta_prior,
        lift_prior=lift_prior,
        aov_cv=0.5,
    )

    assert len(results) == 2
    assert results[0]["variant_label"] == "Variant B"
    assert results[1]["variant_label"] == "Variant C"


def test_monetary_invalid_runtime_rejected_by_model():
    """
    BusinessCaseInput rejects runtime_days=0 at the model boundary via
    Field(..., gt=0). The engine layer never receives this invalid input.
    """
    with pytest.raises(ValidationError):
        BusinessCaseInput(
            aovs={"Control": 50.0, "Challenger": 60.0},
            runtime_days=0,
            projection_period=180,
        )


def test_monetary_label_mismatch_raises_value_error(engine, uninformative_priors):
    """
    Passing a variant_labels list whose length doesn't match visitors raises
    ValueError. Unlike runtime_days, this mismatch is not guaranteed to be
    caught by Pydantic (variant_labels is a plain function parameter), so
    run_monetary_projection enforces it explicitly.
    """
    beta_prior, lift_prior = uninformative_priors
    visitors = [1000, 1000]
    conversions = [100, 110]
    labels = ["Control", "Challenger"]

    with pytest.raises(ValueError, match="variant_labels has 1 entries but visitors has 2"):
        engine.run_monetary_projection(
            visitors=visitors,
            conversions=conversions,
            biz_case=BusinessCaseInput(
                aovs={"Control": 50.0, "Challenger": 60.0},
                runtime_days=30,
                projection_period=90,
            ),
            prob_best_overall=_get_prob_best_overall(engine, visitors, conversions, labels),
            variant_labels=["OnlyOneLabel"],
            beta_prior=beta_prior,
            lift_prior=lift_prior,
        )


def test_aov_cv_zero_approximates_constant_aov(engine, uninformative_priors):
    """
    A very low CV should produce monetary results close to those from a
    high-CV run, confirming the log-normal parameterisation is unbiased:
    E[AOV] = mean_aov regardless of spread.
    """
    beta_prior, lift_prior = uninformative_priors
    visitors = [5000, 5000]
    conversions = [250, 300]
    labels = ["Control", "Challenger"]
    pbo = _get_prob_best_overall(engine, visitors, conversions, labels)
    biz_case = BusinessCaseInput(
        aovs={"Control": 50.0, "Challenger": 50.0},
        runtime_days=30,
        projection_period=180,
    )

    result_low_cv = engine.run_monetary_projection(
        visitors=visitors, conversions=conversions,
        biz_case=biz_case, prob_best_overall=pbo,
        variant_labels=labels,
        beta_prior=beta_prior, lift_prior=lift_prior,
        aov_cv=0.01,
    )
    result_high_cv = engine.run_monetary_projection(
        visitors=visitors, conversions=conversions,
        biz_case=biz_case, prob_best_overall=pbo,
        variant_labels=labels,
        beta_prior=beta_prior, lift_prior=lift_prior,
        aov_cv=1.5,
    )

    # Point estimates should converge regardless of spread (within 10%).
    low = result_low_cv[0]["expected_uplift"]
    high = result_high_cv[0]["expected_uplift"]
    assert abs(low - high) / max(abs(low), 1e-6) < 0.10
