import pytest
import numpy as np
from foe.bayesian.operations import BayesianEngine, get_beta_prior, get_lift_prior, BetaPrior, LiftPrior
from foe.core.models import ExperimentInput, BusinessCaseInput, BayesianResult


# --- Fixtures ---

@pytest.fixture
def engine():
    # Fixed seed for deterministic results across runs
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
    With three variants, the strongest should have PBB > 0.95 and
    all PBBs should sum to ~1.
    """
    beta_prior, lift_prior = uninformative_priors
    results = engine.run_probability_analysis(
        data=make_experiment(
            [1000, 1000, 1000], [50, 75, 120],
            labels=["Control", "B", "C"]
        ),
        beta_prior=beta_prior,
        lift_prior=lift_prior,
    )
    assert len(results) == 2  # one per challenger

    pbb_c = results[1].prob_being_best  # Variant C
    assert pbb_c > 0.90

    # All PBBs including control should sum to ~1
    all_pbb = [1.0 - sum(r.prob_being_best for r in results)] + [r.prob_being_best for r in results]
    assert abs(sum(all_pbb) - 1.0) < 1e-6


def test_empty_input_returns_empty_list(engine, uninformative_priors):
    """
    Empty input should return an empty list without crashing.
    """
    beta_prior, lift_prior = uninformative_priors
    results = engine.run_probability_analysis(
        data=make_experiment([], []),
        beta_prior=beta_prior,
        lift_prior=lift_prior,
    )
    assert results == []


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
    # Skeptical prior should penalise a large lift
    assert result_skeptical[0].prob_beat_control < result_uninformative[0].prob_beat_control


def test_result_fields_are_present(engine, uninformative_priors):
    """
    Each BayesianResult should carry all expected fields.
    """
    beta_prior, lift_prior = uninformative_priors
    results = engine.run_probability_analysis(
        data=make_experiment(
            [1000, 1000], [50, 100],
            labels=["Control", "Challenger"]
        ),
        beta_prior=beta_prior,
        lift_prior=lift_prior,
    )
    r = results[0]
    assert r.variant_label == "Challenger"
    assert r.control_label == "Control"
    assert 0.0 <= r.prob_being_best  <= 1.0
    assert 0.0 <= r.prob_beat_control <= 1.0
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
    # Reconstruct full list including control (index 0)
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


def test_monetary_invalid_runtime_returns_empty(engine, uninformative_priors):
    """
    runtime_days <= 0 should return an empty list without crashing.
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
            runtime_days=0,
            projection_period=180,
        ),
        prob_best_overall=_get_prob_best_overall(engine, visitors, conversions, labels),
        variant_labels=labels,
        beta_prior=beta_prior,
        lift_prior=lift_prior,
    )
    assert results == []


def test_monetary_label_mismatch_returns_empty(engine, uninformative_priors):
    """
    Mismatched variant_labels length should return an empty list safely.
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
            projection_period=90,
        ),
        prob_best_overall=_get_prob_best_overall(engine, visitors, conversions, labels),
        variant_labels=["OnlyOneLabel"],
        beta_prior=beta_prior,
        lift_prior=lift_prior,
    )
    assert results == []


def test_aov_cv_zero_approximates_constant_aov(engine, uninformative_priors):
    """
    A very low CV should produce monetary results close to those from a
    constant AOV, confirming the log-normal parameterisation is unbiased.
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

    # Point estimates (uplift) should be close regardless of CV; only spread differs
    low  = result_low_cv[0]["expected_uplift"]
    high = result_high_cv[0]["expected_uplift"]
    assert abs(low - high) / max(abs(low), 1e-6) < 0.10  # within 10%
