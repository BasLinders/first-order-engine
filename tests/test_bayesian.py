import pytest
import numpy as np
from foe.bayesian.operations import BayesianEngine
from foe.core.models import ExperimentInput, BusinessCaseInput

@pytest.fixture
def engine():
    return BayesianEngine()

def test_probability_of_being_best(engine):
    """
    A variant with double the conversion rate should have PBB near 100%.
    """
    results = engine.run_probability_analysis(
        visitors=[1000, 1000],
        conversions=[50, 100]  # 5% vs 10%
    )

    assert "prob_being_best" in results
    assert len(results["prob_being_best"]) == 2

    control_pbb, challenger_pbb = results["prob_being_best"]

    assert challenger_pbb > 0.99
    assert control_pbb < 0.01
    assert abs(sum(results["prob_being_best"]) - 1.0) < 1e-6  # must sum to 1

def test_flat_test_risk_assessment(engine):
    """
    Identical conversion rates should produce roughly 50/50 PBB.
    """
    results = engine.run_probability_analysis(
        visitors=[1000, 1000],
        conversions=[100, 100]
    )

    for pbb in results["prob_being_best"]:
        assert 0.45 <= pbb <= 0.55

    assert abs(sum(results["prob_being_best"]) - 1.0) < 1e-6

def test_returns_samples_when_requested(engine):
    """
    Samples array should only be present when explicitly requested.
    """
    without = engine.run_probability_analysis(
        visitors=[1000, 1000],
        conversions=[50, 100]
    )
    assert "samples" not in without

    with_samples = engine.run_probability_analysis(
        visitors=[1000, 1000],
        conversions=[50, 100],
        return_samples=True
    )
    assert "samples" in with_samples
    assert with_samples["samples"].shape == (2, 100000)

def test_three_variants(engine):
    """
    PBB should still sum to 1 across three variants,
    with the strongest variant dominating.
    """
    results = engine.run_probability_analysis(
        visitors=[1000, 1000, 1000],
        conversions=[50, 75, 120]  # 5%, 7.5%, 12%
    )

    assert len(results["prob_being_best"]) == 3
    assert results["prob_being_best"][2] > 0.95  # Variant C should dominate
    assert abs(sum(results["prob_being_best"]) - 1.0) < 1e-6

def test_empty_input(engine):
    """
    Empty input should return empty prob_being_best without crashing.
    """
    results = engine.run_probability_analysis(visitors=[], conversions=[])
    assert results["prob_being_best"] == []
    
def test_business_case_revenue_projection(engine):
    """
    Higher CR * higher AOV should produce positive uplift
    and a net positive total contribution.
    """
    visitors = [1000, 1000]
    conversions = [100, 110]
    labels = ["Control", "Challenger"]

    # Get PBB first, as the handler would
    prob_results = engine.run_probability_analysis(
        visitors=visitors,
        conversions=conversions
    )

    biz_case = BusinessCaseInput(
        aovs={"Control": 50.0, "Challenger": 60.0},
        runtime_days=30,
        projection_period=180
    )

    results = engine.run_monetary_projection(
        visitors=visitors,
        conversions=conversions,
        biz_case=biz_case,
        prob_best_overall=prob_results["prob_being_best"],
        variant_labels=labels
    )

    assert len(results) == 1  # One non-control variant
    challenger = results[0]

    assert challenger["variant_label"] == "Challenger"
    assert challenger["variant_index"] == 1
    assert challenger["expected_uplift"] > 0
    assert challenger["expected_total_contribution"] > 0
    assert 0.0 <= challenger["prob_beat_control"] <= 1.0
    assert 0.0 <= challenger["prob_best_overall"] <= 1.0
    assert "conclusion" in challenger


def test_monetary_returns_one_result_per_non_control_variant(engine):
    """
    Three variants should produce two result dicts (excluding control).
    """
    visitors = [1000, 1000, 1000]
    conversions = [50, 75, 120]
    labels = ["Control", "Variant B", "Variant C"]

    prob_results = engine.run_probability_analysis(
        visitors=visitors,
        conversions=conversions
    )

    biz_case = BusinessCaseInput(
        aovs={"Control": 50.0, "Variant B": 50.0, "Variant C": 50.0},
        runtime_days=30,
        projection_period=90
    )

    results = engine.run_monetary_projection(
        visitors=visitors,
        conversions=conversions,
        biz_case=biz_case,
        prob_best_overall=prob_results["prob_being_best"],
        variant_labels=labels
    )

    assert len(results) == 2
    assert results[0]["variant_label"] == "Variant B"
    assert results[1]["variant_label"] == "Variant C"
    

def test_monetary_returns_empty_on_invalid_runtime(engine):
    """
    runtime_days <= 0 should return an empty list, not crash.
    """
    visitors = [1000, 1000]
    conversions = [100, 110]
    labels = ["Control", "Challenger"]

    prob_results = engine.run_probability_analysis(
        visitors=visitors,
        conversions=conversions
    )

    biz_case = BusinessCaseInput(
        aovs={"Control": 50.0, "Challenger": 60.0},
        runtime_days=0,  # Invalid
        projection_period=180
    )

    results = engine.run_monetary_projection(
        visitors=visitors,
        conversions=conversions,
        biz_case=biz_case,
        prob_best_overall=prob_results["prob_being_best"],
        variant_labels=labels
    )

    assert results == []


def test_monetary_label_mismatch_returns_empty(engine):
    """
    Mismatched variant_labels length should return empty list safely.
    """
    visitors = [1000, 1000]
    conversions = [100, 110]

    prob_results = engine.run_probability_analysis(
        visitors=visitors,
        conversions=conversions
    )

    biz_case = BusinessCaseInput(
        aovs={"Control": 50.0, "Challenger": 60.0},
        runtime_days=30,
        projection_period=90
    )

    results = engine.run_monetary_projection(
        visitors=visitors,
        conversions=conversions,
        biz_case=biz_case,
        prob_best_overall=prob_results["prob_being_best"],
        variant_labels=["OnlyOneLabel"]  # Wrong length
    )

    assert results == []
