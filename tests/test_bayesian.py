import pytest
import numpy as np
from foe.bayesian.operations import BayesianEngine
from foe.core.models import ExperimentInput, BusinessCaseInput

@pytest.fixture
def engine():
    return BayesianEngine()

def test_probability_of_being_best(engine):
    """
    Test that a variant with double the conversion rate has a 
    probability of being best (PBB) near 100%.
    """
    data = ExperimentInput(
        visitors=[1000, 1000],
        conversions=[50, 100], # 5% vs 10%
        labels=["Control", "Challenger"]
    )
    
    # Run analysis
    results = engine.run_probability_analysis(data)
    challenger = results[0]
    
    assert challenger.prob_being_best > 0.99
    assert "Clear Winner" in challenger.conclusion
    assert challenger.expected_loss < 0.001 # Minimal risk in rolling this out

def test_flat_test_risk_assessment(engine):
    """
    If results are identical, the engine should indicate 
    roughly 50/50 probability and a higher expected loss.
    """
    data = ExperimentInput(
        visitors=[1000, 1000],
        conversions=[100, 100],
        labels=["Control", "Variant"]
    )
    
    results = engine.run_probability_analysis(data)
    variant = results[0]
    
    # Near 50%
    assert 0.45 <= variant.prob_being_best <= 0.55
    assert "Inconclusive" in variant.conclusion

def test_business_case_revenue_projection(engine):
    """
    Test that revenue projections correctly compound
    based on AOV and conversion rate lift.
    """
    visitors = [1000, 1000]
    conversions = [100, 110]  # 10% lift in CR
    labels = ["Control", "Challenger"]
    data = ExperimentInput(visitors=visitors, conversions=conversions, labels=labels)

    # $50 AOV vs $60 AOV
    biz_case = BusinessCaseInput(
        aovs={"Control": 50.0, "Challenger": 60.0},
        runtime_days=30,       # Data collected over a month
        projection_period=180  # Project 6 months forward
    )

    prob_results = engine.run_probability_analysis(data)
    prob_best_overall = [1.0 - prob_results[0].prob_being_best, prob_results[0].prob_being_best]

    results = engine.run_monetary_projection(visitors, conversions, biz_case, prob_best_overall, labels)

    assert len(results) == 1
    assert results[0]['expected_uplift'] > 0
    assert results[0]['expected_total_contribution'] > 0
