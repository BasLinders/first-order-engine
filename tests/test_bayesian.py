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
    results = engine.run_probability_analysis(
        visitors=data.visitors,
        conversions=data.conversions
    )
    challenger_pbb = results["prob_being_best"][1]
    challenger_loss = results["expected_loss"][1]
    
    assert challenger_pbb > 0.99
    assert challenger_loss < 0.001
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
    
    results = engine.run_probability_analysis(
        visitors=data.visitors,
        conversions=data.conversions
    )
    variant = results[0]
    
    # Near 50%
    for pbb in results["prob_being_best"]:
        assert 0.45 <= pbb <= 0.55
    assert "Inconclusive" in variant.conclusion

def test_business_case_revenue_projection(engine):
    """
    Test that revenue projections correctly compound 
    based on AOV and conversion rate lift.
    """
    # 1000 visitors/day baseline
    data = ExperimentInput(
        visitors=[1000, 1000],
        conversions=[100, 110] # 10% lift in CR
    )
    
    # $50 AOV vs $60 AOV
    biz_case = BusinessCaseInput(
        aovs={"Control": 50.0, "Challenger": 60.0},
        runtime_days=30, # Data collected over a month
        projection_period=180 # Project 6 months forward
    )
    
    # Assuming run_business_synthesis exists in your engine
    projection = engine.run_monetary_projection(data, biz_case)
    
    # Check that the projected revenue is greater than the baseline
    # (Higher CR * Higher AOV should result in significant projected gains)
    assert projection['projected_incremental_revenue'] > 0
    assert "Cumulative Growth" in projection['summary']
