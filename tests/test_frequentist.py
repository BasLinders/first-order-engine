import pytest
import numpy as np
from axiom.frequentist.operations import FrequentistEngine
from axiom.core.models import ExperimentInput, AlternativeHypothesis

@pytest.fixture
def engine():
    return FrequentistEngine()

def test_identical_variants_yield_p_value_one(engine):
    """
    If Control and Treatment are identical, p-value should be 1.0 
    and uplift should be 0.0.
    """
    data = ExperimentInput(
        visitors=[1000, 1000],
        conversions=[100, 100],
        labels=["Control", "Challenger"]
    )
    
    results = engine.run_synthesis(data)
    challenger = results[0]
    
    assert challenger.p_value == pytest.approx(1.0)
    assert challenger.uplift == 0.0
    assert challenger.is_significant is False

def test_significant_winner_detection(engine):
    """
    Test a clear winner scenario.
    """
    # 10% CR vs 15% CR with high sample size
    data = ExperimentInput(
        visitors=[2000, 2000],
        conversions=[200, 300],
        labels=["Control", "Winner"]
    )
    
    results = engine.run_synthesis(data)
    winner = results[0]
    
    assert winner.conversion_rate == 0.15
    assert winner.uplift == pytest.approx(0.5) # (15-10)/10
    assert winner.p_value < 0.01
    assert winner.is_significant is True
    assert "Significant Positive Impact" in winner.conclusion

def test_one_sided_hypothesis(engine):
    """
    Verify that 'greater' alternative hypothesis produces 
    a different p-value than two-sided.
    """
    visitors = [1000, 1000]
    conversions = [100, 120]
    
    two_sided = engine.run_synthesis(ExperimentInput(
        visitors=visitors, conversions=conversions, 
        alternative=AlternativeHypothesis.TWO_SIDED
    ))[0]
    
    one_sided = engine.run_synthesis(ExperimentInput(
        visitors=visitors, conversions=conversions, 
        alternative=AlternativeHypothesis.GREATER
    ))[0]
    
    # In a winning scenario, one-sided p-value should be exactly half of two-sided
    assert one_sided.p_value == pytest.approx(two_sided.p_value / 2)

def test_input_validation_integration():
    """
    Ensure the core validators.py logic correctly raises errors 
    when accessed through the ExperimentInput model.
    """
    with pytest.raises(ValueError, match="Conversions .* exceed visitors"):
        ExperimentInput(
            visitors=[100, 100],
            conversions=[150, 50] # Impossible!
        )

    with pytest.raises(ValueError, match="at least two variants"):
        ExperimentInput(
            visitors=[100],
            conversions=[10]
        )

def test_cuped_reduction_adjustment(engine):
    """
    Verify that providing a reduction_factor < 1 reduces the 
    standard error and tightens the confidence intervals.
    """
    visitors = [5000, 5000]
    conversions = [500, 550]
    
    standard = engine.run_synthesis(ExperimentInput(
        visitors=visitors, conversions=conversions, reduction_factor=1.0
    ))[0]
    
    cuped = engine.run_synthesis(ExperimentInput(
        visitors=visitors, conversions=conversions, reduction_factor=0.8
    ))[0]
    
    # CUPED standard error should be lower
    assert cuped.standard_error < standard.standard_error
    
    # Confidence interval should be narrower
    std_width = standard.ci_diff[1] - standard.ci_diff[0]
    cuped_width = cuped.ci_diff[1] - cuped.ci_diff[0]
    assert cuped_width < std_width
