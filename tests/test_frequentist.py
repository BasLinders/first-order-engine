import pytest
from pydantic import ValidationError

from foe.frequentist.operations import FrequentistEngine
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
