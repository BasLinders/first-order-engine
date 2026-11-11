"""
Power for decision risk is evaluated at a pre-specified MDE, not at the
observed effect (where it is a function of the p-value and adds nothing).
"""

import pytest

from foe.core.models import AlternativeHypothesis, ExperimentInput
from foe.frequentist.operations import FrequentistEngine as Engine, apply_sidak

GREATER = AlternativeHypothesis.GREATER
LESS = AlternativeHypothesis.LESS
TWO_SIDED = AlternativeHypothesis.TWO_SIDED


def _power(**kw):
    args = dict(baseline_rate=0.05, n_ctrl=10000, n_chal=10000, mde_relative=0.10,
                alpha=0.05, alternative=TWO_SIDED)
    args.update(kw)
    return Engine.calculate_analytical_power(**args)


def test_power_matches_textbook_value():
    """n per arm from the standard formula should give ~80% power."""
    from scipy.stats import norm
    p0, p1 = 0.05, 0.055
    z = norm.ppf(0.975) + norm.ppf(0.80)
    n = int(z ** 2 * (p0 * (1 - p0) + p1 * (1 - p1)) / (p1 - p0) ** 2) + 1
    assert _power(n_ctrl=n, n_chal=n) == pytest.approx(0.80, abs=0.005)


def test_analytical_and_simulated_power_agree():
    analytical = _power(n_ctrl=20000, n_chal=20000)
    simulated = Engine.run_vectorized_bootstrap_power(
        baseline_rate=0.05, n_ctrl=20000, n_chal=20000, mde_relative=0.10,
        alpha=0.05, n_bootstraps=20000, random_state=3,
    )
    assert simulated == pytest.approx(analytical, abs=0.02)


def test_one_sided_beats_two_sided_and_less_points_down():
    assert _power(alternative=GREATER) > _power(alternative=TWO_SIDED)
    assert _power(alternative=LESS) > _power(alternative=TWO_SIDED)
    # LESS evaluates a decrease, GREATER an increase: similar but not identical
    assert _power(alternative=LESS) != _power(alternative=GREATER)


def test_power_drops_with_sidak_alpha_and_overdispersion():
    base = _power()
    assert _power(alpha=apply_sidak(0.05, 4)) < base
    assert _power(reduction_factor=2.0) < base


@pytest.mark.parametrize("kw", [
    {"mde_relative": 0.0},
    {"mde_relative": -0.1},
    {"baseline_rate": 0.0},
    {"baseline_rate": 0.95, "mde_relative": 0.10},
    {"baseline_rate": 0.05, "mde_relative": 1.5, "alternative": LESS},
])
def test_invalid_power_inputs_raise(kw):
    with pytest.raises(ValueError):
        _power(**kw)


def test_power_does_not_depend_on_observed_challenger():
    """Two tests with the same design but different outcomes get the same power."""
    engine = Engine()
    risks = []
    for chal_conv in (500, 540, 620):
        data = ExperimentInput(visitors=[10000, 10000], conversions=[500, chal_conv])
        result = engine.run_synthesis(data)[0]
        risks.append(Engine.assess_result_risk(
            data, result, mde_relative=0.10, prior=0.5))
    assert risks[0]["power"] == pytest.approx(risks[1]["power"])
    assert risks[1]["power"] == pytest.approx(risks[2]["power"])
    assert [r["metric"] for r in risks] == [
        "false_negative_discovery_rate",
        "false_negative_discovery_rate",
        "false_positive_risk",
    ]


def test_assess_result_risk_uses_sidak_alpha_and_label():
    data = ExperimentInput(
        visitors=[10000, 10000, 10000], conversions=[500, 520, 600],
        labels=["Control", "B", "C"],
    )
    results = Engine().run_synthesis(data)
    risk = Engine.assess_result_risk(data, results[1], mde_relative=0.10, prior=0.5)
    alpha = apply_sidak(0.05, 3)
    assert risk["alpha"] == pytest.approx(alpha)
    assert risk["mde_relative"] == 0.10
    expected_power = Engine.calculate_analytical_power(0.05, 10000, 10000, 0.10, alpha)
    assert risk["power"] == pytest.approx(expected_power)
    assert risk["value"] == pytest.approx(
        Engine.calculate_false_positive_risk(alpha, expected_power, 0.5)
    )


def test_assess_result_risk_rejects_foreign_result():
    data = ExperimentInput(visitors=[1000, 1000], conversions=[
                           50, 60], labels=["A", "B"])
    other = ExperimentInput(visitors=[1000, 1000], conversions=[
                            50, 60], labels=["A", "X"])
    result = Engine().run_synthesis(other)[0]
    with pytest.raises(ValueError, match="does not match"):
        Engine.assess_result_risk(data, result, mde_relative=0.1, prior=0.5)


def test_risk_wording_mentions_planned_effect():
    msg = Engine.generate_conclusion_statement("B", True, 0.2, false_positive_risk=0.40)
    assert "planned effect size" in msg
    assert "provisional" in msg
