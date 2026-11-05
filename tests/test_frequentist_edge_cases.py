"""
Regression tests for edge-case bugs in the frequentist engine:
zero-conversion control, zero standard error, Lin adjustment inference,
tail-probability underflow, and the φ degrees-of-freedom correction.
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from foe.core.models import AlternativeHypothesis, ExperimentInput
from foe.frequentist.confidence import compute_newcombe_interval, compute_non_inferiority
from foe.frequentist.operations import FrequentistEngine, apply_sidak

GREATER = AlternativeHypothesis.GREATER
LESS = AlternativeHypothesis.LESS
TWO_SIDED = AlternativeHypothesis.TWO_SIDED


@pytest.fixture
def engine():
    return FrequentistEngine()


def _run(engine, visitors, conversions, **kw):
    return engine.run_synthesis(
        ExperimentInput(visitors=visitors, conversions=conversions, **kw)
    )


# --------------------------------------------------------------------- #
#  Zero-conversion control
# --------------------------------------------------------------------- #


def test_zero_control_winner_is_positive_with_undefined_uplift(engine):
    r = _run(engine, [1000, 1000], [0, 30])[0]
    assert r.is_significant
    assert r.uplift is None
    assert r.conclusion.startswith("Significant Positive Impact")
    assert "undefined relative impact" in r.conclusion
    assert r.ci_diff[0] > 0
    assert r.warning is None


def test_zero_control_flat_result(engine):
    r = _run(engine, [1000, 1000], [0, 1])[0]
    assert not r.is_significant
    assert r.uplift is None


def test_conclusion_direction_from_absolute_diff():
    msg = FrequentistEngine.generate_conclusion_statement(
        "B", True, None, absolute_diff=0.03
    )
    assert msg.startswith("Significant Positive Impact")
    msg = FrequentistEngine.generate_conclusion_statement(
        "B", True, None, false_positive_risk=0.05, absolute_diff=-0.03
    )
    assert msg.startswith("Significant Negative Impact")


def test_conclusion_requires_direction_when_lift_undefined():
    with pytest.raises(ValueError, match="absolute_diff"):
        FrequentistEngine.generate_conclusion_statement("B", True, None)


# --------------------------------------------------------------------- #
#  Zero standard error (degenerate arms)
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("conversions", [[0, 0], [1000, 1000]])
def test_identical_degenerate_arms(engine, conversions):
    r = _run(engine, [1000, 1000], conversions)[0]
    assert r.p_value == pytest.approx(1.0)
    assert not r.is_significant
    lo, hi = r.ci_diff
    assert lo < 0 < hi  # not a zero-width point
    assert r.warning is not None


def test_opposite_degenerate_arms_are_significant(engine):
    r = _run(engine, [1000, 1000], [0, 1000])[0]
    assert r.is_significant
    assert r.p_value < 1e-10
    assert r.conclusion.startswith("Significant Positive Impact")
    assert r.ci_diff[0] > 0.99
    assert r.ci_diff[1] <= 1.0
    assert r.warning is not None


def test_opposite_degenerate_tiny_arms_not_overconfident(engine):
    """0/2 vs 2/2 should not be treated as near-certain evidence."""
    r = _run(engine, [2, 2], [0, 2])[0]
    assert r.p_value > 0.01


def test_newcombe_interval_one_sided():
    lo, hi = compute_newcombe_interval(0, 100, 10, 100, 0.05, GREATER)
    assert hi == float("inf") and lo > 0
    lo, hi = compute_newcombe_interval(10, 100, 0, 100, 0.05, LESS)
    assert lo == float("-inf") and hi < 0


# --------------------------------------------------------------------- #
#  Tail-probability precision
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("z", [8.5, 10.0, 20.0])
def test_ztest_large_z_does_not_underflow(z):
    assert FrequentistEngine.run_ztest(z, 1.0, GREATER) == pytest.approx(norm.sf(z))
    assert FrequentistEngine.run_ztest(z, 1.0, TWO_SIDED) == pytest.approx(
        2 * norm.sf(z)
    )
    assert FrequentistEngine.run_ztest(z, 1.0, GREATER) > 0


def test_non_inferiority_large_z_does_not_underflow():
    r = compute_non_inferiority(p_ctrl=0.10, p_chal=0.20, se_diff=0.01, margin=0.01)
    assert r["p_value"] == pytest.approx(norm.sf(11.0))
    assert r["p_value"] > 0


# --------------------------------------------------------------------- #
#  Lin adjustment
# --------------------------------------------------------------------- #


def _lin_frame(labels=("control", "Alt"), n=4000, seed=1):
    rng = np.random.default_rng(seed)
    variant = rng.choice(list(labels), n)
    pre = rng.normal(size=n)
    effect = sum(0.02 * i * (variant == lab) for i, lab in enumerate(labels))
    prob = np.clip(0.10 + effect + 0.02 * np.clip(pre, -1, 1), 0.01, 0.99)
    converted = (rng.random(n) < prob).astype(float)
    return pd.DataFrame({"variant": variant, "pre": pre, "converted": converted})


def test_lin_ci_respects_alpha():
    df = _lin_frame()
    widths = []
    for a in (0.01, 0.05, 0.20):
        ci = FrequentistEngine.run_lin_adjustment(
            df, "converted", "pre", alpha=a, control_label="control"
        )[0]["ci"]
        widths.append(ci[1] - ci[0])
    assert widths[0] > widths[1] > widths[2]


def test_lin_explicit_control_label():
    df = _lin_frame()
    res = FrequentistEngine.run_lin_adjustment(
        df, "converted", "pre", control_label="control"
    )
    assert [r["variant"] for r in res] == ["Alt"]
    assert res[0]["control"] == "control"
    assert res[0]["absolute_lift"] > 0


def test_lin_unknown_control_label_raises():
    with pytest.raises(ValueError, match="control_label"):
        FrequentistEngine.run_lin_adjustment(
            _lin_frame(), "converted", "pre", control_label="nope"
        )


def test_lin_two_sided_p_matches_statsmodels():
    import statsmodels.formula.api as smf

    df = _lin_frame()
    res = FrequentistEngine.run_lin_adjustment(
        df, "converted", "pre", control_label="control"
    )[0]
    d = df.assign(cov_centered=df["pre"] - df["pre"].mean())
    model = smf.ols(
        "converted ~ C(variant, Treatment(reference='control')) * cov_centered", data=d
    ).fit(cov_type="HC3")
    term = "C(variant, Treatment(reference='control'))[T.Alt]"
    assert res["p_value"] == pytest.approx(model.pvalues[term])
    assert res["ci"] == pytest.approx(model.conf_int(alpha=0.05).loc[term].tolist())


def test_lin_one_sided():
    df = _lin_frame()
    two = FrequentistEngine.run_lin_adjustment(
        df, "converted", "pre", control_label="control"
    )[0]
    greater = FrequentistEngine.run_lin_adjustment(
        df, "converted", "pre", control_label="control", alternative=GREATER
    )[0]
    assert greater["p_value"] == pytest.approx(two["p_value"] / 2)
    assert math.isinf(greater["ci"][1])


def test_lin_applies_sidak_for_three_variants():
    df = _lin_frame(labels=("control", "B", "C"))
    res = FrequentistEngine.run_lin_adjustment(
        df, "converted", "pre", alpha=0.05, control_label="control"
    )
    assert len(res) == 2
    assert all(r["alpha"] == pytest.approx(apply_sidak(0.05, 3)) for r in res)


# --------------------------------------------------------------------- #
#  φ degrees-of-freedom correction
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("date_col", [None, "date"])
def test_phi_unbiased_under_binomial_null_with_uneven_traffic(date_col):
    """Mean φ ≈ 1 under a pure binomial null, even with campaign spike days."""
    rng = np.random.default_rng(7)
    days = 28
    traffic = np.r_[np.full(days - 1, 2000), [80000]]
    dates = pd.date_range("2024-01-01", periods=days)
    phis = [
        FrequentistEngine.calculate_aggregate_variance_factor(
            pd.DataFrame(
                {"date": dates, "visitors": traffic, "conversions": rng.binomial(traffic, 0.03)}
            ),
            date_col=date_col,
        )["phi"]
        for _ in range(1500)
    ]
    assert np.mean(phis) == pytest.approx(1.0, abs=0.05)
