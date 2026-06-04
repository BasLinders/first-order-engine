import numpy as np
import pandas as pd
import pytest

from foe.frequentist.operations import FrequentistEngine, apply_sidak
from foe.frequentist.confidence import (
    compute_interval_difference,
    compute_non_inferiority,
)
from foe.core.models import AlternativeHypothesis

Engine = FrequentistEngine
TWO_SIDED = AlternativeHypothesis.TWO_SIDED
GREATER = AlternativeHypothesis.GREATER
LESS = AlternativeHypothesis.LESS


# --------------------------------------------------------------------- #
#  apply_sidak edges
# --------------------------------------------------------------------- #


def test_apply_sidak_two_variants_is_identity():
    """Two variants => no correction; alpha is returned unchanged."""
    assert apply_sidak(0.05, 2) == 0.05


def test_apply_sidak_single_comparison_guard():
    """num_comparisons <= 1 returns alpha unchanged."""
    assert apply_sidak(0.05, 1) == 0.05


def test_apply_sidak_three_variants_tightens():
    """Three variants tighten alpha below the raw level."""
    assert apply_sidak(0.05, 3) == pytest.approx(1 - 0.95**0.5)


# --------------------------------------------------------------------- #
#  Risk-aware conclusion wording
# --------------------------------------------------------------------- #


def test_conclusion_not_significant_with_risk_returns_flat():
    """A non-significant result ignores FPR and returns the flat text."""
    msg = Engine.generate_conclusion_statement("B", False, 0.0, false_positive_risk=0.5)
    assert "Inconclusive / Flat" in msg


def test_conclusion_positive_low_risk():
    """Low FPR keeps a positive winner framed as well supported."""
    msg = Engine.generate_conclusion_statement("B", True, 0.2, false_positive_risk=0.04)
    assert "Significant Positive Impact" in msg
    assert "well supported" in msg


def test_conclusion_positive_high_risk_is_provisional():
    """A high FPR on a winner downgrades the wording to provisional."""
    msg = Engine.generate_conclusion_statement("B", True, 0.2, false_positive_risk=0.40)
    assert "provisional" in msg


def test_conclusion_negative_low_risk():
    """Low FPR on a loser frames discarding as well supported."""
    msg = Engine.generate_conclusion_statement(
        "B", True, -0.2, false_positive_risk=0.04
    )
    assert "Significant Negative Impact" in msg
    assert "well supported" in msg


def test_conclusion_negative_high_risk_confirm():
    """A high FPR on a loser asks for confirmation before discarding."""
    msg = Engine.generate_conclusion_statement(
        "B", True, -0.2, false_positive_risk=0.40
    )
    assert "confirm before discarding" in msg


def test_conclusion_default_path_positive_significant():
    """No FPR supplied reproduces the original winner wording."""
    msg = Engine.generate_conclusion_statement("B", True, 0.2)
    assert "confidently roll this out" in msg


def test_conclusion_default_path_negative_significant():
    """No FPR supplied reproduces the original loser wording."""
    msg = Engine.generate_conclusion_statement("B", True, -0.2)
    assert "confidently discard" in msg


# --------------------------------------------------------------------- #
#  resolve_prior_probability
# --------------------------------------------------------------------- #


def test_resolve_prior_presets():
    """Named presets map to their documented probabilities."""
    assert Engine.resolve_prior_probability("skeptical") == 0.10
    assert Engine.resolve_prior_probability("neutral") == 0.50
    assert Engine.resolve_prior_probability("optimistic") == 0.90


def test_resolve_prior_custom_value():
    """A custom prior in range is returned directly."""
    assert Engine.resolve_prior_probability("custom", 0.33) == 0.33


def test_resolve_prior_custom_missing_raises():
    """Custom mode without a value raises."""
    with pytest.raises(ValueError, match="must be provided"):
        Engine.resolve_prior_probability("custom")


def test_resolve_prior_custom_out_of_range_raises():
    """A custom prior outside [0, 1] raises."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        Engine.resolve_prior_probability("custom", 1.5)


def test_resolve_prior_unknown_mode_raises():
    """An unrecognised mode raises."""
    with pytest.raises(ValueError, match="Unknown sensitivity_mode"):
        Engine.resolve_prior_probability("wishful")


# --------------------------------------------------------------------- #
#  FPR / FNDR
# --------------------------------------------------------------------- #


def test_fpr_known_value():
    """FPR matches the hand-computed Bayes value."""
    fpr = Engine.calculate_false_positive_risk(0.05, 0.80, 0.50)
    assert fpr == pytest.approx(0.0588, abs=1e-4)


def test_fndr_known_value():
    """FNDR matches the hand-computed Bayes value."""
    fndr = Engine.calculate_false_negative_discovery_rate(0.05, 0.80, 0.50)
    assert fndr == pytest.approx(0.1739, abs=1e-4)


def test_fpr_prior_zero_is_certain_false_positive():
    """With no real effects possible, any hit is a false positive."""
    assert Engine.calculate_false_positive_risk(0.05, 0.80, 0.0) == 1.0


def test_fndr_prior_one_is_certain_miss():
    """If an effect always exists, a null result always missed it."""
    fndr = Engine.calculate_false_negative_discovery_rate(0.05, 0.80, 1.0)
    assert fndr == 1.0


def test_fpr_zero_denominator_returns_zero():
    """Degenerate inputs hit the guarded zero-denominator branch."""
    assert Engine.calculate_false_positive_risk(0.0, 0.0, 0.0) == 0.0


def test_fndr_zero_denominator_returns_zero():
    """Degenerate inputs hit the guarded zero-denominator branch."""
    fndr = Engine.calculate_false_negative_discovery_rate(0.0, 1.0, 1.0)
    assert fndr == 0.0


# --------------------------------------------------------------------- #
#  assess_decision_risk (all tails, both branches)
# --------------------------------------------------------------------- #


def test_assess_significant_two_sided():
    """A significant two-sided result reports false-positive risk."""
    r = Engine.assess_decision_risk(True, 0.05, 0.8, 0.5, TWO_SIDED)
    assert r["metric"] == "false_positive_risk"
    assert "detected difference" in r["label"]
    assert r["elevated"] is False


def test_assess_significant_greater():
    """A significant GREATER result labels apparent improvement."""
    r = Engine.assess_decision_risk(True, 0.05, 0.8, 0.5, GREATER)
    assert "apparent improvement" in r["label"]


def test_assess_significant_less():
    """A significant LESS result labels apparent harm."""
    r = Engine.assess_decision_risk(True, 0.05, 0.8, 0.5, LESS)
    assert "apparent harm" in r["label"]


def test_assess_not_significant_greater():
    """A flat GREATER result reports the false-negative rate."""
    r = Engine.assess_decision_risk(False, 0.05, 0.1, 0.5, GREATER)
    assert r["metric"] == "false_negative_discovery_rate"
    assert "real improvement" in r["label"]
    assert r["elevated"] is True


def test_assess_not_significant_less():
    """A flat LESS result labels missed harm."""
    r = Engine.assess_decision_risk(False, 0.05, 0.1, 0.5, LESS)
    assert "real harm" in r["label"]


def test_assess_not_significant_two_sided():
    """A flat two-sided result labels a missed difference."""
    r = Engine.assess_decision_risk(False, 0.05, 0.1, 0.5, TWO_SIDED)
    assert "a real difference" in r["label"]


# --------------------------------------------------------------------- #
#  calculate_aggregate_variance_factor
# --------------------------------------------------------------------- #


def _clean_daily(days=28, p=0.04, seed=0):
    """Build a clean daily aggregate frame for dispersion tests."""
    rng = np.random.default_rng(seed)
    v = rng.integers(800, 1300, days)
    c = rng.binomial(v, p)
    dates = pd.date_range("2024-01-01", periods=days)
    return pd.DataFrame({"date": dates, "visitors": v, "conversions": c})


def test_aggregate_variance_clean_near_one():
    """Clean binomial traffic yields phi near 1 with the full key set."""
    out = Engine.calculate_aggregate_variance_factor(_clean_daily())
    assert 0.5 < out["phi"] < 2.0
    assert out["dow_controlled"] is False
    assert out["clipped"] is False
    expected_keys = {
        "reduction_factor",
        "phi",
        "regime",
        "n_periods",
        "n_effective",
        "dow_controlled",
        "clipped",
    }
    assert expected_keys.issubset(out)


def test_aggregate_variance_dow_controlled_flag():
    """Passing date_col enables day-of-week control."""
    out = Engine.calculate_aggregate_variance_factor(_clean_daily(), date_col="date")
    assert out["dow_controlled"] is True


def test_aggregate_variance_dow_control_lowers_seasonal_phi():
    """Day-of-week control removes weekly seasonality from phi."""
    days = 28
    rng = np.random.default_rng(3)
    v = rng.integers(800, 1300, days)
    dates = pd.date_range("2024-01-01", periods=days)
    dow = dates.dayofweek.to_numpy()
    c = rng.binomial(v, np.where(dow >= 5, 0.02, 0.05))
    df = pd.DataFrame({"date": dates, "visitors": v, "conversions": c})
    phi_plain = Engine.calculate_aggregate_variance_factor(df)["phi"]
    phi_dow = Engine.calculate_aggregate_variance_factor(df, date_col="date")["phi"]
    assert phi_dow < phi_plain


def test_aggregate_variance_upper_clip_and_flag():
    """A single fully-converting day is clipped and flagged."""
    df = _clean_daily()
    spike = df["conversions"].to_numpy().copy()
    spike[5] = int(df["visitors"].iloc[5])
    df = df.assign(conversions=spike)
    out = Engine.calculate_aggregate_variance_factor(df)
    assert out["clipped"] is True
    assert out["reduction_factor"] == 5.0
    assert out["regime"] == "high_noise"


def test_aggregate_variance_min_periods_raises():
    """Too few daily rows raises."""
    with pytest.raises(ValueError, match="daily rows are required"):
        Engine.calculate_aggregate_variance_factor(_clean_daily(days=10))


def test_aggregate_variance_nonpositive_visitors_raises():
    """Zero or negative visitor counts raise."""
    df = _clean_daily()
    df.loc[0, "visitors"] = 0
    with pytest.raises(ValueError, match="zero or negative"):
        Engine.calculate_aggregate_variance_factor(df)


def test_aggregate_variance_negative_conversions_raises():
    """Negative conversion counts raise."""
    df = _clean_daily()
    df.loc[0, "conversions"] = -1
    with pytest.raises(ValueError, match="negative values"):
        Engine.calculate_aggregate_variance_factor(df)


def test_aggregate_variance_conversions_exceed_raises():
    """Conversions exceeding visitors raise."""
    df = _clean_daily()
    df.loc[0, "conversions"] = int(df.loc[0, "visitors"]) + 1
    with pytest.raises(ValueError, match="exceeds"):
        Engine.calculate_aggregate_variance_factor(df)


def test_aggregate_variance_zero_binomial_var_returns_neutral():
    """All-converting days hit the zero-binomial-variance branch."""
    df = pd.DataFrame({"visitors": [1000] * 14, "conversions": [1000] * 14})
    out = Engine.calculate_aggregate_variance_factor(df)
    assert out["phi"] == 1.0
    assert out["regime"] == "neutral"
    assert out["clipped"] is False


# --------------------------------------------------------------------- #
#  Inference primitives
# --------------------------------------------------------------------- #


def test_run_ztest_greater_and_less_branches():
    """GREATER and LESS p-values are complementary tails."""
    greater = Engine.run_ztest(0.02, 0.01, GREATER)
    lesser = Engine.run_ztest(0.02, 0.01, LESS)
    assert 0.0 < greater < 0.5
    assert 0.5 < lesser < 1.0
    assert greater == pytest.approx(1 - lesser)


def test_analytical_power_one_sided_branch():
    """One-sided power exercises the single-tail z_alpha path."""
    power = Engine.calculate_analytical_power(0.03, 0.01, 0.05, GREATER)
    assert 0.0 < power <= 1.0


def test_bootstrap_power_unpooled_se_path():
    """The unpooled-SE bootstrap path returns a valid proportion."""
    p = Engine.run_vectorized_bootstrap_power(
        ctrl_conv=100,
        ctrl_n=1000,
        chal_conv=140,
        chal_n=1000,
        alpha=0.05,
        n_bootstraps=3000,
        alternative=TWO_SIDED,
        use_pooled_se=False,
    )
    assert 0.0 <= p <= 1.0


def test_bootstrap_power_zero_n_returns_zero():
    """A zero sample size short-circuits to zero power."""
    power = Engine.run_vectorized_bootstrap_power(
        ctrl_conv=0, ctrl_n=0, chal_conv=10, chal_n=100
    )
    assert power == 0.0


# --------------------------------------------------------------------- #
#  Variance reduction: CUPED / Lin
# --------------------------------------------------------------------- #


def test_apply_cuped_reduces_variance_when_correlated():
    """CUPED lowers outcome variance when the covariate correlates."""
    rng = np.random.default_rng(7)
    pre = rng.normal(10, 2, 600)
    y = 0.7 * pre + rng.normal(0, 1, 600)
    df = pd.DataFrame({"y": y, "pre": pre})
    out = Engine.apply_cuped(df, "y", "pre")
    assert "y_cuped" in out.columns
    assert out["y_cuped"].var() < df["y"].var()


def test_apply_cuped_zero_variance_covariate_is_noop():
    """A constant covariate yields theta=0, leaving the outcome intact."""
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "pre": [5.0] * 4})
    out = Engine.apply_cuped(df, "y", "pre")
    assert (out["y_cuped"] == out["y"]).all()


def test_run_lin_adjustment_returns_per_challenger_results():
    """Lin adjustment returns one result row per challenger."""
    # Baseline is sorted(variants)[0]; "Control" < "Treatment", so the
    # control label must sort first. A label like "B" would sort ahead of
    # "Control" and silently become the baseline.
    rng = np.random.default_rng(11)
    n = 600
    variant = np.array(["Control"] * 300 + ["Treatment"] * 300)
    pre = rng.normal(0.0, 1.0, n)
    prob = 0.20 + 0.05 * (variant == "Treatment") + 0.05 * pre
    prob = np.clip(prob, 0.01, 0.99)
    converted = (rng.random(n) < prob).astype(int)
    df = pd.DataFrame({"variant": variant, "converted": converted, "pre": pre})
    res = Engine.run_lin_adjustment(df, "converted", "pre")
    assert len(res) == 1
    row = res[0]
    assert row["variant"] == "Treatment"
    expected_keys = {
        "p_value",
        "is_significant",
        "absolute_lift",
        "relative_lift",
        "ci",
        "std_err",
    }
    assert expected_keys.issubset(row)


# --------------------------------------------------------------------- #
#  confidence.py degenerate paths
# --------------------------------------------------------------------- #


def test_compute_interval_difference_zero_se_collapses_to_point():
    """A zero SE collapses the interval to the point estimate."""
    lo, hi = compute_interval_difference(0.01, 0.0, 0.05, TWO_SIDED)
    assert lo == 0.01
    assert hi == 0.01


def test_compute_non_inferiority_zero_se_is_invalid():
    """A non-positive SE is reported as invalid, not non-inferior."""
    r = compute_non_inferiority(
        p_ctrl=0.10, p_chal=0.10, se_diff=0.0, margin=0.02, alpha=0.05
    )
    assert r["is_non_inferior"] is False
    assert "Invalid data" in r["conclusion"]
