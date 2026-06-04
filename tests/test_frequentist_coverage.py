import numpy as np
import pandas as pd
import pytest

from foe.frequentist.operations import FrequentistEngine, apply_sidak
from foe.frequentist.confidence import (
    compute_interval_difference,
    compute_non_inferiority,
)
from foe.core.models import AlternativeHypothesis


def test_conclusion_default_path_negative_significant():
    # No FPR supplied -> original wording for a significant negative result.
    msg = FrequentistEngine.generate_conclusion_statement("B", True, -0.2)
    assert "Significant Negative Impact" in msg
    assert "confidently discard" in msg


def test_conclusion_default_path_positive_significant():
    msg = FrequentistEngine.generate_conclusion_statement("B", True, 0.2)
    assert "confidently roll this out" in msg


def test_compute_interval_difference_zero_se_collapses_to_point():
    lo, hi = compute_interval_difference(0.01, 0.0, 0.05, AlternativeHypothesis.TWO_SIDED)
    assert lo == 0.01 and hi == 0.01


def test_compute_non_inferiority_zero_se_is_invalid():
    r = compute_non_inferiority(p_ctrl=0.10, p_chal=0.10, se_diff=0.0, margin=0.02, alpha=0.05)
    assert r["is_non_inferior"] is False
    assert "Invalid data" in r["conclusion"]


# --------------------------------------------------------------------------- #
#  apply_sidak edges
# --------------------------------------------------------------------------- #

def test_apply_sidak_two_variants_is_identity():
    assert apply_sidak(0.05, 2) == 0.05


def test_apply_sidak_single_comparison_guard():
    # num_comparisons <= 1 returns alpha unchanged
    assert apply_sidak(0.05, 1) == 0.05


def test_apply_sidak_three_variants_tightens():
    assert apply_sidak(0.05, 3) == pytest.approx(1 - 0.95 ** 0.5)


# --------------------------------------------------------------------------- #
#  Risk-aware conclusion wording
# --------------------------------------------------------------------------- #

def test_conclusion_not_significant_with_risk_returns_flat():
    msg = FrequentistEngine.generate_conclusion_statement("B", False, 0.0, false_positive_risk=0.5)
    assert "Inconclusive / Flat" in msg


def test_conclusion_positive_low_risk():
    msg = FrequentistEngine.generate_conclusion_statement("B", True, 0.2, false_positive_risk=0.04)
    assert "Significant Positive Impact" in msg
    assert "well supported" in msg


def test_conclusion_positive_high_risk_is_provisional():
    msg = FrequentistEngine.generate_conclusion_statement("B", True, 0.2, false_positive_risk=0.40)
    assert "provisional" in msg


def test_conclusion_negative_low_risk():
    msg = FrequentistEngine.generate_conclusion_statement("B", True, -0.2, false_positive_risk=0.04)
    assert "Significant Negative Impact" in msg
    assert "well supported" in msg


def test_conclusion_negative_high_risk_confirm():
    msg = FrequentistEngine.generate_conclusion_statement("B", True, -0.2, false_positive_risk=0.40)
    assert "confirm before discarding" in msg


# --------------------------------------------------------------------------- #
#  resolve_prior_probability
# --------------------------------------------------------------------------- #

def test_resolve_prior_presets():
    assert FrequentistEngine.resolve_prior_probability("skeptical") == 0.10
    assert FrequentistEngine.resolve_prior_probability("neutral") == 0.50
    assert FrequentistEngine.resolve_prior_probability("optimistic") == 0.90


def test_resolve_prior_custom_value():
    assert FrequentistEngine.resolve_prior_probability("custom", 0.33) == 0.33


def test_resolve_prior_custom_missing_raises():
    with pytest.raises(ValueError, match="must be provided"):
        FrequentistEngine.resolve_prior_probability("custom")


def test_resolve_prior_custom_out_of_range_raises():
    with pytest.raises(ValueError, match="between 0 and 1"):
        FrequentistEngine.resolve_prior_probability("custom", 1.5)


def test_resolve_prior_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown sensitivity_mode"):
        FrequentistEngine.resolve_prior_probability("wishful")


# --------------------------------------------------------------------------- #
#  FPR / FNDR
# --------------------------------------------------------------------------- #

def test_fpr_known_value():
    assert FrequentistEngine.calculate_false_positive_risk(0.05, 0.80, 0.50) == pytest.approx(0.0588, abs=1e-4)


def test_fndr_known_value():
    assert FrequentistEngine.calculate_false_negative_discovery_rate(0.05, 0.80, 0.50) == pytest.approx(0.1739, abs=1e-4)


def test_fpr_prior_zero_is_certain_false_positive():
    assert FrequentistEngine.calculate_false_positive_risk(0.05, 0.80, 0.0) == 1.0


def test_fndr_prior_one_is_certain_miss():
    assert FrequentistEngine.calculate_false_negative_discovery_rate(0.05, 0.80, 1.0) == 1.0


def test_fpr_zero_denominator_returns_zero():
    # alpha=0 and prior=0 -> denominator 0 -> guarded 0.0
    assert FrequentistEngine.calculate_false_positive_risk(0.0, 0.0, 0.0) == 0.0


def test_fndr_zero_denominator_returns_zero():
    # power=1 (beta=0) and prior=1 -> denominator 0 -> guarded 0.0
    assert FrequentistEngine.calculate_false_negative_discovery_rate(0.0, 1.0, 1.0) == 0.0


# --------------------------------------------------------------------------- #
#  assess_decision_risk (all tails, both branches)
# --------------------------------------------------------------------------- #

def test_assess_significant_two_sided():
    r = FrequentistEngine.assess_decision_risk(True, 0.05, 0.8, 0.5, AlternativeHypothesis.TWO_SIDED)
    assert r["metric"] == "false_positive_risk"
    assert "detected difference" in r["label"]
    assert r["elevated"] is False


def test_assess_significant_greater():
    r = FrequentistEngine.assess_decision_risk(True, 0.05, 0.8, 0.5, AlternativeHypothesis.GREATER)
    assert "apparent improvement" in r["label"]


def test_assess_significant_less():
    r = FrequentistEngine.assess_decision_risk(True, 0.05, 0.8, 0.5, AlternativeHypothesis.LESS)
    assert "apparent harm" in r["label"]


def test_assess_not_significant_greater():
    r = FrequentistEngine.assess_decision_risk(False, 0.05, 0.1, 0.5, AlternativeHypothesis.GREATER)
    assert r["metric"] == "false_negative_discovery_rate"
    assert "real improvement" in r["label"]
    assert r["elevated"] is True


def test_assess_not_significant_less():
    r = FrequentistEngine.assess_decision_risk(False, 0.05, 0.1, 0.5, AlternativeHypothesis.LESS)
    assert "real harm" in r["label"]


def test_assess_not_significant_two_sided():
    r = FrequentistEngine.assess_decision_risk(False, 0.05, 0.1, 0.5, AlternativeHypothesis.TWO_SIDED)
    assert "a real difference" in r["label"]


# --------------------------------------------------------------------------- #
#  calculate_aggregate_variance_factor
# --------------------------------------------------------------------------- #

def _clean_daily(days=28, p=0.04, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.integers(800, 1300, days)
    c = rng.binomial(v, p)
    return pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=days), "visitors": v, "conversions": c}
    )


def test_aggregate_variance_clean_near_one():
    out = FrequentistEngine.calculate_aggregate_variance_factor(_clean_daily())
    assert 0.5 < out["phi"] < 2.0
    assert out["dow_controlled"] is False
    assert out["clipped"] is False
    assert {"reduction_factor", "phi", "regime", "n_periods",
            "n_effective", "dow_controlled", "clipped"}.issubset(out)


def test_aggregate_variance_dow_controlled_flag():
    out = FrequentistEngine.calculate_aggregate_variance_factor(_clean_daily(), date_col="date")
    assert out["dow_controlled"] is True


def test_aggregate_variance_dow_control_lowers_seasonal_phi():
    days = 28
    rng = np.random.default_rng(3)
    v = rng.integers(800, 1300, days)
    dow = pd.date_range("2024-01-01", periods=days).dayofweek.to_numpy()
    c = rng.binomial(v, np.where(dow >= 5, 0.02, 0.05))
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=days), "visitors": v, "conversions": c})
    phi_plain = FrequentistEngine.calculate_aggregate_variance_factor(df)["phi"]
    phi_dow = FrequentistEngine.calculate_aggregate_variance_factor(df, date_col="date")["phi"]
    assert phi_dow < phi_plain


def test_aggregate_variance_upper_clip_and_flag():
    df = _clean_daily()
    spike = df["conversions"].to_numpy().copy()
    spike[5] = int(df["visitors"].iloc[5])  # one fully-converting day
    df = df.assign(conversions=spike)
    out = FrequentistEngine.calculate_aggregate_variance_factor(df)
    assert out["clipped"] is True
    assert out["reduction_factor"] == 5.0
    assert out["regime"] == "high_noise"


def test_aggregate_variance_min_periods_raises():
    with pytest.raises(ValueError, match="daily rows are required"):
        FrequentistEngine.calculate_aggregate_variance_factor(_clean_daily(days=10))


def test_aggregate_variance_nonpositive_visitors_raises():
    df = _clean_daily()
    df.loc[0, "visitors"] = 0
    with pytest.raises(ValueError, match="zero or negative"):
        FrequentistEngine.calculate_aggregate_variance_factor(df)


def test_aggregate_variance_negative_conversions_raises():
    df = _clean_daily()
    df.loc[0, "conversions"] = -1
    with pytest.raises(ValueError, match="negative values"):
        FrequentistEngine.calculate_aggregate_variance_factor(df)


def test_aggregate_variance_conversions_exceed_raises():
    df = _clean_daily()
    df.loc[0, "conversions"] = int(df.loc[0, "visitors"]) + 1
    with pytest.raises(ValueError, match="exceeds"):
        FrequentistEngine.calculate_aggregate_variance_factor(df)


def test_aggregate_variance_zero_binomial_var_returns_neutral():
    # All days fully convert -> p(1-p)=0 -> expected_binomial_var == 0 branch.
    df = pd.DataFrame({"visitors": [1000] * 14, "conversions": [1000] * 14})
    out = FrequentistEngine.calculate_aggregate_variance_factor(df)
    assert out["phi"] == 1.0
    assert out["regime"] == "neutral"
    assert out["clipped"] is False


# --------------------------------------------------------------------------- #
#  Inference primitives: one-sided power, ztest tails, bootstrap unpooled
# --------------------------------------------------------------------------- #

def test_run_ztest_greater_and_less_branches():
    g = FrequentistEngine.run_ztest(0.02, 0.01, AlternativeHypothesis.GREATER)
    l = FrequentistEngine.run_ztest(0.02, 0.01, AlternativeHypothesis.LESS)
    assert 0.0 < g < 0.5      # positive effect -> small upper-tail p
    assert 0.5 < l < 1.0      # positive effect -> large lower-tail p
    assert g == pytest.approx(1 - l)


def test_analytical_power_one_sided_branch():
    power = FrequentistEngine.calculate_analytical_power(
        0.03, 0.01, 0.05, AlternativeHypothesis.GREATER
    )
    assert 0.0 < power <= 1.0


def test_bootstrap_power_unpooled_se_path():
    p = FrequentistEngine.run_vectorized_bootstrap_power(
        ctrl_conv=100, ctrl_n=1000, chal_conv=140, chal_n=1000,
        alpha=0.05, n_bootstraps=3000,
        alternative=AlternativeHypothesis.TWO_SIDED, use_pooled_se=False,
    )
    assert 0.0 <= p <= 1.0


def test_bootstrap_power_zero_n_returns_zero():
    assert FrequentistEngine.run_vectorized_bootstrap_power(
        ctrl_conv=0, ctrl_n=0, chal_conv=10, chal_n=100
    ) == 0.0


# --------------------------------------------------------------------------- #
#  Variance reduction: CUPED / Lin (previously untested)
# --------------------------------------------------------------------------- #

def test_apply_cuped_reduces_variance_when_correlated():
    rng = np.random.default_rng(7)
    pre = rng.normal(10, 2, 600)
    y = 0.7 * pre + rng.normal(0, 1, 600)
    df = pd.DataFrame({"y": y, "pre": pre})
    out = FrequentistEngine.apply_cuped(df, "y", "pre")
    assert "y_cuped" in out.columns
    assert out["y_cuped"].var() < df["y"].var()


def test_apply_cuped_zero_variance_covariate_is_noop():
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "pre": [5.0, 5.0, 5.0, 5.0]})
    out = FrequentistEngine.apply_cuped(df, "y", "pre")
    assert (out["y_cuped"] == out["y"]).all()


def test_run_lin_adjustment_returns_per_challenger_results():
    # NOTE: run_lin_adjustment uses sorted(variants)[0] as the baseline, so the
    # control label must sort first. "Control" < "Treatment" alphabetically;
    # a label like "B" would (surprisingly) sort ahead of "Control".
    rng = np.random.default_rng(11)
    n = 600
    variant = np.array(["Control"] * 300 + ["Treatment"] * 300)
    pre = rng.normal(0.0, 1.0, n)
    prob = np.clip(0.20 + 0.05 * (variant == "Treatment") + 0.05 * pre, 0.01, 0.99)
    converted = (rng.random(n) < prob).astype(int)
    df = pd.DataFrame({"variant": variant, "converted": converted, "pre": pre})

    res = FrequentistEngine.run_lin_adjustment(df, "converted", "pre")
    assert len(res) == 1
    row = res[0]
    assert row["variant"] == "Treatment"
    assert {"p_value", "is_significant", "absolute_lift",
            "relative_lift", "ci", "std_err"}.issubset(row)
