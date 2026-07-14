import math

import numpy as np
import pandas as pd
import pytest

from foe.continuous.operations import ContinuousMetricEngine
from foe.core.models import (
    AlternativeHypothesis,
    AnalysisUnit,
    ContinuousApproach,
    ContinuousMetricConfig,
)

GROUP_COL = "experience_variant_label"


@pytest.fixture
def engine():
    return ContinuousMetricEngine()


def make_df(groups: dict, kpi: str = "revenue") -> pd.DataFrame:
    """Builds a long-format frame from {label: array-like} group data."""
    labels, values = [], []
    for label, arr in groups.items():
        arr = np.asarray(arr)
        labels.extend([label] * len(arr))
        values.append(arr)
    return pd.DataFrame({GROUP_COL: labels, kpi: np.concatenate(values)})


# --------------------------------------------------------------------- #
#  Smoke tests: run_comparison_suite, heuristic path
# --------------------------------------------------------------------- #


def test_heuristic_normal_homogeneous_uses_standard_anova(engine):
    """
    Normal residuals + equal variance across 3 groups should route to
    Standard ANOVA and detect the constructed 15-unit mean shift in group C.
    """
    rng = np.random.default_rng(0)
    n = 200
    df = make_df({
        "A": rng.normal(100, 15, n),
        "B": rng.normal(100, 15, n),
        "C": rng.normal(115, 15, n),
    })
    result = engine.run_comparison_suite(df, ContinuousMetricConfig(kpi="revenue"))

    assert result.test_name == "Standard ANOVA"
    assert result.is_normal is True
    assert result.is_homogeneous is True
    assert result.is_significant is True
    assert result.posthoc_results is None  # heuristic path never posthocs


def test_heuristic_identical_groups_not_significant(engine):
    """Two draws from the same normal distribution should not be significant."""
    rng = np.random.default_rng(0)
    n = 200
    a = rng.normal(100, 15, n)
    b = rng.normal(100, 15, n)
    df = make_df({"A": a, "B": b})
    result = engine.run_comparison_suite(df, ContinuousMetricConfig(kpi="revenue"))

    assert result.test_name == "Standard ANOVA"
    assert result.is_significant is False
    assert "No Significant Difference" in result.conclusion


def test_heuristic_normal_heterogeneous_uses_welch(engine):
    """
    Normal residuals with heterogeneous variance (verified: p_norm ~= 0.68,
    levene p ~= 1e-13 at seed=15) must route to Welch's ANOVA rather than
    crashing -- regression test for the aov["p-unc"]/aov["p_unc"] column
    name mismatch against the installed pingouin version.
    """
    rng = np.random.default_rng(15)
    n = 500
    df = make_df({
        "A": rng.normal(100, 10, n),
        "B": rng.normal(100, 15, n),
    })
    result = engine.run_comparison_suite(df, ContinuousMetricConfig(kpi="revenue"))

    assert result.test_name == "Welch's ANOVA"
    assert result.is_normal is True
    assert result.is_homogeneous is False
    assert 0.0 <= result.p_value <= 1.0


def test_heuristic_non_normal_two_groups_uses_mann_whitney(engine):
    """Skewed (exponential) data with 2 groups should route to Mann-Whitney U."""
    rng = np.random.default_rng(0)
    n = 200
    df = make_df({
        "A": rng.exponential(20, n),
        "B": rng.exponential(30, n),
    })
    result = engine.run_comparison_suite(df, ContinuousMetricConfig(kpi="revenue"))

    assert result.test_name == "Mann-Whitney U"
    assert result.is_normal is False
    assert result.is_significant is True


def test_heuristic_non_normal_three_groups_uses_kruskal_wallis(engine):
    """Skewed (exponential) data with 3 groups should route to Kruskal-Wallis."""
    rng = np.random.default_rng(0)
    n = 200
    df = make_df({
        "A": rng.exponential(20, n),
        "B": rng.exponential(20, n),
        "C": rng.exponential(35, n),
    })
    result = engine.run_comparison_suite(df, ContinuousMetricConfig(kpi="revenue"))

    assert result.test_name == "Kruskal-Wallis"
    assert result.is_significant is True


def test_heuristic_small_sample_skips_normality_test(engine):
    """
    With fewer than 20 residuals, normaltest's kurtosis component is
    unreliable, so the engine short-circuits to is_normal=False rather
    than calling normaltest.
    """
    rng = np.random.default_rng(0)
    df = make_df({"A": rng.normal(50, 5, 5), "B": rng.normal(50, 5, 5)})
    result = engine.run_comparison_suite(df, ContinuousMetricConfig(kpi="revenue"))
    assert result.is_normal is False


def test_heuristic_summary_stats_shape(engine):
    """summary_stats should carry one {label, mean, std, count} record per group."""
    rng = np.random.default_rng(0)
    n = 50
    df = make_df({"A": rng.normal(50, 5, n), "B": rng.normal(50, 5, n)})
    result = engine.run_comparison_suite(df, ContinuousMetricConfig(kpi="revenue"))

    labels = {row[GROUP_COL] for row in result.summary_stats}
    assert labels == {"A", "B"}
    for row in result.summary_stats:
        assert row["count"] == n
        assert {"mean", "std", "count"}.issubset(row)


# --------------------------------------------------------------------- #
#  Smoke tests: run_comparison_suite, Gamma path
# --------------------------------------------------------------------- #


def test_gamma_per_transaction_detects_winner(engine):
    """A materially higher-scale Gamma variant should be detected as significant."""
    rng = np.random.default_rng(0)
    n = 300
    df = make_df({
        "Control": rng.gamma(shape=2.0, scale=25.0, size=n),   # mean 50
        "Winner": rng.gamma(shape=2.0, scale=35.0, size=n),    # mean 70
    })
    config = ContinuousMetricConfig(
        kpi="revenue",
        approach=ContinuousApproach.GAMMA_GLM,
        unit=AnalysisUnit.PER_TRANSACTION,
        control_label="Control",
    )
    result = engine.run_comparison_suite(df, config)

    assert result.test_name == "Gamma GLM (Likelihood Ratio Test)"
    assert result.is_significant is True
    assert result.is_normal is None  # diagnostics are heuristic-only
    assert result.is_homogeneous is None
    assert result.warnings == []


def test_gamma_per_transaction_identical_groups_not_significant(engine):
    """Two draws from the same Gamma distribution should not be significant."""
    rng = np.random.default_rng(0)
    n = 300
    df = make_df({
        "Control": rng.gamma(shape=2.0, scale=25.0, size=n),
        "B": rng.gamma(shape=2.0, scale=25.0, size=n),
    })
    config = ContinuousMetricConfig(
        kpi="revenue", approach=ContinuousApproach.GAMMA_GLM, control_label="Control",
    )
    result = engine.run_comparison_suite(df, config)
    assert result.is_significant is False


def test_gamma_per_visitor_hurdle_model_keeps_zeros(engine):
    """
    per_visitor Gamma path must use the two-part hurdle test name and detect
    a variant with both a higher conversion rate and equal spend distribution.
    """
    rng = np.random.default_rng(0)
    n_visitors = 1000
    conv_a = rng.random(n_visitors) < 0.05
    spend_a = np.where(conv_a, rng.gamma(2.0, 25.0, n_visitors), 0.0)
    conv_b = rng.random(n_visitors) < 0.08
    spend_b = np.where(conv_b, rng.gamma(2.0, 25.0, n_visitors), 0.0)
    df = make_df({"Control": spend_a, "B": spend_b})

    config = ContinuousMetricConfig(
        kpi="revenue",
        approach=ContinuousApproach.GAMMA_GLM,
        unit=AnalysisUnit.PER_VISITOR,
        control_label="Control",
    )
    result = engine.run_comparison_suite(df, config)

    assert result.test_name == "Two-Part Hurdle (Likelihood Ratio Test)"
    assert result.is_significant is True


def test_gamma_three_groups_triggers_posthoc(engine):
    """
    A significant global Gamma test with 3+ groups and a control_label should
    produce one Bonferroni-adjusted posthoc comparison per non-control variant.
    """
    rng = np.random.default_rng(0)
    n = 300
    df = make_df({
        "Control": rng.gamma(shape=2.0, scale=25.0, size=n),
        "Winner": rng.gamma(shape=2.0, scale=35.0, size=n),
        "BigWinner": rng.gamma(shape=2.0, scale=45.0, size=n),
    })
    config = ContinuousMetricConfig(
        kpi="revenue", approach=ContinuousApproach.GAMMA_GLM, control_label="Control",
    )
    result = engine.run_comparison_suite(df, config)

    assert result.is_significant is True
    assert result.posthoc_results is not None
    assert {r.comparison for r in result.posthoc_results} == {
        "Winner vs Control", "BigWinner vs Control",
    }
    for r in result.posthoc_results:
        assert r.p_adj_bonferroni >= r.p_value  # Bonferroni only inflates p
        assert r.is_significant is True


def test_gamma_global_test_unfittable_group_treated_as_non_significant(engine):
    """
    A group with zero variance (all-identical values) can't fit a Gamma MLE,
    so ll_alt is nan; the engine must fall back to p=1.0 with a warning
    rather than propagating the nan.
    """
    rng = np.random.default_rng(0)
    df = make_df({
        "A": [5.0] * 10,  # zero variance -> fit_gamma is unfittable
        "B": rng.gamma(2.0, 25.0, 10),
    })
    result = engine.run_comparison_suite(
        df, ContinuousMetricConfig(kpi="revenue", approach=ContinuousApproach.GAMMA_GLM)
    )
    assert result.p_value == 1.0
    assert result.is_significant is False
    assert any("could not be fit on all groups" in w for w in result.warnings)


def test_gamma_posthoc_unfittable_pair_reports_non_significant(engine):
    """
    run_gamma_posthoc's own nan guard: a degenerate (zero-variance) variant
    paired against the control can't fit an alt model, so the pairwise
    comparison must report p=1.0 / not significant rather than raising.
    """
    rng = np.random.default_rng(0)
    df = make_df({
        "Control": rng.gamma(2.0, 25.0, 300),
        "Degenerate": [7.0] * 10,
    })
    results = ContinuousMetricEngine.run_gamma_posthoc(
        df, "revenue", GROUP_COL, "Control", AnalysisUnit.PER_TRANSACTION, alpha=0.05,
    )
    assert len(results) == 1
    assert results[0].comparison == "Degenerate vs Control"
    assert results[0].p_value == 1.0
    assert results[0].is_significant is False


def test_gamma_three_groups_significant_without_control_label_warns(engine):
    """
    A significant 3+ group Gamma test with no control_label should skip
    posthoc and add an explanatory warning instead of raising.
    """
    rng = np.random.default_rng(0)
    n = 300
    df = make_df({
        "Control": rng.gamma(shape=2.0, scale=25.0, size=n),
        "Winner": rng.gamma(shape=2.0, scale=35.0, size=n),
        "BigWinner": rng.gamma(shape=2.0, scale=45.0, size=n),
    })
    config = ContinuousMetricConfig(kpi="revenue", approach=ContinuousApproach.GAMMA_GLM)
    result = engine.run_comparison_suite(df, config)

    assert result.posthoc_results is None
    assert any("no control_label" in w for w in result.warnings)


# --------------------------------------------------------------------- #
#  Smoke tests: run_comparison_suite, Negative Binomial path (count data)
# --------------------------------------------------------------------- #


def test_negbin_two_groups_detects_winner(engine):
    """
    A discrete count KPI (e.g. items per visitor) must route to Negative
    Binomial regression rather than the normality/variance tree, and detect
    a materially higher-mean variant as significant.
    """
    rng = np.random.default_rng(0)
    n = 500
    df = make_df({
        "Control": rng.negative_binomial(n=3, p=3 / (3 + 1.5), size=n).astype(float),
        "Winner": rng.negative_binomial(n=3, p=3 / (3 + 2.5), size=n).astype(float),
    }, kpi="items")
    config = ContinuousMetricConfig(kpi="items", unit=AnalysisUnit.PER_VISITOR, control_label="Control")
    result = engine.run_comparison_suite(df, config)

    assert result.test_name == "Negative Binomial Regression (LRT)"
    assert result.is_significant is True
    assert result.is_normal is None  # diagnostics don't apply to the count-data path
    assert result.is_homogeneous is None
    assert result.dispersion_alpha is not None
    assert result.dispersion_alpha > 0
    assert result.posthoc_results is None  # only 2 groups -- no posthoc needed


def test_negbin_identical_groups_not_significant(engine):
    """Two draws from the same Negative Binomial distribution should not be significant."""
    rng = np.random.default_rng(1)
    n = 500
    df = make_df({
        "Control": rng.negative_binomial(n=3, p=3 / (3 + 1.5), size=n).astype(float),
        "B": rng.negative_binomial(n=3, p=3 / (3 + 1.5), size=n).astype(float),
    }, kpi="items")
    result = engine.run_comparison_suite(
        df, ContinuousMetricConfig(kpi="items", unit=AnalysisUnit.PER_VISITOR)
    )
    assert result.test_name == "Negative Binomial Regression (LRT)"
    assert result.is_significant is False


def test_negbin_three_groups_triggers_posthoc(engine):
    """
    A significant global Negative Binomial test with 3+ groups and a
    control_label should produce one Bonferroni-adjusted posthoc comparison
    per non-control variant.
    """
    rng = np.random.default_rng(2)
    n = 300
    df = make_df({
        "Control": rng.negative_binomial(n=3, p=3 / (3 + 1.0), size=n).astype(float),
        "Winner": rng.negative_binomial(n=3, p=3 / (3 + 2.0), size=n).astype(float),
        "BigWinner": rng.negative_binomial(n=3, p=3 / (3 + 3.0), size=n).astype(float),
    }, kpi="items")
    config = ContinuousMetricConfig(kpi="items", unit=AnalysisUnit.PER_VISITOR, control_label="Control")
    result = engine.run_comparison_suite(df, config)

    assert result.is_significant is True
    assert result.posthoc_results is not None
    assert {r.comparison for r in result.posthoc_results} == {
        "Winner vs Control", "BigWinner vs Control",
    }
    for r in result.posthoc_results:
        assert r.p_adj_bonferroni >= r.p_value
        assert r.is_significant is True


def test_negbin_three_groups_significant_without_control_label_warns(engine):
    """A significant 3+ group NB test with no control_label should skip posthoc and warn."""
    rng = np.random.default_rng(5)
    n = 300
    df = make_df({
        "A": rng.negative_binomial(n=3, p=3 / (3 + 1.0), size=n).astype(float),
        "B": rng.negative_binomial(n=3, p=3 / (3 + 2.0), size=n).astype(float),
        "C": rng.negative_binomial(n=3, p=3 / (3 + 3.0), size=n).astype(float),
    }, kpi="items")
    result = engine.run_comparison_suite(
        df, ContinuousMetricConfig(kpi="items", unit=AnalysisUnit.PER_VISITOR)
    )
    assert result.posthoc_results is None
    assert any("no control_label" in w for w in result.warnings)


def test_negbin_gate_skips_metrics_with_negative_values(engine):
    """
    A metric with negative values (e.g. profit) can't be a count, even if it
    happens to be integer-valued -- must fall through to the heuristic tree.
    """
    rng = np.random.default_rng(3)
    n = 200
    df = make_df({
        "A": rng.normal(10, 5, n).round(),
        "B": rng.normal(12, 5, n).round(),
    }, kpi="profit")
    result = engine.run_comparison_suite(df, ContinuousMetricConfig(kpi="profit"))
    assert result.test_name != "Negative Binomial Regression (LRT)"


def test_negbin_gate_skips_high_cardinality_integers(engine):
    """
    A high-cardinality integer KPI (e.g. a continuous metric that happens to
    be whole numbers) shouldn't trip the count gate under default thresholds.
    """
    rng = np.random.default_rng(4)
    n = 200
    df = make_df({
        "A": rng.integers(1, 5000, n).astype(float),
        "B": rng.integers(1, 5000, n).astype(float),
    }, kpi="units")
    result = engine.run_comparison_suite(df, ContinuousMetricConfig(kpi="units"))
    assert result.test_name != "Negative Binomial Regression (LRT)"


def test_negbin_gate_thresholds_are_configurable(engine):
    """
    Raising count_max_unique should route a wider-cardinality integer KPI to
    Negative Binomial that the default threshold would have skipped.
    """
    rng = np.random.default_rng(6)
    n = 200
    df = make_df({
        "A": rng.integers(0, 80, n).astype(float),
        "B": rng.integers(0, 80, n).astype(float),
    }, kpi="tickets")

    default_result = engine.run_comparison_suite(df, ContinuousMetricConfig(kpi="tickets"))
    custom_result = engine.run_comparison_suite(
        df, ContinuousMetricConfig(kpi="tickets", count_max_unique=100)
    )
    assert default_result.test_name != "Negative Binomial Regression (LRT)"
    assert custom_result.test_name == "Negative Binomial Regression (LRT)"


# --------------------------------------------------------------------- #
#  is_count_kpi gate (unit tests)
# --------------------------------------------------------------------- #


def test_is_count_kpi_true_for_small_integer_range(engine):
    assert ContinuousMetricEngine.is_count_kpi(pd.Series([0, 1, 2, 3, 1, 2, 0, 4])) is True


def test_is_count_kpi_false_for_negative_values(engine):
    assert ContinuousMetricEngine.is_count_kpi(pd.Series([-1, 1, 2, 3])) is False


def test_is_count_kpi_false_for_non_integer_values(engine):
    assert ContinuousMetricEngine.is_count_kpi(pd.Series([1.5, 2.3, 4.1])) is False


def test_is_count_kpi_false_for_high_cardinality(engine):
    assert ContinuousMetricEngine.is_count_kpi(pd.Series(range(1000))) is False


def test_is_count_kpi_false_for_empty_series(engine):
    assert ContinuousMetricEngine.is_count_kpi(pd.Series([], dtype=float)) is False


def test_is_count_kpi_ratio_threshold_overrides_absolute_cap(engine):
    """
    A KPI with more distinct values than max_unique_for_check is still a
    count if the distinct-value-to-row-count ratio is low enough (a large
    dataset where a genuine count metric has many distinct values in
    absolute terms but few relative to total rows).
    """
    s = pd.Series(list(range(60)) * 100)  # 60 distinct values, 6000 rows -> ratio 0.01
    assert ContinuousMetricEngine.is_count_kpi(s, max_unique_for_check=50, max_unique_ratio=0.05) is True


# --------------------------------------------------------------------- #
#  Zero-handling / unit warnings
# --------------------------------------------------------------------- #


def test_per_transaction_drops_zero_rows_and_warns(engine):
    """PER_TRANSACTION must exclude zero-value rows and report how many."""
    rng = np.random.default_rng(0)
    n = 200
    df = make_df({
        "A": np.concatenate([rng.normal(50, 10, n), np.zeros(50)]),
        "B": np.concatenate([rng.normal(52, 10, n), np.zeros(50)]),
    })
    result = engine.run_comparison_suite(
        df, ContinuousMetricConfig(kpi="revenue", unit=AnalysisUnit.PER_TRANSACTION)
    )
    assert any("excluded 100 zero-value" in w for w in result.warnings)
    assert sum(row["count"] for row in result.summary_stats) == 2 * n


def test_per_visitor_keeps_zero_rows_no_warning(engine):
    """PER_VISITOR must keep zero rows and not warn when zeros are present."""
    rng = np.random.default_rng(0)
    n = 200
    df = make_df({
        "A": np.concatenate([rng.normal(50, 10, n), np.zeros(50)]),
        "B": np.concatenate([rng.normal(52, 10, n), np.zeros(50)]),
    })
    result = engine.run_comparison_suite(
        df, ContinuousMetricConfig(kpi="revenue", unit=AnalysisUnit.PER_VISITOR)
    )
    assert result.warnings == []
    assert sum(row["count"] for row in result.summary_stats) == 2 * (n + 50)


def test_per_visitor_warns_when_no_zeros_present(engine):
    """PER_VISITOR on zero-free data is equivalent to PER_TRANSACTION -- warn about it."""
    rng = np.random.default_rng(0)
    n = 200
    df = make_df({"A": rng.normal(50, 10, n), "B": rng.normal(52, 10, n)})
    result = engine.run_comparison_suite(
        df, ContinuousMetricConfig(kpi="revenue", unit=AnalysisUnit.PER_VISITOR)
    )
    assert any("No zero-value rows found" in w for w in result.warnings)


# --------------------------------------------------------------------- #
#  Validation integration
# --------------------------------------------------------------------- #


def test_single_variant_rejected(engine):
    """validate_continuous_data's cross-field check must surface as ValueError."""
    df = pd.DataFrame({GROUP_COL: ["A"] * 3, "revenue": [1, 2, 3]})
    with pytest.raises(ValueError, match="at least 2 variants"):
        engine.run_comparison_suite(df, ContinuousMetricConfig(kpi="revenue"))


def test_gamma_rejects_negative_values(engine):
    """The Gamma family is undefined for negative values; heuristic isn't."""
    df = make_df({"A": [-5.0, 1.0, 2.0], "B": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="non-negative values"):
        engine.run_comparison_suite(
            df, ContinuousMetricConfig(kpi="revenue", approach=ContinuousApproach.GAMMA_GLM)
        )


def test_missing_kpi_column_rejected(engine):
    df = pd.DataFrame({GROUP_COL: ["A", "B"], "other_col": [1, 2]})
    with pytest.raises(ValueError, match="revenue.*missing"):
        engine.run_comparison_suite(df, ContinuousMetricConfig(kpi="revenue"))


def test_per_transaction_insufficient_positive_rows_rejected(engine):
    """Each variant needs at least 2 strictly-positive rows for per_transaction."""
    df = make_df({"A": [5.0, 0.0, 0.0], "B": [5.0, 5.0, 5.0]})
    with pytest.raises(ValueError, match="at least 2 positive-value rows"):
        engine.run_comparison_suite(
            df, ContinuousMetricConfig(kpi="revenue", unit=AnalysisUnit.PER_TRANSACTION)
        )


# --------------------------------------------------------------------- #
#  generate_continuous_conclusion
# --------------------------------------------------------------------- #


def test_conclusion_significant_mentions_test_and_unit(engine):
    msg = ContinuousMetricEngine.generate_continuous_conclusion(
        "Revenue", True, "Standard ANOVA", 0.0123, unit=AnalysisUnit.PER_TRANSACTION,
    )
    assert "Significant Variance Detected" in msg
    assert "Standard ANOVA" in msg
    assert "p=0.012" in msg
    assert "value per transaction" in msg


def test_conclusion_not_significant_uses_per_visitor_phrase(engine):
    msg = ContinuousMetricEngine.generate_continuous_conclusion(
        "Revenue", False, "Mann-Whitney U", 0.5, unit=AnalysisUnit.PER_VISITOR,
    )
    assert "No Significant Difference" in msg
    assert "revenue per visitor" in msg


# --------------------------------------------------------------------- #
#  Outlier detection / winsorization
# --------------------------------------------------------------------- #


def test_detect_outliers_ols_flags_extreme_point(engine):
    rng = np.random.default_rng(0)
    n = 100
    a = rng.normal(50, 5, n)
    a[0] = 500.0  # extreme outlier
    b = rng.normal(50, 5, n)
    df = make_df({"A": a, "B": b})

    mask = ContinuousMetricEngine.detect_outliers_ols(df, "revenue")
    assert mask[0] is True
    assert sum(mask) == 1


def test_detect_outliers_ols_empty_frame_returns_all_false(engine):
    df = pd.DataFrame({GROUP_COL: [], "revenue": []})
    assert ContinuousMetricEngine.detect_outliers_ols(df, "revenue") == []


def test_winsorize_series_std_method_caps_extreme_value(engine):
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 1000.0])
    clipped, lower, upper = ContinuousMetricEngine.winsorize_series(
        s, method="Standard Deviation", param=2.0
    )
    assert clipped[-1] == pytest.approx(upper)
    assert clipped[-1] < 1000.0
    assert clipped[:-1] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_winsorize_series_percentile_method(engine):
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 1000.0])
    clipped, lower, upper = ContinuousMetricEngine.winsorize_series(
        s, method="Percentile", param=10.0
    )
    assert clipped[-1] == pytest.approx(upper)
    assert lower < clipped[0] or clipped[0] == pytest.approx(lower)


def test_winsorize_series_all_nan_returns_original(engine):
    s = pd.Series([np.nan, np.nan])
    clipped, lower, upper = ContinuousMetricEngine.winsorize_series(s)
    assert all(math.isnan(v) for v in clipped)
    assert lower == 0.0
    assert upper == 0.0


# --------------------------------------------------------------------- #
#  Gamma fitting primitives
# --------------------------------------------------------------------- #


def test_neg_log_likelihood_invalid_params_returns_inf(engine):
    """Non-positive shape/scale are outside the Gamma domain; guard returns inf."""
    data = np.array([1.0, 2.0, 3.0])
    assert ContinuousMetricEngine.neg_log_likelihood([-1.0, 2.0], data) == float("inf")
    assert ContinuousMetricEngine.neg_log_likelihood([2.0, 0.0], data) == float("inf")


def test_fit_gamma_recovers_known_shape_and_scale(engine):
    """MLE fit on a large Gamma(k=2, theta=25) sample should recover close params."""
    rng = np.random.default_rng(0)
    data = rng.gamma(shape=2.0, scale=25.0, size=2000)
    k, theta, log_lik = ContinuousMetricEngine.fit_gamma(data)
    assert k == pytest.approx(2.0, rel=0.15)
    assert theta == pytest.approx(25.0, rel=0.15)
    assert math.isfinite(log_lik)


def test_fit_unit_model_per_transaction_degenerate_returns_nan(engine):
    """Fewer than 2 positive values can't fit a Gamma; returns (nan, 2)."""
    ll, n_params = ContinuousMetricEngine.fit_unit_model([5.0], AnalysisUnit.PER_TRANSACTION)
    assert math.isnan(ll)
    assert n_params == 2


def test_fit_unit_model_per_visitor_all_zero_returns_bernoulli_only(engine):
    """All-zero per_visitor data has a degenerate Bernoulli (p=0) and no Gamma part."""
    ll, n_params = ContinuousMetricEngine.fit_unit_model(np.zeros(50), AnalysisUnit.PER_VISITOR)
    assert ll == 0.0
    assert n_params == 3


def test_fit_unit_model_per_visitor_empty_returns_nan(engine):
    ll, n_params = ContinuousMetricEngine.fit_unit_model([], AnalysisUnit.PER_VISITOR)
    assert math.isnan(ll)
    assert n_params == 3


# --------------------------------------------------------------------- #
#  Negative Binomial fitting primitives
# --------------------------------------------------------------------- #


def test_fit_negbin_recovers_known_mean_via_intercept(engine):
    """
    An intercept-only NB2 fit's implied mean (exp(intercept)) should be close
    to the sample mean of a large negative-binomial draw.
    """
    rng = np.random.default_rng(0)
    data = rng.negative_binomial(n=3, p=3 / (3 + 2.0), size=2000).astype(float)
    log_lik, alpha, res = ContinuousMetricEngine.fit_negbin(data)
    assert math.isfinite(log_lik)
    assert alpha > 0
    implied_mean = math.exp(res.params[0])
    assert implied_mean == pytest.approx(data.mean(), rel=0.15)


def test_run_negbin_lrt_two_groups_matches_hand_lrt(engine):
    """
    _run_negbin_lrt's global test should equal a hand-rolled LRT: fit an
    intercept-only null and an intercept+dummy alt, compare 2*(ll_alt-ll_null)
    against chi2(df=1).
    """
    rng = np.random.default_rng(7)
    n = 400
    a = rng.negative_binomial(n=3, p=3 / (3 + 1.0), size=n).astype(float)
    b = rng.negative_binomial(n=3, p=3 / (3 + 2.0), size=n).astype(float)
    df = make_df({"A": a, "B": b}, kpi="items")

    p_value, lr_stat, disp_alpha = ContinuousMetricEngine._run_negbin_lrt(
        df, "items", GROUP_COL
    )

    data = df["items"].to_numpy(dtype=float)
    ll_null, _, _ = ContinuousMetricEngine.fit_negbin(data)
    dummy = (df[GROUP_COL] == "B").to_numpy(dtype=float).reshape(-1, 1)
    exog_alt = np.column_stack([np.ones(len(data)), dummy])
    ll_alt, _, _ = ContinuousMetricEngine.fit_negbin(data, exog_alt)
    expected_lr = max(2.0 * (ll_alt - ll_null), 0.0)

    assert lr_stat == pytest.approx(expected_lr, rel=1e-6)
    assert disp_alpha > 0
    assert 0.0 <= p_value <= 1.0


@pytest.mark.filterwarnings("ignore")
def test_run_negbin_lrt_raises_on_non_convergent_fit(engine):
    """
    Degenerate, tiny samples can fail to converge to a finite log-likelihood.
    _run_negbin_lrt must raise rather than let a nan p_value leak out (a nan
    would fail ContinuousMetricResult's p_value >= 0.0 constraint downstream).
    Warnings are silenced: statsmodels emits convergence/separation warnings
    for this intentionally degenerate input, which is exactly what's under test.
    """
    df = pd.DataFrame({GROUP_COL: ["A", "A", "B", "B"], "items": [0.0, 0.0, 1.0, 1.0]})
    with pytest.raises(Exception):
        ContinuousMetricEngine._run_negbin_lrt(df, "items", GROUP_COL)


@pytest.mark.filterwarnings("ignore")
def test_negbin_global_test_unfittable_treated_as_non_significant(engine):
    """
    Mirrors the Gamma-path nan guard: a degenerate/non-convergent global NB
    fit must fall back to p=1.0 with a warning instead of propagating a nan
    into the result.
    """
    df = pd.DataFrame({GROUP_COL: ["A", "A", "B", "B"], "items": [0.0, 0.0, 1.0, 1.0]})
    result = engine.run_comparison_suite(
        df, ContinuousMetricConfig(kpi="items", unit=AnalysisUnit.PER_VISITOR)
    )
    assert result.p_value == 1.0
    assert result.is_significant is False
    assert any("could not be fit" in w for w in result.warnings)


@pytest.mark.filterwarnings("ignore")
def test_negbin_posthoc_unfittable_pair_reports_non_significant(engine):
    """run_negbin_posthoc's own guard: a non-convergent pair reports p=1.0 rather than raising."""
    df = pd.DataFrame({
        GROUP_COL: ["Control", "Control", "Tiny", "Tiny"],
        "items": [0.0, 0.0, 1.0, 1.0],
    })
    results = ContinuousMetricEngine.run_negbin_posthoc(df, "items", GROUP_COL, "Control")
    assert len(results) == 1
    assert results[0].comparison == "Tiny vs Control"
    assert results[0].p_value == 1.0
    assert results[0].is_significant is False


# --------------------------------------------------------------------- #
#  estimate_monetary_impact_per_variant
# --------------------------------------------------------------------- #


def test_monetary_impact_per_visitor_scales_by_mean_diff(engine):
    """
    unit=PER_VISITOR fixes both rates at 1.0, so diff_value is exactly the
    mean difference and point_estimate = diff * daily_visitors * period.
    """
    result = ContinuousMetricEngine.estimate_monetary_impact_per_variant(
        mean_ctrl=50.0, std_ctrl=20.0, n_ctrl=1000,
        mean_chal=55.0, std_chal=22.0, n_chal=1000,
        unit=AnalysisUnit.PER_VISITOR,
        daily_visitors=2000,
        projection_period=183,
    )
    assert result["rate_ctrl"] == 1.0
    assert result["rate_chal"] == 1.0
    assert result["point_estimate"] == pytest.approx(5.0 * 2000 * 183)
    assert result["ci_low"] < result["point_estimate"] < result["ci_high"]


def test_monetary_impact_per_transaction_requires_visitor_counts(engine):
    """Per-transaction projection cannot derive an order rate without visitor totals."""
    with pytest.raises(ValueError, match="visitors_ctrl and visitors_chal"):
        ContinuousMetricEngine.estimate_monetary_impact_per_variant(
            mean_ctrl=50.0, std_ctrl=20.0, n_ctrl=100,
            mean_chal=55.0, std_chal=22.0, n_chal=120,
            unit=AnalysisUnit.PER_TRANSACTION,
            daily_visitors=2000,
        )


def test_monetary_impact_per_transaction_folds_in_order_rate(engine):
    """
    Per-transaction point estimate = (rate_chal*mean_chal - rate_ctrl*mean_ctrl)
    scaled by daily_visitors and projection_period.
    """
    result = ContinuousMetricEngine.estimate_monetary_impact_per_variant(
        mean_ctrl=50.0, std_ctrl=20.0, n_ctrl=100,
        mean_chal=55.0, std_chal=22.0, n_chal=120,
        unit=AnalysisUnit.PER_TRANSACTION,
        daily_visitors=2000,
        visitors_ctrl=2000, visitors_chal=2000,
        projection_period=183,
    )
    rate_ctrl = 100 / 2000
    rate_chal = 120 / 2000
    expected_diff = rate_chal * 55.0 - rate_ctrl * 50.0
    assert result["rate_ctrl"] == pytest.approx(rate_ctrl)
    assert result["rate_chal"] == pytest.approx(rate_chal)
    assert result["point_estimate"] == pytest.approx(expected_diff * 2000 * 183)


def test_monetary_impact_one_sided_greater_has_infinite_upper_bound(engine):
    result = ContinuousMetricEngine.estimate_monetary_impact_per_variant(
        mean_ctrl=50.0, std_ctrl=20.0, n_ctrl=1000,
        mean_chal=55.0, std_chal=22.0, n_chal=1000,
        unit=AnalysisUnit.PER_VISITOR,
        daily_visitors=2000,
        alternative=AlternativeHypothesis.GREATER,
    )
    assert result["ci_high"] == float("inf")
    assert math.isfinite(result["ci_low"])
    assert math.isfinite(result["point_estimate"])


def test_monetary_impact_zero_n_variant_treated_as_zero_se(engine):
    """n=0 for a variant must not raise a divide-by-zero; SE contribution is 0."""
    result = ContinuousMetricEngine.estimate_monetary_impact_per_variant(
        mean_ctrl=0.0, std_ctrl=0.0, n_ctrl=0,
        mean_chal=55.0, std_chal=22.0, n_chal=1000,
        unit=AnalysisUnit.PER_VISITOR,
        daily_visitors=2000,
    )
    assert math.isfinite(result["point_estimate"])
    assert math.isfinite(result["ci_low"])


# --------------------------------------------------------------------- #
#  generate_monetary_conclusion
# --------------------------------------------------------------------- #


def test_monetary_conclusion_positive_reads_as_gain(engine):
    result = {"point_estimate": 292800.0, "ci_low": -2273.0, "ci_high": 587873.0, "projection_period": 183}
    msg = ContinuousMetricEngine.generate_monetary_conclusion("Winner", result, is_significant=True)
    assert "gain" in msg
    assert "Winner" in msg
    assert "not statistically significant" not in msg


def test_monetary_conclusion_negative_reads_as_loss(engine):
    result = {"point_estimate": -100.0, "ci_low": -2273.0, "ci_high": 587873.0, "projection_period": 183}
    msg = ContinuousMetricEngine.generate_monetary_conclusion("Loser", result, is_significant=False)
    assert "loss" in msg
    assert "not statistically significant" in msg


# --------------------------------------------------------------------- #
#  run_business_case
# --------------------------------------------------------------------- #


def test_run_business_case_skips_control_returns_one_row_per_challenger(engine):
    group_stats = {
        "Control": {"mean": 50.0, "std": 20.0, "count": 1000},
        "Winner": {"mean": 60.0, "std": 22.0, "count": 1000},
        "Loser": {"mean": 45.0, "std": 18.0, "count": 1000},
    }
    results = engine.run_business_case(
        group_stats=group_stats,
        control_label="Control",
        unit=AnalysisUnit.PER_VISITOR,
        daily_visitors=2000,
        significance_by_variant={"Winner": True, "Loser": True},
    )
    variants = {r["variant"] for r in results}
    assert variants == {"Winner", "Loser"}

    winner = next(r for r in results if r["variant"] == "Winner")
    loser = next(r for r in results if r["variant"] == "Loser")
    assert winner["point_estimate"] > 0
    assert loser["point_estimate"] < 0
    assert "gain" in winner["conclusion"]
    assert "loss" in loser["conclusion"]


def test_run_business_case_per_transaction_uses_visitor_counts(engine):
    group_stats = {
        "Control": {"mean": 50.0, "std": 20.0, "count": 100},
        "Winner": {"mean": 55.0, "std": 22.0, "count": 120},
    }
    results = engine.run_business_case(
        group_stats=group_stats,
        control_label="Control",
        unit=AnalysisUnit.PER_TRANSACTION,
        daily_visitors=2000,
        visitor_counts={"Control": 2000, "Winner": 2000},
    )
    assert len(results) == 1
    assert results[0]["rate_chal"] == pytest.approx(120 / 2000)
