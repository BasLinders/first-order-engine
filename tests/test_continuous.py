import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

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
    # Significant with 3 groups -> all-pairs Tukey HSD.
    assert result.posthoc_test_name == "Tukey HSD"
    assert {r.comparison for r in result.posthoc_results} == {
        "B vs A", "C vs A", "C vs B",
    }


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
    levene p ~= 1e-13 at seed=15) must route to Welch's ANOVA.
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
    assert "difference in Revenue per transaction between the variants" in msg


def test_conclusion_not_significant_uses_per_visitor_phrase(engine):
    msg = ContinuousMetricEngine.generate_continuous_conclusion(
        "Revenue", False, "Mann-Whitney U", 0.5, unit=AnalysisUnit.PER_VISITOR,
    )
    assert "No Significant Difference" in msg
    assert "p=0.500" in msg
    assert "difference in Revenue per visitor across the variants" in msg


@pytest.mark.parametrize("unit, phrase", [
    (AnalysisUnit.PER_VISITOR, "per visitor"),
    (AnalysisUnit.PER_TRANSACTION, "per transaction"),
])
@pytest.mark.parametrize("significant", [True, False])
def test_conclusion_non_revenue_kpi_names_only_the_unit(engine, unit, phrase, significant):
    """A count KPI must not be described as revenue or value."""
    msg = ContinuousMetricEngine.generate_continuous_conclusion(
        "total_item_quantity", significant, "Negative Binomial LRT", 0.01, unit=unit,
    )
    assert f"total_item_quantity {phrase}" in msg
    assert "revenue" not in msg.lower()
    assert "value" not in msg.lower()


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


def _uneven_normal_frame(n_large, n_small, seed):
    rng = np.random.default_rng(seed)
    return make_df({
        "A": rng.normal(100, 10, n_large),
        "B": rng.normal(100, 10, n_small),
    })


@pytest.mark.parametrize("threshold", [2.0, 3.0])
def test_detect_outliers_ols_does_not_flag_small_variant_by_size(engine, threshold):
    """
    At a 90/10 split, leverage (1/n_g) and DFFITS used to flag the whole
    small variant. On normal data its flagged share must now be close to
    the large variant's: about the two-sided normal tail beyond threshold.
    """
    small_shares, large_shares = [], []
    for seed in range(10):
        df = _uneven_normal_frame(1800, 200, seed)
        mask = np.array(
            ContinuousMetricEngine.detect_outliers_ols(df, "revenue", threshold)
        )
        is_small = (df[GROUP_COL] == "B").to_numpy()
        small_shares.append(mask[is_small].mean())
        large_shares.append(mask[~is_small].mean())

    small, large = np.mean(small_shares), np.mean(large_shares)
    expected = 2 * stats.norm.sf(threshold)  # 0.0455 at k=2, 0.0027 at k=3
    assert max(small_shares) < 0.2  # never the whole variant
    assert small == pytest.approx(expected, abs=0.015)
    assert abs(small - large) < 0.015


def test_detect_outliers_ols_flags_extreme_value_in_small_variant(engine):
    df = _uneven_normal_frame(1800, 200, seed=1)
    spike_pos = int(np.flatnonzero((df[GROUP_COL] == "B").to_numpy())[0])
    df.loc[df.index[spike_pos], "revenue"] = 200.0  # 10 SD above the mean
    mask = ContinuousMetricEngine.detect_outliers_ols(df, "revenue", threshold=3.0)
    assert mask[spike_pos] is True
    assert sum(mask) < 20  # the spike plus normal-tail rows, not the variant


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


def test_winsorize_series_exclude_zeros_leaves_zeros_and_uses_nonzero_bounds(engine):
    """
    Per-visitor data: a positive lower cap must never lift a zero (a
    non-converting visitor), and the caps come from the buyers only.
    """
    buyers = [90.0, 95.0, 100.0, 105.0, 110.0, 400.0]
    s = pd.Series([0.0] * 6 + buyers)
    clipped, lower, upper = ContinuousMetricEngine.winsorize_series(
        s, method="Percentile", param=10.0, exclude_zeros=True
    )
    exp_lower, exp_upper = np.percentile(buyers, [10.0, 90.0])
    assert lower == pytest.approx(exp_lower)
    assert upper == pytest.approx(exp_upper)
    assert clipped[:6] == [0.0] * 6
    assert clipped[6] == pytest.approx(exp_lower)  # 90 lifted to the lower cap
    assert clipped[-1] == pytest.approx(exp_upper)


def test_winsorize_series_exclude_zeros_keeps_nan(engine):
    s = pd.Series([0.0, np.nan, 10.0, 12.0, 11.0])
    clipped, _, _ = ContinuousMetricEngine.winsorize_series(s, exclude_zeros=True)
    assert clipped[0] == 0.0
    assert math.isnan(clipped[1])


def test_winsorize_series_exclude_zeros_all_zero_returns_original(engine):
    s = pd.Series([0.0, 0.0, 0.0])
    clipped, lower, upper = ContinuousMetricEngine.winsorize_series(
        s, exclude_zeros=True
    )
    assert clipped == [0.0, 0.0, 0.0]
    assert (lower, upper) == (0.0, 0.0)


def test_winsorize_series_central_percentile_keeps_central_share(engine):
    """'Central Percentile' 95 keeps the middle 95%: 2.5th and 97.5th percentiles."""
    s = pd.Series(np.arange(1.0, 201.0))
    _, lower, upper = ContinuousMetricEngine.winsorize_series(
        s, method="Central Percentile", param=95.0
    )
    exp = np.percentile(s, [2.5, 97.5])
    assert (lower, upper) == (pytest.approx(exp[0]), pytest.approx(exp[1]))


def test_winsorize_series_central_percentile_equals_tail_percentile(engine):
    """Central 95 == 2.5 per tail in the legacy 'Percentile' convention."""
    s = pd.Series(np.random.default_rng(3).lognormal(3, 1, 500))
    a = ContinuousMetricEngine.winsorize_series(s, "Central Percentile", 95.0)
    b = ContinuousMetricEngine.winsorize_series(s, "Percentile", 2.5)
    assert a == b


@pytest.mark.parametrize("param", [0.0, -5.0, 100.5])
def test_winsorize_series_central_percentile_rejects_out_of_range(engine, param):
    with pytest.raises(ValueError, match="Central Percentile"):
        ContinuousMetricEngine.winsorize_series(
            pd.Series([1.0, 2.0, 3.0]), "Central Percentile", param
        )


# --------------------------------------------------------------------- #
#  detect_outliers (zero-aware, size-dependent method)
# --------------------------------------------------------------------- #


def _per_visitor_frame(n_buyers=40, n_zeros=10, spike=140.0, seed=0):
    rng = np.random.default_rng(seed)
    a = np.r_[rng.normal(100, 5, n_buyers), np.zeros(n_zeros)]
    b = np.r_[rng.normal(100, 5, n_buyers), np.zeros(n_zeros)]
    a[0] = spike
    return make_df({"A": a, "B": b})


def test_detect_outliers_never_flags_zero_rows(engine):
    df = _per_visitor_frame()
    res = ContinuousMetricEngine.detect_outliers(df, "revenue")
    mask = np.array(res.mask)
    assert res.method == "studentized_residual"
    assert not mask[df["revenue"].to_numpy() == 0].any()
    assert mask[0]  # the 140 spike among ~N(100, 5) buyers
    assert res.n_considered == int((df["revenue"] != 0).sum())
    assert res.n_outliers == int(mask.sum())


def test_detect_outliers_including_zeros_hides_the_real_outlier(engine):
    """
    Contrast with the test above: with the zero mass in the fit, the spread is
    inflated so much that the 140 spike among ~N(100, 5) buyers is missed.
    This is why exclude_zeros defaults to True.
    """
    df = _per_visitor_frame()
    res = ContinuousMetricEngine.detect_outliers(df, "revenue", exclude_zeros=False)
    assert res.mask[0] is False
    assert res.n_considered == len(df)


def test_detect_outliers_large_data_uses_per_variant_sd(engine):
    df = _per_visitor_frame()
    res = ContinuousMetricEngine.detect_outliers(
        df, "revenue", large_n_threshold=10
    )
    assert res.method == "per_variant_sd"
    mask = np.array(res.mask)
    # Hand check on variant A's buyers.
    a_buyers = df.loc[(df[GROUP_COL] == "A") & (df["revenue"] != 0), "revenue"]
    lo = a_buyers.mean() - 3 * a_buyers.std()
    hi = a_buyers.mean() + 3 * a_buyers.std()
    expected = ((a_buyers < lo) | (a_buyers > hi)).to_numpy()
    assert (mask[a_buyers.index] == expected).all()
    assert mask[0]
    assert not mask[df["revenue"].to_numpy() == 0].any()


def test_detect_outliers_mask_aligns_positionally_with_custom_index(engine):
    df = _per_visitor_frame()
    df.index = np.arange(len(df))[::-1] * 7  # non-default, descending index
    res = ContinuousMetricEngine.detect_outliers(df, "revenue")
    assert len(res.mask) == len(df)
    assert res.mask[0] is True  # still the first row positionally


def test_detect_outliers_missing_values_are_never_flagged(engine):
    df = _per_visitor_frame()
    df.loc[df.index[5], "revenue"] = np.nan
    res = ContinuousMetricEngine.detect_outliers(df, "revenue")
    assert res.mask[5] is False
    assert len(res.mask) == len(df)


def test_detect_outliers_custom_group_col(engine):
    df = _per_visitor_frame().rename(columns={GROUP_COL: "arm"})
    res = ContinuousMetricEngine.detect_outliers(df, "revenue", group_col="arm")
    assert res.mask[0] is True


def test_detect_outliers_all_zero_returns_method_none(engine):
    df = make_df({"A": np.zeros(5), "B": np.zeros(5)})
    res = ContinuousMetricEngine.detect_outliers(df, "revenue")
    assert res.method == "none"
    assert res.mask == [False] * 10
    assert res.n_outliers == 0


# --------------------------------------------------------------------- #
#  Native one-way tests and post-hocs: fixed reference values
#  (computed with pingouin 0.7.0, statsmodels 0.14.6, scikit-posthocs; FOE
#  does not depend on those libraries)
# --------------------------------------------------------------------- #


REF_GROUPS = {
    "A": [12.1, 14.3, 11.8, 13.5, 12.9, 14.0, 13.2],
    "B": [15.2, 16.8, 14.9, 17.5, 16.1, 15.7],
    "C": [11.0, 19.5, 8.2, 22.4, 13.3, 17.9, 9.6, 20.1],
}
REF_LABELS = list(REF_GROUPS)
REF_ARRAYS = [np.array(v) for v in REF_GROUPS.values()]


def test_welch_anova_matches_reference():
    f_stat, df1, df2, p = ContinuousMetricEngine.welch_anova(REF_ARRAYS)
    assert f_stat == pytest.approx(14.276554, rel=1e-6)
    assert df1 == 2
    assert df2 == pytest.approx(11.272218, rel=1e-6)
    assert p == pytest.approx(0.000814, rel=1e-3)


def test_eta_squared_matches_reference():
    assert ContinuousMetricEngine.eta_squared(REF_ARRAYS) == pytest.approx(
        0.125539, rel=1e-5
    )


def test_welch_anova_rejects_zero_variance_group():
    with pytest.raises(ValueError, match="zero variance"):
        ContinuousMetricEngine.welch_anova([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])


def test_welch_anova_rejects_group_with_one_row():
    with pytest.raises(ValueError, match="2\\+ rows"):
        ContinuousMetricEngine.welch_anova([[1.0, 2.0, 3.0], [5.0]])


def test_tukey_hsd_matches_reference():
    res = ContinuousMetricEngine.tukey_hsd(REF_LABELS, REF_ARRAYS)
    # statsmodels pairwise_tukeyhsd: meandiff = later - earlier label.
    expected = [
        ("B vs A", 2.91904762, -1.96764208, 7.80573732, 0.30337082),
        ("C vs A", 2.13571429, -2.41018017, 6.68160874, 0.46886434),
        ("C vs B", -0.78333333, -5.52696687, 3.96030021, 0.90723175),
    ]
    assert [r.comparison for r in res] == [e[0] for e in expected]
    for r, (_, diff, lo, hi, p) in zip(res, expected):
        assert r.mean_diff == pytest.approx(diff, rel=1e-7)
        assert r.ci_low == pytest.approx(lo, rel=1e-7)
        assert r.ci_high == pytest.approx(hi, rel=1e-7)
        assert r.p_value == pytest.approx(p, rel=1e-6)
        assert r.p_adj == r.p_value
        assert r.adjustment == "tukey"
        assert r.statistic_name == "q"
        assert r.is_significant is False


def test_games_howell_matches_reference():
    res = ContinuousMetricEngine.games_howell(REF_LABELS, REF_ARRAYS)
    # pingouin reports A - B; FOE reports B vs A, so diff and t flip sign.
    expected = [
        ("B vs A", 2.919048, 5.480428, 0.000612),
        ("C vs A", 2.135714, 1.102656, 0.540643),
        ("C vs B", -0.783333, -0.402369, 0.915584),
    ]
    assert [r.comparison for r in res] == [e[0] for e in expected]
    for r, (_, diff, t, p) in zip(res, expected):
        assert r.mean_diff == pytest.approx(diff, rel=1e-5)
        assert r.statistic == pytest.approx(t, rel=1e-5)
        assert r.p_value == pytest.approx(p, rel=1e-3)
        assert r.adjustment == "games-howell"
        assert r.ci_low < r.mean_diff < r.ci_high
    assert [r.is_significant for r in res] == [True, False, False]


def test_games_howell_ci_excludes_zero_iff_significant():
    res = ContinuousMetricEngine.games_howell(REF_LABELS, REF_ARRAYS)
    for r in res:
        assert (r.ci_low > 0 or r.ci_high < 0) == r.is_significant


def test_dunn_test_matches_reference_with_ties():
    labels = ["A", "B", "C"]
    groups = [[1, 2, 2, 3, 4], [3, 4, 4, 5, 6, 6], [6, 7, 7, 8, 9]]
    res = {r.comparison: r for r in ContinuousMetricEngine.dunn_test(labels, groups)}
    expected = {  # (raw p, Bonferroni p) from scikit-posthocs posthoc_dunn
        "B vs A": (0.096686, 0.290058),
        "C vs A": (0.000563, 0.001690),
        "C vs B": (0.052269, 0.156808),
    }
    assert set(res) == set(expected)
    for comp, (p, p_adj) in expected.items():
        # Reference values are rounded to 6 decimals.
        assert res[comp].p_value == pytest.approx(p, abs=1e-6)
        assert res[comp].p_adj == pytest.approx(p_adj, abs=1e-6)
        assert res[comp].adjustment == "bonferroni"
        assert res[comp].statistic_name == "z"
    assert res["C vs A"].statistic > 0  # C ranks higher than A
    assert res["C vs A"].is_significant is True
    assert res["C vs B"].is_significant is False


# --------------------------------------------------------------------- #
#  run_comparison_suite: post-hocs, effect sizes, diagnostics
# --------------------------------------------------------------------- #


def test_heuristic_anova_reports_f_eta_squared_and_diagnostics(engine):
    rng = np.random.default_rng(0)
    groups = {k: rng.normal(m, 15, 200) for k, m in [("A", 100), ("B", 100), ("C", 115)]}
    result = engine.run_comparison_suite(
        make_df(groups), ContinuousMetricConfig(kpi="revenue")
    )
    assert result.test_statistic_name == "F"
    assert result.test_statistic > 0
    assert result.effect_size_name == "eta_squared"
    assert result.effect_size == pytest.approx(
        ContinuousMetricEngine.eta_squared(list(groups.values())), rel=1e-9
    )
    assert result.normality_p >= 0.05 and result.normality_stat is not None
    assert result.levene_p >= 0.05 and result.levene_stat is not None


def test_heuristic_welch_three_groups_runs_games_howell(engine):
    # Verified at seed=15: normality p ~= 0.31, Levene p ~= 3e-17.
    rng = np.random.default_rng(15)
    df = make_df({
        "A": rng.normal(100, 8, 400),
        "B": rng.normal(103, 11, 400),
        "C": rng.normal(100, 14, 400),
    })
    result = engine.run_comparison_suite(df, ContinuousMetricConfig(kpi="revenue"))
    assert result.test_name == "Welch's ANOVA"
    assert result.is_significant is True
    assert result.posthoc_test_name == "Games-Howell"
    assert len(result.posthoc_results) == 3
    assert all(r.adjustment == "games-howell" for r in result.posthoc_results)
    assert result.effect_size_name == "partial_eta_squared"
    f_ref, _, _, p_ref = ContinuousMetricEngine.welch_anova(
        [g["revenue"].to_numpy() for _, g in df.groupby(GROUP_COL)]
    )
    assert result.test_statistic == pytest.approx(f_ref)
    assert result.p_value == pytest.approx(p_ref)


def test_heuristic_kruskal_runs_dunn_and_eta_squared_h(engine):
    rng = np.random.default_rng(0)
    groups = {"A": rng.exponential(20, 200), "B": rng.exponential(20, 200),
              "C": rng.exponential(35, 200)}
    result = engine.run_comparison_suite(
        make_df(groups), ContinuousMetricConfig(kpi="revenue")
    )
    assert result.test_name == "Kruskal-Wallis"
    assert result.posthoc_test_name == "Dunn (Bonferroni)"
    assert len(result.posthoc_results) == 3
    for r in result.posthoc_results:
        assert r.p_adj >= r.p_value
    h = result.test_statistic
    assert result.test_statistic_name == "H"
    assert result.effect_size_name == "eta_squared_h"
    assert result.effect_size == pytest.approx((h - 3 + 1) / (600 - 3))


def test_heuristic_kruskal_eta_squared_h_clamped_at_zero(engine):
    """With no effect, H < k - 1 makes the raw formula negative; report 0."""
    rng = np.random.default_rng(0)
    for seed in range(50):
        rng = np.random.default_rng(seed)
        df = make_df({k: rng.exponential(20, 100) for k in "ABC"})
        result = engine.run_comparison_suite(df, ContinuousMetricConfig(kpi="revenue"))
        if result.test_name == "Kruskal-Wallis" and result.test_statistic < 2:
            assert result.effect_size == 0.0
            return
    pytest.fail("no seed produced H < k - 1")


def test_heuristic_mann_whitney_rank_biserial(engine):
    rng = np.random.default_rng(0)
    a, b = rng.exponential(20, 200), rng.exponential(30, 200)
    result = engine.run_comparison_suite(
        make_df({"A": a, "B": b}), ContinuousMetricConfig(kpi="revenue")
    )
    from scipy.stats import mannwhitneyu
    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    assert result.test_statistic_name == "U"
    assert result.test_statistic == pytest.approx(u)
    assert result.effect_size_name == "rank_biserial"
    assert result.effect_size == pytest.approx(1 - 2 * u / (200 * 200))
    assert result.effect_size > 0  # B (second in sort order) is larger
    assert result.posthoc_results is None


def test_heuristic_no_posthoc_when_not_significant(engine):
    rng = np.random.default_rng(0)
    df = make_df({k: rng.normal(100, 15, 200) for k in "ABC"})
    result = engine.run_comparison_suite(df, ContinuousMetricConfig(kpi="revenue"))
    assert result.is_significant is False
    assert result.posthoc_results is None
    assert result.posthoc_test_name is None


def test_heuristic_posthoc_uses_config_alpha(engine):
    """Post-hoc significance follows config.alpha, not a hardcoded 0.05."""
    rng = np.random.default_rng(0)
    df = make_df({"A": rng.normal(100, 15, 200), "B": rng.normal(100, 15, 200),
                  "C": rng.normal(104, 15, 200)})
    strict = engine.run_comparison_suite(
        df, ContinuousMetricConfig(kpi="revenue", alpha=0.2)
    )
    for r in strict.posthoc_results or []:
        assert r.is_significant == (r.p_adj < 0.2)


def test_heuristic_posthoc_labels_match_group_values_for_unsorted_input(engine):
    """
    Labels must follow the data, not row order: a frame listed C, A, B must
    still report C's comparison with C's mean.
    """
    rng = np.random.default_rng(1)
    df = pd.concat([
        make_df({"C": rng.normal(130, 10, 100)}),
        make_df({"A": rng.normal(100, 10, 100)}),
        make_df({"B": rng.normal(100, 10, 100)}),
    ], ignore_index=True)
    result = engine.run_comparison_suite(df, ContinuousMetricConfig(kpi="revenue"))
    by_comp = {r.comparison: r for r in result.posthoc_results}
    means = df.groupby(GROUP_COL)["revenue"].mean()
    assert by_comp["C vs A"].mean_diff == pytest.approx(means["C"] - means["A"])
    assert by_comp["C vs A"].mean_diff > 20


def test_gamma_path_reports_lr_statistic_and_no_heuristic_diagnostics(engine):
    rng = np.random.default_rng(0)
    df = make_df({"A": rng.gamma(2.0, 30.0, 300), "B": rng.gamma(2.0, 45.0, 300)})
    result = engine.run_comparison_suite(df, ContinuousMetricConfig(
        kpi="revenue", approach=ContinuousApproach.GAMMA_GLM,
    ))
    assert result.test_statistic_name == "LR"
    assert result.test_statistic > 0
    assert result.effect_size is None
    assert result.normality_p is None and result.levene_p is None


def test_negbin_path_reports_variance_mean_ratio(engine):
    rng = np.random.default_rng(0)
    a = rng.negative_binomial(2, 0.4, 400)
    b = rng.negative_binomial(2, 0.3, 400)
    df = make_df({"A": a, "B": b})
    result = engine.run_comparison_suite(df, ContinuousMetricConfig(
        kpi="revenue", unit=AnalysisUnit.PER_VISITOR,
    ))
    assert result.test_name == "Negative Binomial Regression (LRT)"
    pooled = np.r_[a, b].astype(float)
    assert result.variance_mean_ratio == pytest.approx(
        pooled.var(ddof=1) / pooled.mean()
    )
    assert result.test_statistic_name == "LR"


# --------------------------------------------------------------------- #
#  PairwiseComparisonResult back-compat
# --------------------------------------------------------------------- #


def test_gamma_posthoc_result_alias_and_deprecated_fields(engine):
    from foe.core.models import GammaPosthocResult, PairwiseComparisonResult
    assert GammaPosthocResult is PairwiseComparisonResult
    r = ContinuousMetricEngine._lrt_posthoc_result("B", "A", 4.2, 0.02, 2, 0.05)
    assert r.lrt_stat == r.statistic == 4.2
    assert r.p_adj_bonferroni == r.p_adj == pytest.approx(0.04)
    assert (r.group_a, r.group_b) == ("B", "A")


def test_p_adj_bonferroni_undefined_for_tukey():
    r = ContinuousMetricEngine.tukey_hsd(REF_LABELS, REF_ARRAYS)[0]
    with pytest.raises(AttributeError, match="p_adj"):
        r.p_adj_bonferroni


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


@pytest.mark.parametrize("shape", [0.05, 0.4, 1.8, 12.0, 500.0])
def test_fit_gamma_solves_both_score_equations(shape):
    """
    At the exact MLE: k * theta == mean(x), and the profile equation
    log k - digamma(k) == log(mean(x)) - mean(log(x)) holds to machine precision.
    """
    from scipy.special import digamma
    x = np.random.default_rng(3).gamma(shape, 10.0, 3000)
    k, theta, log_lik = ContinuousMetricEngine.fit_gamma(x)
    assert k * theta == pytest.approx(x.mean(), rel=1e-13)
    s = np.log(x.mean()) - np.log(x).mean()
    assert np.log(k) - digamma(k) == pytest.approx(s, rel=1e-11)
    assert log_lik == pytest.approx(
        -ContinuousMetricEngine.neg_log_likelihood([k, theta], x), rel=1e-14
    )


def test_fit_gamma_is_a_local_maximum():
    from scipy import stats
    x = np.random.default_rng(11).gamma(1.7, 40.0, 2500)
    k, theta, ll = ContinuousMetricEngine.fit_gamma(x)
    for dk, dt in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
        kk, tt = k * (1 + 1e-5 * dk), theta * (1 + 1e-5 * dt)
        assert stats.gamma.logpdf(x, a=kk, scale=tt).sum() < ll


def test_fit_gamma_never_worse_than_generic_optimiser():
    """
    Regression for the previous fit: scipy.optimize.minimize from method-of-
    moments starts stopped short of the maximum on ~18% of random samples
    (log-likelihood up to 0.7 low). The exact fit must reach at least as high
    a log-likelihood on every sample.
    """
    from scipy.optimize import minimize
    rng = np.random.default_rng(1)
    shortfalls = []
    for _ in range(150):
        x = rng.gamma(rng.uniform(0.5, 4), rng.uniform(5, 80), int(rng.integers(200, 3000)))
        _, _, ll_exact = ContinuousMetricEngine.fit_gamma(x)
        m, v = x.mean(), x.var(ddof=1)
        res = minimize(ContinuousMetricEngine.neg_log_likelihood, [m * m / v, v / m],
                       args=(x,), bounds=((1e-5, None), (1e-5, None)))
        shortfalls.append(ll_exact - (-res.fun))
    assert min(shortfalls) > -1e-8          # never worse
    assert max(shortfalls) > 1e-3           # and the old fit really did fall short


def test_fit_gamma_drops_missing_values():
    x = np.random.default_rng(2).gamma(2.0, 5.0, 500)
    with_nan = np.r_[x, np.nan, np.nan]
    assert ContinuousMetricEngine.fit_gamma(with_nan) == ContinuousMetricEngine.fit_gamma(x)


@pytest.mark.parametrize("data, match", [
    ([1.0, 2.0, 0.0], "x > 0"),
    ([1.0, -2.0, 3.0], "x > 0"),
    ([4.0, 4.0, 4.0], "equal"),
    ([4.0], "two observations"),
    ([np.nan, 3.0], "two observations"),
])
def test_fit_gamma_rejects_invalid_input(data, match):
    with pytest.raises(ValueError, match=match):
        ContinuousMetricEngine.fit_gamma(data)


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


def test_run_negbin_lrt_raises_on_all_zero_sample(engine):
    """
    An all-zero sample carries no information about any mean or the
    dispersion. _run_negbin_lrt must raise rather than let a meaningless
    p-value out; callers treat that as unfittable.
    """
    df = pd.DataFrame({GROUP_COL: ["A", "A", "B", "B"], "items": [0.0, 0.0, 0.0, 0.0]})
    with pytest.raises(ValueError, match="every count is zero"):
        ContinuousMetricEngine._run_negbin_lrt(df, "items", GROUP_COL)


def test_run_negbin_lrt_tests_an_all_zero_variant(engine):
    """
    A variant with only zero counts is fitted at mean 0 and tested normally
    (deliberate since the native rewrite; statsmodels sometimes reported such
    data as unfittable). Tiny case checked by hand: Poisson boundary (alpha = 0),
    LR = 2 * [ll(means 0, 1) - ll(mean 0.5)] = 4 log 2, df = 1.
    """
    df = pd.DataFrame({GROUP_COL: ["A", "A", "B", "B"], "items": [0.0, 0.0, 1.0, 1.0]})
    p_value, lr_stat, disp_alpha = ContinuousMetricEngine._run_negbin_lrt(df, "items", GROUP_COL)
    assert disp_alpha == 0.0
    assert lr_stat == pytest.approx(4.0 * math.log(2.0), rel=1e-12)
    assert p_value == pytest.approx(float(stats.chi2.sf(4.0 * math.log(2.0), 1)), rel=1e-12)


def test_negbin_global_test_unfittable_treated_as_non_significant(engine):
    """
    Mirrors the Gamma-path nan guard: an unfittable global NB fit (all-zero
    counts) must fall back to p=1.0 with a warning instead of propagating a
    nan into the result.
    """
    df = pd.DataFrame({GROUP_COL: ["A"] * 3 + ["B"] * 3, "items": [0.0] * 6})
    out = engine._run_negbin_path(df, "items", GROUP_COL, 2, 0.05, None)
    assert out.p_value == 1.0
    assert any("could not be fit" in w for w in out.extra_warnings)


def test_negbin_posthoc_unfittable_pair_reports_non_significant(engine):
    """run_negbin_posthoc's own guard: an unfittable (all-zero) pair reports p=1.0 rather than raising."""
    df = pd.DataFrame({
        GROUP_COL: ["Control", "Control", "Tiny", "Tiny"],
        "items": [0.0, 0.0, 0.0, 0.0],
    })
    results = ContinuousMetricEngine.run_negbin_posthoc(df, "items", GROUP_COL, "Control")
    assert len(results) == 1
    assert results[0].comparison == "Tiny vs Control"
    assert results[0].p_value == 1.0
    assert results[0].is_significant is False


@pytest.mark.filterwarnings("ignore")
def test_negbin_underdispersed_counts_fit_at_poisson_boundary(engine):
    """
    Items per order (always >= 1, variance < mean) puts the NB2 alpha at the
    Poisson boundary (exactly 0). That is not separation: the LRT must still
    run, match R's Poisson GLM LRT, and raise no "could not be fit" warning.
    """
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        GROUP_COL: ["Control"] * 500 + ["B"] * 500,
        "total_item_quantity": np.concatenate(
            [1 + rng.poisson(0.5, 500), 1 + rng.poisson(0.8, 500)]
        ),
    })
    y = df["total_item_quantity"]
    assert y.var() / y.mean() < 1  # the scenario under test

    result = engine.run_comparison_suite(
        df, ContinuousMetricConfig(kpi="total_item_quantity", unit=AnalysisUnit.PER_TRANSACTION)
    )
    # anova(glm(y ~ 1, poisson), glm(y ~ g, poisson), test = "LRT"), R 4.6.2
    ref_lr, ref_p = 12.355849055620013, 0.00043960701722645736

    assert result.test_name == "Negative Binomial Regression (LRT)"
    assert not any("could not be fit" in w for w in result.warnings)
    assert result.is_significant is True
    assert result.p_value < 0.05
    assert result.test_statistic == pytest.approx(ref_lr, rel=1e-10)
    assert result.p_value == pytest.approx(ref_p, rel=1e-9)
    assert result.dispersion_alpha == 0.0


@pytest.mark.filterwarnings("ignore")
def test_negbin_posthoc_underdispersed_three_variants(engine):
    """Pairwise NB LRTs on underdispersed counts give finite p-values; a real lift is significant."""
    rng = np.random.default_rng(3)
    n = 500
    df = make_df({
        "Control": 1 + rng.poisson(0.5, n),
        "Same": 1 + rng.poisson(0.5, n),
        "Lift": 1 + rng.poisson(0.9, n),
    }, kpi="items")

    results = {
        r.comparison: r
        for r in ContinuousMetricEngine.run_negbin_posthoc(df, "items", GROUP_COL, "Control")
    }
    assert set(results) == {"Same vs Control", "Lift vs Control"}
    for r in results.values():
        assert math.isfinite(r.p_value)
        assert 0.0 <= r.p_value <= 1.0
    assert results["Lift vs Control"].is_significant is True
    assert results["Lift vs Control"].p_value < 1e-3

    # The global test routes through the same LRT and then the posthoc.
    result = engine.run_comparison_suite(
        df,
        ContinuousMetricConfig(kpi="items", unit=AnalysisUnit.PER_TRANSACTION, control_label="Control"),
    )
    assert result.is_significant is True
    assert not any("could not be fit" in w for w in result.warnings)
    assert result.posthoc_results is not None


@pytest.mark.filterwarnings("ignore")
def test_negbin_equidispersed_null_not_significant(engine):
    """Poisson counts with no lift: not significant, and no "could not be fit" warning."""
    rng = np.random.default_rng(11)
    n = 500
    df = make_df({
        "Control": rng.poisson(2.0, n).astype(float),
        "B": rng.poisson(2.0, n).astype(float),
    }, kpi="items")
    result = engine.run_comparison_suite(
        df, ContinuousMetricConfig(kpi="items", unit=AnalysisUnit.PER_VISITOR)
    )
    assert result.test_name == "Negative Binomial Regression (LRT)"
    assert result.is_significant is False
    assert not any("could not be fit" in w for w in result.warnings)


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
