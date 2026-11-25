"""
foe.stats.proportions and foe.stats.oneway against reference values from R
(DescTools::BinomDiffCI(method = "score"), lm / anova / rstandard), pinned
here so the tests don't depend on statsmodels.
"""

import math

import numpy as np
import pandas as pd
import pytest

from foe.behavioral.operations import BehavioralEngine
from foe.continuous.operations import ContinuousMetricEngine
from foe.core.models import AlternativeHypothesis
from foe.frequentist.boschloo import BoschlooEngine
from foe.frequentist.confidence import compute_newcombe_interval
from foe.stats import oneway
from foe.stats.proportions import newcombe_diff_interval, wilson_interval

# (x1, n1, x2, n2, alpha, low, high) from DescTools::BinomDiffCI(method = "score"), R 4.6.2
NEWCOMBE_R = [
    (0, 10, 0, 10, 0.05, -0.27753279986288909, 0.27753279986288909),
    (0, 10, 0, 10, 0.10, -0.21294197008340685, 0.21294197008340685),
    (10, 10, 10, 10, 0.05, -0.27753279986288909, 0.27753279986288909),
    (10, 10, 10, 10, 0.10, -0.21294197008340682, 0.21294197008340682),
    (0, 5, 5, 5, 0.05, -1, -0.38554900569034389),
    (0, 5, 5, 5, 0.10, -1, -0.5034462826450109),
    (3, 7, 9, 11, 0.05, -0.68978917646653848, 0.046443167157006859),
    (3, 7, 9, 11, 0.10, -0.6597119909018645, -0.015871629529156772),
    (120, 1000, 150, 1000, 0.05, -0.060008258570126545, -1.3782434204044625e-05),
    (120, 1000, 150, 1000, 0.10, -0.055164379301483352, -0.0048514872756019005),
    (499, 500, 1, 500, 0.05, 0.98293166725166947, 0.99832901684619291),
    (499, 500, 1, 500, 0.10, 0.98622194952121711, 0.99819724809653498),
]

Y = [12.1, 9.8, 11.4, 10.2, 30.5, 8.7, 14.2,
    13.9, 15.1, 12.8, 16.4, 7.5, 9.9, 10.6, 11.8]
G = list("AAAAABBBBBCCCCC")
# lm(y ~ g), R 4.6.2
RSTANDARD_R = [
    -0.53657400207069228, -0.99365555939017214, -0.6756857803853169, -0.91416311463895861,
    3.1200784564851411, -0.84261991436286632, 0.25040120096632318, 0.19078186740291306,
    0.42925920165655418, -0.027822355662924809, 1.0254525372906576, -0.74325435842384879,
    -0.26629968991656616, -0.12718791160194218, 0.11128942265169939,
]
RESID_R = [
    -2.7, -5.0, -3.4, -4.6, 15.7, -4.24, 1.26, 0.96, 2.16, - \
        0.14, 5.16, -3.74, -1.34, -0.64, 0.56,
]
ANOVA_R = dict(ss_between=31.705333333333346, ss_within=379.80399999999997,
               f_stat=0.50086886920622242, p_value=0.61812892575554979)


# --- proportions -------------------------------------------------------------

@pytest.mark.parametrize("x1,n1,x2,n2,alpha,low,high", NEWCOMBE_R)
def test_newcombe_matches_r(x1, n1, x2, n2, alpha, low, high):
    got = newcombe_diff_interval(x1, n1, x2, n2, alpha)
    assert got == pytest.approx((low, high), rel=1e-12, abs=1e-15)


def test_wilson_known_value_and_bounds():
    # Wilson 95% for 0/10: (0, z^2 / (n + z^2)).
    z2 = 1.959963984540054 ** 2
    assert wilson_interval(0, 10) == pytest.approx((0.0, z2 / (10 + z2)), abs=1e-15)
    for x, n in [(0, 3), (3, 3), (1, 1000), (999, 1000)]:
        lo, hi = wilson_interval(x, n)
        assert 0.0 <= lo <= x / n <= hi <= 1.0


@pytest.mark.parametrize("bad", [(1, 0), (-1, 5), (6, 5)])
def test_proportions_reject_invalid_counts(bad):
    with pytest.raises(ValueError):
        wilson_interval(*bad)


def test_newcombe_one_sided_and_clipping_conventions():
    # Two-sided at 2*alpha supplies the one-sided bound; the other side is open.
    two = compute_newcombe_interval(120, 1000, 150, 1000, 0.10)
    greater = compute_newcombe_interval(
        120, 1000, 150, 1000, 0.05, AlternativeHypothesis.GREATER)
    less = compute_newcombe_interval(
        120, 1000, 150, 1000, 0.05, AlternativeHypothesis.LESS)
    assert greater == (two[0], math.inf)
    assert less == (-math.inf, two[1])
    # p_chal - p_ctrl with ctrl at 5/5 and chal at 0/5 hits the -1 bound.
    assert compute_newcombe_interval(5, 5, 0, 5)[0] == -1.0


def test_boschloo_delegates_to_shared_newcombe():
    for alt in AlternativeHypothesis:
        assert BoschlooEngine.newcombe_interval(3, 40, 9, 41, 0.05, alt) == \
            compute_newcombe_interval(3, 40, 9, 41, 0.05, alt)


# --- one-way model -----------------------------------------------------------

def test_oneway_matches_r():
    np.testing.assert_allclose(oneway.residuals(Y, G), RESID_R, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        oneway.studentized_residuals_internal(Y, G), RSTANDARD_R, rtol=1e-12)
    groups = [np.array(Y[i:i + 5]) for i in (0, 5, 10)]
    aov = oneway.anova(groups)
    for k, v in ANOVA_R.items():
        assert getattr(aov, k) == pytest.approx(v, rel=1e-12)
    assert (aov.df_between, aov.df_within) == (2, 12)


def test_oneway_incomplete_rows_get_nan_and_dont_count():
    y = Y + [np.nan, 5.0]
    g = G + ["A", None]
    r = oneway.studentized_residuals_internal(y, g)
    assert r.shape == (17,)
    assert np.isnan(r[15]) and np.isnan(r[16])
    np.testing.assert_allclose(r[:15], RSTANDARD_R, rtol=1e-12)
    # A label whose only row is incomplete doesn't add a group (k stays 3).
    r2 = oneway.studentized_residuals_internal(Y + [np.nan], G + ["D"])
    np.testing.assert_allclose(r2[:15], RSTANDARD_R, rtol=1e-12)


def test_oneway_degenerate_cases_are_nan_not_round_off():
    # Singleton group: leverage 1, residual 0/0.
    assert np.isnan(oneway.studentized_residuals_internal(
        [1.0, 2, 3, 4, 10], list("AAAAB"))[4])
    # Zero within-group variance: exact zeros, never studentized into noise.
    r = oneway.studentized_residuals_internal([1.0, 1, 1, 2, 2, 2], list("AAABBB"))
    assert np.all(np.isnan(r))
    aov = oneway.anova([[1.0, 1, 1], [2.0, 2, 2]])
    assert aov.f_stat == math.inf and aov.p_value == 0.0
    assert math.isnan(oneway.anova([[5.0, 5], [5.0, 5]]).f_stat)
    with pytest.raises(ValueError):
        oneway.anova([[1.0, 2.0]])
    # N = k leaves no residual degrees of freedom.
    assert math.isnan(oneway.anova([[1.0], [2.0]]).p_value)
    # No complete rows at all.
    assert np.all(np.isnan(oneway.studentized_residuals_internal(
        [np.nan, np.nan], ["A", "B"])))
    with pytest.raises(ValueError):
        oneway.residuals([1.0, 2.0], ["A"])


# --- engine fixes: row-aligned outlier masks with missing values -------------

def _frame_with_missing():
    df = pd.DataFrame({"value": Y + [np.nan, 4.0],
                      "experience_variant_label": G + ["A", None]})
    df.index = range(100, 117)  # non-default index must not matter
    return df


def test_continuous_outlier_mask_is_row_aligned_with_missing_values():
    df = _frame_with_missing()
    mask = ContinuousMetricEngine.detect_outliers_ols(df, "value", threshold=3.0)
    assert len(mask) == len(df)
    assert mask == [i == 4 for i in range(17)]  # only the 30.5 (rstandard 3.12)


def test_behavioral_outlier_mask_is_row_aligned_with_missing_values():
    df = _frame_with_missing()
    mask = BehavioralEngine.detect_outliers_mask(df, "value", threshold=3.0)
    assert mask == [i == 4 for i in range(17)]


def test_continuous_outlier_mask_accepts_column_names_with_spaces():
    df = pd.DataFrame({"order value": Y, "variant label": G})
    mask = ContinuousMetricEngine.detect_outliers_ols(
        df, "order value", threshold=3.0, group_col="variant label"
    )
    assert mask == [i == 4 for i in range(15)]
