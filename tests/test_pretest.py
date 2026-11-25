import numpy as np
import pytest
from scipy.stats import norm

from foe.pretest.operations import PretestEngine
from foe.core.models import (
    AlternativeHypothesis,
    AnalysisUnit,
    ExperimentInput,
    VarianceScaling,
)
from foe.frequentist.operations import FrequentistEngine

Engine = PretestEngine
TWO_SIDED = AlternativeHypothesis.TWO_SIDED


# --------------------------------------------------------------------- #
#  compound_per_visitor_moments
# --------------------------------------------------------------------- #


def test_compound_moments_match_manual_delta_method():
    """mean = p*m; var = p*v + m**2*p*(1-p), per the documented formula."""
    p, m, v = 0.10, 50.0, 400.0
    mean, variance = Engine.compound_per_visitor_moments(m, v, p)

    assert mean == pytest.approx(p * m)
    assert variance == pytest.approx(p * v + m**2 * p * (1 - p))


def test_compound_moments_full_conversion_reduces_to_txn_moments():
    """At p=1 every visitor buys, so the per-visitor distribution IS the
    per-transaction distribution: mean=m, variance=v exactly."""
    m, v = 75.0, 900.0
    mean, variance = Engine.compound_per_visitor_moments(m, v, conversion_rate=1.0)

    assert mean == pytest.approx(m)
    assert variance == pytest.approx(v)


def test_compound_moments_match_monte_carlo_simulation():
    """
    End-to-end correctness check: simulate Y = Bernoulli(p) * X directly
    (X ~ Gamma with mean m, variance v among buyers) and confirm the
    closed-form moments match the empirical ones from a large sample.
    """
    rng = np.random.default_rng(0)
    p, m, v = 0.20, 60.0, 1200.0
    n = 2_000_000

    shape = m**2 / v
    scale = v / m
    buys = rng.random(n) < p
    x = np.where(buys, rng.gamma(shape, scale, size=n), 0.0)

    mean, variance = Engine.compound_per_visitor_moments(m, v, p)
    assert mean == pytest.approx(x.mean(), rel=0.02)
    assert variance == pytest.approx(x.var(), rel=0.05)


def test_compound_moments_variance_can_exceed_txn_variance():
    """
    The zero mass adds a between-group term (m**2*p*(1-p)) on top of the
    scaled-down within-group term (p*v). Whether the total exceeds the
    buyer-only spend variance `v` depends on the parameters (the crossover
    is at p == v/m**2) - it is not universal, which is exactly why RPV can't
    be planned by dropping in the buyer-only (txn) moments unexamined. Here,
    a low-CV buyer distribution (v small relative to m**2) makes the
    between-group term dominate and pushes the compound variance above v.
    """
    p, m, v = 0.10, 50.0, 100.0  # CV among buyers = 0.2; p (0.1) > v/m**2 (0.04)
    _, variance = Engine.compound_per_visitor_moments(m, v, p)

    assert variance > v


def test_compound_moments_variance_can_be_below_txn_variance():
    """The other side of the same crossover: a high-CV buyer distribution
    (v large relative to m**2) can leave the compound variance BELOW v, even
    though the per-visitor MEAN is always scaled down to p*m. It's the mean
    rescaling - not a guaranteed variance increase - that RPV planning must
    never skip (see test_mde_table_rpv_understates_effect_if_txn_moments_used_naively)."""
    p, m, v = 0.10, 50.0, 400.0  # p (0.1) < v/m**2 (0.16)
    _, variance = Engine.compound_per_visitor_moments(m, v, p)

    assert variance < v


# --------------------------------------------------------------------- #
#  calculate_mde_table_rpv
# --------------------------------------------------------------------- #


def test_mde_table_rpv_matches_manual_continuous_delegation():
    """The RPV table should be identical to feeding the compound moments
    straight into calculate_mde_table_continuous by hand."""
    p, m, v = 0.10, 50.0, 400.0
    weekly_visitors = 14_000

    rpv_result = Engine.calculate_mde_table_rpv(
        num_variants=2,
        weekly_visitors=weekly_visitors,
        conversion_rate=p,
        txn_mean=m,
        txn_variance=v,
        risk_pct=95,
        trust_pct=80,
    )

    mean, variance = Engine.compound_per_visitor_moments(m, v, p)
    manual_result = Engine.calculate_mde_table_continuous(
        num_variants=2,
        weekly_units=weekly_visitors,
        mean=mean,
        variance=variance,
        risk_pct=95,
        trust_pct=80,
        unit=AnalysisUnit.PER_VISITOR,
    )

    assert rpv_result["table"] == manual_result["table"]
    assert rpv_result["unit"] == AnalysisUnit.PER_VISITOR.value


def test_mde_table_rpv_understates_effect_if_txn_moments_used_naively():
    """
    Regression guard for the actual bug this feature fixes: naively passing
    buyer-only (txn) moments into the continuous path (skipping the compound
    conversion) understates variance and therefore UNDERSTATES the true MDE
    (looks more sensitive than it really is) at a fixed traffic volume.
    """
    p, m, v = 0.10, 50.0, 400.0
    weekly_visitors = 14_000

    correct = Engine.calculate_mde_table_rpv(
        num_variants=2, weekly_visitors=weekly_visitors, conversion_rate=p,
        txn_mean=m, txn_variance=v, risk_pct=95, trust_pct=80,
    )
    naive = Engine.calculate_mde_table_continuous(
        num_variants=2, weekly_units=weekly_visitors, mean=m, variance=v,
        risk_pct=95, trust_pct=80, unit=AnalysisUnit.PER_VISITOR,
    )

    for correct_row, naive_row in zip(correct["table"], naive["table"]):
        assert correct_row["MDE"] > naive_row["MDE"]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(weekly_visitors=0, conversion_rate=0.1, txn_mean=50, txn_variance=400),
        dict(weekly_visitors=1000, conversion_rate=0.0, txn_mean=50, txn_variance=400),
        dict(weekly_visitors=1000, conversion_rate=1.5, txn_mean=50, txn_variance=400),
        dict(weekly_visitors=1000, conversion_rate=0.1, txn_mean=0, txn_variance=400),
        dict(weekly_visitors=1000, conversion_rate=0.1, txn_mean=50, txn_variance=0),
    ],
)
def test_mde_table_rpv_rejects_invalid_inputs(kwargs):
    result = Engine.calculate_mde_table_rpv(
        num_variants=2, risk_pct=95, trust_pct=80, **kwargs
    )
    assert result["table"] == []


# --------------------------------------------------------------------- #
#  calculate_fixed_sample_size_rpv
# --------------------------------------------------------------------- #


def test_sample_size_rpv_matches_manual_continuous_delegation():
    p, m, v = 0.10, 50.0, 400.0

    rpv_result = Engine.calculate_fixed_sample_size_rpv(
        conversion_rate=p, txn_mean=m, txn_variance=v,
        mde_relative=0.10, num_variants=2,
    )

    mean, variance = Engine.compound_per_visitor_moments(m, v, p)
    manual_result = Engine.calculate_fixed_sample_size_continuous(
        mean=mean, variance=variance, mde_relative=0.10, num_variants=2,
        unit=AnalysisUnit.PER_VISITOR,
    )

    assert rpv_result["n_per_variant"] == manual_result["n_per_variant"]
    assert rpv_result["unit"] == AnalysisUnit.PER_VISITOR.value


def test_sample_size_rpv_larger_than_naive_txn_moments():
    """Same regression guard as the MDE table: skipping the compound
    conversion understates variance, so the naive path under-recommends
    sample size relative to the statistically correct RPV requirement."""
    p, m, v = 0.10, 50.0, 400.0

    correct = Engine.calculate_fixed_sample_size_rpv(
        conversion_rate=p, txn_mean=m, txn_variance=v,
        mde_relative=0.10, num_variants=2,
    )
    naive = Engine.calculate_fixed_sample_size_continuous(
        mean=m, variance=v, mde_relative=0.10, num_variants=2,
        unit=AnalysisUnit.PER_VISITOR,
    )

    assert correct["n_per_variant"] > naive["n_per_variant"]


def test_sample_size_rpv_rejects_invalid_inputs():
    result = Engine.calculate_fixed_sample_size_rpv(
        conversion_rate=0.0, txn_mean=50, txn_variance=400,
        mde_relative=0.10, num_variants=2,
    )
    assert result["n_per_variant"] == 0
    assert result["total_n"] == 0


# --------------------------------------------------------------------- #
#  calculate_power_rpv
# --------------------------------------------------------------------- #


def test_power_rpv_matches_manual_continuous_delegation():
    p, m, v = 0.10, 50.0, 400.0

    rpv_result = Engine.calculate_power_rpv(
        conversion_rate=p, txn_mean=m, txn_variance=v,
        expected_lift=0.10, n_per_variant=5000, num_variants=2,
    )

    mean, variance = Engine.compound_per_visitor_moments(m, v, p)
    manual_result = Engine.calculate_power_continuous(
        mean=mean, variance=variance, expected_lift=0.10,
        n_per_variant=5000, num_variants=2, unit=AnalysisUnit.PER_VISITOR,
    )

    assert rpv_result["power"] == pytest.approx(manual_result["power"])


def test_power_rpv_roundtrips_with_sample_size_rpv():
    """Plugging the RPV sample-size recommendation back into the RPV power
    method should meet (at least) the target power it was sized for."""
    p, m, v = 0.10, 50.0, 400.0
    target_power = 0.80

    sizing = Engine.calculate_fixed_sample_size_rpv(
        conversion_rate=p, txn_mean=m, txn_variance=v,
        mde_relative=0.10, num_variants=2, power=target_power,
    )
    power_check = Engine.calculate_power_rpv(
        conversion_rate=p, txn_mean=m, txn_variance=v,
        expected_lift=0.10, n_per_variant=sizing["n_per_variant"],
        num_variants=2, target_power=target_power,
    )

    assert power_check["power"] >= target_power - 1e-6
    assert power_check["is_powered"] is True


def test_power_rpv_rejects_invalid_inputs():
    result = Engine.calculate_power_rpv(
        conversion_rate=0.1, txn_mean=50, txn_variance=400,
        expected_lift=0.10, n_per_variant=0, num_variants=2,
    )
    assert result["power"] == 0.0


# --------------------------------------------------------------------- #
#  Pretest power / sample size target run_synthesis's unpooled z-test
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("baseline_cr, mde", [(0.01, 0.20), (0.002, 0.50), (0.05, 0.10), (0.30, 0.05)])
@pytest.mark.parametrize("alternative", [TWO_SIDED, AlternativeHypothesis.GREATER])
def test_binomial_sample_size_and_power_round_trip(baseline_cr, mde, alternative):
    n = Engine.calculate_fixed_sample_size(
        baseline_cr, mde, num_variants=2, power=0.80, alternative=alternative
    )["n_per_variant"]
    at_n = Engine.calculate_power_binomial(
        baseline_cr, mde, n, num_variants=2, alternative=alternative
    )["power"]
    assert at_n == pytest.approx(0.80, abs=1e-3)
    assert at_n >= 0.80


@pytest.mark.parametrize(
    "scaling", [VarianceScaling.EQUAL, VarianceScaling.CV_CONSTANT]
)
@pytest.mark.parametrize("cv, mde", [(1.5, 0.05), (0.8, 0.03), (2.0, 0.10)])
def test_continuous_sample_size_and_power_round_trip(scaling, cv, mde):
    n = Engine.calculate_fixed_sample_size_continuous(
        1.0, cv**2, mde, num_variants=2, power=0.80, variance_scaling=scaling
    )["n_per_variant"]
    at_n = Engine.calculate_power_continuous(
        1.0, cv**2, mde, n, num_variants=2, variance_scaling=scaling
    )["power"]
    assert at_n == pytest.approx(0.80, abs=1e-3)
    assert at_n >= 0.80


def test_continuous_equal_variance_outputs_unchanged():
    """EQUAL scaling has identical H0/H1 variances, so the unpooled fix must
    not move it. Pinned values were computed before the fix."""
    power = Engine.calculate_power_continuous(1.0, 1.5**2, 0.20, 500, num_variants=2)
    assert power["power"] == pytest.approx(0.5589, abs=1e-4)
    power = Engine.calculate_power_continuous(1.0, 0.8**2, 0.30, 150, num_variants=2)
    assert power["power"] == pytest.approx(0.9011, abs=1e-4)
    for cv, mde, expected_n in [(1.5, 0.20, 883), (0.8, 0.30, 112), (2.0, 0.05, 25117)]:
        n = Engine.calculate_fixed_sample_size_continuous(
            1.0, cv**2, mde, num_variants=2)
        assert n["n_per_variant"] == expected_n


def test_cv_constant_power_below_equal_variance_power():
    """A larger treatment variance raises the unpooled test's threshold too,
    so CV-constant power must sit below the equal-variance figure."""
    kwargs = dict(mean=1.0, variance=1.5**2, expected_lift=0.20,
                  n_per_variant=500, num_variants=2)
    equal = Engine.calculate_power_continuous(**kwargs)["power"]
    cv_const = Engine.calculate_power_continuous(
        **kwargs, variance_scaling=VarianceScaling.CV_CONSTANT
    )["power"]
    assert cv_const < equal
    assert cv_const == pytest.approx(0.480, abs=0.005)  # Monte Carlo: 0.483


def _simulate_unpooled_ztest_rejects(p1, p2, n, alpha, sims, rng):
    """Vectorised copy of run_synthesis's two-sided unpooled Wald z-test
    (phi = 1). Returns (rejection flags, control counts, treatment counts)."""
    c1 = rng.binomial(n, p1, size=sims)
    c2 = rng.binomial(n, p2, size=sims)
    r1, r2 = c1 / n, c2 / n
    se = np.sqrt(r1 * (1 - r1) / n + r2 * (1 - r2) / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (r2 - r1) / se, 0.0)
    return 2 * norm.sf(np.abs(z)) < alpha, c1, c2


def test_simulated_ztest_matches_run_synthesis():
    """Guards the vectorised replica used in the Monte Carlo test below."""
    rng = np.random.default_rng(7)
    rejects, c1, c2 = _simulate_unpooled_ztest_rejects(
        0.01, 0.012, 20_000, 0.05, 200, rng)
    engine = FrequentistEngine()
    for i in range(len(c1)):
        res = engine.run_synthesis(
            ExperimentInput(visitors=[20_000, 20_000],
                            conversions=[int(c1[i]), int(c2[i])])
        )[0]
        assert res.is_significant == bool(rejects[i])


@pytest.mark.parametrize(
    "p1, lift, n",
    [(0.01, 0.20, 20_000), (0.002, 0.50, 10_000), (0.05, 0.10, 30_000)],
)
def test_binomial_power_matches_monte_carlo_of_run_synthesis(p1, lift, n):
    rng = np.random.default_rng(20261117)
    rejects, _, _ = _simulate_unpooled_ztest_rejects(
        p1, p1 * (1 + lift), n, 0.05, 50_000, rng)
    predicted = Engine.calculate_power_binomial(p1, lift, n, num_variants=2)["power"]
    assert rejects.mean() == pytest.approx(predicted, abs=0.01)


# --------------------------------------------------------------------- #
#  Binomial MDE tables invert the unpooled sample size
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("baseline_cr", [0.002, 0.01, 0.03, 0.30])
@pytest.mark.parametrize("num_variants, alternative", [(2, TWO_SIDED), (2, AlternativeHypothesis.GREATER), (3, TWO_SIDED)])
def test_binomial_mde_table_inverts_sample_size_and_power(baseline_cr, num_variants, alternative):
    weekly = 40_000
    table = Engine.calculate_mde_table(
        num_variants, weekly, int(weekly * baseline_cr), risk_pct=95, trust_pct=80,
        alternative=alternative,
    )["table"]
    for row in table:
        n, mde = row["Visitors_Per_Variant"], row["MDE"] / 100
        sized = Engine.calculate_fixed_sample_size(
            baseline_cr, mde, num_variants, power=0.80, alternative=alternative
        )["n_per_variant"]
        assert sized in (n, n + 1)  # exact inverse up to the ceil in the sample size
        power = Engine.calculate_power_binomial(
            baseline_cr, mde, n, num_variants, alternative=alternative
        )["power"]
        assert power == pytest.approx(0.80, abs=1e-3)


def test_binomial_mde_at_recommended_n_is_the_planned_lift():
    """Low-baseline case where the old 2*p*q MDE reported 44.7% for the
    39,143 visitors needed to detect a 50% lift."""
    n = Engine.calculate_fixed_sample_size(0.002, 0.50, 2)["n_per_variant"]
    mde = Engine._relative_mde_binomial(0.002, n, norm.ppf(0.975), norm.ppf(0.80))
    assert mde == pytest.approx(50.0, abs=0.01)


def test_forecast_binomial_mde_matches_static_table():
    """A flat forecast with a binomial KPI must reproduce calculate_mde_table."""
    records = [
        {"ds": d, "pred_count": 10_000.0, "pred_value": 30.0}
        for d in range(42)
    ]
    forecast = Engine.calculate_mde_from_forecast(
        records, 2, risk_pct=95, trust_pct=80)["table"]
    static = Engine.calculate_mde_table(
        2, 70_000, 210, risk_pct=95, trust_pct=80)["table"]
    for f_row, s_row in zip(forecast, static):
        assert f_row["MDE"] == pytest.approx(s_row["MDE"], rel=1e-12)


# --------------------------------------------------------------------- #
#  Continuous MDE tables honour variance_scaling
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("scaling", [VarianceScaling.EQUAL, VarianceScaling.CV_CONSTANT])
@pytest.mark.parametrize("cv", [0.8, 1.5, 6.5])
@pytest.mark.parametrize("num_variants, alternative", [(2, TWO_SIDED), (2, AlternativeHypothesis.GREATER), (3, TWO_SIDED)])
def test_continuous_mde_table_inverts_sample_size_and_power(scaling, cv, num_variants, alternative):
    table = Engine.calculate_mde_table_continuous(
        num_variants, 6_000, 1.0, cv**2, risk_pct=95, trust_pct=80,
        alternative=alternative, variance_scaling=scaling,
    )["table"]
    for row in table:
        n, mde = row["Units_Per_Variant"], row["MDE"] / 100
        sized = Engine.calculate_fixed_sample_size_continuous(
            1.0, cv**2, mde, num_variants, power=0.80, alternative=alternative,
            variance_scaling=scaling,
        )["n_per_variant"]
        assert sized in (n, n + 1)  # exact inverse up to the ceil in the sample size
        power = Engine.calculate_power_continuous(
            1.0, cv**2, mde, n, num_variants, alternative=alternative,
            variance_scaling=scaling,
        )["power"]
        assert power == pytest.approx(0.80, abs=1e-3)


def test_continuous_mde_table_equal_variance_unchanged():
    """EQUAL keeps the closed form (z_a + z_b) * sqrt(2 var / n) / mean."""
    table = Engine.calculate_mde_table_continuous(
        2, 10_000, 55.0, 70.0**2, 95, 80)["table"]
    k = norm.ppf(0.975) + norm.ppf(0.80)
    for row in table:
        expected = k * np.sqrt(2 * 70.0**2 / row["Units_Per_Variant"]) / 55.0 * 100
        assert row["MDE"] == pytest.approx(expected, rel=1e-12)


def test_cv_constant_mde_exceeds_equal_variance_mde():
    kwargs = dict(num_variants=2, weekly_units=4_000, mean=1.0, variance=1.5**2,
                  risk_pct=95, trust_pct=80)
    equal = Engine.calculate_mde_table_continuous(**kwargs)["table"]
    cv_const = Engine.calculate_mde_table_continuous(
        **kwargs, variance_scaling=VarianceScaling.CV_CONSTANT
    )["table"]
    assert all(c["MDE"] > e["MDE"] for c, e in zip(cv_const, equal))


def test_cv_constant_mde_is_none_when_no_lift_reaches_power():
    """n <= K * CV**2: even an unbounded lift cannot reach the target power,
    because the treatment variance grows with the lift."""
    k = (norm.ppf(0.975) + norm.ppf(0.80)) ** 2
    cv = 6.5
    weekly_units = int(2 * k * cv**2 / 3)  # week 1 and 2 below the bound, week 3 at it
    result = Engine.calculate_mde_table_continuous(
        2, weekly_units, 1.0, cv**2, 95, 80, variance_scaling=VarianceScaling.CV_CONSTANT
    )
    mdes = [row["MDE"] for row in result["table"]]
    assert mdes[0] is None and mdes[1] is None
    assert all(m is not None for m in mdes[3:])
    assert "2 weeks of volume is too little" in result["conclusion"]


def test_mde_table_rpv_passes_variance_scaling_through():
    kwargs = dict(num_variants=2, weekly_visitors=50_000, conversion_rate=0.03,
                  txn_mean=70.0, txn_variance=80.0**2, risk_pct=95, trust_pct=80)
    rpv = Engine.calculate_mde_table_rpv(
        **kwargs, variance_scaling=VarianceScaling.CV_CONSTANT)
    mean, var = Engine.compound_per_visitor_moments(70.0, 80.0**2, 0.03)
    manual = Engine.calculate_mde_table_continuous(
        2, 50_000, mean, var, 95, 80, unit=AnalysisUnit.PER_VISITOR,
        variance_scaling=VarianceScaling.CV_CONSTANT,
    )
    assert rpv["table"] == manual["table"]
    assert rpv["variance_scaling"] == VarianceScaling.CV_CONSTANT.value


def test_forecast_cv_constant_mde_matches_static_table():
    records = [{"ds": d, "pred_count": 1_000.0, "pred_value": 1_900.0}
        for d in range(42)]
    forecast = Engine.calculate_mde_from_forecast(
        records, 2, 95, 80, kpi_cv=1.5, variance_scaling=VarianceScaling.CV_CONSTANT
    )["table"]
    static = Engine.calculate_mde_table_continuous(
        2, 7_000, 1.9, (1.5 * 1.9) ** 2, 95, 80, variance_scaling=VarianceScaling.CV_CONSTANT
    )["table"]
    for f_row, s_row in zip(forecast, static):
        assert f_row["MDE"] == pytest.approx(s_row["MDE"], rel=1e-12)
