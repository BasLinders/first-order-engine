import numpy as np
import pytest

from foe.pretest.operations import PretestEngine
from foe.core.models import AlternativeHypothesis, AnalysisUnit

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
