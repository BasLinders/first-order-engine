import pytest

from foe.core.priors import get_informed_beta_prior


def test_informed_prior_standard_case():
    """weight=0.1 on 1000 visitors / 100 conversions should give Beta(11, 91)."""
    alpha, beta = get_informed_beta_prior(
        hist_conversions=100, hist_visitors=1000, weight=0.1
    )
    assert alpha == pytest.approx(11.0)
    assert beta == pytest.approx(91.0)


def test_informed_prior_zero_weight_gives_uniform():
    """weight=0.0 should produce a flat Beta(1, 1) regardless of history."""
    alpha, beta = get_informed_beta_prior(
        hist_conversions=500, hist_visitors=1000, weight=0.0
    )
    assert alpha == pytest.approx(1.0)
    assert beta == pytest.approx(1.0)


def test_informed_prior_weight_clamped_above_one():
    """weight > 1.0 is clamped to 1.0: full history used as pseudo-observations."""
    alpha, beta = get_informed_beta_prior(
        hist_conversions=100, hist_visitors=1000, weight=5.0
    )
    assert alpha == pytest.approx(101.0)
    assert beta == pytest.approx(901.0)


def test_informed_prior_negative_visitors_raises():
    with pytest.raises(ValueError, match="hist_visitors"):
        get_informed_beta_prior(hist_conversions=10, hist_visitors=-1)


def test_informed_prior_negative_conversions_raises():
    with pytest.raises(ValueError, match="hist_conversions"):
        get_informed_beta_prior(hist_conversions=-1, hist_visitors=100)


def test_informed_prior_conversions_exceed_visitors_raises():
    with pytest.raises(ValueError, match="cannot exceed"):
        get_informed_beta_prior(hist_conversions=200, hist_visitors=100)
