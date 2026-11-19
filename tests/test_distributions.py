"""
foe.stats.distributions against scipy.stats as an independent oracle.

scipy.stats is used here only as a test reference; FOE itself computes these
from scipy.special primitives.
"""

import math

import numpy as np
import pytest
from scipy import stats as sps

from foe.stats.distributions import beta, chi2, f, gamma, norm

RTOL = 1e-12

X = np.concatenate([np.linspace(-40, 40, 401), [-np.inf, np.inf, 0.0]])
Q = np.concatenate([np.logspace(-300, -1, 60), np.linspace(0.01, 0.99, 99), 1 - np.logspace(-15, -1, 30), [0.0, 1.0]])


def close(a, b, rtol=RTOL):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=0, equal_nan=True)


# --- normal ------------------------------------------------------------------

@pytest.mark.parametrize("loc,scale", [(0.0, 1.0), (2.5, 0.3), (-1.0, 7.0)])
def test_norm_matches_scipy(loc, scale):
    close(norm.cdf(X, loc, scale), sps.norm.cdf(X, loc, scale))
    close(norm.sf(X, loc, scale), sps.norm.sf(X, loc, scale))
    close(norm.pdf(X, loc, scale), sps.norm.pdf(X, loc, scale))
    close(norm.logpdf(X[np.isfinite(X)], loc, scale), sps.norm.logpdf(X[np.isfinite(X)], loc, scale))
    close(norm.ppf(Q, loc, scale), sps.norm.ppf(Q, loc, scale))
    close(norm.isf(Q, loc, scale), sps.norm.isf(Q, loc, scale))


def test_norm_tails_keep_precision():
    # sf and isf must not go through 1 - cdf / ppf(1 - q).
    assert norm.sf(10.0) == pytest.approx(7.61985302416047e-24, rel=1e-12)
    assert norm.isf(1e-20) == pytest.approx(9.262340089798408, rel=1e-12)


def test_norm_known_values():
    assert norm.ppf(0.975) == pytest.approx(1.959963984540054, rel=1e-15)
    assert norm.cdf(0.0) == 0.5
    assert norm.pdf(0.0) == pytest.approx(1 / math.sqrt(2 * math.pi), rel=1e-15)


def test_norm_invalid_inputs():
    assert np.isnan(norm.cdf(1.0, 0.0, 0.0))
    assert np.isnan(norm.cdf(1.0, 0.0, -1.0))
    assert np.isnan(norm.ppf(1.5))
    assert np.isnan(norm.isf(-0.1))
    assert np.isnan(norm.cdf(np.nan))
    assert norm.ppf(0.0) == -np.inf and norm.ppf(1.0) == np.inf


def test_scalar_in_scalar_out():
    for v in (norm.cdf(0.3), chi2.sf(2.0, 1), f.sf(1.0, 2, 3), gamma.logpdf(1.0, 2.0), beta.pdf(0.2, 2, 3)):
        assert np.ndim(v) == 0
        assert isinstance(float(v), float)
    assert norm.ppf([0.1, 0.9]).shape == (2,)


# --- chi-squared -------------------------------------------------------------

@pytest.mark.parametrize("df", [1, 2, 3, 7.5, 50, 1000])
def test_chi2_matches_scipy(df):
    x = np.concatenate([np.linspace(0, 3 * df + 50, 300), [-1.0, np.inf]])
    close(chi2.sf(x, df), sps.chi2.sf(x, df))
    close(chi2.ppf(Q, df), sps.chi2.ppf(Q, df), rtol=1e-10)


def test_chi2_invalid_inputs():
    assert np.isnan(chi2.sf(1.0, 0))
    assert np.isnan(chi2.ppf(1.2, 3))
    assert np.isnan(chi2.sf(np.nan, 3))
    assert chi2.sf(-1.0, 3) == 1.0


# --- F -----------------------------------------------------------------------

@pytest.mark.parametrize("dfn,dfd", [(1, 1), (1, 30), (2, 10), (4, 995), (12.5, 3.2)])
def test_f_matches_scipy(dfn, dfd):
    x = np.concatenate([np.linspace(0, 60, 400), [-1.0, np.inf, 1e6]])
    close(f.sf(x, dfn, dfd), sps.f.sf(x, dfn, dfd), rtol=1e-11)


def test_f_invalid_inputs():
    assert np.isnan(f.sf(1.0, 0, 5))
    assert np.isnan(f.sf(1.0, 5, -1))
    assert np.isnan(f.sf(np.nan, 2, 5))


# --- gamma -------------------------------------------------------------------

@pytest.mark.parametrize("a,scale", [(0.5, 1.0), (1.0, 2.0), (3.7, 0.25), (150.0, 10.0)])
def test_gamma_logpdf_matches_scipy(a, scale):
    x = np.concatenate([np.linspace(1e-6, 50 * scale + a * scale * 3, 400), [-1.0]])
    close(gamma.logpdf(x, a=a, scale=scale), sps.gamma.logpdf(x, a=a, scale=scale), rtol=1e-11)


def test_gamma_logpdf_at_zero_and_invalid():
    assert gamma.logpdf(0.0, a=1.0, scale=2.0) == pytest.approx(sps.gamma.logpdf(0.0, a=1.0, scale=2.0))
    assert gamma.logpdf(0.0, a=0.5) == np.inf
    assert gamma.logpdf(0.0, a=2.0) == -np.inf
    assert np.isnan(gamma.logpdf(1.0, a=0.0))
    assert np.isnan(gamma.logpdf(1.0, a=1.0, scale=0.0))


# --- beta --------------------------------------------------------------------

@pytest.mark.parametrize("a,b", [(1, 1), (0.5, 0.5), (2, 5), (31, 970), (1500.5, 30000.5)])
def test_beta_matches_scipy(a, b):
    x = np.concatenate([np.linspace(0, 1, 401), [-0.1, 1.1]])
    close(beta.pdf(x, a, b), sps.beta.pdf(x, a, b), rtol=1e-9)
    # Below ~1e-100 scipy.stats.beta.ppf's root-finder gives up and returns a
    # wrong value, so the oracle comparison stops there; the round-trip test
    # below covers the deep tail.
    q = Q[Q > 1e-90]
    close(beta.ppf(q, a, b), sps.beta.ppf(q, a, b), rtol=1e-10)


@pytest.mark.parametrize("a,b", [(2, 5), (31, 970), (1500.5, 30000.5)])
def test_beta_ppf_round_trips_in_deep_tail(a, b):
    from scipy.special import betainc

    q = np.logspace(-300, -1, 300)
    x = beta.ppf(q, a, b)
    assert np.all(np.isfinite(x)) and np.all(x > 0)
    close(betainc(a, b, x), q, rtol=1e-9)


def test_beta_invalid_inputs():
    assert np.isnan(beta.pdf(0.5, 0, 1))
    assert np.isnan(beta.ppf(0.5, 1, -2))
    assert np.isnan(beta.ppf(1.5, 2, 2))
    assert beta.pdf(1.2, 2, 2) == 0.0
