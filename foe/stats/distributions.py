"""
Probability distributions used by FOE, built on scipy.special primitives.

The API mirrors scipy.stats (``norm.cdf(x, loc, scale)``, ``chi2.sf(x, df)``,
...) so engine code reads the same, but each formula is written out here.
Only the special functions (normal CDF and its inverse, the regularized
incomplete gamma and beta functions and their inverses, log-gamma, log-beta)
come from scipy.

Conventions, matching scipy.stats:
- Inputs broadcast like numpy arrays. A scalar input returns a numpy scalar.
- Invalid parameters (scale <= 0, df <= 0, shape <= 0) and probabilities
  outside [0, 1] return nan.
- Points outside the support have density 0 (log-density -inf), CDF 0 or 1.
"""

import numpy as np
from scipy import special

_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _out(x: np.ndarray):
    """Returns a numpy scalar for 0-d results, the array otherwise."""
    return x[()]


class _Norm:
    """Normal distribution N(loc, scale^2)."""

    @staticmethod
    def _z(x, loc, scale):
        x, loc, scale = np.broadcast_arrays(_arr(x), _arr(loc), _arr(scale))
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (x - loc) / scale
        return np.where(scale > 0, z, np.nan), scale

    def cdf(self, x, loc=0.0, scale=1.0):
        z, _ = self._z(x, loc, scale)
        return _out(special.ndtr(z))

    def sf(self, x, loc=0.0, scale=1.0):
        # ndtr(-z) instead of 1 - ndtr(z): keeps precision in the upper tail.
        z, _ = self._z(x, loc, scale)
        return _out(special.ndtr(-z))

    def ppf(self, q, loc=0.0, scale=1.0):
        q, loc, scale = np.broadcast_arrays(_arr(q), _arr(loc), _arr(scale))
        valid = (scale > 0) & (q >= 0) & (q <= 1)
        with np.errstate(invalid="ignore"):
            x = loc + scale * special.ndtri(np.where(valid, q, 0.5))
        return _out(np.where(valid, x, np.nan))

    def isf(self, q, loc=0.0, scale=1.0):
        # -ndtri(q) instead of ndtri(1 - q): keeps precision for small q.
        q, loc, scale = np.broadcast_arrays(_arr(q), _arr(loc), _arr(scale))
        valid = (scale > 0) & (q >= 0) & (q <= 1)
        with np.errstate(invalid="ignore"):
            x = loc - scale * special.ndtri(np.where(valid, q, 0.5))
        return _out(np.where(valid, x, np.nan))

    def logpdf(self, x, loc=0.0, scale=1.0):
        z, scale = self._z(x, loc, scale)
        with np.errstate(divide="ignore", invalid="ignore"):
            return _out(-0.5 * z * z - _LOG_SQRT_2PI - np.log(scale))

    def pdf(self, x, loc=0.0, scale=1.0):
        return _out(np.exp(_arr(self.logpdf(x, loc, scale))))


class _Chi2:
    """Chi-squared distribution with df degrees of freedom."""

    def sf(self, x, df):
        x, df = np.broadcast_arrays(_arr(x), _arr(df))
        valid = df > 0
        with np.errstate(invalid="ignore"):
            p = special.gammaincc(np.where(valid, df, 1.0) / \
                                  2.0, np.maximum(x, 0.0) / 2.0)
        p = np.where(x < 0, 1.0, p)
        return _out(np.where(valid & ~np.isnan(x), p, np.nan))

    def ppf(self, q, df):
        q, df = np.broadcast_arrays(_arr(q), _arr(df))
        valid = (df > 0) & (q >= 0) & (q <= 1)
        with np.errstate(invalid="ignore"):
            x = 2.0 * special.gammaincinv(np.where(valid,
                                          df, 1.0) / 2.0, np.where(valid, q, 0.5))
        return _out(np.where(valid, x, np.nan))


class _F:
    """F distribution with (dfn, dfd) degrees of freedom."""

    def sf(self, x, dfn, dfd):
        # P(F > x) = I_{dfd / (dfd + dfn x)}(dfd / 2, dfn / 2)
        x, dfn, dfd = np.broadcast_arrays(_arr(x), _arr(dfn), _arr(dfd))
        valid = (dfn > 0) & (dfd > 0) & ~np.isnan(x)
        dfn_s = np.where(valid, dfn, 1.0)
        dfd_s = np.where(valid, dfd, 1.0)
        xp = np.maximum(np.where(valid, x, 0.0), 0.0)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            w = np.where(np.isinf(xp), 0.0, dfd_s / (dfd_s + dfn_s * xp))
            p = special.betainc(dfd_s / 2.0, dfn_s / 2.0, w)
        return _out(np.where(valid, p, np.nan))


class _Gamma:
    """Gamma distribution with shape a and scale."""

    def logpdf(self, x, a, scale=1.0):
        x, a, scale = np.broadcast_arrays(_arr(x), _arr(a), _arr(scale))
        valid = (a > 0) & (scale > 0)
        a_s = np.where(valid, a, 1.0)
        scale_s = np.where(valid, scale, 1.0)
        y = x / scale_s
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = special.xlogy(a_s - 1.0, y) - y - \
                               special.gammaln(a_s) - np.log(scale_s)
        lp = np.where(x < 0, -np.inf, lp)
        return _out(np.where(valid, lp, np.nan))


class _Beta:
    """Beta distribution with shapes a and b on [0, 1]."""

    def pdf(self, x, a, b):
        x, a, b = np.broadcast_arrays(_arr(x), _arr(a), _arr(b))
        valid = (a > 0) & (b > 0)
        a_s = np.where(valid, a, 1.0)
        b_s = np.where(valid, b, 1.0)
        inside = (x >= 0) & (x <= 1)
        xi = np.where(inside, x, 0.5)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            lp = (
                special.xlogy(a_s - 1.0, xi)
                + special.xlog1py(b_s - 1.0, -xi)
                - special.betaln(a_s, b_s)
            )
            d = np.exp(lp)
        d = np.where(inside, d, 0.0)
        return _out(np.where(valid & ~np.isnan(x), d, np.nan))

    def ppf(self, q, a, b):
        q, a, b = np.broadcast_arrays(_arr(q), _arr(a), _arr(b))
        valid = (a > 0) & (b > 0) & (q >= 0) & (q <= 1)
        a_s, b_s = np.where(valid, a, 1.0), np.where(valid, b, 1.0)
        q_s = np.where(valid, q, 0.5)
        with np.errstate(invalid="ignore", divide="ignore"):
            x = special.betaincinv(a_s, b_s, q_s)
            # betaincinv returns nan for extremely small q (e.g. q < 1e-186 at
            # a=2, b=5). There x is tiny and the leading series term
            # I_x(a, b) ~ x^a / (a B(a, b)) is exact to machine precision.
            x = np.where(
                np.isnan(x) & (q_s > 0),
                np.exp((np.log(q_s) + np.log(a_s) + special.betaln(a_s, b_s)) / a_s),
                x,
            )
        return _out(np.where(valid, x, np.nan))


norm = _Norm()
chi2 = _Chi2()
f = _F()
gamma = _Gamma()
beta = _Beta()

__all__ = ["norm", "chi2", "f", "gamma", "beta"]
