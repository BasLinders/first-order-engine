"""
Maximum-likelihood fits with closed-form or one-dimensional profiles.

gamma_mle     Two-parameter Gamma (shape k, scale theta).
nb2_oneway    Negative Binomial (NB2) with one mean per group and a shared
              dispersion alpha, i.e. the regression y ~ group with a log link.

Both reduce to a single root-finding problem solved to machine precision, so
they reach the exact maximum where a general-purpose optimiser can stop short
(statsmodels' NegativeBinomial, which FOE used before, left alpha ~1e-4
relative off on typical data and returned optimiser noise of ~1e-5..1e-7 for
alpha at the Poisson boundary).
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import digamma, gammaln, xlogy

from foe.stats.distributions import gamma as _gamma_dist

_EPS = float(np.finfo(float).eps)


# ----------------------------------------------------------------------------
# Gamma
# ----------------------------------------------------------------------------

def gamma_mle(data) -> tuple[float, float, float]:
    """
    Exact MLE of a Gamma(shape k, scale theta). Returns (k, theta, log_lik).

    The MLE of theta given k is mean(x) / k. Substituting it into the score
    equation for k leaves the one-dimensional profile equation

        log k - digamma(k) = log(mean(x)) - mean(log(x)) =: s,

    whose left side falls strictly from +inf to 0, so it has exactly one root
    for any s > 0 (s > 0 holds whenever the data are not all equal). It is
    solved with Brent's method to machine precision, starting from Minka's
    (2002) closed-form approximation.

    Missing values are dropped. Raises ValueError if fewer than two values
    remain, any value is <= 0, or all values are equal.
    """
    x = pd.Series(data).dropna().to_numpy(dtype=float)
    if x.size < 2:
        raise ValueError("Need at least two observations to fit a Gamma.")
    if np.any(x <= 0):
        raise ValueError("The Gamma is defined for x > 0; got non-positive values.")

    mean_x = float(x.mean())
    s = math.log(mean_x) - float(np.log(x).mean())
    if not s > 0:
        raise ValueError("All values are equal; the Gamma shape is unbounded.")

    def profile(k: float) -> float:
        return math.log(k) - float(digamma(k)) - s

    # Minka's approximation is within ~1.5% of the root; widen to a bracket.
    k0 = (3.0 - s + math.sqrt((s - 3.0) ** 2 + 24.0 * s)) / (12.0 * s)
    lo, hi = k0 / 2.0, k0 * 2.0
    while profile(lo) <= 0:
        lo /= 2.0
    while profile(hi) >= 0:
        hi *= 2.0
    k = brentq(profile, lo, hi, xtol=1e-15, rtol=4 * _EPS, maxiter=200)

    theta = mean_x / k
    log_lik = float(_gamma_dist.logpdf(x, a=k, scale=theta).sum())
    return float(k), float(theta), log_lik


# ----------------------------------------------------------------------------
# Negative Binomial (NB2), one mean per group
# ----------------------------------------------------------------------------

# Above this largest count, the sums over j < y switch from an exact running
# sum to the log-gamma / digamma form (accurate unless alpha is tiny).
_CUMSUM_MAX_COUNT = 100_000
# Log-spaced alpha grid on which the profile score is scanned for maxima.
_ALPHA_GRID = np.logspace(-10.0, 6.0, 161)


@dataclass(frozen=True)
class NB2OneWayFit:
    """
    alpha:        dispersion (Var = mu + alpha mu^2); exactly 0.0 at the
                  Poisson boundary (counts not overdispersed).
    log_lik:      maximised log-likelihood.
    labels:       group labels, in first-appearance order.
    group_means:  fitted mean per group (the group's sample mean).
    group_totals: sum of the counts per group (0 means the group's log-mean
                  is -inf, i.e. the fit is separated).
    codes:        group index of each observation.
    """
    alpha: float
    log_lik: float
    labels: list
    group_means: np.ndarray
    group_totals: np.ndarray
    codes: np.ndarray


class _NB2Profile:
    """NB2 log-likelihood and its alpha-derivative at fixed means mu."""

    def __init__(self, y: np.ndarray, mu: np.ndarray):
        # Collapse to unique (y, mu) pairs with weights: count KPIs have few
        # distinct values, so every evaluation is cheap.
        pairs, w = np.unique(np.column_stack([y, mu]), axis=0, return_counts=True)
        self.y = pairs[:, 0]
        self.mu = pairs[:, 1]
        self.w = w.astype(float)
        self.yi = self.y.astype(np.int64)
        self.max_y = int(self.yi.max()) if self.yi.size else 0
        self.exact_sums = self.max_y <= _CUMSUM_MAX_COUNT
        self.j = np.arange(self.max_y, dtype=float)
        self.const = float(np.sum(self.w * (xlogy(self.y, self.mu) - gammaln(self.y + 1.0))))
        self.score0 = 0.5 * float(np.sum(self.w * ((self.y - self.mu) ** 2 - self.y)))

    def _sum_over_j(self, terms: np.ndarray) -> np.ndarray:
        """sum_{j < y} terms[j] for each unique y."""
        return np.concatenate([[0.0], np.cumsum(terms)])[self.yi]

    def loglik(self, a: float) -> float:
        if a == 0.0:
            return self.const - float(np.sum(self.w * self.mu))
        if self.exact_sums:
            s = self._sum_over_j(np.log1p(a * self.j))
        else:
            r = 1.0 / a
            s = gammaln(self.y + r) - gammaln(r) - self.y * math.log(r)
        return self.const + float(np.sum(
            self.w * (s - (self.y + 1.0 / a) * np.log1p(a * self.mu))
        ))

    def score(self, a: float) -> float:
        """d loglik / d alpha. At alpha = 0 this is 1/2 sum[(y - mu)^2 - y]."""
        if a == 0.0:
            return self.score0
        if self.exact_sums:
            f = self._sum_over_j(self.j / (1.0 + a * self.j))
        else:
            r = 1.0 / a
            f = (self.y - r * (digamma(self.y + r) - digamma(r))) / a
        x = a * self.mu
        # g(x) = log(1 + x) - x / (1 + x), via its series where it cancels.
        small = x < 1e-3
        xs = np.where(small, x, 0.0)
        g_series = xs**2 / 2 - 2 * xs**3 / 3 + 3 * xs**4 / 4 - 4 * xs**5 / 5 + 5 * xs**6 / 6
        g = np.where(small, g_series, np.log1p(x) - x / (1.0 + x))
        return float(np.sum(self.w * (f - self.y * self.mu / (1.0 + x) + g / (a * a))))

    def maximise(self) -> tuple[float, float]:
        """Returns (alpha_hat, log_lik) over alpha >= 0."""
        best = (0.0, self.loglik(0.0))
        if not np.any(self.y > 0):  # all zeros: alpha is not identified
            return best
        # The grid is scanned even when the score at 0 is <= 0 (boundary is a
        # local maximum), so a second, interior maximum can't be missed.
        grid = np.concatenate([[0.0], _ALPHA_GRID])
        sc = np.array([self.score(a) for a in grid])
        while sc[-1] > 0 and grid[-1] < 1e15:  # extremely overdispersed: extend
            grid = np.append(grid, grid[-1] * 10.0)
            sc = np.append(sc, self.score(grid[-1]))
        # Each + -> - sign change of the score brackets a local maximum.
        for i in np.flatnonzero((sc[:-1] > 0) & (sc[1:] <= 0)):
            a = brentq(self.score, grid[i], grid[i + 1], xtol=1e-300, rtol=4 * _EPS, maxiter=500)
            ll = self.loglik(a)
            if ll > best[1]:
                best = (float(a), ll)
        return best


def nb2_oneway(y, groups=None) -> NB2OneWayFit:
    """
    MLE of an NB2 model with one mean per group and a shared dispersion:
    log mu_i = beta_{g(i)}, Var(y_i) = mu_i + alpha mu_i^2.

    For any fixed alpha the score equations for the group means are solved by
    mu_g = mean(y in g), so the fit reduces to the profile likelihood in
    alpha. Its derivative at alpha = 0 is 1/2 sum[(y - mu)^2 - y], so for
    counts that are not overdispersed the Poisson boundary alpha = 0 is a
    local maximum. Every local maximum (the boundary and each sign change of
    the score on a log grid of alpha, solved to machine precision) is
    compared and the highest is returned, so uniqueness is not assumed.

    The log-likelihood uses sum_{j<y} log(1 + alpha j) instead of the
    textbook log-gamma form, which cancels catastrophically near the
    boundary (~1e-5 error at alpha = 1e-7, which is how statsmodels reports a
    slightly higher log-likelihood at a non-optimal mean there).

    y must be non-negative integers (counts) without missing values; groups
    (labels, one per y) defaults to a single group. Raises ValueError for
    invalid counts or mismatched lengths.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("y must be a non-empty 1-D array of counts.")
    if not np.all(np.isfinite(y)) or np.any(y < 0) or np.any(y != np.round(y)):
        raise ValueError("NB2 needs non-negative integer counts without missing values.")
    if groups is None:
        codes, labels = np.zeros(y.size, dtype=np.int64), [None]
    else:
        codes, uniq = pd.factorize(pd.Series(groups).to_numpy(), use_na_sentinel=True)
        if codes.shape != y.shape:
            raise ValueError("y and groups must have the same length.")
        if np.any(codes < 0):
            raise ValueError("groups contains missing labels.")
        labels = list(uniq)
    n_g = np.bincount(codes, minlength=len(labels)).astype(float)
    totals = np.bincount(codes, weights=y, minlength=len(labels))
    means = totals / n_g

    alpha, log_lik = _NB2Profile(y, means[codes]).maximise()
    return NB2OneWayFit(alpha, log_lik, labels, means, totals, codes)
