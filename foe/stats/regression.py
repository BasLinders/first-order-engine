"""
Ordinary least squares with heteroskedasticity-consistent (HC3) standard
errors.

The fit uses the SVD pseudoinverse of the design matrix, singular values at
or below 1e-15 times the largest are treated as zero, as statsmodels' OLS
does. A rank-deficient design (e.g. a covariate that is constant, so its
centered column is all zeros) therefore still fits: the coefficients are the
minimum-norm solution and the identified ones are unaffected.

HC3 (MacKinnon & White 1985):
    Cov(b) = X^+ diag(e_i^2 / (1 - h_i)^2) X^+'
with X^+ the pseudoinverse and h_i the leverages. Inference is normal-based.
Matches R's sandwich::vcovHC(lm(...), type = "HC3") and statsmodels'
OLS(...).fit(cov_type="HC3").
"""

from dataclasses import dataclass

import numpy as np

_RCOND = 1e-15
# Leverage within this of 1 is treated as an exact fit (computed leverages of
# exactly-fit rows land at 1 +/- a few machine epsilons).
_LEVERAGE_ONE_TOL = 1e-8
_DEPENDS_TOL = 1e-8


@dataclass(frozen=True)
class OLSResult:
    params: np.ndarray
    bse: np.ndarray
    cov: np.ndarray
    resid: np.ndarray
    leverage: np.ndarray
    rank: int


def ols_hc3(X, y) -> OLSResult:
    """
    Fits y = X b by least squares and returns HC3 standard errors.
    X must already contain the intercept column; rows must be complete.

    Rows with leverage 1 (e.g. a group of two rows with its own intercept
    and slope, which it fits exactly) have an undefined HC3 weight, 0/0.
    Coefficients that depend on such rows get nan standard errors; the other
    coefficients are unaffected. Point of interest: statsmodels instead
    returns round-off noise here as a finite, falsely precise standard error
    for the dependent coefficients (e.g. 2.18 for a 2-user variant).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("X must be 2-D with one row per element of y.")
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
        raise ValueError("X and y must be finite; drop incomplete rows first.")

    u, s, vt = np.linalg.svd(X, full_matrices=False)
    keep = s > _RCOND * s.max() if s.size else np.zeros(0, dtype=bool)
    u_k, s_k, vt_k = u[:, keep], s[keep], vt[keep]
    pinv = (vt_k.T / s_k) @ u_k.T

    params = pinv @ y
    resid = y - X @ params
    leverage = np.sum(u_k * u_k, axis=1)
    exact_fit = leverage > 1.0 - _LEVERAGE_ONE_TOL
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(exact_fit, 0.0, (resid / (1.0 - leverage)) ** 2)
    cov = (pinv * w) @ pinv.T
    if exact_fit.any():
        # A coefficient depends on a row when its pseudoinverse weight on
        # that row is non-negligible relative to its largest weight.
        scale = np.max(np.abs(pinv), axis=1, keepdims=True)
        depends = np.any(np.abs(pinv[:, exact_fit]) > _DEPENDS_TOL * scale, axis=1)
        cov[depends, :] = np.nan
        cov[:, depends] = np.nan
    with np.errstate(invalid="ignore"):
        bse = np.sqrt(np.diag(cov))
    return OLSResult(params, bse, cov, resid, leverage, int(keep.sum()))
