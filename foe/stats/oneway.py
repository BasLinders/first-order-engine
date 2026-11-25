"""
The one-way model y ~ group, computed from group sums.

With one categorical factor, OLS fitted values are the group means and each
row's leverage is 1 / n_g, so residuals, the ANOVA table and studentized
residuals all have closed forms. Results match statsmodels'
``ols("y ~ C(g)")`` (``resid``, ``anova_lm``, ``resid_studentized_internal``)
and R's ``lm`` / ``anova`` / ``rstandard``.

Row-aligned functions take values and labels of equal length. Rows where the
value or the label is missing are left out of the fit and get nan in the
output, so the result always lines up with the input rows.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from foe.stats.distributions import f


@dataclass(frozen=True)
class OneWayAnova:
    ss_between: float
    ss_within: float
    df_between: int
    df_within: int
    f_stat: float
    p_value: float


def _fit(values, labels):
    """
    Group structure of the complete rows. Returns (valid, y, codes, n_g,
    mean_g): the complete-row mask, their values, a dense group index per
    complete row, and the size and mean of each group that has a complete row.
    """
    y_all = np.asarray(values, dtype=float)
    raw_codes, _ = pd.factorize(pd.Series(labels).to_numpy(), use_na_sentinel=True)
    if y_all.shape != raw_codes.shape:
        raise ValueError("values and labels must have the same length.")
    valid = (raw_codes >= 0) & ~np.isnan(y_all)
    y = y_all[valid]
    # Re-index so groups without any complete row don't count toward k.
    _, codes = np.unique(raw_codes[valid], return_inverse=True)
    n_g = np.bincount(codes).astype(float)
    mean_g = np.bincount(codes, weights=y) / n_g if y.size else np.array([])
    return valid, y, codes, n_g, mean_g


def residuals(values, labels) -> np.ndarray:
    """Row-aligned residuals y - mean(y in its group); nan for incomplete rows."""
    valid, y, codes, _, mean_g = _fit(values, labels)
    out = np.full(valid.shape, np.nan)
    out[valid] = y - mean_g[codes]
    return out


def studentized_residuals_internal(values, labels) -> np.ndarray:
    """
    Row-aligned internally studentized residuals e / (s * sqrt(1 - h)), with
    s^2 = SS_within / (N - k) and leverage h = 1 / n_g. A row alone in its
    group (h = 1) or a fit with N = k gives nan. Incomplete rows get nan.
    """
    valid, y, codes, n_g, mean_g = _fit(values, labels)
    out = np.full(valid.shape, np.nan)
    if y.size == 0:
        return out
    e = y - mean_g[codes]
    df_within = y.size - n_g.size
    h = 1.0 / n_g[codes]
    with np.errstate(divide="ignore", invalid="ignore"):
        mse = float(np.sum(e * e)) / df_within if df_within > 0 else np.nan
        out[valid] = e / np.sqrt(mse * (1.0 - h))
    return out


def anova(groups: Sequence[Sequence[float]]) -> OneWayAnova:
    """
    One-way ANOVA F test across groups (each an array of values, no missing
    values). Equal to type I/II/III sums of squares, since there is one factor.
    """
    arrs = [np.asarray(g, dtype=float) for g in groups]
    if len(arrs) < 2 or any(a.size == 0 for a in arrs):
        raise ValueError("anova needs at least two non-empty groups.")
    n = np.array([a.size for a in arrs], dtype=float)
    means = np.array([a.mean() for a in arrs])
    grand = float(np.concatenate(arrs).mean())
    ss_between = float(np.sum(n * (means - grand) ** 2))
    ss_within = float(sum(np.sum((a - m) ** 2) for a, m in zip(arrs, means)))
    df_between = len(arrs) - 1
    df_within = int(n.sum()) - len(arrs)
    if df_within <= 0:
        return OneWayAnova(ss_between, ss_within, df_between, df_within, float("nan"), float("nan"))
    # numpy division: zero within-group variance gives F = inf (p = 0), or
    # nan when the group means are also equal, instead of raising.
    with np.errstate(divide="ignore", invalid="ignore"):
        f_stat = float(np.float64(ss_between / df_between) / \
                       np.float64(ss_within / df_within))
    p_value = float(f.sf(f_stat, df_between, df_within))
    return OneWayAnova(ss_between, ss_within, df_between, df_within, f_stat, p_value)
