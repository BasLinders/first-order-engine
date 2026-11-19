"""
Confidence intervals for binomial proportions.

Conventions match statsmodels' ``proportion_confint(method="wilson")`` and
``confint_proportions_2indep(method="newcomb", compare="diff")``, which FOE
used before, and R's ``DescTools::BinomDiffCI(method="score")``, which
r_validation checks against.
"""

import math
from typing import Tuple

from foe.stats.distributions import norm


def _check_counts(count: float, nobs: float) -> None:
    if nobs <= 0:
        raise ValueError(f"nobs must be positive, got {nobs}.")
    if count < 0 or count > nobs:
        raise ValueError(f"count must be in [0, nobs], got count={count}, nobs={nobs}.")


def wilson_interval(count: float, nobs: float, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Two-sided Wilson score interval for a single proportion count / nobs at
    level 1 - alpha. No continuity correction. Always within [0, 1].
    """
    _check_counts(count, nobs)
    p = count / nobs
    z = float(norm.isf(alpha / 2.0))
    z2 = z * z
    denom = 1.0 + z2 / nobs
    center = (p + z2 / (2.0 * nobs)) / denom
    half = z * math.sqrt(p * (1.0 - p) / nobs + z2 / (4.0 * nobs * nobs)) / denom
    return center - half, center + half


def newcombe_diff_interval(
    count1: float, nobs1: float, count2: float, nobs2: float, alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Newcombe's hybrid score interval (method 10, Newcombe 1998) for
    p1 - p2, two-sided at level 1 - alpha. Combines the two Wilson intervals:

        low  = d - sqrt((p1 - l1)^2 + (u2 - p2)^2)
        high = d + sqrt((u1 - p1)^2 + (p2 - l2)^2)

    Well defined at zero counts, unlike the Wald interval.
    """
    _check_counts(count1, nobs1)
    _check_counts(count2, nobs2)
    p1, p2 = count1 / nobs1, count2 / nobs2
    l1, u1 = wilson_interval(count1, nobs1, alpha)
    l2, u2 = wilson_interval(count2, nobs2, alpha)
    d = p1 - p2
    low = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    high = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return low, high
