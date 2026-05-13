from scipy.stats import norm
from typing import Tuple, Dict, Any
from foe.core.models import AlternativeHypothesis


def compute_interval_difference(
    diff_cr: float,
    se_diff: float,
    alpha: float = 0.05,
    alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
) -> Tuple[float, float]:
    """
    Computes the (1 - alpha) confidence interval for the difference between
    Variant and Control conversion rates.

    The interval is directional: one-sided tests return a bound at +/-inf on
    the unconstrained side, which correctly reflects the hypothesis being tested.
    Two-sided tests return a symmetric interval around the observed difference.

    Args:
        diff_cr:     Absolute difference in conversion rates (p_chal - p_ctrl).
        se_diff:     Standard error of the difference (unpooled, φ-adjusted).
        alpha:       Significance level (e.g. 0.05 for a 95% CI). Must match
                     the alpha used in run_synthesis (1 - confidence_level).
        alternative: Direction of the hypothesis test. Controls whether the
                     interval is one- or two-sided.
                     GREATER   -> lower bound only: (diff - z*se, +inf)
                     LESS      -> upper bound only: (-inf, diff + z*se)
                     TWO_SIDED -> symmetric:        (diff - z*se, diff + z*se)
    """
    if se_diff <= 0:
        return (float(diff_cr), float(diff_cr))

    if alternative == AlternativeHypothesis.GREATER:
        z = norm.ppf(1 - alpha)
        return (float(diff_cr - z * se_diff), float("inf"))

    elif alternative == AlternativeHypothesis.LESS:
        z = norm.ppf(1 - alpha)
        return (float("-inf"), float(diff_cr + z * se_diff))

    else:  # TWO_SIDED
        z = norm.ppf(1 - alpha / 2)
        moe = z * se_diff
        return (float(diff_cr - moe), float(diff_cr + moe))


def _generate_ni_conclusion(
    is_non_inferior: bool,
    margin: float,
    lower_bound_diff: float,
    alpha: float,
) -> str:
    """
    Generates a stakeholder-friendly NI conclusion string.

    Args:
        is_non_inferior:  Whether the NI test passed.
        margin:           Non-inferiority margin as a proportion (e.g. 0.01).
        lower_bound_diff: One-sided lower bound of the difference.
        alpha:            Significance level used (e.g. 0.05).
    """
    confidence_pct = (1 - alpha) * 100

    if is_non_inferior:
        return (
            f"Non-inferiority established: We are {confidence_pct:.0f}% confident that the "
            f"challenger is not worse than the control by more than our accepted margin of "
            f"{margin:.2%}. The worst-case scenario is a difference of "
            f"{lower_bound_diff:+.2%}, allowing for a safe rollout."
        )
    return (
        f"Inconclusive / Too Risky: We cannot confidently guarantee that the challenger "
        f"performs within the acceptable margin of {margin:.2%}. The worst-case performance "
        f"could drop to {lower_bound_diff:+.2%}, which violates our safety threshold."
    )


def compute_non_inferiority(
    p_ctrl: float,
    p_chal: float,
    se_diff: float,
    margin: float,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Calculates Non-Inferiority (NI). Determines whether a challenger is
    not significantly worse than the control by more than a predefined margin.

    Hypothesis:
        H0: (p_chal - p_ctrl) <= -margin  (challenger is inferior)
        H1: (p_chal - p_ctrl) >  -margin  (challenger is non-inferior)

    Rejecting H0 (p_value <= alpha) means we have sufficient evidence that
    the challenger is within the acceptable performance margin.

    Args:
        p_ctrl:  Baseline conversion rate.
        p_chal:  Challenger conversion rate.
        se_diff: Standard error of the difference (unpooled, φ-adjusted).
                 Must match the SE used in run_synthesis.
        margin:  Non-inferiority margin as a proportion between 0 and 1
                 (e.g. 0.01 for a 1% margin). Raises ValueError otherwise.
        alpha:   Significance level (e.g. 0.05 for 95% confidence). Must
                 match the alpha used in run_synthesis (1 - confidence_level).
    """
    if not (0 < margin < 1):
        raise ValueError(
            f"margin must be a proportion between 0 and 1 "
            f"(e.g. 0.01 for 1%), got {margin}."
        )

    if se_diff <= 0:
        return {
            "p_value": 1.0,
            "alpha": float(alpha),
            "lower_bound_diff": float(p_chal - p_ctrl),
            "is_non_inferior": False,
            "margin_used": margin,
            "conclusion": "Invalid data: standard error is zero or negative.",
        }

    diff = p_chal - p_ctrl

    # Z-statistic: shift the observed difference by the margin so that
    # the null boundary sits at zero.
    z_stat_ni = (diff + margin) / se_diff

    # One-sided p-value: probability of observing this Z or larger under H0.
    p_value_ni = float(1 - norm.cdf(z_stat_ni))

    # One-sided lower bound: worst-case scenario for the difference.
    # If this stays above -margin, we can reject inferiority.
    z_crit_ni = norm.ppf(1 - alpha)
    lower_bound_diff = float(diff - z_crit_ni * se_diff)

    is_ni = bool(p_value_ni <= alpha)

    return {
        "p_value": p_value_ni,
        "alpha": float(alpha),
        "lower_bound_diff": lower_bound_diff,
        "is_non_inferior": is_ni,
        "margin_used": margin,
        "conclusion": _generate_ni_conclusion(is_ni, margin, lower_bound_diff, alpha),
    }
