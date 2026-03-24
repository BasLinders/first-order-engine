import numpy as np
from scipy.stats import norm
from typing import Tuple, Dict, Any
from engine.core.models import AlternativeHypothesis

def compute_interval_difference(
    diff_cr: float,
    se_diff: float,
    alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Computes the 1-alpha Confidence Interval for the difference between 
    Variant and Control. Standard two-sided Frequentist approach.
    
    Args:
        diff_cr: The absolute difference in conversion rates (p_chal - p_ctrl).
        se_diff: The standard error of the difference (usually unpooled).
        alpha: Significance level (default 0.05 for 95% CI).
    """
    if se_diff <= 0:
        return (float(diff_cr), float(diff_cr))
        
    # Standard two-sided Z-critical value
    z_critical = norm.ppf(1 - alpha / 2)
    moe = z_critical * se_diff
    
    return (float(diff_cr - moe), float(diff_cr + moe))

def compute_non_inferiority(
    p_ctrl: float,
    p_chal: float,
    se_diff: float,
    margin: float,
    confidence_level: float = 95.0
) -> Dict[str, Any]:
    """
    Calculates Non-Inferiority (NI). This determines if a challenger is 
    'not significantly worse' than the control by more than a predefined margin.
    
    Args:
        p_ctrl: Baseline conversion rate.
        p_chal: Challenger conversion rate.
        se_diff: Standard error of the difference.
        margin: The non-inferiority margin (as a positive float, e.g., 0.01 for 1%).
        confidence_level: The percentage for the one-sided bound (e.g., 95.0).
    """
    # Guard against zero-variance/empty data
    if se_diff <= 0:
        return {
            "p_value": 1.0,
            "lower_bound_diff": float(p_chal - p_ctrl),
            "is_non_inferior": False,
            "margin_used": margin
        }

    # 1. Calculate the Z-statistic for Non-Inferiority
    # H0: (p_chal - p_ctrl) <= -margin (Challenger is inferior)
    # H1: (p_chal - p_ctrl) > -margin  (Challenger is non-inferior)
    diff = p_chal - p_ctrl
    z_stat_ni = (diff + margin) / se_diff
    
    # 2. P-value for the one-sided test
    p_value_ni = 1 - norm.cdf(z_stat_ni)
    
    # 3. One-sided Alpha calculation
    alpha_ni = 1 - (confidence_level / 100)
    z_crit_ni = norm.ppf(1 - alpha_ni)
    
    # 4. The NI Lower Bound
    # This represents the 'worst-case scenario' for the difference.
    # If this value is greater than -margin, we reject inferiority.
    lower_bound_diff = diff - (z_crit_ni * se_diff)
    
    return {
        "p_value": float(p_value_ni),
        "lower_bound_diff": float(lower_bound_diff),
        "is_non_inferior": bool(p_value_ni <= alpha_ni),
        "margin_used": margin,
        "confidence_level": confidence_level
    }
