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
    Variant and Control. Standard two-sided approach.
    """
    if se_diff == 0:
        return (diff_cr, diff_cr)
        
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
    Calculates non-inferiority. This determines if the challenger 
    is 'not significantly worse' than the control by more than the margin.
    """
    if se_diff == 0:
        return {
            "p_value": 1.0,
            "lower_bound_diff": p_chal - p_ctrl,
            "is_non_inferior": False,
            "margin_used": margin
        }

    # 1. Calculate Z-stat for Non-Inferiority
    # H0: Difference <= -Margin | H1: Difference > -Margin
    diff = p_chal - p_ctrl
    z_stat_ni = (diff + margin) / se_diff
    
    # 2. P-value for the one-sided test
    p_value_ni = 1 - norm.cdf(z_stat_ni)
    
    # 3. Alpha calculation (one-sided)
    alpha_ni = 1 - (confidence_level / 100)
    z_crit_ni = norm.ppf(1 - alpha_ni)
    
    # 4. The NI Lower Bound (one-sided confidence interval)
    # If this bound is > -margin, we have non-inferiority.
    lower_bound_diff = diff - (z_crit_ni * se_diff)
    
    return {
        "p_value": float(p_value_ni),
        "lower_bound_diff": float(lower_bound_diff),
        "is_non_inferior": bool(p_value_ni <= alpha_ni),
        "margin_used": margin,
        "confidence_level": confidence_level
    }
