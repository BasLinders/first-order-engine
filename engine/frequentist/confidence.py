import numpy as np
from scipy.stats import norm
from typing import Tuple, Dict
from engine.core.models import AlternativeHypothesis

def compute_interval_difference(
    diff_cr: float,
    se_diff: float,
    alpha: float
) -> Tuple[float, float]:
    """
    Computes the Confidence Interval for the difference between 
    Variant and Control (Two-sided).
    """
    z_critical = norm.ppf(1 - alpha / 2)
    moe = z_critical * se_diff
    return (diff_cr - moe, diff_cr + moe)

def compute_non_inferiority(
    p_ctrl: float,
    p_chal: float,
    se_diff: float,
    margin: float,
    confidence_level: float
) -> Dict:
    """
    Calculates non-inferiority using unpooled SE.
    """
    # Non-inferiority Z-stat: (Difference + Margin) / SE
    z_stat_ni = (p_chal - p_ctrl + margin) / se_diff
    p_value_ni = 1 - norm.cdf(z_stat_ni)
    
    alpha_ni = 1 - (confidence_level / 100)
    z_crit_ni = norm.ppf(1 - alpha_ni)
    
    # Lower bound of the difference for NI
    lower_bound_diff = (p_chal - p_ctrl) - (z_crit_ni * se_diff)
    
    return {
        "p_value": p_value_ni,
        "lower_bound_diff": lower_bound_diff,
        "is_non_inferior": p_value_ni <= alpha_ni
    }
