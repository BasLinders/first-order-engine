import numpy as np
from scipy.stats import norm
from typing import Tuple, Dict

def compute_ci(
    rate: float, 
    n: int, 
    alpha: float, 
    reduction_factor: float = 1.0
) -> Tuple[float, float]:
    """Computes the confidence interval for a single proportion."""
    z_crit = norm.ppf(1 - alpha / 2)
    se = np.sqrt((rate * (1 - rate) * reduction_factor) / n)
    
    lower = rate - z_crit * se
    upper = rate + z_crit * se
    return (lower, upper)

def compute_non_inferiority(
    p_ctrl: float,
    p_chal: float,
    n_ctrl: int,
    n_chal: int,
    margin: float,
    confidence_level: float,
    reduction_factor: float = 1.0
) -> Dict:
    """
    Calculates non-inferiority stats based on the Z-test logic from hexkit.
    """
    se_unpooled = np.sqrt(
        (p_ctrl * (1 - p_ctrl) * reduction_factor / n_ctrl) + 
        (p_chal * (1 - p_chal) * reduction_factor / n_chal)
    )
    
    # Non-inferiority Z-stat: (Difference + Margin) / SE
    z_stat_ni = (p_chal - p_ctrl + margin) / se_unpooled
    p_value_ni = 1 - norm.cdf(z_stat_ni)
    
    alpha_ni = 1 - (confidence_level / 100)
    z_crit_ni = norm.ppf(1 - alpha_ni)
    lower_bound_diff = (p_chal - p_ctrl) - (z_crit_ni * se_unpooled)
    
    return {
        "p_value": p_value_ni,
        "lower_bound_diff": lower_bound_diff,
        "is_non_inferior": p_value_ni <= alpha_ni
    }
