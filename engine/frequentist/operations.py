import numpy as np
from scipy.stats import norm
from typing import List

def apply_sidak(alpha: float, num_variants: int) -> float:
    """Calculates the adjusted alpha threshold using Šidák correction."""
    num_comparisons = num_variants - 1
    if num_comparisons <= 1:
        return alpha
    return 1 - (1 - alpha) ** (1 / num_comparisons)

def apply_cuped(rho: float) -> float:
    """Calculates the variance reduction factor based on correlation rho."""
    return 1 - (rho ** 2)

def run_ztest(
    """
    Runs the z-test.
    """
    p_ctrl: float, 
    p_chal: float, 
    n_ctrl: int, 
    n_chal: int, 
    reduction_factor: float = 1.0
) -> float:
    """
    Performs a standard two-sided Z-test for proportions.
    Returns the p-value.
    """
    # Pooled proportion
    p_pooled = (p_ctrl * n_ctrl + p_chal * n_chal) / (n_ctrl + n_chal)
    
    # Standard Error (Pooled, with CUPED adjustment)
    se = np.sqrt(
        p_pooled * (1 - p_pooled) * reduction_factor * (1/n_ctrl + 1/n_chal)
    )
    
    z_score = (p_chal - p_ctrl) / se
    p_value = 2 * (1 - norm.cdf(abs(z_score)))
    return p_value
