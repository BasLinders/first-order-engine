import numpy as np
from typing import Tuple, Dict, List

def calculate_msprt_boundaries(alpha: float, beta: float, num_variants: int = 1) -> Tuple[float, float]:
    """
    Calculates the 'Always-Valid' boundaries.
    Crossing the upper bound = Success.
    Crossing the lower bound = Futility.
    """
    # Boundary A (Upper): Conservative threshold for multiple comparisons
    upper = np.log(num_variants / alpha)
    # Boundary B (Lower): Conservative threshold for futility
    lower = np.log(beta)
    
    return upper, lower

def calculate_msprt_llr(
    visitors_base: int, 
    conversions_base: int, 
    visitors_var: int, 
    conversions_var: int, 
    tau: float = 0.01
) -> float:
    """
    Calculates the Log-Likelihood Ratio for a single variant comparison.
    """
    if visitors_base == 0 or visitors_var == 0:
        return 0.0

    p_pool = (conversions_base + conversions_var) / (visitors_base + visitors_var)
    if p_pool <= 0 or p_pool >= 1:
        return 0.0

    # Approximate variance of the difference
    var = p_pool * (1 - p_pool) * (1/visitors_base + 1/visitors_var)
    if var == 0:
        return 0.0

    diff = (conversions_var / visitors_var) - (conversions_base / visitors_base)

    # mSPRT LLR Formula: mixture of normal distributions
    llr = 0.5 * (np.log(var / (var + tau)) + (diff**2 / var) * (tau / (var + tau)))
    return float(llr)
