import numpy as np
from scipy.stats import norm
from typing import List, Optional
from engine.core.models import AlternativeHypothesis

def apply_sidak(alpha: float, num_variants: int) -> float:
    """Calculates adjusted alpha for multiple comparisons."""
    num_comparisons = num_variants - 1
    if num_comparisons <= 1:
        return alpha
    return 1 - (1 - alpha) ** (1 / num_comparisons)

def run_ztest(
    diff_cr: float, 
    se_diff: float, 
    alternative: AlternativeHypothesis
) -> float:
    """
    Performs the Z-test using the Unpooled Variance approach.
    Matches the branching logic for 'Greater', 'Less', and 'Two-sided'.
    """
    if se_diff == 0:
        return 1.0
        
    z_stat = diff_cr / se_diff
    
    if alternative == AlternativeHypothesis.GREATER:
        return 1 - norm.cdf(z_stat)
    elif alternative == AlternativeHypothesis.LESS:
        return norm.cdf(z_stat)
    else: # Two-sided
        return 2 * (1 - norm.cdf(abs(z_stat)))

def calculate_observed_power(
    diff_cr: float,
    se_diff: float,
    alpha: float,
    alternative: AlternativeHypothesis
) -> float:
    """
    Analytical Power calculation logic extracted from hexkit.
    """
    if se_diff == 0:
        return 1.0
        
    z_delta = abs(diff_cr) / se_diff
    
    if alternative in [AlternativeHypothesis.GREATER, AlternativeHypothesis.LESS]:
        z_alpha = norm.ppf(1 - alpha)
        return norm.cdf(z_delta - z_alpha)
    else: # Two-sided
        z_alpha = norm.ppf(1 - alpha / 2)
        return norm.cdf(z_delta - z_alpha) + norm.cdf(-z_delta - z_alpha)
