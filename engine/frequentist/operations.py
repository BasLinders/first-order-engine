import numpy as np
import concurrent.futures
from scipy.stats import norm
from typing import List, Optional
from engine.core.models import AlternativeHypothesis

# --- Corrections ---

def apply_sidak(alpha: float, num_variants: int) -> float:
    """Calculates adjusted alpha for multiple comparisons."""
    num_comparisons = num_variants - 1
    if num_comparisons <= 1:
        return alpha
    return 1 - (1 - alpha) ** (1 / num_comparisons)

# --- Statistical Tests --- 

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

# --- Power Functions ---

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

def _bootstrap_sample_is_significant(
    data_ctrl: np.ndarray, 
    data_chal: np.ndarray, 
    alpha: float, 
    alternative: AlternativeHypothesis
) -> bool:
    """
    Private helper: Performs a single bootstrap resample and returns 
    if the result was significant.
    """
    # Resample with replacement
    sample_ctrl = np.random.choice(data_ctrl, size=len(data_ctrl), replace=True)
    sample_chal = np.random.choice(data_chal, size=len(data_chal), replace=True)
    
    # Calculate pooled SE for the bootstrap iteration
    p_ctrl, p_chal = np.mean(sample_ctrl), np.mean(sample_chal)
    n_ctrl, n_chal = len(sample_ctrl), len(sample_chal)
    
    p_pooled = (np.sum(sample_ctrl) + np.sum(sample_chal)) / (n_ctrl + n_chal)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n_ctrl + 1 / n_chal))
    
    if se == 0:
        return False
        
    z_stat = (p_chal - p_ctrl) / se
    
    # P-value calculation based on tail
    if alternative == AlternativeHypothesis.GREATER:
        p_val = 1 - norm.cdf(z_stat)
    elif alternative == AlternativeHypothesis.LESS:
        p_val = norm.cdf(z_stat)
    else: # Two-sided
        p_val = 2 * (1 - norm.cdf(abs(z_stat)))
        
    return p_val < alpha

def run_bootstrap_power(
    conversions_ctrl: int,
    visitors_ctrl: int,
    conversions_chal: int,
    visitors_chal: int,
    alpha: float,
    alternative: AlternativeHypothesis,
    n_bootstraps: int = 10000
) -> float:
    """
    Calculates observed power via bootstrapping. 
    Uses ThreadPoolExecutor for parallel execution.
    """
    # Create the raw Bernoulli arrays (1s and 0s)
    data_ctrl = np.concatenate([np.ones(conversions_ctrl), np.zeros(visitors_ctrl - conversions_ctrl)])
    data_chal = np.concatenate([np.ones(conversions_chal), np.zeros(visitors_chal - conversions_chal)])
    
    significant_count = 0
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map the private helper across the number of bootstraps
        futures = [
            executor.submit(_bootstrap_sample_is_significant, data_ctrl, data_chal, alpha, alternative) 
            for _ in range(n_bootstraps)
        ]
        
        for future in concurrent.futures.as_completed(futures):
            if future.result():
                significant_count += 1
                
    return significant_count / n_bootstraps
