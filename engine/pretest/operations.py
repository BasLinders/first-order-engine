import numpy as np
import pandas as pd
from scipy.stats import norm
from prophet import Prophet
from typing import List, Dict, Union

def holm_bonferroni_correction(num_variants: int, alpha: float, tails: str) -> float:
    """Adjusts alpha for multiple comparisons."""
    num_comparisons = num_variants - 1
    adjusted_alphas = alpha / np.arange(num_comparisons, 0, -1)
    
    if tails == 'Two-sided':
        z_scores = norm.ppf(1 - adjusted_alphas / 2)
    else:
        z_scores = norm.ppf(1 - adjusted_alphas)
    return np.max(z_scores)

def get_z_alpha(num_variants: int, alpha: float, tails: str) -> float:
    """Handles Z-score selection based on variant count and tails."""
    if num_variants > 2:
        return holm_bonferroni_correction(num_variants, alpha, tails)
    
    if tails == 'One-sided':
        return norm.ppf(1 - alpha)
    return norm.ppf(1 - alpha / 2)

def calculate_mde_table(
    num_variants: int, 
    baseline_visitors: int, 
    baseline_conversions: int, 
    risk_pct: float, 
    trust_pct: float, 
    tails: str
) -> List[Dict]:
    """Calculates a 6-week MDE projection for static traffic."""
    alpha = 1 - (risk_pct / 100)
    z_power = norm.ppf(trust_pct / 100)
    z_alpha = get_z_alpha(num_variants, alpha, tails)
    
    baseline_rate = baseline_conversions / baseline_visitors
    weekly_visitors = int(np.ceil(baseline_visitors / num_variants))
    
    results = []
    for week in range(1, 7):
        n = weekly_visitors * week
        se = np.sqrt(2 * baseline_rate * (1 - baseline_rate) / n)
        mde_rel = ((z_alpha + z_power) * se / baseline_rate) * 100
        results.append({"Week": week, "Visitors": n, "MDE": mde_rel})
    return results
