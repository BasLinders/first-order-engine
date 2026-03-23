import scipy.stats as stats
import statistics
from typing import List, Dict, Optional

def calculate_srm(visitor_counts: List[int], expected_proportions: List[float]) -> Dict:
    """
    Performs the Chi-squared test to check for Sample Ratio Mismatch.
    """
    total_visitors = sum(visitor_counts)
    sum_props = sum(expected_proportions)

    if sum_props == 0:
        raise ValueError("Total proportions must be greater than zero.")

    # Normalize proportions (the 'User Adjustment' logic)
    expected_distribution = [p / sum_props for p in expected_proportions]
    
    # Calculate expected frequencies: E = Total * p
    expected_counts = [total_visitors * p for p in expected_distribution]
    
    # Perform the chi-squared test
    chi2, p_value = stats.chisquare(f_obs=visitor_counts, f_exp=expected_counts)
    
    return {
        "p_value": p_value,
        "expected_counts": expected_counts,
        "mean_expected": statistics.mean(expected_counts),
        "is_mismatch": p_value < 0.01  # Your specific threshold
    }
