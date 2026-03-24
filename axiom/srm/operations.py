import pandas as pd
import scipy.stats as stats
import numpy as np
from typing import Dict, List, Any

class SRMEngine:
    """
    Advanced diagnostic suite for Sample Ratio Mismatch (SRM).
    """

    @staticmethod
    def calculate_chi_squared(observed: List[int], expected: List[float]) -> Dict[str, Any]:
        """Standard Chi-Squared Goodness of Fit."""
        total = sum(observed)
        expected_counts = [total * (p / sum(expected)) for p in expected]
        chi2, p_val = stats.chisquare(f_obs=observed, f_exp=expected_counts)
        
        return {
            "p_value": float(p_val),
            "is_mismatch": p_val < 0.01,
            "severity": (max(observed)/sum(observed)) - (expected[0]/sum(expected))
        }

    def diagnose_segments(self, df: pd.DataFrame, dimensions: List[str], 
                         variant_col: str, expected_ratio: List[float]) -> pd.DataFrame:
        """
        Runs SRM checks across multiple dimensions to find the root cause.
        """
        report = []
        for dim in dimensions:
            segments = df[dim].unique()
            for seg in segments:
                counts = df[df[dim] == seg][variant_col].value_counts().sort_index().tolist()
                
                # Only test if we have data for all variants
                if len(counts) == len(expected_ratio):
                    res = self.calculate_chi_squared(counts, expected_ratio)
                    report.append({
                        "dimension": dim,
                        "segment": seg,
                        "p_value": res["p_value"],
                        "status": "FAIL" if res["is_mismatch"] else "PASS"
                    })
        
        return pd.DataFrame(report).sort_values("p_value")

    @staticmethod
    def get_srm_thresholds(total_n: int, alpha: float = 0.01) -> Dict[str, float]:
        """
        Calculates what 'Observed %' would trigger an SRM at this sample size.
        Helps stakeholders understand the sensitivity of the test.
        """
        # Critical value for Chi-square with 1 dof (for A/B tests)
        critical_value = stats.chi2.ppf(1 - alpha, df=1)
        
        # Solving for the proportion difference that hits the critical value
        # This is an approximation for a 50/50 split
        margin = np.sqrt(critical_value / (4 * total_n))
        
        return {
            "lower_bound_pct": 0.5 - margin,
            "upper_bound_pct": 0.5 + margin,
            "total_sample": total_n
        }
