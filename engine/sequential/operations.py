import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional

class SequentialEngine:
    """
    Core engine for Mixture Sequential Probability Ratio Testing (mSPRT).
    Enables continuous monitoring of A/B tests without alpha inflation.
    """

    @staticmethod
    def calculate_boundaries(alpha: float, beta: float, num_variants: int = 1) -> Tuple[float, float]:
        """
        Calculates mSPRT stopping boundaries. 
        Uses a conservative approach to maintain FWER for multiple variants.
        """
        # Upper boundary: Crossing this allows rejecting H0 (Success)
        # We divide alpha by num_variants (Bonferroni-style) for multi-arm safety
        upper = np.log(num_variants / alpha)
        
        # Lower boundary: Crossing this suggests Futility (Accepting H0)
        lower = np.log(beta)
        
        return upper, lower

    @staticmethod
    def calculate_llr_vectorized(
        n_ctrl: np.ndarray, 
        x_ctrl: np.ndarray, 
        n_var: np.ndarray, 
        x_var: np.ndarray, 
        tau: float = 0.01
    ) -> np.ndarray:
        """
        Vectorized LLR calculation for high-performance trajectory mapping.
        Formula: LLR = 0.5 * (ln(V / (V + tau)) + (diff^2 / V) * (tau / (V + tau)))
        """
        # conversion rates
        p_ctrl = x_ctrl / n_ctrl
        p_var = x_var / n_var
        
        # Pooled conversion rate for variance estimation
        p_pool = (x_ctrl + x_var) / (n_ctrl + n_var)
        
        # Bernoulli variance of the difference
        # We add a tiny epsilon to avoid division by zero
        eps = 1e-10
        variance = (p_pool * (1 - p_pool) * (1/n_ctrl + 1/n_var)) + eps
        
        diff = p_var - p_ctrl
        
        # mSPRT Log-Likelihood Ratio
        llr = 0.5 * (np.log(variance / (variance + tau)) + 
                    (diff**2 / variance) * (tau / (variance + tau)))
        
        return np.nan_to_num(llr, nan=0.0)

    @staticmethod
    def estimate_remaining_time(
        current_llr: float, 
        upper_bound: float, 
        total_visitors: int, 
        days_elapsed: int
    ) -> Dict[str, float]:
        """
        Predicts the required sample size and days to reach significance.
        Uses LLR-velocity to project the trajectory.
        """
        if current_llr <= 0 or days_elapsed <= 0:
            return {"est_visitors_needed": np.inf, "est_days_needed": np.inf}

        avg_daily_vis = total_visitors / days_elapsed
        llr_per_visitor = current_llr / total_visitors
        
        remaining_llr = upper_bound - current_llr
        
        if llr_per_visitor <= 0:
            return {"est_visitors_needed": np.inf, "est_days_needed": np.inf}
            
        est_vis = remaining_llr / llr_per_visitor
        est_days = est_vis / avg_daily_vis
        
        return {
            "est_visitors_needed": round(est_vis),
            "est_days_needed": round(est_days, 1)
        }

    def process_test_trajectory(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """
        Orchestrates the LLR calculation for all variants in the dataset.
        """
        results = []
        upper, lower = self.calculate_boundaries(params['alpha'], params['beta'], params['num_variants'])
        
        # Separate Control
        ctrl_df = df[df['variant_name'] == 'Control'].sort_values('measurement_date')
        
        for variant in df['variant_name'].unique():
            if variant == 'Control': continue
            
            var_df = df[df['variant_name'] == variant].sort_values('measurement_date')
            merged = pd.merge(var_df, ctrl_df, on='measurement_date', suffixes=('_var', '_ctrl'))
            
            merged['llr'] = self.calculate_llr_vectorized(
                merged['visitors_ctrl'].values, merged['conversions_ctrl'].values,
                merged['visitors_var'].values, merged['conversions_var'].values,
                tau=params['tau']
            )
            
            merged['upper_bound'] = upper
            merged['lower_bound'] = lower
            results.append(merged)
            
        return pd.concat(results) if results else pd.DataFrame()
