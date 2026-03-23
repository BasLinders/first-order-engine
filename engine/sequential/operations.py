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
        Vectorized LLR calculation with safe division and optimized math.
        """
        # Safe division to prevent RuntimeWarnings on day 0
        p_ctrl = np.divide(x_ctrl, n_ctrl, out=np.zeros_like(x_ctrl, dtype=float), where=n_ctrl!=0)
        p_var = np.divide(x_var, n_var, out=np.zeros_like(x_var, dtype=float), where=n_var!=0)
        
        # Pooled conversion rate
        n_total = n_ctrl + n_var
        p_pool = np.divide(x_ctrl + x_var, n_total, out=np.zeros_like(n_total, dtype=float), where=n_total!=0)
        
        eps = 1e-10
        # Safe reciprocal for variance calculation
        inv_n_ctrl = np.divide(1.0, n_ctrl, out=np.zeros_like(n_ctrl, dtype=float), where=n_ctrl!=0)
        inv_n_var = np.divide(1.0, n_var, out=np.zeros_like(n_var, dtype=float), where=n_var!=0)
        
        variance = (p_pool * (1 - p_pool) * (inv_n_ctrl + inv_n_var)) + eps
        
        diff = p_var - p_ctrl
        
        # Optimized squaring (diff * diff is faster than diff**2)
        llr = 0.5 * (np.log(variance / (variance + tau)) + 
                    ((diff * diff) / variance) * (tau / (variance + tau)))
        
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
        Orchestrates LLR calculation, ensuring data is cumulative and dates align safely.
        Assumes input df has: ['measurement_date', 'variant_name', 'visitors', 'conversions']
        """
        results = []
        upper, lower = self.calculate_boundaries(params['alpha'], params['beta'], params['num_variants'])
        
        # 1. Guarantee data is cumulative per variant
        df = df.sort_values(['variant_name', 'measurement_date'])
        df['visitors'] = df.groupby('variant_name')['visitors'].cumsum()
        df['conversions'] = df.groupby('variant_name')['conversions'].cumsum()
        
        # Separate Control and set index for easier joining
        ctrl_df = df[df['variant_name'] == 'Control'].set_index('measurement_date')
        
        for variant in df['variant_name'].unique():
            if variant == 'Control': 
                continue
            
            var_df = df[df['variant_name'] == variant].set_index('measurement_date')
            
            # 2. Outer join handles missing dates (e.g., variant started later or data dropped)
            merged = var_df.join(ctrl_df, how='outer', lsuffix='_var', rsuffix='_ctrl')
            
            # Forward-fill missing cumulative totals, then fill leading NaNs with 0
            merged = merged.ffill().fillna(0)
            
            # Bring measurement_date back as a column
            merged = merged.reset_index()
            merged['variant_name'] = variant 
            
            # 3. Calculate LLR
            merged['llr'] = self.calculate_llr_vectorized(
                merged['visitors_ctrl'].values, merged['conversions_ctrl'].values,
                merged['visitors_var'].values, merged['conversions_var'].values,
                tau=params['tau']
            )
            
            merged['upper_bound'] = upper
            merged['lower_bound'] = lower
            results.append(merged)
            
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
