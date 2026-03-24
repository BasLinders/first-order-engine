import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Union
from enum import Enum

class TestType(Enum):
    ONE_SAMPLE = "one_sample"
    MULTI_SAMPLE = "multi_sample"

class SequentialEngine:
    """
    Core engine for Mixture Sequential Probability Ratio Testing (mSPRT).
    Pure statistical logic, strictly decoupled from any UI, routing, or database layers.
    """

    @staticmethod
    def calculate_boundaries(alpha: float, beta: float, num_variants: int = 1) -> Tuple[float, float]:
        """Calculates mSPRT stopping boundaries."""
        upper = np.log(num_variants / alpha)
        lower = np.log(beta)
        return upper, lower

    @staticmethod
    def calculate_llr_vectorized(
        n_var: np.ndarray, 
        x_var: np.ndarray, 
        n_ctrl: Optional[np.ndarray] = None, 
        x_ctrl: Optional[np.ndarray] = None, 
        tau: float = 0.01,
        fixed_baseline_cr: Optional[float] = None
    ) -> np.ndarray:
        """
        Vectorized LLR calculation. 
        Expects raw numpy arrays of cumulative counts.
        """
        llr = np.zeros_like(x_var, dtype=float)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            if fixed_baseline_cr is not None:
                # ONE-SAMPLE LOGIC
                valid_mask = n_var > 0
                p_base = fixed_baseline_cr
                p_var = x_var / np.maximum(n_var, 1) 
                
                variance = (p_var * (1 - p_var)) / np.maximum(n_var, 1)
                diff = p_var - p_base
                
            else:
                # MULTI-SAMPLE LOGIC
                if n_ctrl is None or x_ctrl is None:
                    raise ValueError("Multi-sample test requires Control arrays.")
                    
                valid_mask = (n_ctrl > 0) & (n_var > 0)
                
                p_ctrl = x_ctrl / np.maximum(n_ctrl, 1)
                p_var = x_var / np.maximum(n_var, 1)
                
                n_total = np.maximum(n_ctrl + n_var, 1)
                p_pool = (x_ctrl + x_var) / n_total
                
                variance = p_pool * (1 - p_pool) * (1.0 / np.maximum(n_ctrl, 1) + 1.0 / np.maximum(n_var, 1))
                diff = p_var - p_ctrl

            variance = np.where(variance <= 0, np.nan, variance)
            
            calc = 0.5 * (np.log(variance / (variance + tau)) + 
                         ((diff * diff) / variance) * (tau / (variance + tau)))
            
            llr = np.where(valid_mask & ~np.isnan(calc), calc, 0.0)
            
        return llr

    @staticmethod
    def estimate_remaining_time(
        current_llr: float, 
        upper_bound: float, 
        total_visitors: int, 
        days_elapsed: int
    ) -> Dict[str, Union[float, int]]:
        """Predicts the required sample size and days to reach significance based on linear velocity."""
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

    def process_test_trajectory(
        self, 
        df: pd.DataFrame, 
        test_type: TestType,
        tau: float,
        alpha: float,
        beta: float,
        num_variants: int = 1,
        baseline_cr: Optional[float] = None,
        control_group_name: str = 'Control'
    ) -> pd.DataFrame:
        """
        Orchestrates LLR calculation across a DataFrame.
        Assumes input df has: ['measurement_date', 'variant_name', 'visitors', 'conversions']
        Assumes data is ALREADY CUMULATIVE.
        """
        if df.empty:
            return pd.DataFrame()

        results = []
        upper, lower = self.calculate_boundaries(alpha, beta, num_variants)
        
        # Defend against duplicate dates and sort
        df = df.groupby(['variant_name', 'measurement_date']).last().reset_index()
        variants_to_test = [v for v in df['variant_name'].unique() if v != control_group_name]
        
        if test_type == TestType.MULTI_SAMPLE:
            if control_group_name not in df['variant_name'].values:
                return pd.DataFrame() # Missing control data
                
            ctrl_df = df[df['variant_name'] == control_group_name].set_index('measurement_date')
            
            for variant in variants_to_test:
                var_df = df[df['variant_name'] == variant].set_index('measurement_date')
                
                # Outer join aligns dates. Forward-fill handles staggered starts.
                merged = var_df.join(ctrl_df, how='outer', lsuffix='_var', rsuffix='_ctrl')
                merged = merged.ffill().fillna(0).reset_index()
                merged['variant_name'] = variant 
                
                merged['llr'] = self.calculate_llr_vectorized(
                    n_var=merged['visitors_var'].values, 
                    x_var=merged['conversions_var'].values,
                    n_ctrl=merged['visitors_ctrl'].values, 
                    x_ctrl=merged['conversions_ctrl'].values,
                    tau=tau
                )
                
                merged['upper_bound'] = upper
                merged['lower_bound'] = lower
                results.append(merged)
                
        elif test_type == TestType.ONE_SAMPLE:
            if baseline_cr is None:
                raise ValueError("baseline_cr must be provided for One-Sample tests.")
                
            for variant in variants_to_test:
                merged = df[df['variant_name'] == variant].copy()
                
                merged['llr'] = self.calculate_llr_vectorized(
                    n_var=merged['visitors'].values, 
                    x_var=merged['conversions'].values,
                    tau=tau,
                    fixed_baseline_cr=baseline_cr
                )
                
                merged['upper_bound'] = upper
                merged['lower_bound'] = lower
                results.append(merged)
                
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
