import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Union
from enum import Enum

# Inherit from str to ensure clean JSON serialization over APIs
class TestType(str, Enum):
    ONE_SAMPLE = "one_sample"
    MULTI_SAMPLE = "multi_sample"

class SequentialEngine:
    """
    Core engine for Mixture Sequential Probability Ratio Testing (mSPRT).
    """

    @staticmethod
    def conditional_power_check(
        current_llr: float,
        upper_bound: float,
        total_visitors: int,
        max_visitors: int
    ) -> Dict[str, Union[bool, float]]:

        """
        Projects whether the test can reach significance before the visitor cap.
        Returns can_recover and projected_llr at cap.
        """

        if total_visitors == 0:
            return {"can_recover": True, "projected_llr": 0.0}

        remaining = max_visitors - total_visitors

        if remaining <= 0:
            return {"can_recover": current_llr >= upper_bound, "projected_llr": current_llr}

        llr_per_visitor = current_llr / total_visitors
        projected_llr = current_llr + (llr_per_visitor * remaining)

        return {
            "can_recover": projected_llr >= upper_bound,
            "projected_llr": round(projected_llr, 4)
        }

    @staticmethod
    def generate_sequential_conclusion(
        variant_name: str,
        current_llr: float,
        upper_bound: float,
        lower_bound: float,
        days_elapsed: int
    ) -> str:
        """
        Evaluates the current LLR against the stopping boundaries and generates
        a definitive, UI-agnostic status update.
        """

        if current_llr >= upper_bound:
            return (
                f"Winner Declared: '{variant_name}' has crossed the upper significance boundary "
                f"after {days_elapsed} days. The test can be stopped early, and the variant can "
                "be confidently rolled out."
            )

        elif current_llr <= lower_bound:
            return (
                f"Futility Reached: '{variant_name}' has crossed the lower boundary after "
                f"{days_elapsed} days. The variant is highly unlikely to result in a positive "
                "impact. The test should be stopped to prevent further loss."
            )

        else:
            return (
                f"Test Running: '{variant_name}' is currently between the decision boundaries "
                f"(LLR: {current_llr:.2f}). Continue collecting data until a boundary is crossed "
                "or the maximum sample size is reached."
            )

    @staticmethod
    def calculate_boundaries(
        alpha: float, beta: float, num_variants: int = 1
    ) -> Tuple[float, float]:

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

        with np.errstate(divide="ignore", invalid="ignore"):

            if fixed_baseline_cr is not None:
                # ONE-SAMPLE LOGIC
                valid_mask = n_var > 0
                p_base = fixed_baseline_cr

                p_var = x_var / np.maximum(n_var, 1)

                variance = (p_base * (1 - p_base)) / np.maximum(n_var, 1)
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
                variance = (
                    p_pool
                    * (1 - p_pool)
                    * (1.0 / np.maximum(n_ctrl, 1) + 1.0 / np.maximum(n_var, 1))
                )
                diff = p_var - p_ctrl

            variance = np.where(variance <= 0, np.nan, variance)
            calc = 0.5 * (
                np.log(variance / (variance + tau))
                + ((diff * diff) / variance) * (tau / (variance + tau))
            )

            llr = np.where(valid_mask & ~np.isnan(calc), calc, 0.0)

        return llr

    @staticmethod
    def estimate_remaining_time(
        current_llr: float, upper_bound: float, total_visitors: int, days_elapsed: int
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

    @staticmethod
    def _assign_status(
        merged: pd.DataFrame,
        upper: float,
        lower: float,
        visitors_col: str,
        max_visitors: Optional[int]
    ) -> pd.Series:
        """Assigns a status label to each row based on LLR position and optional visitor cap."""
        base = np.where(
            merged['llr'] >= upper, 'winner',
            np.where(merged['llr'] <= lower, 'loser', 'continue')
        )

        if max_visitors is not None:
            return np.where(
                merged[visitors_col] >= max_visitors,
                np.where(base == 'continue', 'cap_reached', base),
                base
            )
        return base
        
    def process_test_trajectory(
        self,
        df: pd.DataFrame,
        test_type: TestType,
        tau: float,
        alpha: float,
        beta: float,
        num_variants: int = 1,
        baseline_cr: Optional[float] = None,
        max_visitors: Optional[int] = None,
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
        df = df.groupby(["variant_name", "measurement_date"]).last().reset_index()
        
        variants_to_test = [
            v for v in df["variant_name"].unique() if v != control_group_name
        ]

        if test_type == TestType.MULTI_SAMPLE:
            if control_group_name not in df["variant_name"].values:

                return pd.DataFrame()  # Missing control data

            ctrl_df = df[df["variant_name"] == control_group_name].set_index(
                "measurement_date"
            )
            
            for variant in variants_to_test:
                var_df = df[df["variant_name"] == variant].set_index("measurement_date")

                # Outer join aligns dates. Forward-fill handles staggered starts.
                merged = var_df.join(
                    ctrl_df, how="outer", lsuffix="_var", rsuffix="_ctrl"
                )

                merged = merged.ffill().fillna(0).reset_index()
                merged["variant_name"] = variant
                merged["llr"] = self.calculate_llr_vectorized(
                    n_var=merged["visitors_var"].values,
                    x_var=merged["conversions_var"].values,
                    n_ctrl=merged["visitors_ctrl"].values,
                    x_ctrl=merged["conversions_ctrl"].values,
                    tau=tau
                )

                merged['upper_bound'] = upper
                merged['lower_bound'] = lower
                merged['max_visitors'] = max_visitors if max_visitors is not None else np.nan

                # Add a simple status flag for UI charting colors
                merged['status'] = self._assign_status(merged, upper, lower, 'visitors_var', max_visitors)

                results.append(merged)
                
        elif test_type == TestType.ONE_SAMPLE:
            if baseline_cr is None:
                raise ValueError("baseline_cr must be provided for One-Sample tests.")

            for variant in variants_to_test:
                merged = df[df["variant_name"] == variant].copy()
                
                merged["llr"] = self.calculate_llr_vectorized(
                    n_var=merged["visitors"].values,
                    x_var=merged["conversions"].values,
                    tau=tau,
                    fixed_baseline_cr=baseline_cr
                )

                merged['upper_bound'] = upper
                merged['lower_bound'] = lower
                merged['status'] = self._assign_status(merged, upper, lower, 'visitors', max_visitors)

                results.append(merged)

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
