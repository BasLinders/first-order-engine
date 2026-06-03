import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Tuple, Dict, Optional, Union
from enum import Enum


# Inherit from str to ensure clean JSON serialization over APIs
class TestType(str, Enum):
    ONE_SAMPLE = "one_sample"
    MULTI_SAMPLE = "multi_sample"

class GSDSpendingMethod(str, Enum):
    POCOCK = "pocock"
    OBRIEN_FLEMING = "obrien_fleming"


class SequentialEngine:
    """
    Core engine for Mixture Sequential Probability Ratio Testing (mSPRT)
    and Group Sequential Design (GSD).
    """

    # ---------------------------------------------------------
    # mSPRT METHODS
    # ---------------------------------------------------------

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
        Assumes data is ALREADY CUMULATIVE.
        """
        if df.empty:
            return pd.DataFrame()

        results = []
        upper, lower = self.calculate_boundaries(alpha, beta, num_variants)

        df = df.groupby(["variant_name", "measurement_date"]).last().reset_index()
        variants_to_test = [v for v in df["variant_name"].unique() if v != control_group_name]

        if test_type == TestType.MULTI_SAMPLE:
            if control_group_name not in df["variant_name"].values:
                return pd.DataFrame() 

            ctrl_df = df[df["variant_name"] == control_group_name].set_index("measurement_date")

            for variant in variants_to_test:
                var_df = df[df["variant_name"] == variant].set_index("measurement_date")
                merged = var_df.join(ctrl_df, how="outer", lsuffix="_var", rsuffix="_ctrl")
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

    # ---------------------------------------------------------
    # GROUP SEQUENTIAL DESIGN (GSD) METHODS
    # ---------------------------------------------------------

    @staticmethod
    def lan_demets_alpha_spent(alpha: float, t: float, method: GSDSpendingMethod) -> float:
        """
        Calculates the cumulative Type I error (alpha) spent at information fraction t.
        """
        # Clamp t to ensure safe bounds [0.0, 1.0]
        t = max(0.0, min(1.0, t))
        
        if t == 0:
            return 0.0
        if t == 1:
            return alpha

        if method == GSDSpendingMethod.OBRIEN_FLEMING:
            z_alpha = stats.norm.ppf(1 - alpha / 2)
            return 2 * (1 - stats.norm.cdf(z_alpha / np.sqrt(t)))
            
        elif method == GSDSpendingMethod.POCOCK:
            return alpha * np.log(1 + (np.e - 1) * t)
        
        raise ValueError(f"Unsupported GSD spending method: {method}")

    @staticmethod
    def calculate_gsd_z_score_vectorized(
        n_var: np.ndarray,
        x_var: np.ndarray,
        n_ctrl: Optional[np.ndarray] = None,
        x_ctrl: Optional[np.ndarray] = None,
        fixed_baseline_cr: Optional[float] = None
    ) -> np.ndarray:
        """Vectorized standard Z-score calculation for GSD boundaries."""
        with np.errstate(divide="ignore", invalid="ignore"):
            p_var = x_var / np.maximum(n_var, 1)
            
            if fixed_baseline_cr is not None:
                # ONE-SAMPLE LOGIC
                p_base = fixed_baseline_cr
                se = np.sqrt((p_base * (1 - p_base)) / np.maximum(n_var, 1))
                z_score = (p_var - p_base) / se
            else:
                # MULTI-SAMPLE LOGIC
                p_ctrl = x_ctrl / np.maximum(n_ctrl, 1)
                p_pool = (x_var + x_ctrl) / np.maximum(n_var + n_ctrl, 1)
                
                se = np.sqrt(p_pool * (1 - p_pool) * (1.0 / np.maximum(n_var, 1) + 1.0 / np.maximum(n_ctrl, 1)))
                z_score = (p_var - p_ctrl) / se
                
        return np.where(np.isnan(z_score) | np.isinf(z_score), 0.0, z_score)

    def process_gsd_trajectory(
        self,
        df: pd.DataFrame,
        test_type: TestType,
        alpha: float,
        max_visitors: int,
        spending_method: GSDSpendingMethod = GSDSpendingMethod.OBRIEN_FLEMING,
        baseline_cr: Optional[float] = None,
        control_group_name: str = 'Control'
    ) -> pd.DataFrame:
        """
        Orchestrates Group Sequential Design (GSD) evaluation across a DataFrame.
        Evaluates discrete looks based on the information fraction (t).
        """
        if df.empty or max_visitors <= 0:
            return pd.DataFrame()

        results = []
        df = df.groupby(["variant_name", "measurement_date"]).last().reset_index()
        variants_to_test = [v for v in df["variant_name"].unique() if v != control_group_name]

        if test_type == TestType.MULTI_SAMPLE:
            if control_group_name not in df["variant_name"].values:
                return pd.DataFrame()

            ctrl_df = df[df["variant_name"] == control_group_name].set_index("measurement_date")

            for variant in variants_to_test:
                var_df = df[df["variant_name"] == variant].set_index("measurement_date")
                merged = var_df.join(ctrl_df, how="outer", lsuffix="_var", rsuffix="_ctrl")
                merged = merged.ffill().fillna(0).reset_index()
                merged["variant_name"] = variant

                # Information fraction t = Current Total Sample / Max Total Sample
                total_current_visitors = merged["visitors_var"] + merged["visitors_ctrl"]
                merged["info_fraction_t"] = (total_current_visitors / max_visitors).clip(upper=1.0)

                merged["z_score"] = self.calculate_gsd_z_score_vectorized(
                    n_var=merged["visitors_var"].values,
                    x_var=merged["conversions_var"].values,
                    n_ctrl=merged["visitors_ctrl"].values,
                    x_ctrl=merged["conversions_ctrl"].values
                )
                
                # Calculate spent alpha for each look
                merged["alpha_spent"] = merged["info_fraction_t"].apply(
                    lambda t: self.lan_demets_alpha_spent(alpha, t, spending_method)
                )
                
                # Marginal boundary approximation for UI visualization
                merged["marginal_z_bound"] = merged["alpha_spent"].apply(
                    lambda a: stats.norm.ppf(1 - a / 2) if a > 0 else np.inf
                )

                merged["status"] = np.where(
                    np.abs(merged["z_score"]) >= merged["marginal_z_bound"],
                    "winner" if spending_method else "significant", # Customize labeling as needed
                    np.where(total_current_visitors >= max_visitors, "cap_reached", "continue")
                )
                
                results.append(merged)

        elif test_type == TestType.ONE_SAMPLE:
            if baseline_cr is None:
                raise ValueError("baseline_cr must be provided for One-Sample tests.")

            for variant in variants_to_test:
                merged = df[df["variant_name"] == variant].copy()

                merged["info_fraction_t"] = (merged["visitors"] / max_visitors).clip(upper=1.0)

                merged["z_score"] = self.calculate_gsd_z_score_vectorized(
                    n_var=merged["visitors"].values,
                    x_var=merged["conversions"].values,
                    fixed_baseline_cr=baseline_cr
                )

                merged["alpha_spent"] = merged["info_fraction_t"].apply(
                    lambda t: self.lan_demets_alpha_spent(alpha, t, spending_method)
                )

                merged["marginal_z_bound"] = merged["alpha_spent"].apply(
                    lambda a: stats.norm.ppf(1 - a / 2) if a > 0 else np.inf
                )

                merged["status"] = np.where(
                    np.abs(merged["z_score"]) >= merged["marginal_z_bound"],
                    "significant",
                    np.where(merged["visitors"] >= max_visitors, "cap_reached", "continue")
                )

                results.append(merged)

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
