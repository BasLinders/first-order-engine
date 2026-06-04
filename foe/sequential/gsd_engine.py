import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Optional
from enum import Enum


# These Enums could be moved to a shared `enums.py` file in the future
class TestType(str, Enum):
    ONE_SAMPLE = "one_sample"
    MULTI_SAMPLE = "multi_sample"


class GSDSpendingMethod(str, Enum):
    POCOCK = "pocock"
    OBRIEN_FLEMING = "obrien_fleming"


class GSDEngine:
    """
    Core engine for Group Sequential Design (GSD) A/B Testing.
    Utilizes Lan-DeMets alpha spending functions to calculate dynamic Z-boundaries.
    """

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
            # 2 * (1 - Phi(Z_{1-alpha/2} / sqrt(t)))
            return 2 * (1 - stats.norm.cdf(z_alpha / np.sqrt(t)))

        elif method == GSDSpendingMethod.POCOCK:
            # alpha * ln(1 + (e - 1) * t)
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
        """
        Vectorized standard Z-score calculation for GSD boundaries.
        Expects raw numpy arrays of cumulative counts.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            p_var = x_var / np.maximum(n_var, 1)

            if fixed_baseline_cr is not None:
                # ONE-SAMPLE LOGIC
                p_base = fixed_baseline_cr
                se = np.sqrt((p_base * (1 - p_base)) / np.maximum(n_var, 1))
                z_score = (p_var - p_base) / se
            else:
                # MULTI-SAMPLE LOGIC
                if n_ctrl is None or x_ctrl is None:
                    raise ValueError("Multi-sample test requires Control arrays.")

                p_ctrl = x_ctrl / np.maximum(n_ctrl, 1)
                p_pool = (x_var + x_ctrl) / np.maximum(n_var + n_ctrl, 1)

                se = np.sqrt(
                    p_pool * (1 - p_pool) * (1.0 / np.maximum(n_var, 1) + 1.0 / np.maximum(n_ctrl, 1))
                )
                z_score = (p_var - p_ctrl) / se

        return np.where(np.isnan(z_score) | np.isinf(z_score), 0.0, z_score)

    def process_test_trajectory(
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

        Note: `max_visitors` is strictly required for GSD to calculate the horizon.
        """
        if df.empty or max_visitors <= 0:
            return pd.DataFrame()

        results = []

        # Defend against duplicate dates and sort
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

                # Assign status based on marginal boundary
                merged["status"] = np.where(
                    np.abs(merged["z_score"]) >= merged["marginal_z_bound"],
                    "significant",
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
