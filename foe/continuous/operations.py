import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro, levene, kruskal, mannwhitneyu
from pingouin import welch_anova


class ContinuousMetricEngine:
    """
    Engine for analyzing continuous metrics (Revenue, Profit, Quantity).
    Handles the statistical decision tree: Normality -> Variance -> Test Selection.
    Strictly returns JSON-serializable primitives for Cloud APIs.
    """

    @staticmethod
    def generate_continuous_conclusion(
        kpi: str, is_significant: bool, test_used: str, p_value: float
    ) -> str:
        """Generates a definitive summary for continuous metric evaluation."""
        if not is_significant:
            return (
                f"No Significant Difference: Based on the {test_used} (p={p_value:.3f}), "
                f"there is no statistically significant difference in {kpi} across the variants. "
                "The metric performs similarly across all groups."
            )

        return (
            f"Significant Variance Detected: The {test_used} (p={p_value:.3f}) detected a "
            f"statistically significant difference in {kpi} between the variants. "
            "Review the summary statistics to identify the winning group."
        )

    @staticmethod
    def detect_outliers_ols(
        df: pd.DataFrame, kpi: str, threshold: float = 3.0
    ) -> list[bool]:
        """
        Uses OLS residuals and influence to detect outliers.
        Returns a boolean list (JSON serializable) instead of the model.
        """
        # Ensure calls don't crash on empty/invalid data
        if df.empty or len(df[kpi].dropna()) < 3:
            return [False] * len(df)

        model = smf.ols(f"{kpi} ~ C(experience_variant_label)", data=df).fit()
        influence = model.get_influence()

        std_resid = np.abs(influence.resid_studentized_internal) > threshold
        leverage = influence.hat_matrix_diag > (
            threshold * (model.df_model + 1) / len(df)
        )
        dffits = np.abs(influence.dffits[0]) > (
            threshold * np.sqrt((model.df_model + 1) / len(df))
        )

        outlier_mask = std_resid | leverage | dffits
        return outlier_mask.tolist()  # Safe for JSON

    @staticmethod
    def winsorize_series(
        series: pd.Series, method: str = "Standard Deviation", param: float = 3.0
    ) -> tuple[list[float], float, float]:
        """Caps outliers without removing data points."""
        series_clean = series.dropna()
        if series_clean.empty:
            return series.tolist(), 0.0, 0.0

        if method == "Standard Deviation":
            mean_val = series_clean.mean()
            std_val = series_clean.std()
            lower = mean_val - (param * std_val)
            upper = mean_val + (param * std_val)
        else:  # Percentile (e.g., param=1 means 1st and 99th percentile)
            lower_p = param
            upper_p = 100.0 - param
            lower, upper = np.percentile(series_clean, [lower_p, upper_p])

        clipped = series.clip(lower, upper)
        return clipped.tolist(), float(lower), float(upper)

    def run_comparison_suite(self, df: pd.DataFrame, kpi: str) -> dict:
        """
        The core decision engine for choosing the right statistical test.
        """
        if df.empty or kpi not in df.columns:
            return {}

        groups = [
            g[kpi].dropna().values
            for _, g in df.groupby("experience_variant_label", observed=True)
        ]
        num_groups = len(groups)

        if num_groups < 2:
            return {"error": "Not enough variants to compare."}

        # 2. Assumption: Normality of Residuals
        model = smf.ols(f"{kpi} ~ C(experience_variant_label)", data=df).fit()

        # Guard against Shapiro limit (N > 5000)
        resid_sample = model.resid.dropna()
        if len(resid_sample) > 5000:
            np.random.seed(
                42
            )  # Safe here as it's purely for diagnostic sampling, not test math
            resid_sample = np.random.choice(resid_sample, 5000, replace=False)

        _, p_norm = shapiro(resid_sample)
        is_normal = bool(p_norm >= 0.05)

        # 3. Assumption: Homogeneity of Variance
        _, p_var = levene(*groups)
        is_homogeneous = bool(p_var >= 0.05)

        # Convert Pandas groupby agg to a nested dictionary for JSON serialization
        summary_df = df.groupby("experience_variant_label", observed=True)[kpi].agg(
            ["mean", "std", "count"]
        )
        summary_stats = summary_df.reset_index().to_dict(orient="records")

        results = {
            "kpi": kpi,
            "is_normal": is_normal,
            "is_homogeneous": is_homogeneous,
            "summary_stats": summary_stats,
        }

        # 4. Statistical Decision Tree
        if is_normal and is_homogeneous:
            results["test_name"] = "Standard ANOVA"
            anova_results = sm.stats.anova_lm(model, typ=2)
            results["p_value"] = float(anova_results["PR(>F)"].iloc[0])

        elif is_normal and not is_homogeneous:
            results["test_name"] = "Welch's ANOVA"
            aov = welch_anova(data=df, dv=kpi, between="experience_variant_label")
            results["p_value"] = float(aov["p-unc"].iloc[0])

        else:
            # Non-Normal -> Non-parametric fallback
            if num_groups > 2:
                results["test_name"] = "Kruskal-Wallis"
                _, p = kruskal(*groups)
                results["p_value"] = float(p)
            else:
                results["test_name"] = "Mann-Whitney U"
                _, p = mannwhitneyu(groups[0], groups[1], alternative="two-sided")
                results["p_value"] = float(p)

        results["is_significant"] = bool(results["p_value"] < 0.05)
        results["conclusion"] = self.generate_continuous_conclusion(
            kpi=kpi,
            is_significant=results["is_significant"],
            test_used=results["test_name"],
            p_value=results["p_value"],
        )

        return results
