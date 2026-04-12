import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro, levene, kruskal, mannwhitneyu
from scipy.optimize import minimize
from scipy import stats
from pingouin import welch_anova

class ContinuousMetricEngine:
    """
    Engine for analyzing continuous metrics (Revenue, Profit, Quantity).
    Handles the statistical decision tree: Normality -> Variance -> Test Selection.
    Strictly returns JSON-serializable primitives for Cloud APIs.
    """

    @staticmethod
    def generate_continuous_conclusion(
        kpi: str,
        is_significant: bool,
        test_used: str,
        p_value: float
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
    def detect_outliers_ols(df: pd.DataFrame, kpi: str, threshold: float = 3.0) -> list[bool]:
        """
        Uses OLS residuals and influence to detect outliers.
        Returns a boolean list (JSON serializable) instead of the model.
        """
        # Ensure calls don't crash on empty/invalid data
        if df.empty or len(df[kpi].dropna()) < 3:
            return [False] * len(df)

        model = smf.ols(f'{kpi} ~ C(experience_variant_label)', data=df).fit()
        influence = model.get_influence()
        
        std_resid = np.abs(influence.resid_studentized_internal) > threshold
        leverage = influence.hat_matrix_diag > (threshold * (model.df_model + 1) / len(df))
        dffits = np.abs(influence.dffits[0]) > (threshold * np.sqrt((model.df_model + 1) / len(df)))
        
        outlier_mask = (std_resid | leverage | dffits)
        return outlier_mask.tolist() # Safe for JSON

    @staticmethod
    def winsorize_series(series: pd.Series, method: str = 'Standard Deviation', param: float = 3.0) -> tuple[list[float], float, float]:
        """Caps outliers without removing data points."""
        series_clean = series.dropna()
        if series_clean.empty:
            return series.tolist(), 0.0, 0.0

        if method == 'Standard Deviation':
            mean_val = series_clean.mean()
            std_val = series_clean.std()
            lower = mean_val - (param * std_val)
            upper = mean_val + (param * std_val)
        else: # Percentile (e.g., param=1 means 1st and 99th percentile)
            lower_p = param
            upper_p = 100.0 - param
            lower, upper = np.percentile(series_clean, [lower_p, upper_p])
        
        clipped = series.clip(lower, upper)
        return clipped.tolist(), float(lower), float(upper)

    @staticmethod
    def neg_log_likelihood(params, data):
        """Negative log-likelihood function for the Gamma distribution."""
        k, theta = params
        if k <= 0 or theta <= 0:
            return np.inf
        return -np.sum(stats.gamma.logpdf(data, a=k, scale=theta))

    @staticmethod
    def fit_gamma(data) -> tuple[float, float, float]:
        """Fits a Gamma model to the data using MLE. Returns k, theta, and log-likelihood."""
        data_clean = data.dropna()
        # Initial guess using Method of Moments
        mean_val = data_clean.mean()
        var_val = data_clean.var()
        initial_params = [mean_val**2 / var_val, var_val / mean_val]
        
        result = minimize(
            ContinuousMetricEngine.neg_log_likelihood, 
            initial_params, 
            args=(data_clean,), 
            bounds=((1e-5, None), (1e-5, None))
        )
        k, theta = result.x
        log_lik = -result.fun
        return float(k), float(theta), float(log_lik)

    @staticmethod
    def run_gamma_posthoc(df: pd.DataFrame, kpi: str, group_col: str, control_label: str) -> list[dict]:
        """Runs pairwise LRTs against a control variant, returning JSON-serializable results."""
        variants = [v for v in df[group_col].unique() if v != control_label]
        posthoc_results = []
        num_comparisons = len(variants)
        
        for variant in variants:
            pair_df = df[df[group_col].isin([control_label, variant])]
            
            # 1. Fit Null Model (Single mean for both)
            null_mean = pair_df[kpi].mean()
            null_log_lik = np.sum(stats.gamma.logpdf(pair_df[kpi], a=1, scale=null_mean)) 
            
            # 2. Fit Alternative Model (Separate means)
            ctrl_data = pair_df[pair_df[group_col] == control_label][kpi]
            var_data = pair_df[pair_df[group_col] == variant][kpi]
            
            alt_log_lik = (
                np.sum(stats.gamma.logpdf(ctrl_data, a=1, scale=ctrl_data.mean())) + 
                np.sum(stats.gamma.logpdf(var_data, a=1, scale=var_data.mean()))
            )
            
            # 3. Likelihood Ratio Test
            lrt_stat = 2 * (alt_log_lik - null_log_lik)
            p_val = stats.chi2.sf(lrt_stat, df=1)
            adj_p = min(p_val * num_comparisons, 1.0)
            
            posthoc_results.append({
                "comparison": f"{variant} vs {control_label}",
                "lrt_stat": float(lrt_stat),
                "p_value": float(p_val),
                "p_adj_bonferroni": float(adj_p),
                "is_significant": bool(adj_p < 0.05)
            })
            
        return posthoc_results

    def run_comparison_suite(self, df: pd.DataFrame, kpi: str, approach: str = "Heuristic (Auto-detect)", control_label: str = None) -> dict:
        """
        The core decision engine for choosing the right statistical test.
        """
        if df.empty or kpi not in df.columns:
            return {}

        groups = [g[kpi].dropna().values for _, g in df.groupby('experience_variant_label', observed=True)]
        num_groups = len(groups)
        
        if num_groups < 2:
            return {"error": "Not enough variants to compare."}

        # Convert Pandas groupby agg to a nested dictionary for JSON serialization
        summary_df = df.groupby('experience_variant_label', observed=True)[kpi].agg(['mean', 'std', 'count'])
        summary_stats = summary_df.reset_index().to_dict(orient='records')

        results = {
            "kpi": kpi,
            "approach_used": approach,
            "summary_stats": summary_stats,
            "posthoc_results": None
        }

        # --- PATH A: GAMMA GLM ---
        if approach == "Gamma GLM (Best for Revenue/Items)":
            results["is_normal"] = False
            results["is_homogeneous"] = False
            results["test_name"] = "Gamma GLM (Likelihood Ratio Test)"
            
            # 1. Null Model (Overall mean)
            global_mean = df[kpi].dropna().mean()
            null_log_lik = np.sum(stats.gamma.logpdf(df[kpi].dropna(), a=1, scale=global_mean))
            
            # 2. Alternative Model (Variant means)
            alt_log_lik = 0
            for group_data in groups:
                alt_log_lik += np.sum(stats.gamma.logpdf(group_data, a=1, scale=group_data.mean()))
            
            # 3. Global LRT
            lrt_stat = 2 * (alt_log_lik - null_log_lik)
            df_model = num_groups - 1
            p_value = stats.chi2.sf(lrt_stat, df=df_model)
            results["p_value"] = float(p_value)
            results["is_significant"] = bool(p_value < 0.05)
            
            # 4. Post-Hoc if required
            if results["is_significant"] and num_groups > 2 and control_label:
                results["posthoc_results"] = self.run_gamma_posthoc(df, kpi, 'experience_variant_label', control_label)

        # --- PATH B: HEURISTIC ---
        else:
            model = smf.ols(f'{kpi} ~ C(experience_variant_label)', data=df).fit()
            resid_sample = model.resid.dropna()
            
            if len(resid_sample) > 5000:
                np.random.seed(42)
                resid_sample = np.random.choice(resid_sample, 5000, replace=False)
                
            _, p_norm = shapiro(resid_sample)
            results["is_normal"] = bool(p_norm >= 0.05)
            
            _, p_var = levene(*groups)
            results["is_homogeneous"] = bool(p_var >= 0.05)
            
            if results["is_normal"] and results["is_homogeneous"]:
                results["test_name"] = "Standard ANOVA"
                anova_results = sm.stats.anova_lm(model, typ=2)
                results["p_value"] = float(anova_results['PR(>F)'].iloc[0])
                
            elif results["is_normal"] and not results["is_homogeneous"]:
                results["test_name"] = "Welch's ANOVA"
                aov = welch_anova(data=df, dv=kpi, between='experience_variant_label')
                results["p_value"] = float(aov['p-unc'].iloc[0])
                
            else: 
                if num_groups > 2:
                    results["test_name"] = "Kruskal-Wallis"
                    _, p = kruskal(*groups)
                    results["p_value"] = float(p)
                else:
                    results["test_name"] = "Mann-Whitney U"
                    _, p = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                    results["p_value"] = float(p)

            results["is_significant"] = bool(results["p_value"] < 0.05)

        # Finalize Conclusion
        results["conclusion"] = self.generate_continuous_conclusion(
            kpi=kpi, 
            is_significant=results["is_significant"], 
            test_used=results["test_name"], 
            p_value=results["p_value"]
        )

        return results
