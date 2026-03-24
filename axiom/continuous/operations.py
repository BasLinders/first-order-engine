import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro, levene, kruskal, mannwhitneyu
from pingouin import welch_anova, pairwise_gameshowell
import scikit_posthocs as sp

class ContinuousMetricEngine:
    """
    Engine for analyzing continuous metrics (Revenue, Profit, Quantity).
    Handles the statistical decision tree: Normality -> Variance -> Test Selection.
    """

    @staticmethod
    def detect_outliers_ols(df, kpi, threshold=3):
        """Uses OLS residuals and influence to detect outliers for smaller datasets."""
        model = smf.ols(f'{kpi} ~ C(experience_variant_label)', data=df).fit()
        influence = model.get_influence()
        
        # Combine Studentized Residuals, Leverage, and DFFITS
        std_resid = np.abs(influence.resid_studentized_internal) > threshold
        leverage = influence.hat_matrix_diag > (threshold * (model.df_model + 1) / len(df))
        dffits = np.abs(influence.dffits[0]) > (threshold * np.sqrt((model.df_model + 1) / len(df)))
        
        return (std_resid | leverage | dffits), model

    @staticmethod
    def winsorize_series(series, method='Standard Deviation', param=3):
        """Caps outliers without removing data points."""
        if method == 'Standard Deviation':
            lower = series.mean() - (param * series.std())
            upper = series.mean() + (param * series.std())
        else: # Percentile
            lower_p = (100.0 - param) / 2.0
            upper_p = 100.0 - lower_p
            lower, upper = np.percentile(series.dropna(), [lower_p, upper_p])
        
        return series.clip(lower, upper), lower, upper

    def run_comparison_suite(self, df, kpi):
        """
        The core decision engine for choosing the right statistical test.
        """
        # 1. Grouping for variance/non-parametric tests
        groups = [g[kpi].values for _, g in df.groupby('experience_variant_label', observed=True)]
        num_groups = len(groups)
        
        # 2. Assumption: Normality of Residuals
        # Refit model to get final residuals
        model = smf.ols(f'{kpi} ~ C(experience_variant_label)', data=df).fit()
        # Shapiro-Wilk is limited to N <= 5000 in many implementations
        _, p_norm = shapiro(model.resid[:5000])
        is_normal = p_norm >= 0.05
        
        # 3. Assumption: Homogeneity of Variance
        _, p_var = levene(*groups)
        is_homogeneous = p_var >= 0.05
        
        results = {
            "is_normal": is_normal,
            "is_homogeneous": is_homogeneous,
            "model": model,
            "summary_stats": df.groupby('experience_variant_label', observed=True)[kpi].agg(['mean', 'std', 'count'])
        }

        # 4. Statistical Decision Tree
        if is_normal and is_homogeneous:
            results["test_name"] = "Standard ANOVA"
            anova_results = sm.stats.anova_lm(model, typ=2)
            results["p_value"] = anova_results['PR(>F)'].iloc[0]
            results["significant"] = results["p_value"] < 0.05
            
        elif is_homogeneous:
            results["test_name"] = "Welch's ANOVA"
            aov = welch_anova(data=df, dv=kpi, between='experience_variant_label')
            results["p_value"] = aov['p-unc'].iloc[0]
            results["significant"] = results["p_value"] < 0.05
            
        else: # Heterogeneous Variances
            if num_groups > 2:
                results["test_name"] = "Kruskal-Wallis"
                _, p = kruskal(*groups)
                results["p_value"] = p
            else:
                results["test_name"] = "Mann-Whitney U"
                _, p = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                results["p_value"] = p
            results["significant"] = results["p_value"] < 0.05

        return results
