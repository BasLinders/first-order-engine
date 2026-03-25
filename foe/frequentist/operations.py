import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from typing import List, Optional, Tuple, Dict, Any

# Updated import path based on our earlier root folder rename
from axiom.core.models import AlternativeHypothesis

def apply_sidak(alpha: float, num_variants: int) -> float:
    """Calculates adjusted alpha for multiple comparisons (A/B/n)."""
    num_comparisons = num_variants - 1
    if num_comparisons <= 1:
        return alpha
    return 1 - (1 - alpha) ** (1 / num_comparisons)

class FrequentistEngine:
    """
    Axiom Frequentist Engine: Implements Variance Reduction (CUPED/Lin),
    Robust OLS Inference, and High-Performance Bootstrapping.
    Stateless design optimized for Cloud Functions.
    """

    @staticmethod
    def generate_conclusion_statement(
        variant_name: str,
        is_significant: bool,
        relative_lift: float
    ) -> str:
        """
        Generates a definitive, UI-agnostic summary of the results in English.
        """
        if not is_significant:
            return (
                f"Inconclusive / Flat: '{variant_name}' shows no statistically "
                "significant difference from the control. We cannot confidently "
                "conclude that this variant had a meaningful impact."
            )

        direction = "positive" if relative_lift > 0 else "negative"
        action = "a clear winner" if relative_lift > 0 else "performing worse than control"
        
        return (
            f"Significant {direction} result: '{variant_name}' is {action} "
            f"with an observed relative impact of {relative_lift:+.2%}. "
            f"You can confidently {'roll this out' if relative_lift > 0 else 'discard this variant'}."
        )

    @staticmethod
    def apply_cuped(
        df: pd.DataFrame, 
        target_kpi: str, 
        pre_period_kpi: str
    ) -> pd.DataFrame:
        """
        Standard CUPED adjustment. 
        Formula: Y_cuped = Y - theta * (X_pre - mean(X_pre))
        """
        cov = df[[target_kpi, pre_period_kpi]].cov().iloc[0, 1]
        var_pre = df[pre_period_kpi].var()
        
        theta = cov / var_pre if var_pre != 0 else 0
        mean_pre = df[pre_period_kpi].mean()
        
        df[f'{target_kpi}_cuped'] = df[target_kpi] - theta * (df[pre_period_kpi] - mean_pre)
        return df

    @staticmethod
    def run_lin_adjustment(
        df: pd.DataFrame, 
        target_kpi: str, 
        pre_period_kpi: str, 
        variant_col: str = 'variant'
    ) -> List[Dict[str, Any]]:
        """
        Lin's Adjustment (2013). More robust than CUPED for heterogeneous effects.
        Regression: Y ~ Treatment * (Covariate - mean(Covariate))
        """
        # 1. Setup Data
        df = df.copy()
        df['cov_centered'] = df[pre_period_kpi] - df[pre_period_kpi].mean()
        
        variants = sorted(df[variant_col].unique())
        baseline = variants[0] # Assumes alphabetical or 'Control' is first
        
        # 2. Fit OLS with Interaction and Robust Standard Errors (HC3)
        formula = f"{target_kpi} ~ C({variant_col}, Treatment(reference='{baseline}')) * cov_centered"
        model = smf.ols(formula, data=df).fit(cov_type='HC3')
        
        results = []
        for challenger in variants[1:]:
            term = f"C({variant_col}, Treatment(reference='{baseline}'))[T.{challenger}]"
            
            p_val = model.pvalues[term]
            ate = model.params[term] # Average Treatment Effect
            control_mean = model.params['Intercept']
            
            results.append({
                "variant": challenger,
                "p_value": float(p_val),
                "is_significant": p_val < 0.05,
                "absolute_lift": float(ate),
                "relative_lift": float(ate / control_mean) if control_mean != 0 else 0,
                "ci": model.conf_int().loc[term].tolist(),
                "std_err": float(model.bse[term])
            })
            
        return results

    @staticmethod
    def run_ztest(
        diff: float, 
        se_diff: float, 
        alternative: AlternativeHypothesis
    ) -> float:
        """Standard Z-test logic for varying tail configurations."""
        if se_diff == 0:
            return 1.0
        
        z_stat = diff / se_diff
        
        if alternative == AlternativeHypothesis.GREATER:
            return 1 - norm.cdf(z_stat)
        elif alternative == AlternativeHypothesis.LESS:
            return norm.cdf(z_stat)
        else: # TWO_SIDED
            return 2 * (1 - norm.cdf(abs(z_stat)))

    @staticmethod
    def calculate_analytical_power(
        diff: float,
        se_diff: float,
        alpha: float,
        alternative: AlternativeHypothesis
    ) -> float:
        """Closed-form power calculation."""
        if se_diff == 0:
            return 0.0
            
        z_delta = abs(diff) / se_diff
        side_multiplier = 2 if alternative == AlternativeHypothesis.TWO_SIDED else 1
        z_alpha = norm.ppf(1 - alpha / side_multiplier)
        
        return float(norm.cdf(z_delta - z_alpha))

    @staticmethod
    def run_vectorized_bootstrap_power(
        ctrl_conv: int, ctrl_n: int, 
        chal_conv: int, chal_n: int, 
        alpha: float = 0.05, 
        n_bootstraps: int = 10000
    ) -> float:
        """
        Calculates observed power via high-performance vectorized bootstrapping.
        Optimized for Cloud Environments: Uses the Binomial distribution to avoid 
        massive memory allocations (OOM errors) when N is very large.
        """
        if ctrl_n == 0 or chal_n == 0:
            return 0.0

        ctrl_p = ctrl_conv / ctrl_n
        chal_p = chal_conv / chal_n

        # Simulate conversion COUNTS directly using the Binomial distribution
        # Size is just (n_bootstraps,) instead of (n_bootstraps, N)
        sim_ctrl_convs = np.random.binomial(n=ctrl_n, p=ctrl_p, size=n_bootstraps)
        sim_chal_convs = np.random.binomial(n=chal_n, p=chal_p, size=n_bootstraps)

        # Calculate simulated conversion rates
        means_ctrl = sim_ctrl_convs / ctrl_n
        means_chal = sim_chal_convs / chal_n

        # Pooled Standard Error (Vectorized)
        p_pooled = (sim_ctrl_convs + sim_chal_convs) / (ctrl_n + chal_n)
        
        # Add a tiny epsilon to avoid division by zero if a simulated pooled p is 0 or 1
        epsilon = 1e-9
        se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1/ctrl_n + 1/chal_n) + epsilon)
        
        z_stats = (means_chal - means_ctrl) / se_pooled
        p_values = 2 * (1 - norm.cdf(np.abs(z_stats)))
        
        return float(np.mean(p_values < alpha))
