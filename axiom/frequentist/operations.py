import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from typing import List, Optional, Tuple, Dict, Any
from engine.core.models import AlternativeHypothesis

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
    """

    @staticmethod
    def apply_cuped(
        df: pd.DataFrame, 
        target_kpi: str, 
        pre_period_kpi: str
    ) -> pd.DataFrame:
        """
        Standard CUPED adjustment. 
        Formula: $Y_{cuped} = Y - \theta(X_{pre} - \bar{X}_{pre})$
        """
        cov = df[[target_kpi, pre_period_kpi]].cov().iloc[0, 1]
        var_pre = df[pre_period_kpi].var()
        
        theta = cov / var_pre if var_pre != 0 else 0
        mean_pre = df[pre_period_kpi].mean()
        
        df[f'{target_kpi}_cuped'] = df[target_kpi] - theta * (df[pre_period_kpi] - mean_pre)
        return df

    def run_lin_adjustment(
        self, 
        df: pd.DataFrame, 
        target_kpi: str, 
        pre_period_kpi: str, 
        variant_col: str = 'variant'
    ) -> List[Dict[str, Any]]:
        """
        Lin's Adjustment (2013). More robust than CUPED for heterogeneous effects.
        Regression: $Y \sim Treatment * (Covariate - \bar{Covariate})$
        """
        # 1. Setup Data
        df = df.copy()
        df['cov_centered'] = df[pre_period_kpi] - df[pre_period_kpi].mean()
        
        variants = sorted(df[variant_col].unique())
        baseline = variants[0] # Assumes alphabetical or 'Control' is first
        
        # 2. Fit OLS with Interaction and Robust Standard Errors (HC3)
        # The interaction term handles cases where the treatment changes the variance.
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
        Stays within NumPy C-extensions to bypass the Python GIL.
        """
        # Create populations
        pop_ctrl = np.array([1]*ctrl_conv + [0]*(ctrl_n - ctrl_conv))
        pop_chal = np.array([1]*chal_conv + [0]*(chal_n - chal_conv))

        # Generate ALL resample indices at once (Matrix: n_bootstraps x N)
        # Note: For extremely large N, consider chunking to avoid MemoryError
        idx_ctrl = np.random.randint(0, ctrl_n, size=(n_bootstraps, ctrl_n))
        idx_chal = np.random.randint(0, chal_n, size=(n_bootstraps, chal_n))

        # Vectorized mean calculation
        means_ctrl = pop_ctrl[idx_ctrl].mean(axis=1)
        means_chal = pop_chal[idx_chal].mean(axis=1)

        # Pooled Standard Error (Vectorized)
        p_pooled = (means_ctrl * ctrl_n + means_chal * chal_n) / (ctrl_n + chal_n)
        se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1/ctrl_n + 1/chal_n))
        
        # Avoid division by zero in edge cases
        se_pooled[se_pooled == 0] = np.inf
        
        z_stats = (means_chal - means_ctrl) / se_pooled
        p_values = 2 * (1 - norm.cdf(np.abs(z_stats)))
        
        return float(np.mean(p_values < alpha))
