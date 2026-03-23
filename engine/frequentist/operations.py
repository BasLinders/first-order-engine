import numpy as np
import pandas as pd
import concurrent.futures
import statsmodels.formula.api as smf
from scipy.stats import norm
from typing import List, Optional, Tuple, Dict, Any
from engine.core.models import AlternativeHypothesis

# --- Corrections ---

def apply_sidak(alpha: float, num_variants: int) -> float:
    """Calculates adjusted alpha for multiple comparisons."""
    num_comparisons = num_variants - 1
    if num_comparisons <= 1:
        return alpha
    return 1 - (1 - alpha) ** (1 / num_comparisons)

class FrequentistEngine:
    """
    Advanced Frequentist methods including Variance Reduction (CUPED) 
    and OLS-based inference.
    """

    @staticmethod
    def apply_cuped(
        df: pd.DataFrame, 
        target_kpi: str, 
        pre_period_kpi: str, 
        variant_col: str = 'experience_variant_label'
    ) -> pd.DataFrame:
        """
        Standard CUPED: Adjusts the post-period KPI using the pre-period covariate.
        Formula: Y_cuped = Y_actual - theta * (X_pre - mean(X_pre))
        """
        # Calculate Theta (covariance / variance of pre-period)
        covariance = df[[target_kpi, pre_period_kpi]].cov().iloc[0, 1]
        variance_pre = df[pre_period_kpi].var()
        
        theta = covariance / variance_pre if variance_pre != 0 else 0
        
        # Apply adjustment
        mean_pre = df[pre_period_kpi].mean()
        df[f'{target_kpi}_cuped'] = df[target_kpi] - theta * (df[pre_period_kpi] - mean_pre)
        
        return df

    def run_lin_adjustment(
        self, 
        df: pd.DataFrame, 
        target_kpi: str, 
        pre_period_kpi: str, 
        variant_col: str = 'experience_variant_label'
    ) -> Dict[str, Any]:
        """
        Implements Lin's Adjustment (2013).
        Regression: Y ~ Treatment + Covariate_Centered + (Treatment * Covariate_Centered)
        
        This is more robust than standard CUPED for heterogeneous treatment effects.
        """
        # 1. Center the covariate (pre-period data)
        df['covariate_centered'] = df[pre_period_kpi] - df[pre_period_kpi].mean()
        
        # 2. Define Treatment Dummy (Assumes 'A' or 'Control' is baseline)
        # We ensure the model treats the first alphabetical variant as baseline
        variants = sorted(df[variant_col].unique())
        baseline = variants[0]
        test_variant = variants[1]

        # 3. Fit OLS with interaction term
        # Lin (2013) recommends: Y = alpha + beta*Treatment + gamma*Covariate + delta*(Treatment * Covariate)
        formula = f"{target_kpi} ~ C({variant_col}, Treatment(reference='{baseline}')) * covariate_centered"
        model = smf.ols(formula, data=df).fit(cov_type='HC3') # Using Robust Standard Errors
        
        # 4. Extract Results
        # The coefficient for the Treatment dummy is our Adjusted Lift
        treatment_key = f"C({variant_col}, Treatment(reference='{baseline}'))[T.{test_variant}]"
        
        p_val = model.pvalues[treatment_key]
        ate = model.params[treatment_key] # Average Treatment Effect
        
        # Calculate relative lift based on the intercept (Control Mean)
        control_mean = model.params['Intercept']
        relative_lift = ate / control_mean if control_mean != 0 else 0

        return {
            "p_value": float(p_val),
            "is_significant": p_val < 0.05,
            "absolute_ate": float(ate),
            "relative_lift": float(relative_lift),
            "confidence_interval": model.conf_int().loc[treatment_key].tolist(),
            "model_summary": model.summary2().tables[1]
        }
        
# --- Statistical Tests --- 

def run_ztest(
    diff_cr: float, 
    se_diff: float, 
    alternative: AlternativeHypothesis
) -> float:
    """
    Performs the Z-test using the Unpooled Variance approach.
    Matches the branching logic for 'Greater', 'Less', and 'Two-sided'.
    """
    if se_diff == 0:
        return 1.0
        
    z_stat = diff_cr / se_diff
    
    if alternative == AlternativeHypothesis.GREATER:
        return 1 - norm.cdf(z_stat)
    elif alternative == AlternativeHypothesis.LESS:
        return norm.cdf(z_stat)
    else: # Two-sided
        return 2 * (1 - norm.cdf(abs(z_stat)))

# --- Power Functions ---

def calculate_observed_power(
    diff_cr: float,
    se_diff: float,
    alpha: float,
    alternative: AlternativeHypothesis
) -> float:
    """
    Analytical Power calculation logic extracted from hexkit.
    """
    if se_diff == 0:
        return 1.0
        
    z_delta = abs(diff_cr) / se_diff
    
    if alternative in [AlternativeHypothesis.GREATER, AlternativeHypothesis.LESS]:
        z_alpha = norm.ppf(1 - alpha)
        return norm.cdf(z_delta - z_alpha)
    else: # Two-sided
        z_alpha = norm.ppf(1 - alpha / 2)
        return norm.cdf(z_delta - z_alpha) + norm.cdf(-z_delta - z_alpha)

def _bootstrap_sample_is_significant(
    data_ctrl: np.ndarray, 
    data_chal: np.ndarray, 
    alpha: float, 
    alternative: AlternativeHypothesis
) -> bool:
    """
    Private helper: Performs a single bootstrap resample and returns 
    if the result was significant.
    """
    # Resample with replacement
    sample_ctrl = np.random.choice(data_ctrl, size=len(data_ctrl), replace=True)
    sample_chal = np.random.choice(data_chal, size=len(data_chal), replace=True)
    
    # Calculate pooled SE for the bootstrap iteration
    p_ctrl, p_chal = np.mean(sample_ctrl), np.mean(sample_chal)
    n_ctrl, n_chal = len(sample_ctrl), len(sample_chal)
    
    p_pooled = (np.sum(sample_ctrl) + np.sum(sample_chal)) / (n_ctrl + n_chal)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n_ctrl + 1 / n_chal))
    
    if se == 0:
        return False
        
    z_stat = (p_chal - p_ctrl) / se
    
    # P-value calculation based on tail
    if alternative == AlternativeHypothesis.GREATER:
        p_val = 1 - norm.cdf(z_stat)
    elif alternative == AlternativeHypothesis.LESS:
        p_val = norm.cdf(z_stat)
    else: # Two-sided
        p_val = 2 * (1 - norm.cdf(abs(z_stat)))
        
    return p_val < alpha

def run_bootstrap_power(
    conversions_ctrl: int,
    visitors_ctrl: int,
    conversions_chal: int,
    visitors_chal: int,
    alpha: float,
    alternative: AlternativeHypothesis,
    n_bootstraps: int = 10000
) -> float:
    """
    Calculates observed power via bootstrapping. 
    Uses ThreadPoolExecutor for parallel execution.
    """
    # Create the raw Bernoulli arrays (1s and 0s)
    data_ctrl = np.concatenate([np.ones(conversions_ctrl), np.zeros(visitors_ctrl - conversions_ctrl)])
    data_chal = np.concatenate([np.ones(conversions_chal), np.zeros(visitors_chal - conversions_chal)])
    
    significant_count = 0
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map the private helper across the number of bootstraps
        futures = [
            executor.submit(_bootstrap_sample_is_significant, data_ctrl, data_chal, alpha, alternative) 
            for _ in range(n_bootstraps)
        ]
        
        for future in concurrent.futures.as_completed(futures):
            if future.result():
                significant_count += 1
                
    return significant_count / n_bootstraps
