import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
from typing import Tuple, List, Optional, Dict, Any

class BehavioralEngine:
    """
    Engine for analyzing visitor-level behavioral metrics.
    Handles skew, outliers, and Welch's t-test inference.
    """

    @staticmethod
    def detect_outliers_mask(df: pd.DataFrame, kpi: str, threshold: float = 3.0, 
                             large_file_limit: int = 10000) -> pd.Series:
        """
        Identifies outliers using IQR (Fast/Large) or OLS Residuals (Precise/Small).
        """
        if df.empty or kpi not in df.columns:
            return pd.Series(False, index=df.index)

        if len(df) > large_file_limit:
            # IQR Method: Vectorized via groupby for massive performance gains
            def iqr_filter(group):
                q1 = group.quantile(0.25)
                q3 = group.quantile(0.75)
                iqr = q3 - q1
                return (group < (q1 - threshold * iqr)) | (group > (q3 + threshold * iqr))
            
            # Apply transformation and fill NaNs (if any) with False
            mask = df.groupby('experience_variant_label')[kpi].transform(iqr_filter)
            return mask.fillna(False).astype(bool)
        else:
            # OLS Studentized Residuals
            # Wrapped in Q() to prevent statsmodels from crashing if column names have spaces
            formula = f"Q('{kpi}') ~ C(Q('experience_variant_label'))"
            model = smf.ols(formula, data=df).fit()
            influence = model.get_influence()
            return pd.Series(np.abs(influence.resid_studentized_internal) > threshold, index=df.index)

    @staticmethod
    def apply_transformations(df: pd.DataFrame, kpi: str, method: str, 
                              params: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Handles Winsorization, Log Transforms, or Removal of outliers.
        """
        df_processed = df.copy()
        meta = {"method": method}

        if method == 'Winsorizing':
            if params.get('logic') == 'Percentile':
                p = params.get('percentile', 99)
                low_p, high_p = (100 - p) / 2, 100 - ((100 - p) / 2)
                lower, upper = np.percentile(df_processed[kpi].dropna(), [low_p, high_p])
            else: # Standard Deviation
                sd_limit = params.get('stdev', 3)
                mean, std = df_processed[kpi].mean(), df_processed[kpi].std()
                lower, upper = mean - (sd_limit * std), mean + (sd_limit * std)
            
            # Behavioral metrics (clicks, time) shouldn't logically be < 0
            lower = max(0.0, float(lower)) 
            upper = float(upper)
            
            df_processed[kpi] = df_processed[kpi].clip(lower, upper)
            meta.update({"lower_cap": lower, "upper_cap": upper})

        elif method == 'Log Transform':
            # log1p safely handles 0-value behavioral events (e.g., 0 revenue -> log(1) = 0)
            df_processed[kpi] = np.log1p(df_processed[kpi])
        
        return df_processed, meta

    def run_welch_inference(self, df: pd.DataFrame, kpi: str, control_label: str, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Performs Welch's t-test. 
        Returns p-value, lift, and descriptive stats.
        """
        variants = df['experience_variant_label'].unique()
        if len(variants) != 2:
            raise ValueError("Welch's t-test engine requires exactly 2 variants in the dataset.")
        
        if control_label not in variants:
            raise ValueError(f"Control label '{control_label}' not found in dataset variants.")

        # Determine the challenger label dynamically
        challenger_label = variants[1] if variants[0] == control_label else variants[0]

        # Extract groups, dropping NaNs to prevent scipy calculation errors
        group_control = df[df['experience_variant_label'] == control_label][kpi].dropna()
        group_challenger = df[df['experience_variant_label'] == challenger_label][kpi].dropna()

        # equal_var=False triggers the Welch's Satterthwaite effective degrees of freedom
        t_stat, p_val = stats.ttest_ind(group_challenger, group_control, equal_var=False)
        
        mean_control = float(group_control.mean())
        mean_challenger = float(group_challenger.mean())
        
        # Calculate lift safely
        lift = (mean_challenger - mean_control) / mean_control if mean_control != 0 else 0.0

        return {
            "p_value": float(p_val),
            "is_significant": bool(p_val < alpha),
            "alpha_used": alpha,
            "lift": lift,
            "group_control": {"label": control_label, "mean": mean_control, "n": len(group_control)},
            "group_challenger": {"label": challenger_label, "mean": mean_challenger, "n": len(group_challenger)}
        }
