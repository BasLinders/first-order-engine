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
        if len(df) > large_file_limit:
            # IQR Method: robust and fast for big data
            outliers_mask = pd.Series([False] * len(df), index=df.index)
            for variant in df['experience_variant_label'].unique():
                subset = df[df['experience_variant_label'] == variant][kpi].dropna()
                if not subset.empty:
                    q1, q3 = subset.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    is_outlier = (subset < (q1 - threshold * iqr)) | (subset > (q3 + threshold * iqr))
                    outliers_mask.loc[subset.index] = is_outlier
            return outliers_mask
        else:
            # OLS Studentized Residuals: Mathematically precise for smaller samples
            formula = f"{kpi} ~ C(experience_variant_label)"
            model = smf.ols(formula, data=df).fit()
            influence = model.get_influence()
            return np.abs(influence.resid_studentized_internal) > threshold

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
                lower, upper = np.percentile(df_processed[kpi], [low_p, high_p])
            else: # Standard Deviation
                sd_limit = params.get('stdev', 3)
                mean, std = df_processed[kpi].mean(), df_processed[kpi].std()
                lower, upper = mean - (sd_limit * std), mean + (sd_limit * std)
            
            lower = max(0, lower) # Behavioral metrics (clicks, time) shouldn't be < 0
            df_processed[kpi] = df_processed[kpi].clip(lower, upper)
            meta.update({"lower_cap": lower, "upper_cap": upper})

        elif method == 'Log Transform':
            # log1p handles 0-value behavioral events safely
            df_processed[kpi] = np.log1p(df_processed[kpi])
        
        return df_processed, meta

    def run_welch_inference(self, df: pd.DataFrame, kpi: str) -> Dict[str, Any]:
        """
        Performs Welch's t-test. 
        Returns p-value, lift, and descriptive stats.
        """
        variants = df['experience_variant_label'].unique()
        if len(variants) != 2:
            raise ValueError("Welch's t-test engine requires exactly 2 variants.")

        # Ensure consistent ordering: [Control, Variant]
        # We assume the first variant encountered is the baseline unless specified
        group_a = df[df['experience_variant_label'] == variants[0]][kpi]
        group_b = df[df['experience_variant_label'] == variants[1]][kpi]

        # equal_var=False triggers the Welch's Satterthwaite effective degrees of freedom
        t_stat, p_val = stats.ttest_ind(group_b, group_a, equal_var=False)
        
        mean_a, mean_b = group_a.mean(), group_b.mean()
        lift = (mean_b - mean_a) / mean_a if mean_a != 0 else 0

        return {
            "p_value": float(p_val),
            "is_significant": p_val < 0.05,
            "lift": float(lift),
            "group_a": {"label": variants[0], "mean": mean_a, "n": len(group_a)},
            "group_b": {"label": variants[1], "mean": mean_b, "n": len(group_b)}
        }
