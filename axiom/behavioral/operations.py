import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
from typing import Tuple, Dict, Any

class BehavioralEngine:
    """
    Engine for analyzing visitor-level behavioral metrics.
    Handles skew, outliers, and Welch's t-test inference.
    Strictly returns JSON-serializable primitives for Cloud APIs.
    """

    @staticmethod
    def generate_behavioral_conclusion(
        kpi: str, 
        is_significant: bool, 
        lift: float, 
        p_value: float,
        challenger_label: str
    ) -> str:
        """Generates a definitive UI statement for the Welch's t-test results."""
        if not is_significant:
            return (
                f"Flat: There is no statistically significant difference in {kpi} "
                f"between the variants (p={p_value:.4f}). The observed lift of {lift:+.2%} "
                "is likely due to random chance."
            )
        
        direction = "Positive" if lift > 0 else "Negative"
        action = "an improvement" if lift > 0 else "a degradation"
        
        return (
            f"Significant {direction} Impact: The challenger variant '{challenger_label}' "
            f"caused a statistically significant {action} in {kpi} (p={p_value:.4f}). "
            f"The relative difference compared to control is {lift:+.2%}."
        )

    @staticmethod
    def detect_outliers_mask(
        df: pd.DataFrame, 
        kpi: str, 
        threshold: float = 3.0, 
        large_file_limit: int = 10000
    ) -> List[bool]:
        """
        Identifies outliers using IQR (Fast/Large) or OLS Residuals (Precise/Small).
        Returns a JSON-serializable list of booleans.
        """
        if df.empty or kpi not in df.columns or 'experience_variant_label' not in df.columns:
            return [False] * len(df)

        if len(df) > large_file_limit:
            # Fully vectorized IQR Method (Fastest)
            q1 = df.groupby('experience_variant_label')[kpi].transform(lambda x: x.quantile(0.25))
            q3 = df.groupby('experience_variant_label')[kpi].transform(lambda x: x.quantile(0.75))
            iqr = q3 - q1
            
            lower_bound = q1 - (threshold * iqr)
            upper_bound = q3 + (threshold * iqr)
            
            mask = (df[kpi] < lower_bound) | (df[kpi] > upper_bound)
            return mask.fillna(False).tolist()
            
        else:
            # OLS Studentized Residuals (Precise)
            # Wrapped in Q() to prevent statsmodels from crashing if column names have spaces
            formula = f"Q('{kpi}') ~ C(Q('experience_variant_label'))"
            model = smf.ols(formula, data=df).fit()
            influence = model.get_influence()
            
            mask = np.abs(influence.resid_studentized_internal) > threshold
            return pd.Series(mask, index=df.index).fillna(False).tolist()

    @staticmethod
    def apply_transformations(
        df: pd.DataFrame, 
        kpi: str, 
        method: str, 
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Handles Winsorization, Log Transforms, or Removal of outliers.
        """
        df_processed = df.copy()
        meta = {"method": method, "kpi": kpi}

        if method == 'Winsorizing':
            if params.get('logic') == 'Percentile':
                p = params.get('percentile', 99)
                low_p = (100 - p) / 2
                high_p = 100 - low_p
                lower, upper = np.percentile(df_processed[kpi].dropna(), [low_p, high_p])
            else: # Standard Deviation
                sd_limit = params.get('stdev', 3)
                mean = df_processed[kpi].mean()
                std = df_processed[kpi].std()
                lower = mean - (sd_limit * std)
                upper = mean + (sd_limit * std)
            
            # Behavioral metrics (clicks, time) shouldn't logically be < 0
            lower = max(0.0, float(lower)) 
            upper = float(upper)
            
            df_processed[kpi] = df_processed[kpi].clip(lower, upper)
            meta.update({"lower_cap": lower, "upper_cap": upper})

        elif method == 'Log Transform':
            # log1p safely handles 0-value behavioral events (e.g., 0 clicks -> log(1) = 0)
            df_processed[kpi] = np.log1p(df_processed[kpi])
        
        return df_processed, meta

    def run_welch_inference(
        self, 
        df: pd.DataFrame, 
        kpi: str, 
        control_label: str, 
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Performs Welch's t-test for unequal variances. 
        Returns p-value, lift, and descriptive stats as JSON-safe dict.
        """
        variants = df['experience_variant_label'].dropna().unique()
        if len(variants) != 2:
            return {"error": "Welch's t-test engine requires exactly 2 variants in the dataset."}
        
        if control_label not in variants:
            return {"error": f"Control label '{control_label}' not found in dataset variants."}

        # Determine the challenger label dynamically
        challenger_label = str(variants[1] if variants[0] == control_label else variants[0])

        # Extract groups, dropping NaNs to prevent scipy calculation errors
        group_control = df[df['experience_variant_label'] == control_label][kpi].dropna()
        group_challenger = df[df['experience_variant_label'] == challenger_label][kpi].dropna()

        # equal_var=False triggers the Welch's Satterthwaite effective degrees of freedom
        t_stat, p_val = stats.ttest_ind(group_challenger, group_control, equal_var=False)
        
        mean_control = float(group_control.mean())
        mean_challenger = float(group_challenger.mean())
        
        # Calculate lift safely
        lift = (mean_challenger - mean_control) / mean_control if mean_control != 0 else 0.0
        is_significant = bool(p_val < alpha)

        return {
            "p_value": float(p_val),
            "is_significant": is_significant,
            "alpha_used": alpha,
            "lift": float(lift),
            "group_control": {"label": str(control_label), "mean": mean_control, "n": len(group_control)},
            "group_challenger": {"label": challenger_label, "mean": mean_challenger, "n": len(group_challenger)},
            "conclusion": self.generate_behavioral_conclusion(
                kpi=kpi,
                is_significant=is_significant,
                lift=lift,
                p_value=float(p_val),
                challenger_label=challenger_label
            )
        }
