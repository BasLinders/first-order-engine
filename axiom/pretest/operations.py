import numpy as np
from scipy.stats import norm
from typing import List, Dict, Any

# Ensure we use the centralized type-safe Enum
from axiom.core.models import AlternativeHypothesis

class PretestEngine:
    """
    Calculates Sample Size, Minimum Detectable Effect (MDE), and duration
    for A/B/n tests. Stateless and optimized for Cloud environments.
    """

    @staticmethod
    def generate_sample_size_conclusion(
        n_per_variant: int,
        num_variants: int,
        mde_relative: float,
        power: float,
        alpha: float
    ) -> str:
        """Generates a definitive UI string for sample size requirements."""
        total_n = n_per_variant * num_variants
        conf = (1 - alpha) * 100
        pow_pct = power * 100
        mde_pct = mde_relative * 100
        
        return (
            f"Required Traffic: You need {n_per_variant:,} visitors per variant "
            f"({total_n:,} total) to reliably detect a {mde_pct:.2f}% relative difference "
            f"with {conf:.1f}% confidence and {pow_pct:.0f}% statistical power."
        )

    @staticmethod
    def generate_mde_table_conclusion(results: List[Dict[str, float]]) -> str:
        """Generates a summary string comparing a 2-week vs 6-week runtime."""
        if len(results) < 6:
            return "Insufficient data to generate a multi-week projection."
            
        week_2_mde = results[1]["MDE"]
        week_6_mde = results[5]["MDE"]
        
        return (
            f"Sensitivity Projection: Running this test for 2 weeks allows you to detect a "
            f"{week_2_mde:.2f}% relative effect. Extending the test to 6 weeks increases "
            f"your sensitivity, allowing you to detect an effect as small as {week_6_mde:.2f}%."
        )

    @staticmethod
    def holm_bonferroni_correction(num_variants: int, alpha: float, alternative: AlternativeHypothesis) -> float:
        """Adjusts alpha for multiple comparisons based on the worst-case step-down."""
        num_comparisons = num_variants - 1
        adjusted_alphas = alpha / np.arange(num_comparisons, 0, -1)
        
        if alternative == AlternativeHypothesis.TWO_SIDED:
            z_scores = norm.ppf(1 - adjusted_alphas / 2)
        else:
            z_scores = norm.ppf(1 - adjusted_alphas)
        
        return float(np.max(z_scores))

    @classmethod
    def get_z_alpha(cls, num_variants: int, alpha: float, alternative: AlternativeHypothesis) -> float:
        """Handles Z-score selection based on variant count and tails."""
        if num_variants > 2:
            return cls.holm_bonferroni_correction(num_variants, alpha, alternative)
        
        if alternative in [AlternativeHypothesis.GREATER, AlternativeHypothesis.LESS]:
            return float(norm.ppf(1 - alpha))
        
        # TWO_SIDED
        return float(norm.ppf(1 - alpha / 2))

    @classmethod
    def calculate_mde_table(
        cls,
        num_variants: int, 
        baseline_visitors: int, 
        baseline_conversions: int, 
        risk_pct: float, 
        trust_pct: float, 
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED
    ) -> Dict[str, Any]:
        """Calculates a 6-week MDE projection for static traffic."""
        if baseline_visitors <= 0 or baseline_conversions < 0:
            return {"table": [], "conclusion": "Invalid baseline data."}

        alpha = 1 - (risk_pct / 100)
        power = trust_pct / 100
        z_power = norm.ppf(power)
        z_alpha = cls.get_z_alpha(num_variants, alpha, alternative)
        
        baseline_rate = baseline_conversions / baseline_visitors
        if baseline_rate == 0:
             return {"table": [], "conclusion": "Baseline conversion rate cannot be zero."}

        # Assumes baseline_visitors represents 1 week of total traffic
        weekly_visitors = int(np.ceil(baseline_visitors / num_variants))
        
        results = []
        for week in range(1, 7):
            n = weekly_visitors * week
            se = np.sqrt(2 * baseline_rate * (1 - baseline_rate) / n)
            mde_rel = ((z_alpha + z_power) * se / baseline_rate) * 100
            
            results.append({
                "Week": int(week), 
                "Visitors_Per_Variant": int(n), 
                "MDE": float(mde_rel)
            })
            
        return {
            "table": results,
            "conclusion": cls.generate_mde_table_conclusion(results)
        }

    @classmethod
    def calculate_fixed_sample_size(
        cls,
        baseline_cr: float,
        mde_relative: float,
        num_variants: int,
        alpha: float = 0.05,
        power: float = 0.80,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED
    ) -> Dict[str, Any]:
        """
        Calculates required sample size per variant using pooled variance.
        """
        if baseline_cr <= 0 or mde_relative <= 0:
            return {"n_per_variant": 0, "total_n": 0, "conclusion": "Invalid inputs."}

        # 1. Parameter setup
        p1 = baseline_cr
        p2 = baseline_cr * (1 + mde_relative)
        p_pooled = (p1 + p2) / 2
        
        # 2. Get adjusted Z-alpha and Z-beta
        z_alpha = cls.get_z_alpha(num_variants, alpha, alternative)
        z_beta = norm.ppf(power)

        # 3. Standard Error components
        term1 = z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled))
        term2 = z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        
        diff = p2 - p1
        
        n_per_variant = int(np.ceil(((term1 + term2) ** 2) / (diff ** 2)))
        
        return {
            "n_per_variant": n_per_variant,
            "total_n": n_per_variant * num_variants,
            "mde_relative_used": mde_relative,
            "power_used": power,
            "alpha_used": alpha,
            "conclusion": cls.generate_sample_size_conclusion(
                n_per_variant=n_per_variant,
                num_variants=num_variants,
                mde_relative=mde_relative,
                power=power,
                alpha=alpha
            )
        }
