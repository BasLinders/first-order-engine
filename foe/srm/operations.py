import pandas as pd
import scipy.stats as stats
import numpy as np
from typing import Dict, List, Any


class SRMEngine:
    """
    Advanced diagnostic suite for Sample Ratio Mismatch (SRM).
    """

    @staticmethod
    def generate_srm_conclusion(
        is_mismatch: bool, p_value: float, severity: float
    ) -> str:
        """Generates a definitive UI statement for the overall SRM check."""
        if is_mismatch:
            direction = "over-indexed" if severity > 0 else "under-indexed"
            return (
                f"CRITICAL ALERT: Sample Ratio Mismatch (SRM) detected (p={p_value:.4g}). "
                f"The observed traffic split deviates significantly from the expected design. "
                f"The primary variant is {direction} by {abs(severity):.2%}. "
                "Do NOT trust the conversion metrics of this test until the tracking or routing bug is resolved."
            )
        return (
            f"PASS: No Sample Ratio Mismatch detected (p={p_value:.4g}). "
            "The traffic split matches the expected design. It is safe to proceed with the analysis."
        )

    @staticmethod
    def calculate_chi_squared(
        observed: List[int], expected: List[float], alpha: float = 0.01
    ) -> Dict[str, Any]:
        """Standard Chi-Squared Goodness of Fit."""
        total = sum(observed)
        if total == 0 or sum(expected) == 0:
            return {
                "p_value": 1.0,
                "is_mismatch": False,
                "severity": 0.0,
                "conclusion": "Invalid data: Total visitors or expected weights sum to zero."
            }

        expected_counts = [total * (p / sum(expected)) for p in expected]

        # Suppress scipy warnings if expected counts are too low
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2, p_val = stats.chisquare(f_obs=observed, f_exp=expected_counts)

        # Calculate severity (deviation of the first variant from expectation)
        severity = (observed[0] / total) - (expected[0] / sum(expected))
        is_mismatch = bool(p_val < alpha)

        return {
            "p_value": float(p_val),
            "is_mismatch": is_mismatch,
            "severity": float(severity),
            "conclusion": SRMEngine.generate_srm_conclusion(
                is_mismatch, float(p_val), float(severity)
            )
        }

    def diagnose_segments(
        self,
        df: pd.DataFrame,
        dimensions: List[str],
        variant_col: str,
        expected_ratio: List[float],
        alpha: float = 0.01
    ) -> List[Dict[str, Any]]:
        """
        Runs SRM checks across multiple dimensions to find the root cause.
        Returns a JSON-serializable list of dictionaries sorted by severity (p-value).
        """
        report = []
        for dim in dimensions:
            if dim not in df.columns:
                continue

            segments = df[dim].dropna().unique()
            for seg in segments:
                # Count visitors per variant in this specific segment
                counts = (
                    df[df[dim] == seg][variant_col].value_counts().sort_index().tolist()
                )

                # Only test if we have data for all expected variants
                if len(counts) == len(expected_ratio) and sum(counts) > 0:
                    res = self.calculate_chi_squared(
                        counts, expected_ratio, alpha=alpha
                    )
                    report.append(
                        {
                            "dimension": str(dim),
                            "segment": str(seg),
                            "total_segment_visitors": sum(counts),
                            "p_value": res["p_value"],
                            "severity": res["severity"],
                            "status": "FAIL (SRM)" if res["is_mismatch"] else "PASS",
                            "conclusion": f"Segment '{seg}' in '{dim}' {'failed' if res['is_mismatch'] else 'passed'} SRM check."
                        }
                    )

        # Sort by lowest p-value to bubble the worst offenders to the top
        return sorted(report, key=lambda x: x["p_value"])

    @staticmethod
    def get_srm_thresholds(
        total_n: int, target_pct: float = 0.50, alpha: float = 0.01
    ) -> Dict[str, Any]:
        """
        Calculates what 'Observed %' would trigger an SRM at this sample size.
        Generalized to handle any expected target percentage.
        """
        if total_n <= 0 or not (0 < target_pct < 1):
            return {}

        # Critical value for Chi-square with 1 dof (for 2-variant tests)
        critical_value = stats.chi2.ppf(1 - alpha, df=1)

        # Generalized formula: chi2 = n * (p_obs - p_exp)^2 / (p_exp * (1 - p_exp))
        # Solving for margin = abs(p_obs - p_exp)
        margin = np.sqrt((critical_value * target_pct * (1 - target_pct)) / total_n)

        lower_bound = target_pct - margin
        upper_bound = target_pct + margin

        return {
            "target_pct": float(target_pct),
            "lower_bound_pct": float(lower_bound),
            "upper_bound_pct": float(upper_bound),
            "total_sample": total_n,
            "conclusion": (
                f"Sensitivity Threshold: For a total sample size of {total_n:,} and a target allocation of {target_pct:.1%}, "
                f"any observed traffic split outside the range of {lower_bound:.2%} to {upper_bound:.2%} "
                "will trigger an SRM alert."
            )
        }
