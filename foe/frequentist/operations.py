import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from typing import List, Dict, Any

from foe.core.models import AlternativeHypothesis, ExperimentInput, FrequentistResult
from foe.frequentist.confidence import compute_interval_difference


def apply_sidak(alpha: float, num_variants: int) -> float:
    """Calculates adjusted alpha for multiple comparisons (A/B/n)."""
    num_comparisons = num_variants - 1
    if num_comparisons <= 1:
        return alpha
    return 1 - (1 - alpha) ** (1 / num_comparisons)


class FrequentistEngine:
    """
    Axiom Frequentist Engine: Implements Variance Reduction (CUPED/Lin/Aggregate),
    Robust OLS Inference, and High-Performance Bootstrapping.
    Stateless design optimized for Cloud Functions.
    """

    @staticmethod
    def generate_conclusion_statement(
        variant_name: str, is_significant: bool, relative_lift: float
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

        if relative_lift > 0:
            return (
                f"Significant Positive Impact: '{variant_name}' is a clear winner "
                f"with an observed relative impact of {relative_lift:+.2%}. "
                "You can confidently roll this out."
            )
        return (
            f"Significant Negative Impact: '{variant_name}' is performing worse than control "
            f"with an observed relative impact of {relative_lift:+.2%}. "
            "You can confidently discard this variant."
        )

    @staticmethod
    def apply_cuped(
        df: pd.DataFrame, target_kpi: str, pre_period_kpi: str
    ) -> pd.DataFrame:
        """
        Standard CUPED adjustment.
        Formula: Y_cuped = Y - theta * (X_pre - mean(X_pre))

        Requires user-level data with a pre-experiment covariate column.
        The adjusted column is written back to the DataFrame as
        '{target_kpi}_cuped' for use in downstream analysis.
        """
        cov = df[[target_kpi, pre_period_kpi]].cov().iloc[0, 1]
        var_pre = df[pre_period_kpi].var()

        theta = cov / var_pre if var_pre != 0 else 0
        mean_pre = df[pre_period_kpi].mean()

        df[f"{target_kpi}_cuped"] = df[target_kpi] - theta * (
            df[pre_period_kpi] - mean_pre
        )
        return df

    @staticmethod
    def run_lin_adjustment(
        df: pd.DataFrame,
        target_kpi: str,
        pre_period_kpi: str,
        variant_col: str = "variant",
        alpha: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """
        Lin's Adjustment (2013). More robust than CUPED for heterogeneous effects.
        Regression: Y ~ Treatment * (Covariate - mean(Covariate))

        Requires user-level data. Uses HC3 robust standard errors.

        Args:
            df:              User-level DataFrame containing target_kpi,
                             pre_period_kpi, and variant_col columns.
            target_kpi:      Name of the outcome column (e.g. 'converted').
            pre_period_kpi:  Name of the pre-experiment covariate column.
            variant_col:     Name of the variant assignment column.
            alpha:           Significance threshold. Should match the
                             confidence_level used in ExperimentInput
                             (i.e. alpha = 1 - confidence_level).
        """
        df = df.copy()
        df["cov_centered"] = df[pre_period_kpi] - df[pre_period_kpi].mean()

        variants = sorted(df[variant_col].unique())
        baseline = variants[0]  # Assumes alphabetical or 'Control' is first

        formula = f"{target_kpi} ~ C({variant_col}, Treatment(reference='{baseline}')) * cov_centered"
        model = smf.ols(formula, data=df).fit(cov_type="HC3")

        results = []
        for challenger in variants[1:]:
            term = (
                f"C({variant_col}, Treatment(reference='{baseline}'))[T.{challenger}]"
            )

            p_val = model.pvalues[term]
            ate = model.params[term]  # Average Treatment Effect
            control_mean = model.params["Intercept"]

            results.append(
                {
                    "variant": challenger,
                    "p_value": float(p_val),
                    "is_significant": bool(p_val < alpha),
                    "absolute_lift": float(ate),
                    "relative_lift": (
                        float(ate / control_mean) if control_mean != 0 else 0
                    ),
                    "ci": model.conf_int().loc[term].tolist(),
                    "std_err": float(model.bse[term]),
                }
            )

        return results

    @staticmethod
    def calculate_aggregate_variance_factor(
        df: pd.DataFrame,
        visitors_col: str = "visitors",
        conversions_col: str = "conversions",
        min_periods: int = 14,
    ) -> Dict[str, Any]:
        """
        Estimates a variance scaling factor (φ) from aggregate historical
        daily data, without requiring user-level observations.

        Compares the observed day-to-day variance of the conversion rate
        against what pure binomial sampling would predict for the same
        traffic volumes. The ratio φ = observed / expected is used to
        scale standard errors in run_synthesis via ExperimentInput.reduction_factor.

        φ < 1  ->  rate is more stable than binomial theory predicts; SE shrinks.
        φ ≈ 1  ->  rate behaves as binomial; no meaningful adjustment.
        φ > 1  ->  overdispersion detected (campaign bursts, seasonality); SE inflates.

        Both variances are visitor-weighted to prevent low-traffic days from
        distorting the estimate.

        Args:
            df:              Daily aggregate DataFrame. Must contain visitors and
                             conversions columns and at least min_periods rows.
            visitors_col:    Column name for daily visitor counts.
            conversions_col: Column name for daily conversion counts.
            min_periods:     Minimum number of daily rows required for a reliable
                             estimate. Raises ValueError if not met.

        Returns:
            Dict with keys:
                reduction_factor  float  Clipped φ; pass directly to
                                         ExperimentInput.reduction_factor.
                phi               float  Raw dispersion ratio (for logging/display).
                regime            str    'stable' | 'neutral' | 'noisy' | 'high_noise'
                n_periods         int    Number of rows used in the calculation.
        """
        n_periods = len(df)
        if n_periods < min_periods:
            raise ValueError(
                f"At least {min_periods} daily rows are required for a reliable "
                f"dispersion estimate; got {n_periods}."
            )

        visitors = df[visitors_col].astype(float)
        conversions = df[conversions_col].astype(float)

        if (visitors <= 0).any():
            raise ValueError(f"'{visitors_col}' contains zero or negative values.")
        if (conversions < 0).any():
            raise ValueError(f"'{conversions_col}' contains negative values.")
        if (conversions > visitors).any():
            raise ValueError(
                f"'{conversions_col}' exceeds '{visitors_col}' on one or more rows — "
                "check your column mapping."
            )

        rates = conversions / visitors
        weights = visitors / visitors.sum()
        weighted_mean = (rates * weights).sum()

        observed_var = (weights * (rates - weighted_mean) ** 2).sum()
        expected_binomial_var = (weights * rates * (1 - rates) / visitors).sum()

        if expected_binomial_var == 0:
            return {
                "reduction_factor": 1.0,
                "phi": 1.0,
                "regime": "neutral",
                "n_periods": n_periods,
            }

        phi = float(observed_var / expected_binomial_var)
        reduction_factor = float(np.clip(phi, 0.10, None))

        if phi < 0.80:
            regime = "stable"
        elif phi < 1.05:
            regime = "neutral"
        elif phi < 1.50:
            regime = "noisy"
        else:
            regime = "high_noise"

        return {
            "reduction_factor": reduction_factor,
            "phi": phi,
            "regime": regime,
            "n_periods": n_periods,
        }

    @staticmethod
    def run_ztest(
        diff: float, se_diff: float, alternative: AlternativeHypothesis
    ) -> float:
        """Standard Z-test logic for varying tail configurations."""
        if se_diff == 0:
            return 1.0

        z_stat = diff / se_diff

        if alternative == AlternativeHypothesis.GREATER:
            return 1 - norm.cdf(z_stat)
        elif alternative == AlternativeHypothesis.LESS:
            return norm.cdf(z_stat)
        else:  # TWO_SIDED
            return 2 * (1 - norm.cdf(abs(z_stat)))

    @staticmethod
    def calculate_analytical_power(
        diff: float, se_diff: float, alpha: float, alternative: AlternativeHypothesis
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
        ctrl_conv: int,
        ctrl_n: int,
        chal_conv: int,
        chal_n: int,
        alpha: float = 0.05,
        n_bootstraps: int = 10000,
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

        # Simulate conversion COUNTS directly using the Binomial distribution.
        # Size is just (n_bootstraps,) instead of (n_bootstraps, N).
        sim_ctrl_convs = np.random.binomial(n=ctrl_n, p=ctrl_p, size=n_bootstraps)
        sim_chal_convs = np.random.binomial(n=chal_n, p=chal_p, size=n_bootstraps)

        means_ctrl = sim_ctrl_convs / ctrl_n
        means_chal = sim_chal_convs / chal_n

        p_pooled = (sim_ctrl_convs + sim_chal_convs) / (ctrl_n + chal_n)

        # Small epsilon guards against division by zero when a simulated
        # pooled proportion lands exactly at 0 or 1.
        epsilon = 1e-9
        se_pooled = np.sqrt(
            p_pooled * (1 - p_pooled) * (1 / ctrl_n + 1 / chal_n) + epsilon
        )

        z_stats = (means_chal - means_ctrl) / se_pooled
        p_values = 2 * (1 - norm.cdf(np.abs(z_stats)))

        return float(np.mean(p_values < alpha))

    def run_synthesis(self, data: ExperimentInput) -> List[FrequentistResult]:
        """
        High-level entry point: takes a validated ExperimentInput and returns
        a FrequentistResult for each challenger vs control.

        If a reduction_factor (φ) has been set on ExperimentInput — either from
        calculate_aggregate_variance_factor, apply_cuped, or any other source —
        it is applied to the variance before taking the square root, which is
        the mathematically correct way to scale a standard error:

            SE = sqrt(phi * p*(1-p) / n)

        This means SE is multiplied by sqrt(phi), not phi directly.
        """
        labels = data.labels or [f"Variant {i}" for i in range(len(data.visitors))]
        p_ctrl = data.conversions[0] / data.visitors[0]
        n_ctrl = data.visitors[0]
        alpha = 1.0 - data.confidence_level

        results = []
        for i in range(1, len(data.visitors)):
            n_chal = data.visitors[i]
            p_chal = data.conversions[i] / n_chal
            diff = p_chal - p_ctrl
            uplift = diff / p_ctrl if p_ctrl != 0 else 0.0

            # φ scales variance, not SE directly. Applying it inside the sqrt
            # ensures a 1% change in φ produces a proportional ~0.5% change in SE,
            # rather than a full 1% change which would over-correct.
            se_diff = (
                p_ctrl * (1 - p_ctrl) * data.reduction_factor / n_ctrl
                + p_chal * (1 - p_chal) * data.reduction_factor / n_chal
            ) ** 0.5

            p_value = self.run_ztest(diff, se_diff, data.alternative)
            is_sig = bool(p_value < alpha)
            ci = compute_interval_difference(diff, se_diff, alpha, data.alternative)
            conclusion = self.generate_conclusion_statement(labels[i], is_sig, uplift)

            results.append(
                FrequentistResult(
                    variant_label=labels[i],
                    control_label=labels[0],
                    conversion_rate=p_chal,
                    standard_error=se_diff,
                    p_value=p_value,
                    uplift=uplift,
                    is_significant=is_sig,
                    ci_diff=ci,
                    conclusion=conclusion,
                )
            )

        return results
