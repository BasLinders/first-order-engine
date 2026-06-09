import numpy as np
from scipy.stats import norm
from typing import List, Dict, Any, Optional

from foe.core.models import AlternativeHypothesis, AnalysisUnit, VarianceScaling


class PretestEngine:
    """
    Calculates Sample Size, Minimum Detectable Effect (MDE), power, and duration
    for A/B/n tests on binomial (conversion) or continuous (revenue, items) KPIs.
    Stateless and optimized for Cloud environments; returns JSON-safe dicts.

    All planning math is expressed in terms of a per-unit (mean, variance):
      * Binomial    -> mean = p,  variance = p * (1 - p)
      * Continuous  -> mean = mu, variance = sigma**2
    so the binomial and continuous paths share one set of formulas. The
    binomial public methods retain their original signatures and outputs.
    """

    # ------------------------------------------------------------------ #
    # Conclusion strings
    # ------------------------------------------------------------------ #

    @staticmethod
    def generate_sample_size_conclusion(
        n_per_variant: int,
        num_variants: int,
        mde_relative: float,
        power: float,
        alpha: float,
        unit_noun: str = "visitors",
    ) -> str:
        """Generates a definitive UI string for sample size requirements."""
        total_n = n_per_variant * num_variants
        conf = (1 - alpha) * 100
        pow_pct = power * 100
        mde_pct = mde_relative * 100

        return (
            f"Required Traffic: You need {n_per_variant:,} {unit_noun} per variant "
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
    def generate_power_conclusion(
        power: float, target_power: float, expected_lift: float, unit_noun: str
    ) -> str:
        """Generates a definitive UI string for a power calculation."""
        lift_pct = expected_lift * 100
        if power >= target_power:
            return (
                f"Adequately Powered: With this setup, the test has a {power * 100:.1f}% "
                f"chance of detecting a {lift_pct:.2f}% relative lift in {unit_noun} — "
                f"meeting your {target_power * 100:.0f}% power target."
            )
        return (
            f"Underpowered: The test has only a {power * 100:.1f}% chance of detecting a "
            f"{lift_pct:.2f}% relative lift in {unit_noun}, below your "
            f"{target_power * 100:.0f}% target. Increase runtime or expected effect."
        )

    # ------------------------------------------------------------------ #
    # Critical-value helpers (unchanged)
    # ------------------------------------------------------------------ #

    @staticmethod
    def holm_bonferroni_correction(
        num_variants: int, alpha: float, alternative: AlternativeHypothesis
    ) -> float:
        """Adjusts alpha for multiple comparisons based on the worst-case step-down."""
        num_comparisons = num_variants - 1
        adjusted_alphas = alpha / np.arange(num_comparisons, 0, -1)

        if alternative == AlternativeHypothesis.TWO_SIDED:
            z_scores = norm.ppf(1 - adjusted_alphas / 2)
        else:
            z_scores = norm.ppf(1 - adjusted_alphas)

        return float(np.max(z_scores))

    @classmethod
    def get_z_alpha(
        cls, num_variants: int, alpha: float, alternative: AlternativeHypothesis
    ) -> float:
        """Handles Z-score selection based on variant count and tails."""
        if num_variants > 2:
            return cls.holm_bonferroni_correction(num_variants, alpha, alternative)

        if alternative in [AlternativeHypothesis.GREATER, AlternativeHypothesis.LESS]:
            return float(norm.ppf(1 - alpha))

        # TWO_SIDED
        return float(norm.ppf(1 - alpha / 2))

    # ------------------------------------------------------------------ #
    # Moment-based core (shared by binomial and continuous)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _relative_mde(
        mean: float, variance: float, n_per_variant: float, z_alpha: float, z_power: float
    ) -> float:
        """Relative MDE (%) for a two-sample test of means with per-unit `variance`."""
        se = np.sqrt(2 * variance / n_per_variant)
        return float((z_alpha + z_power) * se / mean * 100)

    @staticmethod
    def _sample_size_per_variant(
        mde_absolute: float,
        var_null_sum: float,
        var_alt_sum: float,
        z_alpha: float,
        z_power: float,
    ) -> int:
        """
        Per-variant n for a two-sample test:
            n = (z_alpha*sqrt(V0) + z_power*sqrt(V1))**2 / delta**2
        where V0 / V1 are the summed two-group variances under H0 / H1.
        """
        term1 = z_alpha * np.sqrt(var_null_sum)
        term2 = z_power * np.sqrt(var_alt_sum)
        return int(np.ceil(((term1 + term2) ** 2) / (mde_absolute ** 2)))

    @staticmethod
    def _power_two_sample(
        delta: float,
        var_null_sum: float,
        var_alt_sum: float,
        n_per_variant: float,
        z_alpha: float,
        alternative: AlternativeHypothesis,
    ) -> float:
        """
        Exact two-sample z-test power. `z_alpha` is the already-tail-adjusted
        critical value (so TWO_SIDED passes the alpha/2 value). Handles unequal
        variances via separate null/alt standard errors, which is what lets the
        CV-constant scaling change the answer.
        """
        se_null = np.sqrt(var_null_sum / n_per_variant)
        se_alt = np.sqrt(var_alt_sum / n_per_variant)
        if alternative == AlternativeHypothesis.TWO_SIDED:
            return float(
                norm.cdf((delta - z_alpha * se_null) / se_alt)
                + norm.cdf((-delta - z_alpha * se_null) / se_alt)
            )
        return float(norm.cdf((delta - z_alpha * se_null) / se_alt))

    @staticmethod
    def compound_per_visitor_moments(
        txn_mean: float, txn_variance: float, conversion_rate: float
    ) -> tuple[float, float]:
        """
        Per-visitor mean and variance of a revenue-style metric, built from the
        per-transaction (buyer) distribution and the conversion rate p:

            mean   = p * m
            var    = p * v + m**2 * p * (1 - p)

        Lets a per-transaction sample (mean m, variance v) plus a conversion
        rate describe the per-visitor metric, with n then counted in visitors.
        """
        p = conversion_rate
        mean_r = p * txn_mean
        var_r = p * txn_variance + (txn_mean ** 2) * p * (1 - p)
        return float(mean_r), float(var_r)

    # ------------------------------------------------------------------ #
    # MDE tables
    # ------------------------------------------------------------------ #

    @classmethod
    def calculate_mde_table(
        cls,
        num_variants: int,
        baseline_visitors: int,
        baseline_conversions: int,
        risk_pct: float,
        trust_pct: float,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
    ) -> Dict[str, Any]:
        """Calculates a 6-week MDE projection for a binomial KPI (static traffic).

        `baseline_visitors` is total weekly traffic across all variants.
        """
        if baseline_visitors <= 0 or baseline_conversions < 0:
            return {"table": [], "conclusion": "Invalid baseline data."}

        baseline_rate = baseline_conversions / baseline_visitors
        if baseline_rate == 0:
            return {"table": [], "conclusion": "Baseline conversion rate cannot be zero."}

        alpha = 1 - (risk_pct / 100)
        power = trust_pct / 100
        z_power = norm.ppf(power)
        z_alpha = cls.get_z_alpha(num_variants, alpha, alternative)

        weekly_per_variant = int(np.ceil(baseline_visitors / num_variants))

        results = []
        for week in range(1, 7):
            n = weekly_per_variant * week
            mde_rel = cls._relative_mde(
                baseline_rate, baseline_rate * (1 - baseline_rate), n, z_alpha, z_power
            )
            results.append(
                {"Week": int(week), "Visitors_Per_Variant": int(n), "MDE": float(mde_rel)}
            )

        return {"table": results, "conclusion": cls.generate_mde_table_conclusion(results)}

    @classmethod
    def calculate_mde_table_continuous(
        cls,
        num_variants: int,
        weekly_units: int,
        mean: float,
        variance: float,
        risk_pct: float,
        trust_pct: float,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
        unit: AnalysisUnit = AnalysisUnit.PER_TRANSACTION,
    ) -> Dict[str, Any]:
        """Calculates a 6-week MDE projection for a continuous KPI (static volume).

        `weekly_units` is total weekly volume across all variants, counted in the
        analysis unit (visitors for per-visitor, transactions for per-transaction).
        `mean`/`variance` are the per-unit moments (for per-visitor, use
        ``compound_per_visitor_moments``).
        """
        if weekly_units <= 0 or mean <= 0 or variance <= 0:
            return {"table": [], "conclusion": "Invalid baseline data (need volume, mean, variance > 0)."}

        alpha = 1 - (risk_pct / 100)
        power = trust_pct / 100
        z_power = norm.ppf(power)
        z_alpha = cls.get_z_alpha(num_variants, alpha, alternative)

        weekly_per_variant = int(np.ceil(weekly_units / num_variants))

        results = []
        for week in range(1, 7):
            n = weekly_per_variant * week
            mde_rel = cls._relative_mde(mean, variance, n, z_alpha, z_power)
            results.append(
                {"Week": int(week), "Units_Per_Variant": int(n), "MDE": float(mde_rel)}
            )

        return {
            "table": results,
            "unit": unit.value,
            "conclusion": cls.generate_mde_table_conclusion(results),
        }

    # ------------------------------------------------------------------ #
    # Sample size
    # ------------------------------------------------------------------ #

    @classmethod
    def calculate_fixed_sample_size(
        cls,
        baseline_cr: float,
        mde_relative: float,
        num_variants: int,
        alpha: float = 0.05,
        power: float = 0.80,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
    ) -> Dict[str, Any]:
        """Required sample size per variant for a binomial KPI (pooled-variance form)."""
        if baseline_cr <= 0 or mde_relative <= 0:
            return {"n_per_variant": 0, "total_n": 0, "conclusion": "Invalid inputs."}

        p1 = baseline_cr
        p2 = baseline_cr * (1 + mde_relative)
        p_pooled = (p1 + p2) / 2

        z_alpha = cls.get_z_alpha(num_variants, alpha, alternative)
        z_beta = norm.ppf(power)

        n_per_variant = cls._sample_size_per_variant(
            mde_absolute=p2 - p1,
            var_null_sum=2 * p_pooled * (1 - p_pooled),
            var_alt_sum=p1 * (1 - p1) + p2 * (1 - p2),
            z_alpha=z_alpha,
            z_power=z_beta,
        )

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
                alpha=alpha,
            ),
        }

    @classmethod
    def calculate_fixed_sample_size_continuous(
        cls,
        mean: float,
        variance: float,
        mde_relative: float,
        num_variants: int,
        alpha: float = 0.05,
        power: float = 0.80,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
        variance_scaling: VarianceScaling = VarianceScaling.EQUAL,
        unit: AnalysisUnit = AnalysisUnit.PER_TRANSACTION,
    ) -> Dict[str, Any]:
        """Required sample size per variant for a continuous KPI.

        Under CV-constant scaling the treatment variance grows as
        ``variance * (1 + mde_relative)**2``; under equal variance both groups
        share `variance`. This method is the exact inverse of
        ``calculate_power_continuous`` for the same inputs.
        """
        if mean <= 0 or variance <= 0 or mde_relative <= 0:
            return {"n_per_variant": 0, "total_n": 0, "conclusion": "Invalid inputs."}

        mde_absolute = mean * mde_relative
        if variance_scaling == VarianceScaling.CV_CONSTANT:
            var_treat = variance * (1 + mde_relative) ** 2
        else:
            var_treat = variance

        z_alpha = cls.get_z_alpha(num_variants, alpha, alternative)
        z_beta = norm.ppf(power)

        n_per_variant = cls._sample_size_per_variant(
            mde_absolute=mde_absolute,
            var_null_sum=2 * variance,
            var_alt_sum=variance + var_treat,
            z_alpha=z_alpha,
            z_power=z_beta,
        )

        unit_noun = "visitors" if unit == AnalysisUnit.PER_VISITOR else "transactions"
        return {
            "n_per_variant": n_per_variant,
            "total_n": n_per_variant * num_variants,
            "mde_relative_used": mde_relative,
            "power_used": power,
            "alpha_used": alpha,
            "unit": unit.value,
            "variance_scaling": variance_scaling.value,
            "conclusion": cls.generate_sample_size_conclusion(
                n_per_variant=n_per_variant,
                num_variants=num_variants,
                mde_relative=mde_relative,
                power=power,
                alpha=alpha,
                unit_noun=unit_noun,
            ),
        }

    # ------------------------------------------------------------------ #
    # Power
    # ------------------------------------------------------------------ #

    @classmethod
    def calculate_power_binomial(
        cls,
        baseline_cr: float,
        expected_lift: float,
        n_per_variant: int,
        num_variants: int,
        alpha: float = 0.05,
        target_power: float = 0.80,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
    ) -> Dict[str, Any]:
        """Probability of detecting a relative `expected_lift` for a binomial KPI."""
        if baseline_cr <= 0 or expected_lift <= 0 or n_per_variant <= 0:
            return {"power": 0.0, "conclusion": "Invalid inputs."}

        p1 = baseline_cr
        p2 = min(baseline_cr * (1 + expected_lift), 1.0)
        z_alpha = cls.get_z_alpha(num_variants, alpha, alternative)

        power = cls._power_two_sample(
            delta=p2 - p1,
            var_null_sum=2 * p1 * (1 - p1),
            var_alt_sum=p1 * (1 - p1) + p2 * (1 - p2),
            n_per_variant=n_per_variant,
            z_alpha=z_alpha,
            alternative=alternative,
        )
        power = float(min(max(power, 0.0), 1.0))

        return {
            "power": power,
            "target_power": target_power,
            "is_powered": bool(power >= target_power),
            "conclusion": cls.generate_power_conclusion(
                power, target_power, expected_lift, "conversion rate"
            ),
        }

    @classmethod
    def calculate_power_continuous(
        cls,
        mean: float,
        variance: float,
        expected_lift: float,
        n_per_variant: int,
        num_variants: int,
        alpha: float = 0.05,
        target_power: float = 0.80,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
        variance_scaling: VarianceScaling = VarianceScaling.EQUAL,
        unit: AnalysisUnit = AnalysisUnit.PER_TRANSACTION,
    ) -> Dict[str, Any]:
        """Probability of detecting a relative `expected_lift` for a continuous KPI.

        Uses the exact two-sample z-test, so CV-constant scaling (unequal
        treatment variance) is handled correctly. Exact inverse of
        ``calculate_fixed_sample_size_continuous``.
        """
        if mean <= 0 or variance <= 0 or expected_lift <= 0 or n_per_variant <= 0:
            return {"power": 0.0, "conclusion": "Invalid inputs."}

        delta = mean * expected_lift
        if variance_scaling == VarianceScaling.CV_CONSTANT:
            var_treat = variance * (1 + expected_lift) ** 2
        else:
            var_treat = variance

        z_alpha = cls.get_z_alpha(num_variants, alpha, alternative)
        power = cls._power_two_sample(
            delta=delta,
            var_null_sum=2 * variance,
            var_alt_sum=variance + var_treat,
            n_per_variant=n_per_variant,
            z_alpha=z_alpha,
            alternative=alternative,
        )
        power = float(min(max(power, 0.0), 1.0))

        unit_noun = "revenue per visitor" if unit == AnalysisUnit.PER_VISITOR else "order value"
        return {
            "power": power,
            "target_power": target_power,
            "is_powered": bool(power >= target_power),
            "unit": unit.value,
            "variance_scaling": variance_scaling.value,
            "conclusion": cls.generate_power_conclusion(
                power, target_power, expected_lift, unit_noun
            ),
        }

    # ------------------------------------------------------------------ #
    # Seasonal MDE (consumes ForecastingEngine output)
    # ------------------------------------------------------------------ #

    @classmethod
    def calculate_mde_from_forecast(
        cls,
        forecast_records: List[Dict[str, Any]],
        num_variants: int,
        risk_pct: float,
        trust_pct: float,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
        kpi_cv: Optional[float] = None,
        unit: AnalysisUnit = AnalysisUnit.PER_TRANSACTION,
    ) -> Dict[str, Any]:
        """Seasonal 6-week MDE from forecast records (see ForecastingEngine).

        Each record must carry ``pred_count`` (forecasted unit count) and
        ``pred_value`` (forecasted sum-of-metric), sorted by date. The per-unit
        mean for each cumulative week is ``sum(pred_value) / sum(pred_count)``.

        Pass ``kpi_cv`` for a continuous KPI (variance = (cv * mean)**2); leave it
        as None for a binomial KPI (variance = mean * (1 - mean)).
        """
        if not forecast_records:
            return {"table": [], "conclusion": "No forecast data provided."}
        if kpi_cv is not None and kpi_cv <= 0:
            return {"table": [], "conclusion": "Coefficient of variation must be > 0."}

        records = sorted(forecast_records, key=lambda r: r["ds"])

        alpha = 1 - (risk_pct / 100)
        power = trust_pct / 100
        z_power = norm.ppf(power)
        z_alpha = cls.get_z_alpha(num_variants, alpha, alternative)

        results = []
        for week in range(1, 7):
            window = records[: week * 7]
            total_count = float(sum(r["pred_count"] for r in window))
            total_value = float(sum(r["pred_value"] for r in window))

            if total_count <= 0:
                results.append(
                    {"Week": int(week), "Units_Per_Variant": 0, "Mean": None, "MDE": None}
                )
                continue

            mean = total_value / total_count
            if kpi_cv is None:
                mean = float(np.clip(mean, 1e-4, 1 - 1e-4))
                variance = mean * (1 - mean)
            else:
                variance = (kpi_cv * mean) ** 2

            n_per_variant = total_count / num_variants
            if mean <= 0:
                results.append(
                    {"Week": int(week), "Units_Per_Variant": int(n_per_variant),
                     "Mean": float(mean), "MDE": None}
                )
                continue

            mde_rel = cls._relative_mde(mean, variance, n_per_variant, z_alpha, z_power)
            results.append(
                {
                    "Week": int(week),
                    "Units_Per_Variant": int(n_per_variant),
                    "Mean": float(mean),
                    "MDE": float(mde_rel),
                }
            )

        clean = [r for r in results if r["MDE"] is not None]
        conclusion = (
            cls.generate_mde_table_conclusion(clean)
            if len(clean) >= 6
            else "Seasonal MDE projection (some weeks unavailable; see table)."
        )
        return {"table": results, "unit": unit.value, "conclusion": conclusion}
