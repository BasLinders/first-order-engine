import math
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from typing import List, Dict, Any, Optional, Tuple

from foe.core.models import AlternativeHypothesis, ExperimentInput, FrequentistResult
from foe.frequentist.confidence import compute_interval_difference


def apply_sidak(alpha: float, num_variants: int) -> float:
    """Calculates adjusted alpha for multiple comparisons (A/B/n)."""
    num_comparisons = num_variants - 1
    if num_comparisons <= 1:
        return alpha
    return 1 - (1 - alpha) ** (1 / num_comparisons)


# Prior P(H1) presets, shared with the UI tool. Skeptical = most experiments
# don't move the needle; optimistic = strong belief the variant has an effect.
_PRIOR_PRESETS: Dict[str, float] = {
    "skeptical": 0.10,
    "neutral": 0.50,
    "optimistic": 0.90,
}


class FrequentistEngine:
    """
    Axiom Frequentist Engine: Variance Reduction (CUPED/Lin/Aggregate),
    Robust OLS Inference, High-Performance Bootstrapping, and Bayesian
    decision-risk (false-positive / false-negative) reporting.
    Stateless design optimized for Cloud Functions.
    """

    # ------------------------------------------------------------------ #
    #  Conclusions
    # ------------------------------------------------------------------ #

    @staticmethod
    def generate_conclusion_statement(
        variant_name: str,
        is_significant: bool,
        relative_lift: float,
        false_positive_risk: Optional[float] = None,
    ) -> str:
        """
        Generates a UI-agnostic summary of the results in English.

        When `false_positive_risk` is omitted (None) the original wording is
        preserved verbatim. When supplied, the wording is calibrated to that
        risk instead of asserting unconditional confidence.
        """
        if not is_significant:
            return (
                f"Inconclusive / Flat: '{variant_name}' shows no statistically "
                "significant difference from the control. We cannot confidently "
                "conclude that this variant had a meaningful impact."
            )

        # --- Backward-compatible path: no risk supplied -> original wording ---
        if false_positive_risk is None:
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

        # --- Risk-aware path -------------------------------------------------
        provisional = false_positive_risk > 0.20
        if relative_lift > 0:
            base = (
                f"Significant Positive Impact: '{variant_name}' shows an observed "
                f"relative impact of {relative_lift:+.2%}."
            )
            tail = (
                f" The false-positive risk is {false_positive_risk:.0%} given your prior — "
                "treat this as provisional and consider replication before a full rollout."
                if provisional
                else f" The false-positive risk is low ({false_positive_risk:.0%}); rolling this out is well supported."
            )
            return base + tail

        base = (
            f"Significant Negative Impact: '{variant_name}' is performing worse than control "
            f"with an observed relative impact of {relative_lift:+.2%}."
        )
        tail = (
            f" The false-positive risk is {false_positive_risk:.0%} given your prior — "
            "confirm before discarding."
            if provisional
            else f" The false-positive risk is low ({false_positive_risk:.0%}); discarding this variant is well supported."
        )
        return base + tail

    # ------------------------------------------------------------------ #
    #  Variance reduction: user-level (CUPED / Lin)
    # ------------------------------------------------------------------ #

    @staticmethod
    def apply_cuped(
        df: pd.DataFrame, target_kpi: str, pre_period_kpi: str
    ) -> pd.DataFrame:
        """
        Standard CUPED adjustment.
        Formula: Y_cuped = Y - theta * (X_pre - mean(X_pre))

        Requires user-level data with a pre-experiment covariate column.
        The adjusted column is written back as '{target_kpi}_cuped'.
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
        Regression: Y ~ Treatment * (Covariate - mean(Covariate)). HC3 robust SEs.

        Args:
            df:              User-level DataFrame.
            target_kpi:      Outcome column (e.g. 'converted').
            pre_period_kpi:  Pre-experiment covariate column.
            variant_col:     Variant assignment column.
            alpha:           Significance threshold (alpha = 1 - confidence_level).
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
            ate = model.params[term]
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

    # ------------------------------------------------------------------ #
    #  Variance reduction: aggregate overdispersion (φ)
    # ------------------------------------------------------------------ #

    @staticmethod
    def calculate_aggregate_variance_factor(
        df: pd.DataFrame,
        visitors_col: str = "visitors",
        conversions_col: str = "conversions",
        min_periods: int = 14,
        date_col: Optional[str] = None,
        phi_upper: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Estimates a variance scaling factor (φ) from aggregate historical daily
        data, without user-level observations.

        Compares the observed day-to-day variance of the conversion rate against
        what pure binomial sampling would predict for the same traffic volumes.
        The ratio φ = observed / expected scales standard errors in run_synthesis
        via ExperimentInput.reduction_factor (φ scales variance, so SE ∝ √φ).

            φ < 1  -> rate more stable than binomial theory predicts; SE shrinks.
            φ ≈ 1  -> rate behaves as binomial; no meaningful adjustment.
            φ > 1  -> overdispersion (campaign bursts, seasonality); SE inflates.

        Both variances are visitor-weighted so low-traffic days don't distort the
        estimate. Two refinements over a naive ratio (parity with the corrected
        UI estimator):

        1. Degrees-of-freedom correction. The centring baseline is estimated from
           the same rows whose residuals we measure, biasing the observed variance
           DOWNWARD (the anti-conservative direction). We divide by the residual
           dof using Kish's effective sample size n_eff = 1 / Σ wᵢ², so φ → 1 under
           a true binomial null even on short windows. (Reduces to n/(n-k) for
           equal weights.)

        2. Optional day-of-week control. If `date_col` is supplied, the centring
           baseline is the per-weekday rate rather than a single global mean, so
           recurring weekly seasonality is not mistaken for overdispersion.

        Returns a dict: reduction_factor (clipped φ), phi (raw), regime
        ('stable'|'neutral'|'noisy'|'high_noise'), n_periods, n_effective,
        dow_controlled, clipped.
        """
        n_periods = len(df)
        if n_periods < min_periods:
            raise ValueError(
                f"At least {min_periods} daily rows are required for a reliable "
                f"dispersion estimate; got {n_periods}."
            )

        visitors = df[visitors_col].astype(float).to_numpy()
        conversions = df[conversions_col].astype(float).to_numpy()

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

        # --- Centring baseline + parameter count -------------------------
        dow_controlled = False
        if date_col is not None:
            dow = pd.to_datetime(df[date_col]).dt.dayofweek.to_numpy()
            grp = pd.DataFrame({"dow": dow, "v": visitors, "c": conversions})
            agg = grp.groupby("dow").agg(v=("v", "sum"), c=("c", "sum"))
            agg["base"] = agg["c"] / agg["v"]
            expected = agg["base"].reindex(dow).to_numpy()
            n_params = int(np.unique(dow).size)
            dow_controlled = True
        else:
            expected = (rates * weights).sum()  # global weighted mean
            n_params = 1

        observed_var = float((weights * (rates - expected) ** 2).sum())
        expected_binomial_var = float((weights * rates * (1 - rates) / visitors).sum())

        # --- Degrees-of-freedom correction (Kish effective sample size) --
        n_effective = 1.0 / float(np.square(weights).sum())
        dof = n_effective - n_params
        if dof > 0:
            observed_var *= n_effective / dof

        if expected_binomial_var == 0:
            return {
                "reduction_factor": 1.0,
                "phi": 1.0,
                "regime": "neutral",
                "n_periods": n_periods,
                "n_effective": n_effective,
                "dow_controlled": dow_controlled,
                "clipped": False,
            }

        phi = observed_var / expected_binomial_var
        reduction_factor = float(np.clip(phi, 0.10, phi_upper))
        clipped = bool(phi > phi_upper or phi < 0.10)

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
            "phi": float(phi),
            "regime": regime,
            "n_periods": n_periods,
            "n_effective": n_effective,
            "dow_controlled": dow_controlled,
            "clipped": clipped,
        }

    # ------------------------------------------------------------------ #
    #  Inference primitives
    # ------------------------------------------------------------------ #

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
        """Closed-form power at the observed effect size."""
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
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
        use_pooled_se: bool = True,
    ) -> float:
        """
        Observed power via high-performance vectorized bootstrapping.
        Simulates conversion COUNTS via the Binomial distribution to avoid OOM
        when N is large.

        Note: this routine does not apply a reduction_factor; under overdispersion
        prefer `calculate_analytical_power`, which shares the φ-scaled unpooled SE
        used by `run_synthesis`. `use_pooled_se=False` aligns the bootstrap test
        statistic with that unpooled SE (default True preserves prior behavior).
        """
        if ctrl_n == 0 or chal_n == 0:
            return 0.0

        ctrl_p = ctrl_conv / ctrl_n
        chal_p = chal_conv / chal_n

        sim_ctrl_convs = np.random.binomial(n=ctrl_n, p=ctrl_p, size=n_bootstraps)
        sim_chal_convs = np.random.binomial(n=chal_n, p=chal_p, size=n_bootstraps)

        means_ctrl = sim_ctrl_convs / ctrl_n
        means_chal = sim_chal_convs / chal_n

        epsilon = 1e-9
        if use_pooled_se:
            p_pooled = (sim_ctrl_convs + sim_chal_convs) / (ctrl_n + chal_n)
            se = np.sqrt(
                p_pooled * (1 - p_pooled) * (1 / ctrl_n + 1 / chal_n) + epsilon
            )
        else:
            se = np.sqrt(
                means_ctrl * (1 - means_ctrl) / ctrl_n
                + means_chal * (1 - means_chal) / chal_n
                + epsilon
            )

        z_stats = (means_chal - means_ctrl) / se

        if alternative == AlternativeHypothesis.GREATER:
            p_values = 1 - norm.cdf(z_stats)
        elif alternative == AlternativeHypothesis.LESS:
            p_values = norm.cdf(z_stats)
        else:
            p_values = 2 * (1 - norm.cdf(np.abs(z_stats)))

        return float(np.mean(p_values < alpha))

    # ------------------------------------------------------------------ #
    #  Decision risk: false-positive / false-negative
    #
    #  Composable helpers. Deliberately NOT wired into run_synthesis: the
    #  FrequentistResult contract (see test_frequentist.py) carries inference
    #  fields only. Call these from a reporting layer, or extend the model with
    #  observed_power / decision_risk_* and populate them in run_synthesis if you
    #  want them on every result.
    # ------------------------------------------------------------------ #

    @staticmethod
    def resolve_prior_probability(
        sensitivity_mode: str = "neutral",
        custom_prior: Optional[float] = None,
    ) -> float:
        """
        Resolves a prior P(H1) — probability a real effect exists, before data —
        from a named preset or an explicit value.

        Presets: skeptical=0.10, neutral=0.50, optimistic=0.90. With
        sensitivity_mode='custom', `custom_prior` (in [0, 1]) is used directly.
        """
        if sensitivity_mode == "custom":
            if custom_prior is None:
                raise ValueError(
                    "custom_prior must be provided when sensitivity_mode is 'custom'."
                )
            if not 0.0 <= custom_prior <= 1.0:
                raise ValueError(
                    f"custom_prior must be between 0 and 1, got {custom_prior}."
                )
            return float(custom_prior)
        if sensitivity_mode not in _PRIOR_PRESETS:
            raise ValueError(
                f"Unknown sensitivity_mode '{sensitivity_mode}'. "
                f"Expected one of {list(_PRIOR_PRESETS) + ['custom']}."
            )
        return _PRIOR_PRESETS[sensitivity_mode]

    @staticmethod
    def calculate_false_positive_risk(
        alpha: float, power: float, prior: float
    ) -> float:
        """
        P(H0 | significant) — probability a significant result is a false
        positive, by Bayes' rule:

            FPR = α·P(H0) / [ α·P(H0) + power·P(H1) ]

        with P(H1) = prior, P(H0) = 1 - prior.
        """
        p1, p0 = prior, 1.0 - prior
        denominator = (alpha * p0) + (power * p1)
        return (alpha * p0) / denominator if denominator else 0.0

    @staticmethod
    def calculate_false_negative_discovery_rate(
        alpha: float, power: float, prior: float
    ) -> float:
        """
        P(H1 | not significant) — probability a non-significant result missed a
        real effect, by Bayes' rule:

            FNDR = β·P(H1) / [ β·P(H1) + (1-α)·P(H0) ]

        with β = 1 - power. High FNDR on a flat result means underpowered, not
        necessarily inert.
        """
        p1, p0 = prior, 1.0 - prior
        beta = 1.0 - power
        denominator = (beta * p1) + ((1.0 - alpha) * p0)
        return (beta * p1) / denominator if denominator else 0.0

    @classmethod
    def assess_decision_risk(
        cls,
        is_significant: bool,
        alpha: float,
        power: float,
        prior: float,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
    ) -> Dict[str, Any]:
        """
        Selects and computes the relevant decision-risk metric:
          - significant result      -> false-positive risk (FPR)
          - non-significant result  -> false-negative discovery rate (FNDR)

        The numeric value is identical across tails; only the label changes to
        reflect direction.
        """
        if is_significant:
            value = cls.calculate_false_positive_risk(alpha, power, prior)
            if alternative == AlternativeHypothesis.LESS:
                label = "False Positive Risk (apparent harm may be spurious)"
            elif alternative == AlternativeHypothesis.GREATER:
                label = "False Positive Risk (apparent improvement may be spurious)"
            else:
                label = "False Positive Risk (detected difference may be spurious)"
            metric = "false_positive_risk"
        else:
            value = cls.calculate_false_negative_discovery_rate(alpha, power, prior)
            if alternative == AlternativeHypothesis.LESS:
                label = "False Negative Discovery Rate (real harm may have been missed)"
            elif alternative == AlternativeHypothesis.GREATER:
                label = "False Negative Discovery Rate (real improvement may have been missed)"
            else:
                label = "False Negative Discovery Rate (a real difference may have been missed)"
            metric = "false_negative_discovery_rate"

        return {
            "metric": metric,
            "value": float(value),
            "label": label,
            "power": float(power),
            "prior": float(prior),
            "elevated": bool(value > 0.20),
        }

    # ------------------------------------------------------------------ #
    #  Monetary impact (deterministic)
    #
    #  Unlike the Bayesian engine's run_monetary_projection, this does not
    #  simulate a posterior: a frequentist result has no distribution over
    #  the true rate, only a point estimate and a confidence interval. These
    #  helpers translate that point estimate and CI directly into a revenue
    #  range via simple arithmetic, so the claim never exceeds what the CI
    #  actually licenses ("with X% confidence, between low and high").
    # ------------------------------------------------------------------ #

    @staticmethod
    def estimate_monetary_impact(
        diff: float,
        ci_diff: Tuple[float, float],
        daily_visitors: float,
        aov: float,
        projection_period: int = 183,
    ) -> Dict[str, Any]:
        """
        Deterministically scales an absolute conversion-rate difference (and
        its confidence interval) into a monetary range over projection_period
        days, given expected daily traffic and Average Order Value.

        revenue(d) = d * daily_visitors * aov * projection_period

        A one-sided test's infinite CI bound (see FrequentistResult.ci_diff)
        propagates as +/-inf rather than being silently clipped; render it as
        "unbounded" rather than a literal number (see generate_monetary_conclusion).
        """
        def revenue(rate_diff: float) -> float:
            return rate_diff * daily_visitors * aov * projection_period

        return {
            "point_estimate": revenue(diff),
            "ci_low": revenue(ci_diff[0]),
            "ci_high": revenue(ci_diff[1]),
            "daily_visitors": daily_visitors,
            "aov": aov,
            "projection_period": projection_period,
        }

    @staticmethod
    def estimate_monetary_impact_per_variant(
        p_ctrl: float,
        se_ctrl: float,
        aov_ctrl: float,
        p_chal: float,
        se_chal: float,
        aov_chal: float,
        daily_visitors: float,
        alpha: float = 0.05,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
        projection_period: int = 183,
        se_aov_ctrl: float = 0.0,
        se_aov_chal: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Like estimate_monetary_impact, but allows Control and Challenger to
        carry different Average Order Values -- e.g. a pricing or
        merchandising test where AOV itself may differ by variant, not just
        the conversion rate.

        estimate_monetary_impact scales a single rate-difference CI by one
        shared AOV; that is only valid when AOV is identical across variants,
        since revenue(d) = d * aov is not equal to
        p_chal * aov_chal - p_ctrl * aov_ctrl when aov_chal != aov_ctrl. This
        method instead builds the value-weighted difference directly:

            X = p_chal * aov_chal - p_ctrl * aov_ctrl

        By default (se_aov_ctrl = se_aov_chal = 0.0) AOV is treated as a fixed
        known constant, and only conversion-rate uncertainty is propagated:

            SE(X) = sqrt(aov_chal^2 * se_chal^2 + aov_ctrl^2 * se_ctrl^2)

        This understates uncertainty whenever AOV itself is an ESTIMATE (e.g.
        the mean of a finite sample of order values) rather than a truly known
        constant: a "guaranteed" AOV gap can make X look confidently positive
        even when the underlying conversion-rate difference alone is nowhere
        near significant. Passing se_aov_ctrl/se_aov_chal (e.g.
        sample_std_of_order_value / sqrt(n_orders)) propagates that estimation
        uncertainty too, via the standard delta-method variance of a product
        of two independent random variables (dropping the second-order
        Var(p)*Var(aov) cross term):

            Var(p*aov) ~= aov^2 * Var(p) + p^2 * Var(aov)
            SE(X) = sqrt(
                aov_chal^2*se_chal^2 + p_chal^2*se_aov_chal^2
                + aov_ctrl^2*se_ctrl^2 + p_ctrl^2*se_aov_ctrl^2
            )

        assuming p_ctrl and p_chal are independent, and AOV and conversion
        status are independent within a variant. The CI for X reuses
        compute_interval_difference, so a one-sided alternative still
        produces a +/-inf bound rather than being silently clipped.

        Reduces to estimate_monetary_impact's result when aov_ctrl == aov_chal
        and se_aov_ctrl == se_aov_chal == 0.0 (X = aov * diff, SE(X) = aov * se_diff).
        """
        diff_value = (p_chal * aov_chal) - (p_ctrl * aov_ctrl)
        var_chal = (aov_chal ** 2) * (se_chal ** 2) + (p_chal ** 2) * (se_aov_chal ** 2)
        var_ctrl = (aov_ctrl ** 2) * (se_ctrl ** 2) + (p_ctrl ** 2) * (se_aov_ctrl ** 2)
        se_diff_value = math.sqrt(var_chal + var_ctrl)
        ci_diff_value = compute_interval_difference(
            diff_value, se_diff_value, alpha=alpha, alternative=alternative
        )

        def revenue(x: float) -> float:
            return x * daily_visitors * projection_period

        return {
            "point_estimate": revenue(diff_value),
            "ci_low": revenue(ci_diff_value[0]),
            "ci_high": revenue(ci_diff_value[1]),
            "daily_visitors": daily_visitors,
            "aov_ctrl": aov_ctrl,
            "aov_chal": aov_chal,
            "projection_period": projection_period,
        }

    @staticmethod
    def generate_monetary_conclusion(
        variant_name: str,
        monetary_result: Dict[str, Any],
        is_significant: bool,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
    ) -> str:
        """UI-agnostic narrative summary of estimate_monetary_impact's output."""
    
        def fmt(x: float) -> str:
            return "unbounded" if math.isinf(x) else f"{x:,.0f}"
    
        point = monetary_result["point_estimate"]
        ci_low = monetary_result["ci_low"]
        ci_high = monetary_result["ci_high"]
        period = monetary_result["projection_period"]
        range_str = f"{fmt(ci_low)} to {fmt(ci_high)}"
    
        if not is_significant:
            return (
                f"'{variant_name}' is not statistically significant, so this monetary "
                f"range ({range_str} over {period} days) reflects the full uncertainty "
                "in the observed effect -- including no impact at all, or a loss. "
                "Treat it as illustrative, not a business case to act on."
            )
    
        # Significant one-sided cases: only one side of the interval is meaningful.
        # Say so explicitly instead of pairing a real number with "unbounded" as
        # if they were a comparable pair.
        if alternative == AlternativeHypothesis.GREATER:
            return (
                f"'{variant_name}' is projected to generate a monetary uplift of "
                f"at least {fmt(ci_low)} over the next {period} days. This is a "
                "one-sided estimate, so no statistical upper bound is calculated -- "
                f"the point estimate ({fmt(point)}) is your best single guess for "
                "the actual size, not the ceiling."
            )
    
        if alternative == AlternativeHypothesis.LESS:
            return (
                f"'{variant_name}' is projected to cost at most {fmt(abs(ci_high))} "
                f"over the next {period} days relative to control. This is a "
                "one-sided estimate, so no statistical lower bound is calculated -- "
                f"the point estimate ({fmt(point)}) is your best single guess for "
                "the actual size, not the floor."
            )
    
        # TWO_SIDED, significant: both bounds are real and both are meaningful.
        if point >= 0:
            return (
                f"'{variant_name}' is projected to generate a monetary uplift of "
                f"{fmt(point)} over the next {period} days, with {range_str} as the "
                "confidence range around that estimate."
            )
        return (
            f"'{variant_name}' is projected to cost {fmt(abs(point))} over the next "
            f"{period} days relative to control, with {range_str} as the confidence "
            "range around that estimate."
        )

    # ------------------------------------------------------------------ #
    #  Orchestration
    # ------------------------------------------------------------------ #

    def run_synthesis(self, data: ExperimentInput) -> List[FrequentistResult]:
        """
        High-level entry point: takes a validated ExperimentInput and returns a
        FrequentistResult per challenger vs control.

        If a reduction_factor (φ) is set on ExperimentInput — from
        calculate_aggregate_variance_factor, apply_cuped, or elsewhere — it scales
        the variance before the square root:

            SE = sqrt(phi * p*(1-p) / n)

        so SE is multiplied by sqrt(phi), not phi directly.
        """
        labels = data.labels or [f"Variant {i}" for i in range(len(data.visitors))]
        p_ctrl = data.conversions[0] / data.visitors[0]
        n_ctrl = data.visitors[0]
        alpha = apply_sidak(1.0 - data.confidence_level, len(data.visitors))

        results = []
        for i in range(1, len(data.visitors)):
            n_chal = data.visitors[i]
            p_chal = data.conversions[i] / n_chal
            diff = p_chal - p_ctrl
            uplift = diff / p_ctrl if p_ctrl != 0 else 0.0

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
