import math
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from typing import List, Dict, Any, Optional, Tuple

from foe.core.models import AlternativeHypothesis, ExperimentInput, FrequentistResult
from foe.frequentist.confidence import (
    compute_interval_difference,
    compute_newcombe_interval,
)


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
        relative_lift: Optional[float],
        false_positive_risk: Optional[float] = None,
        absolute_diff: Optional[float] = None,
    ) -> str:
        """
        Generates a UI-agnostic summary of the results in English.

        When `false_positive_risk` is omitted (None) the original wording is
        preserved verbatim. When supplied, the wording is calibrated to that
        risk instead of asserting unconditional confidence.

        Direction is taken from `absolute_diff` when supplied, otherwise from
        `relative_lift`. `relative_lift` may be None (control rate of 0, where
        relative lift is undefined); `absolute_diff` is then required to tell
        a winner from a loser.
        """
        if not is_significant:
            return (
                f"Inconclusive / Flat: '{variant_name}' shows no statistically "
                "significant difference from the control. We cannot confidently "
                "conclude that this variant had a meaningful impact."
            )

        direction = absolute_diff if absolute_diff is not None else relative_lift
        if direction is None:
            raise ValueError(
                "absolute_diff is required when relative_lift is undefined (None)."
            )
        is_positive = direction > 0
        impact = (
            f"an observed relative impact of {relative_lift:+.2%}"
            if relative_lift is not None
            else "an undefined relative impact (the control had no conversions)"
        )

        # --- Backward-compatible path: no risk supplied -> original wording ---
        if false_positive_risk is None:
            if is_positive:
                return (
                    f"Significant Positive Impact: '{variant_name}' is a clear winner "
                    f"with {impact}. "
                    "You can confidently roll this out."
                )
            return (
                f"Significant Negative Impact: '{variant_name}' is performing worse than control "
                f"with {impact}. "
                "You can confidently discard this variant."
            )

        # --- Risk-aware path -------------------------------------------------
        provisional = false_positive_risk > 0.20
        if is_positive:
            base = (
                f"Significant Positive Impact: '{variant_name}' shows {impact}."
            )
            tail = (
                f" The false-positive risk is {false_positive_risk:.0%} given your prior and planned effect size — "
                "treat this as provisional and consider replication before a full rollout."
                if provisional
                else f" The false-positive risk is low ({false_positive_risk:.0%}); rolling this out is well supported."
            )
            return base + tail

        base = (
            f"Significant Negative Impact: '{variant_name}' is performing worse than control "
            f"with {impact}."
        )
        tail = (
            f" The false-positive risk is {false_positive_risk:.0%} given your prior and planned effect size — "
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
        control_label: Optional[str] = None,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
    ) -> List[Dict[str, Any]]:
        """
        Lin's Adjustment (2013). More robust than CUPED for heterogeneous effects.
        Regression: Y ~ Treatment * (Covariate - mean(Covariate)). HC3 robust SEs.

        Inference follows run_synthesis: z-based p-value for `alternative`,
        Šidák-adjusted alpha for 3+ variants, and a matching (1 - alpha)
        interval from compute_interval_difference (one-sided -> one bound
        unbounded). HC3 inference in statsmodels is normal-based, so the
        two-sided p-value equals model.pvalues.

        Args:
            df:              User-level DataFrame.
            target_kpi:      Outcome column (e.g. 'converted').
            pre_period_kpi:  Pre-experiment covariate column.
            variant_col:     Variant assignment column.
            alpha:           Family-wise significance level (1 - confidence_level),
                             before Šidák adjustment.
            control_label:   Value of `variant_col` that is the control. Strongly
                             recommended: when omitted, the alphabetically first
                             label is used, which silently picks the wrong
                             baseline for labels like 'Alt' vs 'control'.
            alternative:     Direction of the test (challenger vs control).

        Each result's "alpha" is the adjusted threshold actually applied.
        "relative_lift" is None when the fitted control mean is 0.
        """
        df = df.copy()
        df["cov_centered"] = df[pre_period_kpi] - df[pre_period_kpi].mean()

        variants = sorted(df[variant_col].unique())
        if control_label is None:
            baseline = variants[0]
        elif control_label not in variants:
            raise ValueError(
                f"control_label '{control_label}' not found in '{variant_col}'; "
                f"available: {variants}."
            )
        else:
            baseline = control_label
        challengers = [v for v in variants if v != baseline]
        adj_alpha = apply_sidak(alpha, len(variants))

        formula = f"{target_kpi} ~ C({variant_col}, Treatment(reference='{baseline}')) * cov_centered"
        model = smf.ols(formula, data=df).fit(cov_type="HC3")

        results = []
        for challenger in challengers:
            term = (
                f"C({variant_col}, Treatment(reference='{baseline}'))[T.{challenger}]"
            )

            ate = float(model.params[term])
            se = float(model.bse[term])
            control_mean = float(model.params["Intercept"])
            p_val = FrequentistEngine.run_ztest(ate, se, alternative)

            results.append(
                {
                    "variant": challenger,
                    "control": baseline,
                    "p_value": float(p_val),
                    "alpha": float(adj_alpha),
                    "is_significant": bool(p_val < adj_alpha),
                    "absolute_lift": ate,
                    "relative_lift": (
                        ate / control_mean if control_mean != 0 else None
                    ),
                    "ci": list(
                        compute_interval_difference(ate, se, adj_alpha, alternative)
                    ),
                    "std_err": se,
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
           DOWNWARD (the anti-conservative direction). With visitor weights the
           expectation under a binomial null is exactly (n - k)/n times the
           expected binomial variance, for n periods and k fitted baselines, so
           we scale by n/(n - k) and φ → 1 under a true binomial null even on
           short windows. This uses the raw period count, not Kish's effective
           sample size: Kish's n_eff over-corrects when daily traffic is uneven
           (campaign spikes), inflating φ (about 1.5 with one 40x spike day).
           n_effective is still reported as a traffic-concentration diagnostic.

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

        # --- Degrees-of-freedom correction: n / (n - k) in periods ---------
        n_effective = 1.0 / float(np.square(weights).sum())
        dof = n_periods - n_params
        if dof > 0:
            observed_var *= n_periods / dof

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
        """
        Standard Z-test logic for varying tail configurations. Upper tails use
        norm.sf, which stays accurate where 1 - norm.cdf underflows to 0
        (z above ~8.3).

        A zero SE returns 1.0 (no evidence). run_synthesis avoids reaching
        this with a non-zero difference by falling back to the pooled SE.
        """
        if se_diff == 0:
            return 1.0

        z_stat = diff / se_diff

        if alternative == AlternativeHypothesis.GREATER:
            return float(norm.sf(z_stat))
        elif alternative == AlternativeHypothesis.LESS:
            return float(norm.cdf(z_stat))
        else:  # TWO_SIDED
            return float(min(1.0, 2 * norm.sf(abs(z_stat))))

    # ------------------------------------------------------------------ #
    #  Power at a pre-specified effect (MDE)
    #
    #  Power is evaluated at an effect size fixed BEFORE looking at the
    #  result, never at the observed difference. "Observed power" is a
    #  one-to-one function of the p-value (power = 0.5 exactly at p = alpha),
    #  so it adds no information: it is inflated for significant results
    #  (winner's curse) and mechanically low for flat ones, which would make
    #  the false-positive / false-negative risks below circular.
    # ------------------------------------------------------------------ #

    @staticmethod
    def _alternative_rate(
        baseline_rate: float, mde_relative: float, alternative: AlternativeHypothesis
    ) -> float:
        """
        Challenger rate under H1 for a relative MDE. The effect points in the
        tested direction: a decrease for LESS, an increase otherwise.
        """
        if not 0.0 < baseline_rate < 1.0:
            raise ValueError(
                f"baseline_rate must be strictly between 0 and 1, got {baseline_rate}."
            )
        if mde_relative <= 0:
            raise ValueError(
                "mde_relative must be a positive relative effect (e.g. 0.05 for 5%); "
                f"got {mde_relative}. The direction follows `alternative`."
            )
        sign = -1.0 if alternative == AlternativeHypothesis.LESS else 1.0
        p_alt = baseline_rate * (1.0 + sign * mde_relative)
        if not 0.0 < p_alt < 1.0:
            raise ValueError(
                f"baseline_rate={baseline_rate} with mde_relative={mde_relative} "
                f"implies a challenger rate of {p_alt:.4f}, outside (0, 1)."
            )
        return p_alt

    @classmethod
    def calculate_analytical_power(
        cls,
        baseline_rate: float,
        n_ctrl: int,
        n_chal: int,
        mde_relative: float,
        alpha: float,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
        reduction_factor: float = 1.0,
    ) -> float:
        """
        Closed-form power of run_synthesis's test to detect a pre-specified
        relative effect `mde_relative` (same convention as the pretest module,
        e.g. 0.05 = 5% relative lift).

        Uses the same φ-scaled unpooled SE as run_synthesis, evaluated at the
        rates under H1 (baseline vs baseline·(1 ± mde)). Two-sided power
        includes both rejection tails.

        Args:
            baseline_rate:    Control conversion rate. A planned rate, or the
                              observed control rate (the control is not subject
                              to the winner's curse).
            n_ctrl, n_chal:   Sample sizes per arm.
            mde_relative:     Effect size the test should detect, fixed in
                              advance. Never the observed lift.
            alpha:            The per-comparison alpha actually used for the
                              decision: Šidák-adjusted for A/B/n, as in
                              run_synthesis (see apply_sidak).
            alternative:      Test direction.
            reduction_factor: φ, as on ExperimentInput.
        """
        if n_ctrl <= 0 or n_chal <= 0:
            return 0.0
        p_alt = cls._alternative_rate(baseline_rate, mde_relative, alternative)
        se = (
            reduction_factor
            * (
                baseline_rate * (1 - baseline_rate) / n_ctrl
                + p_alt * (1 - p_alt) / n_chal
            )
        ) ** 0.5
        z_delta = abs(p_alt - baseline_rate) / se

        if alternative == AlternativeHypothesis.TWO_SIDED:
            z_alpha = norm.ppf(1 - alpha / 2)
            return float(norm.cdf(z_delta - z_alpha) + norm.cdf(-z_delta - z_alpha))
        z_alpha = norm.ppf(1 - alpha)
        return float(norm.cdf(z_delta - z_alpha))

    @classmethod
    def run_vectorized_bootstrap_power(
        cls,
        baseline_rate: float,
        n_ctrl: int,
        n_chal: int,
        mde_relative: float,
        alpha: float = 0.05,
        n_bootstraps: int = 10000,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
        use_pooled_se: bool = False,
        random_state: Optional[int] = None,
    ) -> float:
        """
        Simulation-based power to detect a pre-specified relative effect, as a
        cross-check on calculate_analytical_power at small samples where the
        normal approximation is loose. Simulates conversion COUNTS from
        Binomial(n_ctrl, baseline_rate) and Binomial(n_chal, baseline·(1 ± mde))
        and reports the share of simulated tests that reject at `alpha`.

        Arguments as in calculate_analytical_power. `alpha` should be the
        Šidák-adjusted per-comparison alpha for A/B/n tests.

        Defaults to the unpooled SE, matching run_synthesis. `use_pooled_se=True`
        gives the textbook pooled statistic instead. Does not apply a
        reduction_factor (it simulates pure binomial sampling); under
        overdispersion use calculate_analytical_power.
        """
        if n_ctrl <= 0 or n_chal <= 0:
            return 0.0
        p_alt = cls._alternative_rate(baseline_rate, mde_relative, alternative)
        ctrl_n, chal_n = n_ctrl, n_chal

        rng = np.random.default_rng(random_state)
        sim_ctrl_convs = rng.binomial(n=ctrl_n, p=baseline_rate, size=n_bootstraps)
        sim_chal_convs = rng.binomial(n=chal_n, p=p_alt, size=n_bootstraps)

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
            p_values = norm.sf(z_stats)
        elif alternative == AlternativeHypothesis.LESS:
            p_values = norm.cdf(z_stats)
        else:
            p_values = 2 * norm.sf(np.abs(z_stats))

        return float(np.mean(p_values < alpha))

    # ------------------------------------------------------------------ #
    #  Decision risk: false-positive / false-negative
    #
    #  Composable helpers. Deliberately NOT wired into run_synthesis: the
    #  FrequentistResult contract (see test_frequentist.py) carries inference
    #  fields only. Call assess_result_risk from a reporting layer: it takes a
    #  pre-specified MDE and applies the same Šidák alpha as run_synthesis.
    #  The `power` fed into these formulas must be power at that MDE, never
    #  power at the observed effect (see the power section above).
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

        `power` must be the power to detect a pre-specified effect (see
        calculate_analytical_power), and `alpha` the per-comparison alpha
        used for the decision (Šidák-adjusted for A/B/n). Power at the
        observed effect makes FPR look low for every significant result and
        FNDR high for every flat one, by construction. Prefer
        assess_result_risk, which wires both correctly.

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

    @classmethod
    def assess_result_risk(
        cls,
        data: ExperimentInput,
        result: FrequentistResult,
        mde_relative: float,
        prior: float,
        baseline_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Decision risk for one run_synthesis result, with power evaluated at a
        pre-specified relative MDE and the same Šidák-adjusted alpha that
        run_synthesis used for the decision.

        Args:
            data:          The ExperimentInput passed to run_synthesis.
            result:        One FrequentistResult from that call.
            mde_relative:  Effect size fixed in advance (e.g. the MDE from the
                           pretest plan). Never the observed lift.
            prior:         P(H1), e.g. from resolve_prior_probability.
            baseline_rate: Planned control rate. Defaults to the observed
                           control rate.

        Returns assess_decision_risk's dict plus "alpha" and "mde_relative".
        """
        labels = data.labels or [f"Variant {i}" for i in range(len(data.visitors))]
        if result.variant_label not in labels[1:]:
            raise ValueError(
                f"Result for '{result.variant_label}' does not match a challenger "
                f"in the input; challengers are {labels[1:]}."
            )
        i = labels.index(result.variant_label, 1)
        alpha = apply_sidak(1.0 - data.confidence_level, len(data.visitors))
        if baseline_rate is None:
            baseline_rate = data.conversions[0] / data.visitors[0]

        power = cls.calculate_analytical_power(
            baseline_rate=baseline_rate,
            n_ctrl=data.visitors[0],
            n_chal=data.visitors[i],
            mde_relative=mde_relative,
            alpha=alpha,
            alternative=data.alternative,
            reduction_factor=data.reduction_factor,
        )
        risk = cls.assess_decision_risk(
            result.is_significant, alpha, power, prior, data.alternative
        )
        risk.update({"alpha": float(alpha), "mde_relative": float(mde_relative)})
        return risk

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
        """
        UI-agnostic narrative summary of estimate_monetary_impact's output.

        `alternative` must match whatever was passed to estimate_monetary_impact
        (and therefore to compute_interval_difference) when this monetary_result
        was generated. See compute_interval_difference's docstring for the
        underlying convention:

            GREATER   -> lower bound only: (diff - z*se, +inf)
            LESS      -> upper bound only: (-inf, diff + z*se)
            TWO_SIDED -> symmetric:        (diff - z*se, diff + z*se)

        Concretely: for a one-sided test, one side of monetary_result's CI is
        always +/-inf by construction -- that is not a missing value, a fallback,
        or a sign of low confidence, it's the definition of "one-sided." This
        function therefore branches on `alternative` for the significant case, so
        a significant GREATER result reads as an open-ended "at least X" claim
        rather than pairing a real number with "unbounded" as if they formed a
        comparable range. If `alternative` doesn't match what actually produced
        monetary_result, the wording here will misrepresent the interval, since
        this function has no independent way to detect that mismatch.
        """

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

        Edge cases:
          * Control rate 0: uplift is None (relative lift undefined); the
            conclusion's direction comes from the absolute difference.
          * Both arms at 0% or 100% (unpooled SE = 0): the test falls back to
            the pooled SE (the textbook SE under H0, still φ-scaled) and the
            interval to Newcombe's score interval (no φ), and `warning` is set.
            Identical degenerate arms then give p = 1; opposite ones (0% vs
            100%) are correctly significant instead of reported as flat.
        """
        labels = data.labels or [f"Variant {i}" for i in range(len(data.visitors))]
        c_ctrl = data.conversions[0]
        n_ctrl = data.visitors[0]
        p_ctrl = c_ctrl / n_ctrl
        alpha = apply_sidak(1.0 - data.confidence_level, len(data.visitors))
        phi = data.reduction_factor

        results = []
        for i in range(1, len(data.visitors)):
            c_chal = data.conversions[i]
            n_chal = data.visitors[i]
            p_chal = c_chal / n_chal
            diff = p_chal - p_ctrl
            uplift = diff / p_ctrl if p_ctrl != 0 else None

            se_diff = (
                p_ctrl * (1 - p_ctrl) * phi / n_ctrl
                + p_chal * (1 - p_chal) * phi / n_chal
            ) ** 0.5

            warning = None
            if se_diff > 0:
                p_value = self.run_ztest(diff, se_diff, data.alternative)
                ci = compute_interval_difference(diff, se_diff, alpha, data.alternative)
            else:
                p_pool = (c_ctrl + c_chal) / (n_ctrl + n_chal)
                se_pooled = (
                    p_pool * (1 - p_pool) * phi * (1 / n_ctrl + 1 / n_chal)
                ) ** 0.5
                p_value = self.run_ztest(diff, se_pooled, data.alternative)
                ci = compute_newcombe_interval(
                    c_ctrl, n_ctrl, c_chal, n_chal, alpha, data.alternative
                )
                warning = (
                    "Both arms converted at 0% or 100%, so the unpooled standard "
                    "error is zero. The p-value uses the pooled standard error and "
                    "the interval is Newcombe's score interval. Counts this extreme "
                    "are better analysed with an exact test (Boschloo)."
                )

            is_sig = bool(p_value < alpha)
            conclusion = self.generate_conclusion_statement(
                labels[i], is_sig, uplift, absolute_diff=diff
            )

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
                    warning=warning,
                )
            )

        return results
