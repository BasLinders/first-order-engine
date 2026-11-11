import warnings
from dataclasses import dataclass
from typing import Dict, List, Any, Literal, Optional

import numpy as np
from scipy.stats import norm

from foe.core.models import BusinessCaseInput, BayesianResult, ExperimentInput


# BetaPrior and LiftPrior are intentionally plain dataclasses rather than
# Pydantic models: they are internal engine primitives, not API boundary types.

@dataclass(frozen=True)
class BetaPrior:
    alpha: float = 1.0
    beta: float = 1.0


@dataclass(frozen=True)
class LiftPrior:
    mean_log_lift: float
    std_log_lift: float


_LIFT_PRIOR_STD: Dict[str, float] = {
    "narrow": 0.10,
    "medium": 0.25,
    "wide":   1.00,
}

# Original names, kept as aliases so existing callers keep working. They
# describe the prior's width only when expected_lift_pct is 0: a narrow prior
# around a non-zero lift is a confident directional belief, not skepticism.
_LIFT_PRIOR_WIDTH_ALIASES: Dict[str, str] = {
    "skeptical":     "narrow",
    "moderate":      "medium",
    "uninformative": "wide",
}

LiftPriorWidth = Literal[
    "narrow", "medium", "wide", "skeptical", "moderate", "uninformative"
]

# Default projection period used when no BusinessCaseInput is available.
_DEFAULT_PROJECTION_DAYS: int = 30


def get_beta_prior() -> BetaPrior:
    """
    Returns a fixed uninformative prior. At typical sample sizes the data
    dominates regardless; prior beliefs about lift are expressed via get_lift_prior.
    """
    return BetaPrior(alpha=1.0, beta=1.0)


def get_lift_prior(
    expected_lift_pct: float,
    skepticism: LiftPriorWidth,
) -> LiftPrior:
    """
    Builds a normal prior on log-lift, centred at log(1 + expected_lift_pct/100),
    with a width that controls how strongly lifts far from that centre are
    penalised.

    Args:
        expected_lift_pct: Expected relative lift as a percentage (e.g. 5.0
                           for +5%). Use 0.0 for no directional expectation.
                           Must be greater than -100.
        skepticism:        Prior width on the log-lift scale: 'narrow'
                           (sigma 0.10), 'medium' (0.25) or 'wide' (1.00).
                           The original names 'skeptical', 'moderate' and
                           'uninformative' are accepted as aliases.

    Only a prior centred at 0 is skeptical. A narrow prior around a non-zero
    expected lift is a confident belief in that direction (+20% / narrow
    alone gives P(lift > 0) ~ 0.97); see prior_prob_positive_lift.
    """
    width = _LIFT_PRIOR_WIDTH_ALIASES.get(skepticism, skepticism)
    if width not in _LIFT_PRIOR_STD:
        raise ValueError(
            f"skepticism must be one of {list(_LIFT_PRIOR_STD)} (or the aliases "
            f"{list(_LIFT_PRIOR_WIDTH_ALIASES)}), got {skepticism!r}."
        )
    if expected_lift_pct <= -100:
        raise ValueError("expected_lift_pct must be greater than -100.")

    return LiftPrior(
        mean_log_lift=np.log1p(expected_lift_pct / 100.0),
        std_log_lift=_LIFT_PRIOR_STD[width],
    )


def prior_prob_positive_lift(lift_prior: Optional[LiftPrior]) -> float:
    """
    P(lift > 0) implied by the lift prior alone, before seeing any data.

    0.5 means the prior is neutral about direction. Anything else means the
    prior by itself pushes prob_beat_control toward one side. A non-zero
    expected lift combined with a narrow ('skeptical') prior is a strong
    directional belief: +10% / narrow gives ~0.83 here. Callers should
    surface this value so the prior's pull on the decision is visible.

    This is the lift-prior component only; the flat Beta(1, 1) base prior is
    symmetric in log-lift, so it does not move this probability.
    """
    if lift_prior is None:
        return 0.5
    return float(norm.cdf(lift_prior.mean_log_lift / lift_prior.std_log_lift))


def mc_margin(p: float, ess: float, z: float = 1.96) -> float:
    """
    Monte Carlo margin of error for a probability estimated from importance-
    weighted draws: z * sqrt(p * (1 - p) / ess).

    Pass a result's probability (e.g. prob_beat_control) and its
    effective_sample_size. With the default z = 1.96 this is the half-width
    of an approximate 95% interval for the Monte Carlo error alone; it says
    nothing about uncertainty from the data, which the probability itself
    already expresses.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p!r}.")
    if ess <= 0:
        raise ValueError(f"ess must be positive, got {ess!r}.")
    if z <= 0:
        raise ValueError(f"z must be positive, got {z!r}.")
    return float(z * np.sqrt(p * (1.0 - p) / ess))


class BayesianEngine:
    """
    The FOE Bayesian Engine handles Beta-Binomial posterior updates,
    Monte Carlo simulations for Probability of Being Best, lift-prior
    importance weighting, and Decision Theory-based monetary risk projections
    with log-normal AOV sampling.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_posteriors(
        self,
        visitors: List[int],
        conversions: List[int],
        beta_prior: BetaPrior,
        n_samples: int,
    ) -> np.ndarray:
        """
        Returns a (num_variants, n_samples) array of Beta posterior samples,
        using the provided BetaPrior as the starting point for all variants.
        """
        visitors_arr = np.array(visitors, dtype=float)
        conversions_arr = np.array(conversions, dtype=float)
        a_post = beta_prior.alpha + conversions_arr
        b_post = beta_prior.beta + (visitors_arr - conversions_arr)
        return self.rng.beta(
            a_post[:, np.newaxis],
            b_post[:, np.newaxis],
            size=(len(visitors), n_samples),
        )

    @staticmethod
    def _compute_joint_lift_weights(
        samples: np.ndarray,
        lift_prior: Optional[LiftPrior],
    ) -> np.ndarray:
        """
        Importance weights for the joint (num_variants, n_samples) posterior
        draws under the lift prior. Weights sum to 1.

        The lift prior is applied independently to every challenger's lift
        vs control, so the joint prior is the *product* of the per-challenger
        densities (a sum in log space). One weight vector is shared by every
        downstream quantity (P(best), P(beat control), uplift, monetary
        projection), so they all describe the same posterior.

        lift_prior=None means no lift prior: uniform weights, i.e. the plain
        Beta-Binomial posterior.
        """
        n_samples = samples.shape[1]
        if lift_prior is None or samples.shape[0] < 2:
            return np.full(n_samples, 1.0 / n_samples)

        log_lift = np.log(samples[1:]) - np.log(samples[0])
        log_w = norm.logpdf(
            log_lift, lift_prior.mean_log_lift, lift_prior.std_log_lift
        ).sum(axis=0)
        w = np.exp(log_w - log_w.max())
        return w / w.sum()

    @staticmethod
    def _effective_sample_size(weights: np.ndarray) -> float:
        """
        Kish effective sample size of normalised importance weights. Equals
        n_samples when weights are uniform; falls as the prior and the data
        disagree. A low ESS means the weighted estimates rest on few draws
        and carry more Monte Carlo error than n_samples suggests.
        """
        return float(1.0 / np.sum(weights ** 2))

    def _sample_aov(self, mean_aov: float, cv: float, n_samples: int) -> np.ndarray:
        """
        Draws AOV samples from a log-normal distribution with mean mean_aov
        and coefficient of variation cv (sd = cv * mean_aov), parameterised so
        that E[AOV] = mean_aov exactly — variance is added without any bias.
        A non-positive mean_aov yields zeros instead of log(0) = -inf.
        """
        if mean_aov <= 0:
            return np.zeros(n_samples)
        sigma2 = np.log1p(cv ** 2)
        mu = np.log(mean_aov) - sigma2 / 2
        return self.rng.lognormal(mean=mu, sigma=np.sqrt(sigma2), size=n_samples)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def generate_bayesian_conclusion(
        variant_name: str,
        prob_beat_control: float,
        expected_uplift: float,
        expected_risk: float,
        projection_period: int,
    ) -> str:
        uplift_str = f"{expected_uplift:,.0f}"
        risk_str = f"{expected_risk:,.0f}"

        if prob_beat_control >= 0.95:
            return (
                f"Strong Winner: '{variant_name}' has a {prob_beat_control:.1%} probability "
                f"of outperforming the control. Rolling this out carries minimal expected risk "
                f"({risk_str}) with a projected upside of {uplift_str} over the next "
                f"{projection_period} days."
            )
        elif prob_beat_control <= 0.10:
            return (
                f"Clear Loser: '{variant_name}' has only a {prob_beat_control:.1%} chance of "
                f"beating the control. Rolling this out carries an expected monetary risk of "
                f"{risk_str}. It is recommended to discard this variant."
            )
        elif expected_uplift > abs(expected_risk) * 3:
            return (
                f"Asymmetric Bet: '{variant_name}' has a {prob_beat_control:.1%} chance to win. "
                f"While not perfectly certain, the potential upside ({uplift_str}) heavily "
                f"outweighs the expected risk ({risk_str}). Consider rolling out if you have "
                f"a high risk tolerance."
            )
        else:
            return (
                f"Inconclusive: '{variant_name}' has a {prob_beat_control:.1%} probability of "
                f"beating the control. The potential upside ({uplift_str}) does not definitively "
                f"outweigh the risk ({risk_str}). Collect more data or discard if time-constrained."
            )

    @staticmethod
    def _generate_probability_conclusion(
        variant_name: str,
        prob_beat_control: float,
        prob_being_best: float,
    ) -> str:
        """
        Generates a conclusion string for probability-only analysis, where no
        monetary values are available. Kept separate from generate_bayesian_conclusion
        to avoid formatting rate differences (e.g. 0.003) as currency strings.
        """
        if prob_beat_control >= 0.95:
            return (
                f"Strong Winner: '{variant_name}' has a {prob_beat_control:.1%} probability "
                f"of outperforming the control and a {prob_being_best:.1%} chance of being "
                f"the best overall variant. You can confidently roll this out."
            )
        elif prob_beat_control <= 0.10:
            return (
                f"Clear Loser: '{variant_name}' has only a {prob_beat_control:.1%} chance of "
                f"beating the control. It is recommended to discard this variant."
            )
        else:
            return (
                f"Inconclusive: '{variant_name}' has a {prob_beat_control:.1%} probability of "
                f"beating the control. Collect more data before making a shipping decision."
            )

    def run_probability_analysis(
        self,
        data: ExperimentInput,
        beta_prior: Optional[BetaPrior] = None,
        lift_prior: Optional[LiftPrior] = None,
        n_samples: int = 100_000,
    ) -> List[BayesianResult]:
        """
        Calculates Probability of Being Best using vectorized Monte Carlo sampling.
        Applies lift prior importance weights when provided; lift_prior=None
        means no lift prior (plain Beta-Binomial posterior).
        Returns one BayesianResult per challenger variant. The control's
        P(best) is 1 - sum of the challengers' prob_being_best.
        """
        beta_prior = beta_prior or get_beta_prior()

        visitors = data.visitors
        conversions = data.conversions
        labels: List[str] = data.labels or [
            f"Variant {i}" for i in range(len(visitors))
        ]
        num_variants = len(visitors)

        # (num_variants, n_samples)
        samples = self._sample_posteriors(visitors, conversions, beta_prior, n_samples)
        control_samples = samples[0]

        weights = self._compute_joint_lift_weights(samples, lift_prior)
        ess = self._effective_sample_size(weights)

        winner_indices = np.argmax(samples, axis=0)
        prob_best_overall = np.bincount(
            winner_indices, weights=weights, minlength=num_variants
        )

        results = []
        for i in range(1, num_variants):
            challenger_samples = samples[i]

            prob_beat_control = float(
                np.average(challenger_samples > control_samples, weights=weights)
            )
            expected_uplift = float(
                np.average(np.maximum(challenger_samples - control_samples, 0), weights=weights)
            )
            expected_loss = float(
                np.average(np.maximum(control_samples - challenger_samples, 0), weights=weights)
            )
            prob_best = float(prob_best_overall[i])

            conclusion = self._generate_probability_conclusion(
                variant_name=labels[i],
                prob_beat_control=prob_beat_control,
                prob_being_best=prob_best,
            )

            results.append(
                BayesianResult(
                    variant_label=labels[i],
                    control_label=labels[0],
                    prob_being_best=prob_best,
                    prob_beat_control=prob_beat_control,
                    expected_uplift=expected_uplift,
                    expected_loss=expected_loss,
                    conclusion=conclusion,
                    effective_sample_size=ess,
                    n_samples=n_samples,
                )
            )

        return results

    def simulate_uplift_distributions(
        self,
        data: ExperimentInput,
        beta_prior: Optional[BetaPrior] = None,
        lift_prior: Optional[LiftPrior] = None,
        n_samples: int = 20_000,
    ) -> List[np.ndarray]:
        """
        Posterior draws of the relative conversion-rate uplift
        (cr_variant - cr_control) / cr_control, one array per challenger.

        Draws are resampled with the joint lift-prior weights (sampling-
        importance-resampling), so the arrays are unweighted and can be fed
        straight into a histogram. The same resampled indices are used for
        every challenger, keeping the arrays jointly consistent.
        """
        beta_prior = beta_prior or get_beta_prior()

        samples = self._sample_posteriors(data.visitors, data.conversions, beta_prior, n_samples)
        weights = self._compute_joint_lift_weights(samples, lift_prior)
        idx = self.rng.choice(n_samples, size=n_samples, p=weights)

        control = samples[0]
        return [
            ((samples[i] - control) / control)[idx]
            for i in range(1, samples.shape[0])
        ]

    def run_monetary_projection(
        self,
        visitors: List[int],
        conversions: List[int],
        biz_case: BusinessCaseInput,
        prob_best_overall: Optional[List[float]] = None,
        variant_labels: Optional[List[str]] = None,
        beta_prior: Optional[BetaPrior] = None,
        lift_prior: Optional[LiftPrior] = None,
        aov_cv: float = 0.5,
        n_simulations: int = 50_000,
    ) -> List[Dict[str, Any]]:
        """
        Translates conversion rate posteriors into monetary risk and uplift
        projections. AOV is modelled as a log-normal variable to add realistic
        spread without biasing point estimates. Lift prior weights are applied
        to all monetary aggregations; lift_prior=None means no lift prior.

        Variants are compared on conversion *rate*, projected onto the same
        daily traffic per arm (mean visitors per arm / runtime_days). Comparing
        raw conversion volumes would let an uneven traffic split decide which
        arm is "better".

        Each draw's daily revenue difference is
        (cr_variant * aov_variant - cr_control * aov_control) * traffic, so an
        AOV difference counts even at equal conversion rates.

        aov_variant and aov_control are draws of each arm's *mean* AOV: log-
        normal with mean biz_case.aovs[label] and sd aov_cv * AOV / sqrt(orders),
        where aov_cv is the order-to-order coefficient of variation of order
        values and orders is approximated by the arm's conversions. Pass the
        observed CV when it is known; aov_cv=0 treats the AOVs as exact.
        expected_uplift / expected_risk are the projection period times the
        weighted mean of its positive / negative part. prob_beat_control stays
        rate-based (it matches run_probability_analysis); prob_revenue_better
        is P(revenue difference > 0).

        beta_prior defaults to Beta(biz_case.alpha_prior, biz_case.beta_prior);
        an explicit beta_prior takes precedence.

        prob_best_overall in the results is computed from this projection's
        own weighted draws (rate-based, like run_probability_analysis), so it
        is consistent with the other numbers in the same result. The
        prob_best_overall argument is deprecated and ignored.

        variant_labels is required; it defaults to None only so that the
        deprecated prob_best_overall before it can be optional.
        """
        if prob_best_overall is not None:
            warnings.warn(
                "run_monetary_projection(prob_best_overall=...) is deprecated and "
                "ignored: prob_best_overall is now computed from the projection's "
                "own draws.",
                DeprecationWarning,
                stacklevel=2,
            )
        if variant_labels is None:
            raise ValueError("variant_labels is required.")

        beta_prior = beta_prior or BetaPrior(
            alpha=biz_case.alpha_prior, beta=biz_case.beta_prior
        )

        num_variants = len(visitors)

        if len(variant_labels) != num_variants:
            raise ValueError(
                f"variant_labels has {len(variant_labels)} entries but visitors has "
                f"{num_variants}. They must be the same length."
            )

        # (num_variants, n_simulations)
        samples = self._sample_posteriors(visitors, conversions, beta_prior, n_simulations)
        weights = self._compute_joint_lift_weights(samples, lift_prior)
        ess = self._effective_sample_size(weights)
        prob_best = np.bincount(
            np.argmax(samples, axis=0), weights=weights, minlength=num_variants
        )

        daily_visitors_per_arm = float(np.mean(visitors)) / biz_case.runtime_days

        # Each arm's mean AOV is an estimate from its own orders, so it is
        # uncertain: sd = aov_cv * AOV / sqrt(orders), independent per arm,
        # with conversions standing in for orders. This is the same model as
        # ContinuousMetricEngine.estimate_monetary_impact_per_variant
        # (se = std / sqrt(n_orders)) with std = aov_cv * AOV. Draws are made
        # once per arm, so every challenger is compared to the same control.
        aov_draws = [
            self._sample_aov(
                biz_case.aovs.get(label, 0.0),
                aov_cv / np.sqrt(max(conversions[k], 1)),
                n_simulations,
            )
            for k, label in enumerate(variant_labels)
        ]

        control_cr = samples[0]
        control_aov = aov_draws[0]

        results = []
        for i in range(1, num_variants):
            variant_label = variant_labels[i]
            variant_cr = samples[i]
            variant_aov = aov_draws[i]

            # Daily revenue difference at equal traffic per arm, each arm
            # valued at its own AOV.
            rev_diff = (
                (variant_cr * variant_aov - control_cr * control_aov)
                * daily_visitors_per_arm
            )

            # Rate-based, so it matches run_probability_analysis.
            prob_beat_control = float(np.average(variant_cr > control_cr, weights=weights))
            prob_revenue_better = float(np.average(rev_diff > 0, weights=weights))

            period = biz_case.projection_period
            uplift = period * float(np.average(np.maximum(rev_diff, 0.0), weights=weights))
            risk = period * float(np.average(np.maximum(-rev_diff, 0.0), weights=weights))

            conclusion = self.generate_bayesian_conclusion(
                variant_name=variant_label,
                prob_beat_control=prob_beat_control,
                expected_uplift=uplift,
                expected_risk=risk,
                projection_period=biz_case.projection_period,
            )

            results.append({
                "variant_index": i,
                "variant_label": variant_label,
                "prob_beat_control": prob_beat_control,
                "prob_revenue_better": prob_revenue_better,
                "prob_best_overall": float(prob_best[i]),
                "expected_uplift": round(uplift, 2),
                "expected_risk": round(risk, 2),
                "expected_total_contribution": round(uplift - risk, 2),
                "conclusion": conclusion,
                "effective_sample_size": ess,
                "n_samples": n_simulations,
            })

        return results
