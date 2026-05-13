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
    "skeptical":     0.10,
    "moderate":      0.25,
    "uninformative": 1.00,
}

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
    skepticism: Literal["skeptical", "moderate", "uninformative"],
) -> LiftPrior:
    """
    Builds a log-normal lift prior from a directional expectation and a
    skepticism level controlling how strongly large lifts are penalised.

    Args:
        expected_lift_pct: Expected relative lift as a percentage (e.g. 5.0
                           for +5%). Use 0.0 for no directional expectation.
                           Must be greater than -100.
        skepticism:        How tightly the prior constrains implausible lifts.
                           'skeptical' applies strong pressure; 'uninformative'
                           applies almost none.
    """
    if skepticism not in _LIFT_PRIOR_STD:
        raise ValueError(
            f"skepticism must be one of {list(_LIFT_PRIOR_STD)}, got {skepticism!r}."
        )
    if expected_lift_pct <= -100:
        raise ValueError("expected_lift_pct must be greater than -100.")

    return LiftPrior(
        mean_log_lift=np.log1p(expected_lift_pct / 100.0),
        std_log_lift=_LIFT_PRIOR_STD[skepticism],
    )


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

    def _compute_lift_weights(
        self,
        control_samples: np.ndarray,
        challenger_samples: np.ndarray,
        lift_prior: LiftPrior,
    ) -> np.ndarray:
        """
        Importance weights reflecting how plausible each simulated lift is
        under the lift prior. Weights sum to 1.
        """
        log_lift = np.log(challenger_samples) - np.log(control_samples)
        log_w = norm.logpdf(log_lift, lift_prior.mean_log_lift, lift_prior.std_log_lift)
        w = np.exp(log_w - log_w.max())
        return w / w.sum()

    @staticmethod
    def _weighted_mean_masked(
        values: np.ndarray,
        weights: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """
        Computes a weighted mean over the True positions of mask.

        Normalises the masked weight vector before averaging, so the result
        is correct even when the prior concentrates all weight on samples
        that sit outside the mask (which would make weights * mask sum to
        zero and cause np.average to raise ZeroDivisionError).

        Returns 0.0 if no samples pass the mask or the masked weight sum
        is effectively zero.
        """
        masked_weights = weights * mask
        weight_sum = masked_weights.sum()
        if weight_sum <= 0 or not mask.any():
            return 0.0
        normalised = masked_weights / weight_sum
        return float(np.dot(values, normalised))

    def _sample_aov(self, mean_aov: float, cv: float, n_samples: int) -> np.ndarray:
        """
        Draws AOV samples from a log-normal distribution parameterised so that
        E[AOV] = mean_aov exactly — variance is added without any bias.
        """
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
        Applies lift prior importance weights when provided.
        Returns one BayesianResult per challenger variant.
        """
        beta_prior = beta_prior or get_beta_prior()
        lift_prior = lift_prior or get_lift_prior(0.0, "uninformative")

        visitors = data.visitors
        conversions = data.conversions
        labels: List[str] = data.labels or [
            f"Variant {i}" for i in range(len(visitors))
        ]
        num_variants = len(visitors)

        # (num_variants, n_samples)
        samples = self._sample_posteriors(visitors, conversions, beta_prior, n_samples)
        control_samples = samples[0]

        # Average importance weights across all challengers for "best overall"
        if num_variants > 1:
            per_challenger_weights = np.array([
                self._compute_lift_weights(control_samples, samples[i], lift_prior)
                for i in range(1, num_variants)
            ])
            weights = per_challenger_weights.mean(axis=0)
            weights /= weights.sum()
        else:
            weights = np.ones(n_samples) / n_samples

        winner_indices = np.argmax(samples, axis=0)
        prob_best_overall = np.array([
            np.average(winner_indices == i, weights=weights)
            for i in range(num_variants)
        ])

        results = []
        for i in range(1, num_variants):
            challenger_samples = samples[i]
            w = self._compute_lift_weights(control_samples, challenger_samples, lift_prior)

            prob_beat_control = float(
                np.average(challenger_samples > control_samples, weights=w)
            )
            expected_uplift = float(
                np.average(np.maximum(challenger_samples - control_samples, 0), weights=w)
            )
            expected_loss = float(
                np.average(np.maximum(control_samples - challenger_samples, 0), weights=w)
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
                )
            )

        return results

    def run_monetary_projection(
        self,
        visitors: List[int],
        conversions: List[int],
        biz_case: BusinessCaseInput,
        prob_best_overall: List[float],
        variant_labels: List[str],
        beta_prior: Optional[BetaPrior] = None,
        lift_prior: Optional[LiftPrior] = None,
        aov_cv: float = 0.5,
        n_simulations: int = 50_000,
    ) -> List[Dict[str, Any]]:
        """
        Translates conversion rate posteriors into monetary risk and uplift
        projections. AOV is modelled as a log-normal variable to add realistic
        spread without biasing point estimates. Lift prior weights are applied
        to all monetary aggregations.
        """
        beta_prior = beta_prior or get_beta_prior()
        lift_prior = lift_prior or get_lift_prior(0.0, "uninformative")

        num_variants = len(visitors)

        if len(variant_labels) != num_variants:
            raise ValueError(
                f"variant_labels has {len(variant_labels)} entries but visitors has "
                f"{num_variants}. They must be the same length."
            )

        # (num_variants, n_simulations)
        samples = self._sample_posteriors(visitors, conversions, beta_prior, n_simulations)
        daily_vol = (
            samples * np.array(visitors)[:, np.newaxis]
        ) / biz_case.runtime_days

        control_vol = daily_vol[0]
        control_cr = samples[0]
        control_label = variant_labels[0]
        control_aov = biz_case.aovs.get(control_label, 0.0)

        results = []
        for i in range(1, num_variants):
            variant_label = variant_labels[i]
            variant_vol = daily_vol[i]
            variant_cr = samples[i]
            variant_aov = biz_case.aovs.get(variant_label, 0.0)

            weights = self._compute_lift_weights(control_cr, variant_cr, lift_prior)

            sampled_variant_aov = self._sample_aov(variant_aov, aov_cv, n_simulations)
            sampled_control_aov = self._sample_aov(control_aov, aov_cv, n_simulations)

            diff = variant_vol - control_vol
            positive_mask = diff > 0
            negative_mask = diff < 0

            prob_beat_control = float(np.average(positive_mask, weights=weights))
            prob_control_better = float(np.average(negative_mask, weights=weights))

            expected_daily_gain = self._weighted_mean_masked(
                values=diff * sampled_variant_aov,
                weights=weights,
                mask=positive_mask,
            )
            expected_daily_loss = self._weighted_mean_masked(
                values=diff * sampled_control_aov,
                weights=weights,
                mask=negative_mask,
            )

            uplift = expected_daily_gain * biz_case.projection_period * prob_beat_control
            risk = expected_daily_loss * biz_case.projection_period * prob_control_better

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
                "prob_best_overall": prob_best_overall[i],
                "expected_uplift": round(uplift, 2),
                "expected_risk": round(risk, 2),
                "expected_total_contribution": round(uplift + risk, 2),
                "conclusion": conclusion,
            })

        return results
