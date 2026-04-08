import numpy as np
from typing import List, Dict, Any, Optional
from foe.core.models import BusinessCaseInput, BayesianResult, ExperimentInput


class BayesianEngine:
    """
    The FOE Bayesian Engine handles Beta-Binomial posterior updates,
    Monte Carlo simulations for 'Probability of Being Best', and
    Decision Theory-based monetary risk projections.
    """

    def __init__(self, seed: Optional[int] = None):
        # Uses a thread-safe, isolated Random Number Generator
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def generate_bayesian_conclusion(
        variant_name: str,
        prob_beat_control: float,
        expected_uplift: float,
        expected_risk: float,
        projection_period: int,
    ) -> str:
        """
        Translates Bayesian risk distributions into clear, UI-ready business text.
        """
        # Format currency (USD/EUR agnostic for now, just using standard decimal formatting)
        uplift_str = f"{expected_uplift:,.0f}"
        risk_str = f"{expected_risk:,.0f}"

        if prob_beat_control >= 0.95:
            return (
                f"Clear Winner: '{variant_name}' has a {prob_beat_control:.1%} probability "
                f"of outperforming the control. Rolling this out carries minimal expected risk "
                f"({risk_str}) with a projected upside of {uplift_str} over the next {projection_period} days."
            )
        elif prob_beat_control <= 0.10:
            return (
                f"Clear Loser: '{variant_name}' has only a {prob_beat_control:.1%} chance of beating the control. "
                f"Rolling this out carries an expected monetary risk of {risk_str}. It is recommended to discard this variant."
            )
        elif expected_uplift > expected_risk * 3:
            return (
                f"Asymmetric Bet: '{variant_name}' has a {prob_beat_control:.1%} chance to win. "
                f"While not perfectly certain, the potential upside ({uplift_str}) heavily outweighs "
                f"the expected risk ({risk_str}). Consider rolling out if you have a high risk tolerance."
            )
        else:
            return (
                f"Inconclusive: '{variant_name}' has a {prob_beat_control:.1%} probability of beating the control. "
                f"The potential upside ({uplift_str}) does not definitively outweigh the risk ({risk_str}). "
                "Collect more data or discard if time-constrained."
            )

    @staticmethod
    def _calculate_posterior_params(
        conversions: int, visitors: int, a_prior: float = 1.0, b_prior: float = 1.0
    ) -> tuple[float, float]:
        """
        Calculates the posterior parameters for a Beta distribution.
        """
        return a_prior + conversions, b_prior + (visitors - conversions)

    def _sample_posteriors(
        self,
        visitors: List[int],
        conversions: List[int],
        n_samples: int = 100000,
    ) -> np.ndarray:
        """Returns a (num_variants, n_samples) array of Beta posterior samples."""
        visitors_arr = np.array(visitors, dtype=float)
        conversions_arr = np.array(conversions, dtype=float)
        a_post = 1.0 + conversions_arr
        b_post = 1.0 + (visitors_arr - conversions_arr)
        return self.rng.beta(
            a_post[:, np.newaxis],
            b_post[:, np.newaxis],
            size=(len(visitors), n_samples),
        )

    def run_probability_analysis(
        self,
        data: ExperimentInput,
        n_samples: int = 100000,
    ) -> List[BayesianResult]:
        """
        Calculates Probability of Being Best using vectorized Monte Carlo sampling.
        Returns one BayesianResult per challenger variant.
        """
        visitors = data.visitors
        conversions = data.conversions
        labels = data.labels or [f"Variant {i}" for i in range(len(visitors))]

        samples = self._sample_posteriors(visitors, conversions, n_samples)
        winner_indices = np.argmax(samples, axis=0)
        counts = np.bincount(winner_indices, minlength=len(visitors))
        prob_best_overall = counts / n_samples

        control_samples = samples[0]
        results = []
        for i in range(1, len(visitors)):
            challenger_samples = samples[i]
            prob_being_best = float(prob_best_overall[i])
            prob_beat_control = float((challenger_samples > control_samples).mean())
            expected_uplift = float(
                np.mean(np.maximum(challenger_samples - control_samples, 0))
            )
            expected_loss = float(
                np.mean(np.maximum(control_samples - challenger_samples, 0))
            )

            conclusion = self.generate_bayesian_conclusion(
                variant_name=labels[i],
                prob_beat_control=prob_beat_control,
                expected_uplift=expected_uplift,
                expected_risk=expected_loss,
                projection_period=30,
            )
            results.append(
                BayesianResult(
                    variant_label=labels[i],
                    control_label=labels[0],
                    prob_being_best=prob_being_best,
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
        variant_labels: List[str],  # Required to map the dict and build UI strings
        n_simulations: int = 50000,
    ) -> List[Dict[str, Any]]:
        """
        Translates conversion rate probabilities into monetary risk and uplift projections.
        """
        num_variants = len(visitors)
        if (
            biz_case.runtime_days <= 0
            or num_variants < 2
            or len(variant_labels) != num_variants
        ):
            return []

        samples = self._sample_posteriors(visitors, conversions, n_simulations)

        daily_vol_samples = (
            samples * np.array(visitors)[:, np.newaxis]
        ) / biz_case.runtime_days

        control_label = variant_labels[0]
        control_vol = daily_vol_samples[0]
        control_aov = biz_case.aovs.get(control_label, 0.0)
        results = []

        for i in range(1, num_variants):
            variant_label = variant_labels[i]
            variant_vol = daily_vol_samples[i]
            variant_aov = biz_case.aovs.get(variant_label, 0.0)

            diff = variant_vol - control_vol

            gain_samples = np.maximum(diff, 0)
            uplift = float(
                np.mean(gain_samples) * variant_aov * biz_case.projection_period
            )

            loss_samples = np.abs(np.minimum(diff, 0))
            risk = float(
                np.mean(loss_samples) * control_aov * biz_case.projection_period
            )

            prob_beat_control = float((diff > 0).mean())

            conclusion = self.generate_bayesian_conclusion(
                variant_name=variant_label,
                prob_beat_control=prob_beat_control,
                expected_uplift=uplift,
                expected_risk=risk,
                projection_period=biz_case.projection_period,
            )

            results.append(
                {
                    "variant_index": i,
                    "variant_label": variant_label,
                    "prob_beat_control": prob_beat_control,
                    "prob_best_overall": prob_best_overall[i],
                    "expected_uplift": uplift,
                    "expected_risk": risk,
                    "expected_total_contribution": uplift - risk,
                    "conclusion": conclusion,
                }
            )

        return results
