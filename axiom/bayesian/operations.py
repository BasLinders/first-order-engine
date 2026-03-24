import numpy as np
import string
from scipy.stats import beta
from typing import List, Dict, Any, Optional
from axiom.core.models import BusinessCaseInput

class BayesianEngine:
    """
    The Axiom Bayesian Engine handles Beta-Binomial posterior updates, 
    Monte Carlo simulations for 'Probability of Being Best', and 
    Decision Theory-based monetary risk projections.
    """

    def __init__(self, seed: Optional[int] = 42):
        if seed is not None:
            np.random.seed(seed)

    @staticmethod
    def _calculate_posterior_params(
        conversions: int, 
        visitors: int, 
        a_prior: float = 1.0, 
        b_prior: float = 1.0
    ) -> tuple[float, float]:
        """
        Calculates the posterior parameters for a Beta distribution.
        Logic: Alpha = prior + successes; Beta = prior + (trials - successes).
        """
        return a_prior + conversions, b_prior + (visitors - conversions)

    def run_probability_analysis(
        self, 
        visitors: List[int], 
        conversions: List[int], 
        n_samples: int = 100000
    ) -> Dict[str, Any]:
        """
        Calculates Probability of Being Best using vectorized Monte Carlo sampling.
        """
        num_variants = len(visitors)
        if num_variants == 0:
            return {"prob_being_best": [], "samples": np.array([])}

        # Vectorize parameter calculation
        visitors_arr = np.array(visitors)
        conversions_arr = np.array(conversions)
        
        # Using a default flat prior (1,1) if not specified
        a_post = 1.0 + conversions_arr
        b_post = 1.0 + (visitors_arr - conversions_arr)
        
        # Vectorized Sampling: Generates a matrix of shape (num_variants, n_samples)
        # np.random.beta is significantly faster than scipy.stats.beta.rvs for large arrays
        samples = np.random.beta(a_post[:, np.newaxis], b_post[:, np.newaxis], size=(num_variants, n_samples))
        
        # Identify the index of the max value across variants for each sample
        winner_indices = np.argmax(samples, axis=0)
        
        # Calculate probabilities of being best
        counts = np.bincount(winner_indices, minlength=num_variants)
        prob_being_best = (counts / n_samples).tolist()
        
        return {
            "prob_being_best": prob_being_best,
            "samples": samples
        }

    def run_monetary_projection(
        self,
        visitors: List[int],
        conversions: List[int],
        biz_case: BusinessCaseInput,
        prob_best_overall: List[float],
        n_simulations: int = 50000
    ) -> List[Dict[str, Any]]:
        """
        Translates conversion rate probabilities into monetary risk and uplift projections.
        Uses Decision Theory to calculate the 'Expected Value' of choosing a challenger.
        """
        num_variants = len(visitors)
        if biz_case.runtime_days <= 0 or num_variants < 2:
            return []

        # Fresh sampling for the decision context
        analysis = self.run_probability_analysis(visitors, conversions, n_samples=n_simulations)
        samples = analysis['samples']
        
        # Convert CR samples to 'Conversions Per Day'
        # Formula: (CR * visitors_to_date) / days_to_date
        daily_vol_samples = (samples * np.array(visitors)[:, np.newaxis]) / biz_case.runtime_days
        
        control_vol = daily_vol_samples[0]
        control_aov = biz_case.aovs[0]
        results = []

        for i in range(1, num_variants):
            variant_vol = daily_vol_samples[i]
            variant_aov = biz_case.aovs[i]
            
            # Daily difference in conversion volume compared to control
            diff = variant_vol - control_vol
            
            # 1. Expected Uplift (Mean of gains * AOV * Period)
            gain_samples = np.maximum(diff, 0)
            uplift = np.mean(gain_samples) * variant_aov * biz_case.projection_period
            
            # 2. Expected Risk (Mean of losses * AOV * Period)
            # We take the absolute value of the loss
            loss_samples = np.abs(np.minimum(diff, 0))
            risk = np.mean(loss_samples) * control_aov * biz_case.projection_period
            
            # 3. Probability to Beat Control (specific pairwise check)
            prob_beat_control = (diff > 0).mean()
            
            results.append({
                "Variant": string.ascii_uppercase[i],
                "Chance to Beat Control": round(float(prob_beat_control * 100), 2),
                "Chance to be Best Overall": round(float(prob_best_overall[i] * 100), 2),
                "Expected Monetary Uplift": round(float(uplift), 2),
                "Expected Monetary Risk": round(float(risk), 2),
                "Expected Total Contribution": round(float(uplift - risk), 2)
            })

        return results
