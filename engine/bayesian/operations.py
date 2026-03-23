import numpy as np
import pandas as pd
import string
from typing import List, Dict, Any, Optional

class BayesianEngine:
    """
    Engine for Beta-Binomial Bayesian A/B testing and Decision Theory risk modeling.
    """

    def __init__(self, seed: Optional[int] = 42):
        if seed:
            np.random.seed(seed)

    @staticmethod
    def _get_posterior_params(conversions: int, visitors: int, a_prior: float = 1.0, b_prior: float = 1.0):
        """Standard Beta-Binomial update logic."""
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
        # 1. Vectorized Posterior Parameter Calculation
        visitors_arr = np.array(visitors)
        conversions_arr = np.array(conversions)
        
        a_post = 1.0 + conversions_arr
        b_post = 1.0 + (visitors_arr - conversions_arr)
        
        # 2. Vectorized Sampling: Result is (num_variants, n_samples)
        # Using np.random.beta for maximum performance
        samples = np.random.beta(a_post[:, np.newaxis], b_post[:, np.newaxis], size=(len(visitors), n_samples))
        
        # 3. Identify Winners per sample
        winner_indices = np.argmax(samples, axis=0)
        
        # 4. Calculate Probabilities
        counts = np.bincount(winner_indices, minlength=len(visitors))
        prob_being_best = (counts / n_samples).tolist()
        
        return {
            "prob_being_best": prob_being_best,
            "samples": samples
        }

    def run_monetary_projection(
        self,
        visitors: List[int],
        conversions: List[int],
        biz_case: Any, # Expecting BusinessCaseInput model
        prob_best_overall: List[float],
        n_simulations: int = 50000
    ) -> List[Dict[str, Any]]:
        """
        Decision Theory logic: Projections based on 'Expected Loss' and 'Expected Gain'.
        """
        num_variants = len(visitors)
        # Probabilities from a fresh simulation for the Risk context
        analysis = self.run_probability_analysis(visitors, conversions, n_samples=n_simulations)
        samples = analysis['samples']
        
        # Convert CR samples to Volume samples (conversions per day)
        daily_vol_samples = (samples * np.array(visitors)[:, np.newaxis]) / biz_case.runtime_days
        
        control_vol = daily_vol_samples[0]
        control_aov = biz_case.aovs[0]
        results = []

        for i in range(1, num_variants):
            variant_vol = daily_vol_samples[i]
            variant_aov = biz_case.aovs[i]
            
            # Difference in daily conversions
            diff = variant_vol - control_vol
            
            # Uplift: Mean of positive differences * AOV * Days
            # This is the 'Expected Value' of the gain
            gain_samples = np.maximum(diff, 0)
            uplift = np.mean(gain_samples) * variant_aov * biz_case.projection_period
            
            # Risk: Mean of negative differences (as positive value) * AOV * Days
            # This is the 'Expected Loss' (Bayesian Risk)
            loss_samples = np.abs(np.minimum(diff, 0))
            risk = np.mean(loss_samples) * control_aov * biz_case.projection_period
            
            results.append({
                "Variant": string.ascii_uppercase[i],
                "Prob to Beat Control": float((diff > 0).mean()),
                "Prob to be Best": prob_best_overall[i],
                "Expected Uplift": round(float(uplift), 2),
                "Expected Risk": round(float(risk), 2),
                "Net Contribution": round(float(uplift - risk), 2)
            })

        return results
