import numpy as np
import string
from scipy.stats import beta
from typing import List, Dict, Tuple
from engine.core.models import BusinessCaseInput

def get_posterior_parameters(
    conversions: int, 
    visitors: int, 
    prior_alpha: float = 1.0, 
    prior_beta: float = 1.0
) -> Tuple[float, float]:
    """Updates the Beta distribution parameters (Alpha/Beta) based on data."""
    return prior_alpha + conversions, prior_beta + (visitors - conversions)

def run_bayesian_core(
    visitors: List[int],
    conversions: List[int],
    n_samples: int = 100000
) -> Dict:
    """
    Calculates the Probability of Being Best for all variants.
    Returns the samples matrix for further risk analysis.
    """
    all_samples = []
    for v, c in zip(visitors, conversions):
        a, b = get_posterior_parameters(c, v)
        all_samples.append(np.random.beta(a, b, n_samples))
    
    samples_matrix = np.array(all_samples)
    winner_indices = np.argmax(samples_matrix, axis=0)
    
    prob_being_best = [float(np.mean(winner_indices == i)) for i in range(len(visitors))]
    
    return {
        "prob_being_best": prob_being_best,
        "samples_matrix": samples_matrix
    }

def run_multi_variant_risk_assessment(
    visitors: List[int],
    conversions: List[int],
    biz_case: BusinessCaseInput,
    prob_to_be_best: List[float],
    n_simulations: int = 20000,
    seed: int = 42
) -> List[Dict]:
    """
    Monetary Risk Logic: Translates CR probabilities into 6-month revenue projections.
    """
    np.random.seed(seed)
    num_variants = len(visitors)
    if biz_case.runtime_days <= 0:
        return []

    # Generate Daily Conversion Volume Samples
    all_daily_samples = []
    for i in range(num_variants):
        a_post, b_post = get_posterior_parameters(conversions[i], visitors[i], biz_case.alpha_prior, biz_case.beta_prior)
        samples_cr = beta.rvs(a_post, b_post, size=n_simulations)
        daily_vol = (samples_cr * visitors[i]) / biz_case.runtime_days
        all_daily_samples.append(daily_vol)

    control_samples = all_daily_samples[0]
    control_aov = biz_case.aovs[0]
    results = []

    for i in range(1, num_variants):
        challenger_samples = all_daily_samples[i]
        challenger_aov = biz_case.aovs[i]
        diff_samples = challenger_samples - control_samples
        
        # Uplift Calculation
        prob_challenger_better = (diff_samples > 0).mean()
        pos_diffs = diff_samples[diff_samples > 0]
        expected_daily_gain = np.mean(pos_diffs) if len(pos_diffs) > 0 else 0
        uplift_monetary = expected_daily_gain * challenger_aov * biz_case.projection_period * prob_challenger_better
        
        # Risk Calculation
        prob_control_better = (diff_samples < 0).mean()
        neg_diffs = diff_samples[diff_samples < 0]
        expected_daily_loss = np.mean(neg_diffs) if len(neg_diffs) > 0 else 0
        risk_monetary = expected_daily_loss * control_aov * biz_case.projection_period * prob_control_better

        results.append({
            "Variant": string.ascii_uppercase[i],
            "Chance to Beat Control": round(prob_challenger_better * 100, 2),
            "Chance to be Best Overall": round(prob_to_be_best[i] * 100, 2),
            "Expected Monetary Uplift": round(float(uplift_monetary), 2),
            "Expected Monetary Risk": round(float(risk_monetary), 2),
            "Expected Total Contribution": round(float(uplift_monetary + risk_monetary), 2)
        })

    return results
