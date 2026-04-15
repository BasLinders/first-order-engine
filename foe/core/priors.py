import numpy as np
from typing import Tuple

def get_informed_beta_prior(
    hist_conversions: int, 
    hist_visitors: int, 
    weight: float = 0.1
) -> Tuple[float, float]:
    """
    Calculates Alpha and Beta parameters for a Beta Distribution based on 
    historical performance and a confidence weight.
    
    Logic:
    - Weight 0.0: Total ignorance (Beta(1,1)).
    - Weight 1.0: Full historical confidence (Highly skeptical of new data).
    """
    weight = np.clip(weight, 0.0, 1.0)
    
    # Laplace smoothing
    alpha = (hist_conversions * weight) + 1
    beta = ((hist_visitors - hist_conversions) * weight) + 1
    
    return float(alpha), float(beta)
