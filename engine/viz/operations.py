import numpy as np
from scipy.stats import norm
from typing import List, Dict, Tuple
from engine.core.models import AlternativeHypothesis

# --- Frequentist visualizations ---

def get_z_distribution_coords(
    mean: float, 
    std_error: float, 
    n_points: int = 100
) -> Dict[str, List[float]]:
    """
    Generates X and Y coordinates for a Normal Distribution curve.
    Perfect for 'Overlap' or 'Bell Curve' charts in a UI.
    """
    if std_error == 0:
        return {"x": [], "y": []}

    # Range: +/- 4 Standard Errors
    x = np.linspace(mean - 4 * std_error, mean + 4 * std_error, n_points)
    y = norm.pdf(x, mean, std_error)
    
    return {
        "x": x.tolist(),
        "y": y.tolist()
    }

def get_power_curve_coords(
    control_cr: float,
    sample_size: int,
    alpha: float,
    alternative: AlternativeHypothesis,
    mde_range: Tuple[float, float] = (0.01, 0.2),
    n_points: int = 20
) -> Dict[str, List[float]]:
    """
    Calculates Power (Y) for various Minimum Detectable Effects (X).
    Allows the UI to draw a 'Power Sensitivity' curve.
    """
    mde_steps = np.linspace(mde_range[0], mde_range[1], n_points)
    powers = []
    
    # Standard Error for the difference (assuming equal split)
    # This is a simplified pre-test SE calculation
    var_base = control_cr * (1 - control_cr)
    se_diff = np.sqrt((var_base / sample_size) + (var_base / sample_size))
    
    z_alpha = norm.ppf(1 - alpha / (2 if alternative == AlternativeHypothesis.TWO_SIDED else 1))
    
    for mde in mde_steps:
        z_delta = (mde * control_cr) / se_diff
        power = norm.cdf(z_delta - z_alpha)
        powers.append(float(power))
        
    return {
        "mde": mde_steps.tolist(),
        "power": powers
    }

def get_forest_plot_data(
    variant_labels: List[str],
    uplifts: List[float],
    conf_intervals: List[Tuple[float, float]]
) -> List[Dict]:
    """
    Formats uplift and CI data specifically for Forest Plots (Whisker charts).
    """
    forest_data = []
    for label, uplift, (low, high) in zip(variant_labels, uplifts, conf_intervals):
        forest_data.append({
            "label": label,
            "mean": uplift,
            "error_minus": uplift - low,
            "error_plus": high - uplift
        })
    return forest_data
