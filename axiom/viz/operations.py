import string
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import List, Dict, Tuple, Union
from engine.core.models import AlternativeHypothesis

# --- Data Quality visualizations --- 
def get_srm_viz_data(
    visitor_counts: List[int], 
    expected_counts: List[float]
) -> List[Dict]:
    """
    Prepares data in a 'melted' format ready for Altair grouped bar charts.
    Matches the render_results logic in the UI.
    """
    alphabet = string.ascii_uppercase
    num_variants = len(visitor_counts)
    
    melted_data = []
    
    for i in range(num_variants):
        variant_label = alphabet[i]
        
        # Add Observed record
        melted_data.append({
            "Variant": variant_label,
            "Metric": "Observed",
            "Count": visitor_counts[i],
            "opacity": 1.0
        })
        
        # Add Expected record
        melted_data.append({
            "Variant": variant_label,
            "Metric": "Expected",
            "Count": round(expected_counts[i]),
            "opacity": 0.4
        })
        
    return melted_data

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

# --- Bayesian Visualizations --- 

from scipy.stats import beta
from typing import List, Dict, Tuple

def get_bayesian_density_coords(
    alpha: float, 
    beta_param: float, 
    n_points: int = 100
) -> Dict[str, List[float]]:
    """
    Generates X (CR) and Y (Density) for the Beta distribution curve.
    Focuses on the 99.9% density region for a clean 'bell' look.
    """
    # Zoom into the relevant area of the distribution
    x_min, x_max = beta.ppf([0.001, 0.999], alpha, beta_param)
    x = np.linspace(x_min, x_max, n_points)
    y = beta.pdf(x, alpha, beta_param)
    
    return {"x": x.tolist(), "y": y.tolist()}

def get_bayesian_winner_status(
    prob_variant_best: float,
    prob_control_best: float,
    threshold: float
) -> Dict[str, str]:
    """
    Translates probabilities into semantic labels for UI rendering.
    """
    p_best_pct = prob_variant_best * 100
    p_ctrl_pct = prob_control_best * 100

    if p_best_pct >= threshold:
        return {"label": "winner", "color": "green", "class": "success"}
    elif p_ctrl_pct >= threshold:
        return {"label": "loss averted", "color": "red", "class": "danger"}
    else:
        return {"label": "inconclusive", "color": "black", "class": "neutral"}

# --- Sequential (LLR Always Vailid Approach) ---

def get_sequential_viz_data(
    data_points: List[Dict], 
    config: SequentialConfig
) -> Dict:
    """
    Transforms raw time-series data into LLR trajectories for plotting.
    Handles both One-sample and Multi-sample logic.
    """
    import pandas as pd
    df = pd.DataFrame(data_points)
    upper, lower = calculate_msprt_boundaries(config.alpha, config.beta, config.num_variants)
    
    trajectories = []
    variants = [v for v in df['variant_name'].unique() if v != "Control"]
    
    for variant in variants:
        v_df = df[df['variant_name'] == variant].copy()
        
        if config.p0 is not None: # One-sample logic
            v_df['llr'] = v_df.apply(lambda r: calculate_msprt_llr(
                r['visitors'], int(r['visitors'] * config.p0), 
                r['visitors'], r['conversions'], config.tau
            ), axis=1)
        else: # Multi-sample logic
            ctrl_df = df[df['variant_name'] == 'Control']
            merged = pd.merge(v_df, ctrl_df, on='date', suffixes=('_var', '_ctrl'))
            merged['llr'] = merged.apply(lambda r: calculate_msprt_llr(
                r['visitors_ctrl'], r['conversions_ctrl'],
                r['visitors_var'], r['conversions_var'], config.tau
            ), axis=1)
            v_df = merged
            
        trajectories.append(v_df[['date', 'variant_name', 'llr']].to_dict(orient='records'))

    return {
        "upper_bound": upper,
        "lower_bound": lower,
        "trajectories": trajectories
    }

# --- Interaction Analysis visualizations --- 

class InteractionVizEngine:
    """
    Prepares JSON-ready data structures for visualizing test interactions.
    Returns coordinate data.
    """
    @staticmethod
    def get_forest_plot_data(model) -> List[Dict[str, Union[str, float, bool]]]:
        """
        Calculates coefficients and 95% Confidence Intervals for a Forest Plot.
        Matches the logic used in Bas's Interaction Analysis app.
        """
        # Extract params and confidence intervals (ignoring Intercept)
        # Using model.conf_int() to get the [0.025, 0.975] bounds
        params = model.params[1:]
        conf = model.conf_int()[1:]
        
        results_df = pd.DataFrame({
            'Feature': params.index,
            'Coefficient': params.values,
            'Lower': conf[0].values,
            'Upper': conf[1].values
        })

        forest_data = []
        
        # Determine significance and color state
        for _, row in results_df.iterrows():
            # A result is significant if the CI does not cross zero (the null hypothesis)
            is_significant = not (row['Lower'] <= 0 <= row['Upper'])
            
            # Semantic color state (matches common Streamlit themes)
            color_state = "significant" if is_significant else "neutral"

            forest_data.append({
                "label": row['Feature'], # The cleaned name from format_summary_table
                "mean_effect": float(row['Coefficient']), # Log-Odds
                "error_minus": float(row['Coefficient'] - row['Lower']),
                "error_plus": float(row['Upper'] - row['Coefficient']),
                "is_significant": is_significant,
                "color_state": color_state
            })
            
        # Return sorted data (smallest effect size first for y-axis order)
        return sorted(forest_data, key=lambda x: x['mean_effect'])

    @staticmethod
    def get_interaction_plot_coords(df, kpi, segment_column) -> List[Dict[str, Union[str, float]]]:
        """
        Calculates the mean KPI for every 'Variant X Segment' pair.
        Returns data ready for a standard interaction/point plot.
        """
        # Group by both dimensions and calculate the average
        # Using observed=True to prevent issues with empty categorical groups
        means = df.groupby(['experience_variant_label', segment_column], observed=True)[kpi].mean().reset_index()
        
        # Return a simple list of dictionaries (records)
        return means.to_dict(orient='records')
