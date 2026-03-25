import string
import numpy as np
import pandas as pd
from scipy.stats import norm, beta
from typing import List, Dict, Tuple, Union, Any

# Updated imports to match our package structure
from axiom.core.models import AlternativeHypothesis

class VizEngine:
    """
    Translates statistical data into JSON-ready coordinate structures 
    for UI charting libraries (Altair, Plotly, Vega-Lite).
    """

    # --- Data Quality visualizations --- 
    
    @staticmethod
    def get_srm_viz_data(visitor_counts: List[int], expected_counts: List[float]) -> Dict[str, Any]:
        """Prepares data in a 'melted' format ready for grouped bar charts."""
        alphabet = string.ascii_uppercase
        num_variants = len(visitor_counts)
        melted_data = []
        
        for i in range(num_variants):
            variant_label = alphabet[i]
            melted_data.append({"Variant": variant_label, "Metric": "Observed", "Count": visitor_counts[i], "opacity": 1.0})
            melted_data.append({"Variant": variant_label, "Metric": "Expected", "Count": round(expected_counts[i]), "opacity": 0.4})
            
        return {
            "chart_data": melted_data,
            "chart_caption": "Comparison of observed traffic vs. expected allocation. Large visual discrepancies indicate an SRM."
        }

    # --- Frequentist visualizations ---

    @staticmethod
    def get_z_distribution_coords(mean: float, std_error: float, n_points: int = 100) -> Dict[str, Any]:
        """Generates X and Y coordinates for a Normal Distribution curve."""
        if std_error == 0:
            return {"x": [], "y": [], "chart_caption": "Invalid data: Standard error is 0."}

        x = np.linspace(mean - 4 * std_error, mean + 4 * std_error, n_points)
        y = norm.pdf(x, mean, std_error)
        
        return {
            "x": x.tolist(),
            "y": y.tolist(),
            "chart_caption": f"Normal distribution centered at {mean:.4f} with a standard error of {std_error:.4f}."
        }

    @staticmethod
    def get_power_curve_coords(
        control_cr: float,
        sample_size: int,
        alpha: float,
        alternative: AlternativeHypothesis,
        mde_range: Tuple[float, float] = (0.01, 0.2),
        n_points: int = 20
    ) -> Dict[str, Any]:
        """Calculates Power (Y) for various Minimum Detectable Effects (X)."""
        mde_steps = np.linspace(mde_range[0], mde_range[1], n_points)
        powers = []
        
        var_base = control_cr * (1 - control_cr)
        se_diff = np.sqrt((var_base / sample_size) + (var_base / sample_size))
        z_alpha = norm.ppf(1 - alpha / (2 if alternative == AlternativeHypothesis.TWO_SIDED else 1))
        
        target_mde_80 = None
        for mde in mde_steps:
            z_delta = (mde * control_cr) / se_diff
            power = float(norm.cdf(z_delta - z_alpha))
            powers.append(power)
            if target_mde_80 is None and power >= 0.80:
                target_mde_80 = mde
                
        caption = "Power Curve showing test sensitivity."
        if target_mde_80:
            caption = f"At this sample size, the test achieves standard 80% power for an MDE of ~{target_mde_80:.2%}."
            
        return {"mde": mde_steps.tolist(), "power": powers, "chart_caption": caption}

    @staticmethod
    def get_frequentist_forest_plot(variant_labels: List[str], uplifts: List[float], conf_intervals: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Formats uplift and CI data specifically for Whisker charts."""
        forest_data = []
        sig_count = 0
        for label, uplift, (low, high) in zip(variant_labels, uplifts, conf_intervals):
            is_sig = not (low <= 0 <= high)
            if is_sig: sig_count += 1
            forest_data.append({
                "label": label, "mean": float(uplift), "error_minus": float(uplift - low),
                "error_plus": float(high - uplift), "is_significant": is_sig
            })
            
        return {
            "chart_data": forest_data,
            "chart_caption": f"Effect sizes and 95% Confidence Intervals. {sig_count} variant(s) show a statistically significant effect."
        }

    # --- Bayesian Visualizations --- 

    @staticmethod
    def get_bayesian_density_coords(alpha: float, beta_param: float, n_points: int = 100) -> Dict[str, Any]:
        """Generates X (CR) and Y (Density) for the Beta distribution curve."""
        x_min, x_max = beta.ppf([0.001, 0.999], alpha, beta_param)
        x = np.linspace(x_min, x_max, n_points)
        y = beta.pdf(x, alpha, beta_param)
        return {"x": x.tolist(), "y": y.tolist()}

    @staticmethod
    def get_bayesian_winner_status(prob_variant_best: float, prob_control_best: float, threshold: float) -> Dict[str, str]:
        """Translates probabilities into semantic labels for UI rendering."""
        p_best_pct = prob_variant_best * 100
        p_ctrl_pct = prob_control_best * 100

        if p_best_pct >= threshold:
            return {"label": "winner", "color": "green", "class": "success", "summary": f"Clear Winner: Variant has a {p_best_pct:.1f}% chance of being best."}
        elif p_ctrl_pct >= threshold:
            return {"label": "loss averted", "color": "red", "class": "danger", "summary": f"Loss Averted: Control has a {p_ctrl_pct:.1f}% chance of being best."}
        else:
            return {"label": "inconclusive", "color": "black", "class": "neutral", "summary": "Inconclusive: Neither variant meets the probability threshold to declare a winner."}

    # --- Sequential ---

    @staticmethod
    def get_sequential_chart_data(trajectory_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Takes the output from SequentialEngine.process_test_trajectory 
        and maps it to UI coordinates.
        """
        if trajectory_df.empty:
            return {"trajectories": [], "upper_bound": 0.0, "lower_bound": 0.0, "chart_caption": "No sequential data available."}

        # The boundaries are constant per test, so grab them from the first row
        upper = float(trajectory_df['upper_bound'].iloc[0])
        lower = float(trajectory_df['lower_bound'].iloc[0])
        
        # Ensure dates are JSON serializable
        df_safe = trajectory_df.copy()
        if pd.api.types.is_datetime64_any_dtype(df_safe['measurement_date']):
            df_safe['measurement_date'] = df_safe['measurement_date'].dt.strftime('%Y-%m-%d')

        trajectories = []
        for variant in df_safe['variant_name'].unique():
            v_df = df_safe[df_safe['variant_name'] == variant]
            trajectories.append({
                "variant": variant,
                "data": v_df[['measurement_date', 'llr']].rename(columns={'measurement_date': 'date'}).to_dict(orient='records')
            })
            
        return {
            "upper_bound": upper,
            "lower_bound": lower,
            "trajectories": trajectories,
            "chart_caption": "Sequential Log-Likelihood Ratio (LLR) trajectory over time. Test stops when a trajectory crosses the upper or lower boundary."
        }

    # --- Interaction Analysis --- 

    @staticmethod
    def get_interaction_forest_plot(model) -> Dict[str, Any]:
        """Formats interaction model coefficients for a Forest Plot."""
        params = model.params[1:]
        conf = model.conf_int()[1:]
        
        forest_data = []
        interaction_count = 0
        
        for name, coef, low, high in zip(params.index, params.values, conf[0].values, conf[1].values):
            is_significant = not (low <= 0 <= high)
            if is_significant and ":" in str(name): 
                interaction_count += 1
                
            forest_data.append({
                "raw_label": str(name),
                "mean_effect": float(coef),
                "error_minus": float(coef - low),
                "error_plus": float(high - coef),
                "is_significant": is_significant,
                "color_state": "significant" if is_significant else "neutral"
            })
            
        caption = "Interaction Forest Plot."
        if interaction_count > 0:
            caption += f" Detected {interaction_count} statistically significant interaction(s) (synergies/clashes)."
            
        # Return sorted data (smallest effect size first for y-axis order)
        return {
            "chart_data": sorted(forest_data, key=lambda x: x['mean_effect']),
            "chart_caption": caption
        }

    @staticmethod
    def get_interaction_point_coords(df: pd.DataFrame, kpi: str, segment_column: str) -> Dict[str, Any]:
        """Calculates the mean KPI for every 'Variant X Segment' pair."""
        means = df.groupby(['experience_variant_label', segment_column], observed=True)[kpi].mean().reset_index()
        
        # Ensure floats for JSON serialization
        means[kpi] = means[kpi].astype(float)
        
        return {
            "chart_data": means.to_dict(orient='records'),
            "chart_caption": f"Average {kpi} broken down by Variant and {segment_column}."
        }
