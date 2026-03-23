import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from typing import List, Dict, Any

class InteractionEngine:
    """
    Analyzes synergies and clashes between concurrent A/B tests.
    Uses a Generalized Linear Model (GLM) with Binomial family.
    """

    @staticmethod
    def prepare_long_format(input_df: pd.DataFrame, test_cols: List[str]) -> pd.DataFrame:
        """
        Converts aggregated 'Visitor/Conversion' counts into a long-format 
        dataframe suitable for statsmodels frequency weights.
        """
        rows = []
        for _, row in input_df.iterrows():
            # Add Successes (Conversions)
            rows.append({
                **{col: str(row[col]) for col in test_cols}, 
                'conversion': 1, 
                'count': row['conversions']
            })
            # Add Failures (Non-conversions)
            rows.append({
                **{col: str(row[col]) for col in test_cols}, 
                'conversion': 0, 
                'count': row['visitors'] - row['conversions']
            })
        return pd.DataFrame(rows)

    def fit_interaction_model(self, df: pd.DataFrame, test_cols: List[str]):
        """
        Fits a Factorial Logistic Regression model.
        Formula: conversion ~ Test1 * Test2 * ... * TestN
        """
        # Create formula with interaction terms (*)
        formula = "conversion ~ " + " * ".join([f"C({col})" for col in test_cols])
        
        try:
            model = sm.GLM.from_formula(
                formula, 
                data=df, 
                family=sm.families.Binomial(), 
                freq_weights=df['count']
            ).fit()
            
            return model
        except Exception as e:
            raise ValueError(f"Interaction Model Error: {e}")

    @staticmethod
    def format_summary_table(model) -> pd.DataFrame:
        """
        Renames raw statsmodels coefficients into readable variant and 
        interaction names for the UI.
        """
        summary = model.summary2().tables[1]
        new_names = {}
        
        for old_name in summary.index:
            if old_name == 'Intercept':
                new_names[old_name] = 'Baseline (Control Group)'
            elif ':' in old_name:
                # Interaction: C(test1)[T.B]:C(test2)[T.Y] -> test1 (B) & test2 (Y) Interaction
                parts = old_name.split(':')
                clean_parts = [p.replace('C(', '').replace(')', '').replace('[T.', ' (').replace(']', ')') for p in parts]
                new_names[old_name] = " & ".join(clean_parts) + " Clash/Synergy"
            else:
                # Main Effect: C(test1)[T.B] -> test1 (B)
                clean_name = old_name.replace('C(', '').split(')')[0]
                variant_val = old_name.split('[T.')[1].replace(']', '')
                new_names[old_name] = f"{clean_name} ({variant_val})"
        
        summary.index = summary.index.map(new_names)
        return summary
