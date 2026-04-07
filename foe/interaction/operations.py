import re
import pandas as pd
import statsmodels.api as sm
import patsy
from typing import List, Dict, Any

# Pre-compiled regex for parsing statsmodels coefficient names like:
# "C(test1)[T.VariantB]"  →  group(1)="test1", group(2)="VariantB"
_COEF_TERM_RE = re.compile(r"C\((\w+)\)\[T\.([^\]]+)\]")


class InteractionEngine:
    """
    Analyzes synergies and clashes between concurrent A/B tests.
    Strictly returns JSON-serializable primitives for Cloud APIs.
    """

    # A full factorial model produces 2^N terms; beyond 4 tests the model
    # becomes numerically unstable and very hard to interpret.
    MAX_TESTS = 4

    @staticmethod
    def generate_interaction_conclusion(
        term_label: str, coef: float, p_value: float, alpha: float = 0.05
    ) -> str:
        """
        Generates definitive business statements for main effects and interactions.
        """
        is_significant = bool(p_value < alpha)
        is_interaction = "Clash/Synergy" in term_label

        if not is_interaction:
            if not is_significant:
                return "Flat: This variant does not have a statistically significant independent effect."
            direction = "Positive" if coef > 0 else "Negative"
            return f"Significant {direction} Independent Effect: This variant significantly alters conversion rates on its own."

        # Interaction Logic
        if not is_significant:
            return "Independent: These variants do not significantly interfere with each other. It is safe to run them concurrently."

        if coef > 0:
            return (
                "Synergy Detected: Combining these variants yields a higher conversion rate "
                "than the sum of their individual effects. Highly recommended to deploy together."
            )
        else:
            return (
                "Clash/Cannibalization: Combining these variants hurts overall performance, "
                "yielding worse results than expected. Do not deploy these variants to the same users."
            )

    @staticmethod
    def prepare_aggregated_format(
        input_df: pd.DataFrame, test_cols: List[str]
    ) -> pd.DataFrame:
        """Prepares a cleaned, aggregated dataframe for model fitting."""
        df = input_df.copy()
        for col in test_cols:
            df[col] = df[col].astype(str)
        df["non_conversions"] = df["visitors"] - df["conversions"]
        return df

    def run_interaction_analysis(
        self, df: pd.DataFrame, test_cols: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Orchestrates preparation, model fitting, and JSON-safe extraction.
        Builds the design matrix safely using patsy to avoid formula parser bugs.
        """
        self._validate_inputs(df, test_cols)
        processed_df = self.prepare_aggregated_format(df, test_cols)

        # 1. Prepare 2D Endogenous Variable (Response)
        endog = processed_df[["conversions", "non_conversions"]].to_numpy()

        # 2. Build Exogenous Design Matrix via patsy
        formula_rhs = " * ".join([f"C({col})" for col in test_cols])
        exog = patsy.dmatrix(
            f"~ {formula_rhs}", data=processed_df, return_type="dataframe"
        )

        try:
            model = sm.GLM(
                endog=endog,
                exog=exog,
                family=sm.families.Binomial(),
            ).fit()
        except Exception as e:
            raise ValueError(f"Interaction model fitting failed: {e}") from e

        # 3. Extract and parse results into JSON format
        return self._format_summary_table(model)

    def _format_summary_table(self, model) -> List[Dict[str, Any]]:
        """
        Parses the raw statsmodels summary into a JSON-ready list of dicts.
        """
        summary_df = model.summary2().tables[1].copy()
        results = []

        for raw_name, row in summary_df.iterrows():
            clean_name = self._rename_coefficient(str(raw_name))
            coef = float(row["Coef."])
            p_val = float(row["P>|z|"])

            # Skip the intercept/baseline conclusion as it represents the raw control state
            conclusion = (
                "Baseline Group"
                if clean_name == "Baseline (Control Group)"
                else self.generate_interaction_conclusion(
                    term_label=clean_name, coef=coef, p_value=p_val
                )
            )

            results.append(
                {
                    "term": clean_name,
                    "raw_term": str(raw_name),
                    "coefficient": coef,
                    "std_err": float(row["Std.Err."]),
                    "z_score": float(row["z"]),
                    "p_value": p_val,
                    "is_significant": bool(p_val < 0.05),
                    "conclusion": conclusion,
                }
            )

        return results

    def _validate_inputs(self, df: pd.DataFrame, test_cols: List[str]) -> None:
        """Raises ValueError for any input that would cause a bad model fit."""
        if not test_cols:
            raise ValueError("test_cols must contain at least one column name.")

        if len(test_cols) > self.MAX_TESTS:
            raise ValueError(
                f"Cannot fit a factorial model with {len(test_cols)} tests "
                f"(maximum is {self.MAX_TESTS}). The model would produce "
                f"{2 ** len(test_cols)} terms and become numerically unstable."
            )

        missing_cols = [
            c for c in [*test_cols, "visitors", "conversions"] if c not in df.columns
        ]
        if missing_cols:
            raise ValueError(f"Required columns missing from dataframe: {missing_cols}")

        if df["conversions"].lt(0).any():
            raise ValueError("'conversions' column contains negative values.")

        if df["visitors"].lt(0).any():
            raise ValueError("'visitors' column contains negative values.")

        if (df["conversions"] > df["visitors"]).any():
            raise ValueError(
                "Some rows have more conversions than visitors. Check your input data."
            )

        if df[test_cols].isnull().any().any():
            raise ValueError("Test variant columns contain null values.")

    @staticmethod
    def _rename_coefficient(name: str) -> str:
        """Converts a single statsmodels coefficient name to a readable label."""
        if name == "Intercept":
            return "Baseline (Control Group)"

        if ":" in name:
            parts = name.split(":")
            clean_parts = []
            for part in parts:
                m = _COEF_TERM_RE.fullmatch(part.strip())
                clean_parts.append(
                    f"{m.group(1)} ({m.group(2)})" if m else part.strip()
                )
            return " & ".join(clean_parts) + " — Clash/Synergy"

        m = _COEF_TERM_RE.fullmatch(name.strip())
        if m:
            return f"{m.group(1)} ({m.group(2)})"

        return name
