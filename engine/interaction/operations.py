import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import List


# Pre-compiled regex for parsing statsmodels coefficient names like:
# "C(test1)[T.VariantB]"  →  group(1)="test1", group(2)="VariantB"
_COEF_TERM_RE = re.compile(r"C\((\w+)\)\[T\.([^\]]+)\]")

class InteractionEngine:
    """
    Analyzes synergies and clashes between concurrent A/B tests.

    Uses a Generalized Linear Model (GLM) with a Binomial family.
    The model is fit on pre-aggregated (conversions, visitors) count data
    using a two-column binomial response, which produces correct standard
    errors and p-values without inflating the effective sample size.
    """

    # Maximum number of concurrent tests allowed in a single model.
    # A full factorial model produces 2^N terms; beyond 4 tests the model
    # becomes numerically unstable and very hard to interpret.
    MAX_TESTS = 4

    @staticmethod
    def prepare_aggregated_format(
        input_df: pd.DataFrame, test_cols: List[str]
    ) -> pd.DataFrame:
        """
        Prepares a cleaned, aggregated dataframe for model fitting.

        Each row represents one unique combination of test variants.
        The response is kept as (conversions, non_conversions) — a two-column
        binomial response — which is the statistically correct approach for
        pre-aggregated count data.

        Parameters
        ----------
        input_df : pd.DataFrame
            Must contain the test variant columns plus 'visitors' and
            'conversions' integer columns.
        test_cols : list[str]
            Column names identifying the A/B test variant assignments.

        Returns
        -------
        pd.DataFrame
            Cleaned dataframe with string-typed variant columns and an added
            'non_conversions' column.
        """
        df = input_df.copy()

        # Cast variant columns to str so statsmodels treats them as categorical
        for col in test_cols:
            df[col] = df[col].astype(str)

        df["non_conversions"] = df["visitors"] - df["conversions"]
        return df

    def fit_interaction_model(self, df: pd.DataFrame, test_cols: List[str]):
        """
        Fits a full-factorial Logistic Regression (GLM-Binomial) model.

        Formula: cbind(conversions, non_conversions) ~ Test1 * Test2 * ... * TestN

        Using a two-column binomial response (successes, failures) is correct
        for aggregated count data; it does NOT inflate the effective N the way
        freq_weights does, so standard errors and p-values are reliable.

        Parameters
        ----------
        df : pd.DataFrame
            Output of :meth:`prepare_aggregated_format`.
        test_cols : list[str]
            Column names for each concurrent test.

        Returns
        -------
        statsmodels GLMResultsWrapper

        Raises
        ------
        ValueError
            On invalid inputs or model fitting failure.
        """
        self._validate_inputs(df, test_cols)

        # Two-column binomial response: shape (n_rows, 2)
        endog = df[["conversions", "non_conversions"]].to_numpy()

        # Full factorial formula — "*" expands to all main effects + interactions
        formula_rhs = " * ".join([f"C({col})" for col in test_cols])

        try:
            model = sm.GLM(
                endog=endog,
                exog=sm.formula.api.formulaic_matrix(f"~ {formula_rhs}", data=df),
                family=sm.families.Binomial(),
            ).fit()
        except Exception as e:
            raise ValueError(f"Interaction model fitting failed: {e}") from e

        return model

    def fit_interaction_model_from_formula(
        self, df: pd.DataFrame, test_cols: List[str]
    ):
        """
        Alternative entry point using statsmodels formula API directly.

        Preferred when you want statsmodels to handle the design matrix
        construction (handles reference-level encoding automatically).

        Parameters
        ----------
        df : pd.DataFrame
            Output of :meth:`prepare_aggregated_format`.
        test_cols : list[str]
            Column names for each concurrent test.

        Returns
        -------
        statsmodels GLMResultsWrapper
        """
        self._validate_inputs(df, test_cols)

        formula_rhs = " * ".join([f"C({col})" for col in test_cols])
        formula = f"conversions + non_conversions ~ {formula_rhs}"

        try:
            model = sm.formula.glm(
                formula=formula,
                data=df,
                family=sm.families.Binomial(),
            ).fit()
        except Exception as e:
            raise ValueError(f"Interaction model fitting failed: {e}") from e

        return model

    @staticmethod
    def format_summary_table(model) -> pd.DataFrame:
        """
        Renames raw statsmodels coefficient names into human-readable labels.

        Examples
        --------
        ``Intercept``                               → "Baseline (Control Group)"
        ``C(test1)[T.B]``                           → "test1 (B)"
        ``C(test1)[T.B]:C(test2)[T.Y]``             → "test1 (B) & test2 (Y) — Clash/Synergy"

        Parameters
        ----------
        model : statsmodels GLMResultsWrapper

        Returns
        -------
        pd.DataFrame
            A copy of the coefficient summary table with renamed index.
        """
        # .copy() prevents mutating the object held inside the model result
        summary = model.summary2().tables[1].copy()

        summary.index = summary.index.map(InteractionEngine._rename_coefficient)
        return summary

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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
                "Some rows have more conversions than visitors. "
                "Check your input data."
            )

        if df[test_cols].isnull().any().any():
            raise ValueError("Test variant columns contain null values.")

    @staticmethod
    def _rename_coefficient(name: str) -> str:
        """
        Converts a single statsmodels coefficient name to a readable label.

        Uses a pre-compiled regex rather than chained string replacements so
        that changes to statsmodels' formatting surface as unmatched patterns
        (returned verbatim) rather than silently garbled names.
        """
        if name == "Intercept":
            return "Baseline (Control Group)"

        if ":" in name:
            # Interaction term: one part per test involved
            parts = name.split(":")
            clean_parts = []
            for part in parts:
                m = _COEF_TERM_RE.fullmatch(part.strip())
                clean_parts.append(
                    f"{m.group(1)} ({m.group(2)})" if m else part.strip()
                )
            return " & ".join(clean_parts) + " — Clash/Synergy"

        # Main effect term
        m = _COEF_TERM_RE.fullmatch(name.strip())
        if m:
            return f"{m.group(1)} ({m.group(2)})"

        # Fallback: return verbatim so nothing silently disappears
        return name
