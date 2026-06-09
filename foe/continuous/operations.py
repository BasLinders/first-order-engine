import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro, levene, kruskal, mannwhitneyu
from scipy.optimize import minimize
from scipy import stats
from pingouin import welch_anova

from foe.core.models import (
    AnalysisUnit,
    ContinuousApproach,
    ContinuousMetricConfig,
    ContinuousMetricResult,
    GammaPosthocResult,
)
from foe.core.validators import validate_continuous_data


class ContinuousMetricEngine:
    """
    Engine for analyzing continuous metrics (Revenue, Profit, Quantity).

    Two analysis paths:
      * Heuristic    -> normality/variance decision tree (ANOVA / Welch /
                        Kruskal-Wallis / Mann-Whitney).
      * Gamma family -> likelihood-ratio test on fitted distributions. The
                        analysis unit selects the model:
                          - per_transaction: a two-parameter Gamma per variant.
                          - per_visitor:     a two-part hurdle model per variant
                                             (Bernoulli convert x Gamma spend),
                                             which keeps non-converting zeros.

    The row-level data is a ``pandas.DataFrame``; all settings arrive via a
    frozen ``ContinuousMetricConfig`` and the result is a frozen, JSON-
    serializable ``ContinuousMetricResult``.
    """

    # ------------------------------------------------------------------ #
    # Conclusion text
    # ------------------------------------------------------------------ #

    @staticmethod
    def generate_continuous_conclusion(
        kpi: str,
        is_significant: bool,
        test_used: str,
        p_value: float,
        unit: AnalysisUnit = AnalysisUnit.PER_TRANSACTION,
    ) -> str:
        """Generates a definitive, UI-agnostic summary for continuous evaluation."""
        unit_phrase = (
            "revenue per visitor" if unit == AnalysisUnit.PER_VISITOR
            else "value per transaction"
        )
        if not is_significant:
            return (
                f"No Significant Difference: Based on the {test_used} (p={p_value:.3f}), "
                f"there is no statistically significant difference in {kpi} "
                f"({unit_phrase}) across the variants. The metric performs similarly "
                "across all groups."
            )

        return (
            f"Significant Variance Detected: The {test_used} (p={p_value:.3f}) detected a "
            f"statistically significant difference in {kpi} ({unit_phrase}) between the "
            "variants. Review the summary statistics to identify the winning group."
        )

    # ------------------------------------------------------------------ #
    # Outlier helpers (caller-orchestrated; left intact)
    # ------------------------------------------------------------------ #

    @staticmethod
    def detect_outliers_ols(
        df: pd.DataFrame, kpi: str, threshold: float = 3.0
    ) -> list[bool]:
        """
        Uses OLS residuals and influence to detect outliers.
        Returns a boolean list (JSON serializable) instead of the model.
        """
        if df.empty or len(df[kpi].dropna()) < 3:
            return [False] * len(df)

        model = smf.ols(f"{kpi} ~ C(experience_variant_label)", data=df).fit()
        influence = model.get_influence()

        std_resid = np.abs(influence.resid_studentized_internal) > threshold
        leverage = influence.hat_matrix_diag > (
            threshold * (model.df_model + 1) / len(df)
        )
        dffits = np.abs(influence.dffits[0]) > (
            threshold * np.sqrt((model.df_model + 1) / len(df))
        )

        outlier_mask = std_resid | leverage | dffits
        return outlier_mask.tolist()  # Safe for JSON

    @staticmethod
    def winsorize_series(
        series: pd.Series, method: str = "Standard Deviation", param: float = 3.0
    ) -> tuple[list[float], float, float]:
        """Caps outliers without removing data points."""
        series_clean = series.dropna()
        if series_clean.empty:
            return series.tolist(), 0.0, 0.0

        if method == "Standard Deviation":
            mean_val = series_clean.mean()
            std_val = series_clean.std()
            lower = mean_val - (param * std_val)
            upper = mean_val + (param * std_val)
        else:  # Percentile (e.g., param=1 means 1st and 99th percentile)
            lower_p = param
            upper_p = 100.0 - param
            lower, upper = np.percentile(series_clean, [lower_p, upper_p])

        clipped = series.clip(lower, upper)
        return clipped.tolist(), float(lower), float(upper)

    # ------------------------------------------------------------------ #
    # Gamma fitting
    # ------------------------------------------------------------------ #

    @staticmethod
    def neg_log_likelihood(params, data):
        """Negative log-likelihood function for the Gamma distribution."""
        k, theta = params
        if k <= 0 or theta <= 0:
            return np.inf
        return -np.sum(stats.gamma.logpdf(data, a=k, scale=theta))

    @staticmethod
    def fit_gamma(data) -> tuple[float, float, float]:
        """Fits a two-parameter Gamma (shape k, scale theta) by MLE.
        Returns (k, theta, log_likelihood)."""
        data_clean = pd.Series(data).dropna()
        # Initial guess using Method of Moments
        mean_val = data_clean.mean()
        var_val = data_clean.var()
        initial_params = [mean_val**2 / var_val, var_val / mean_val]

        result = minimize(
            ContinuousMetricEngine.neg_log_likelihood,
            initial_params,
            args=(data_clean.to_numpy(),),
            bounds=((1e-5, None), (1e-5, None)),
        )
        k, theta = result.x
        log_lik = -result.fun
        return float(k), float(theta), float(log_lik)

    @staticmethod
    def fit_unit_model(data, unit: AnalysisUnit) -> tuple[float, int]:
        """
        Fit the likelihood model for the chosen unit and return
        (log_likelihood, num_parameters).

        per_transaction -> a single two-parameter Gamma on the positive values
                           (2 params: k, theta).
        per_visitor     -> a two-part hurdle that *accepts zeros*: a Bernoulli on
                           P(value > 0) plus a Gamma on the positive values. The
                           log-likelihoods add; parameters are (p, k, theta) = 3.
        """
        arr = pd.Series(data).dropna().to_numpy(dtype=float)

        if unit == AnalysisUnit.PER_VISITOR:
            n = arr.size
            positives = arr[arr > 0]
            n_pos = positives.size
            if n == 0:
                return float("nan"), 3

            # Bernoulli (conversion) part at its MLE p = n_pos / n.
            p = n_pos / n
            if 0.0 < p < 1.0:
                ll_bern = n_pos * np.log(p) + (n - n_pos) * np.log(1.0 - p)
            else:
                # p == 0 (no buyers) or p == 1 (no zeros): Bernoulli LL is 0 at the MLE.
                ll_bern = 0.0

            # Gamma (spend among buyers) part.
            if n_pos >= 2 and np.var(positives) > 0:
                _, _, ll_gamma = ContinuousMetricEngine.fit_gamma(positives)
            else:
                ll_gamma = 0.0

            return float(ll_bern + ll_gamma), 3

        # per_transaction
        positives = arr[arr > 0]
        if positives.size < 2 or np.var(positives) == 0:
            return float("nan"), 2
        _, _, ll_gamma = ContinuousMetricEngine.fit_gamma(positives)
        return float(ll_gamma), 2

    @staticmethod
    def run_gamma_posthoc(
        df: pd.DataFrame,
        kpi: str,
        group_col: str,
        control_label: str,
        unit: AnalysisUnit = AnalysisUnit.PER_TRANSACTION,
        alpha: float = 0.05,
    ) -> list[GammaPosthocResult]:
        """Pairwise LRTs of each treatment against the control, Bonferroni-adjusted.

        Uses the same unit-appropriate model as the global test, so the pairwise
        degrees of freedom equal the per-group parameter count (2 for a Gamma,
        3 for the two-part hurdle).
        """
        variants = [v for v in df[group_col].unique() if v != control_label]
        num_comparisons = len(variants)
        posthoc_results: list[GammaPosthocResult] = []

        for variant in variants:
            pair_df = df[df[group_col].isin([control_label, variant])]

            # Null: a single model for both groups combined.
            null_ll, n_params = ContinuousMetricEngine.fit_unit_model(
                pair_df[kpi], unit
            )
            # Alternative: separate models per group.
            ctrl = pair_df[pair_df[group_col] == control_label][kpi]
            var = pair_df[pair_df[group_col] == variant][kpi]
            ll_ctrl, _ = ContinuousMetricEngine.fit_unit_model(ctrl, unit)
            ll_var, _ = ContinuousMetricEngine.fit_unit_model(var, unit)
            alt_ll = ll_ctrl + ll_var

            if np.isnan(null_ll) or np.isnan(alt_ll):
                lrt_stat = 0.0
                p_val = 1.0
            else:
                # df = params(Alt) - params(Null) = 2*n_params - n_params = n_params.
                lrt_stat = max(2.0 * (alt_ll - null_ll), 0.0)
                p_val = float(stats.chi2.sf(lrt_stat, df=n_params))

            adj_p = float(min(p_val * num_comparisons, 1.0))
            posthoc_results.append(
                GammaPosthocResult(
                    comparison=f"{variant} vs {control_label}",
                    lrt_stat=float(lrt_stat),
                    p_value=float(p_val),
                    p_adj_bonferroni=adj_p,
                    is_significant=bool(adj_p < alpha),
                )
            )

        return posthoc_results

    # ------------------------------------------------------------------ #
    # Main decision engine
    # ------------------------------------------------------------------ #

    def run_comparison_suite(
        self, df: pd.DataFrame, config: ContinuousMetricConfig
    ) -> ContinuousMetricResult:
        """
        Core engine: validate, apply unit-driven zero handling, then run the
        requested analysis path and return a typed result.

        Outlier handling is the caller's responsibility; pass an already
        winsorized / outlier-removed frame if desired. The unit zero-filter
        composes on top of whatever frame is supplied.
        """
        kpi = config.kpi
        group_col = config.group_col
        unit = config.unit
        approach = config.approach
        alpha = config.alpha

        # 1. Validate (raises ValueError -> 422 upstream).
        validate_continuous_data(
            df, kpi, group_col, approach=approach.value, unit=unit.value
        )

        warnings: list[str] = []

        # 2. Coerce / clean, then apply unit-driven zero handling.
        work = df[[group_col, kpi]].copy()
        work[kpi] = pd.to_numeric(work[kpi], errors="coerce")
        work = work.dropna(subset=[kpi])

        n_zeros = int((work[kpi] == 0).sum())
        if unit == AnalysisUnit.PER_TRANSACTION:
            if n_zeros > 0:
                work = work[work[kpi] != 0]
                warnings.append(
                    f"Per-transaction analysis excluded {n_zeros} zero-value "
                    "(non-order) row(s)."
                )
        else:  # per_visitor
            if n_zeros == 0:
                warnings.append(
                    "No zero-value rows found; per-visitor and per-transaction "
                    "analysis are equivalent for this data."
                )

        groups = [
            g[kpi].to_numpy()
            for _, g in work.groupby(group_col, observed=True)
        ]
        num_groups = len(groups)

        # Summary statistics (computed on the analysis population).
        summary_df = work.groupby(group_col, observed=True)[kpi].agg(
            ["mean", "std", "count"]
        )
        summary_stats = summary_df.reset_index().to_dict(orient="records")

        # 3. Branch once on the approach.
        if approach == ContinuousApproach.GAMMA_GLM:
            test_name, p_value, posthoc = self._run_gamma_path(
                work, kpi, group_col, groups, num_groups, unit, alpha,
                config.control_label, warnings,
            )
            is_normal = None
            is_homogeneous = None
        else:
            test_name, p_value, is_normal, is_homogeneous = self._run_heuristic_path(
                work, kpi, group_col, groups, num_groups,
            )
            posthoc = None

        p_value = float(min(max(p_value, 0.0), 1.0))
        is_significant = bool(p_value < alpha)

        conclusion = self.generate_continuous_conclusion(
            kpi=kpi,
            is_significant=is_significant,
            test_used=test_name,
            p_value=p_value,
            unit=unit,
        )

        return ContinuousMetricResult(
            kpi=kpi,
            approach_used=approach,
            unit=unit,
            test_name=test_name,
            p_value=p_value,
            is_significant=is_significant,
            is_normal=is_normal,
            is_homogeneous=is_homogeneous,
            summary_stats=summary_stats,
            posthoc_results=posthoc,
            conclusion=conclusion,
            warnings=warnings,
        )

    # ------------------------------------------------------------------ #
    # Path implementations
    # ------------------------------------------------------------------ #

    def _run_gamma_path(
        self, work, kpi, group_col, groups, num_groups, unit, alpha,
        control_label, warnings,
    ) -> tuple[str, float, list[GammaPosthocResult] | None]:
        ll_null, n_params = self.fit_unit_model(work[kpi], unit)
        ll_alt = 0.0
        for grp in groups:
            ll_g, _ = self.fit_unit_model(grp, unit)
            ll_alt += ll_g

        if np.isnan(ll_null) or np.isnan(ll_alt):
            # Validation should prevent this; treat as "no detectable difference".
            warnings.append(
                "The likelihood model could not be fit on all groups; the global "
                "test was treated as non-significant."
            )
            return self._gamma_test_name(unit), 1.0, None

        df_diff = n_params * (num_groups - 1)
        lr_stat = max(2.0 * (ll_alt - ll_null), 0.0)
        p_value = float(stats.chi2.sf(lr_stat, df=df_diff))

        posthoc = None
        if p_value < alpha and num_groups > 2:
            if control_label:
                posthoc = self.run_gamma_posthoc(
                    work, kpi, group_col, control_label, unit, alpha
                )
            else:
                warnings.append(
                    "Global Gamma test is significant with 3+ groups, but no "
                    "control_label was provided for post-hoc comparisons."
                )

        return self._gamma_test_name(unit), p_value, posthoc

    @staticmethod
    def _gamma_test_name(unit: AnalysisUnit) -> str:
        return (
            "Two-Part Hurdle (Likelihood Ratio Test)"
            if unit == AnalysisUnit.PER_VISITOR
            else "Gamma GLM (Likelihood Ratio Test)"
        )

    def _run_heuristic_path(
        self, work, kpi, group_col, groups, num_groups,
    ) -> tuple[str, float, bool, bool]:
        model = smf.ols(f"{kpi} ~ C({group_col})", data=work).fit()

        # Normality of residuals (Shapiro is unreliable / capped at N > 5000).
        resid = model.resid.dropna()
        if len(resid) > 5000:
            resid = resid.sample(5000, random_state=42)
        _, p_norm = shapiro(resid)
        is_normal = bool(p_norm >= 0.05)

        # Homogeneity of variance.
        _, p_var = levene(*groups)
        is_homogeneous = bool(p_var >= 0.05)

        if is_normal and is_homogeneous:
            anova_results = sm.stats.anova_lm(model, typ=2)
            return "Standard ANOVA", float(anova_results["PR(>F)"].iloc[0]), is_normal, is_homogeneous

        if is_normal and not is_homogeneous:
            aov = welch_anova(data=work, dv=kpi, between=group_col)
            return "Welch's ANOVA", float(aov["p-unc"].iloc[0]), is_normal, is_homogeneous

        # Non-normal -> non-parametric.
        if num_groups > 2:
            _, p = kruskal(*groups)
            return "Kruskal-Wallis", float(p), is_normal, is_homogeneous

        _, p = mannwhitneyu(groups[0], groups[1], alternative="two-sided")
        return "Mann-Whitney U", float(p), is_normal, is_homogeneous
