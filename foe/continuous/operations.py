import itertools
import math
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import NegativeBinomial
from scipy.stats import normaltest, levene, kruskal, mannwhitneyu, rankdata
from scipy.stats import studentized_range
from scipy.optimize import brentq
from scipy.special import digamma
from scipy import stats

from foe.core.models import (
    AlternativeHypothesis,
    AnalysisUnit,
    ContinuousApproach,
    ContinuousMetricConfig,
    ContinuousMetricResult,
    OutlierDetectionResult,
    PairwiseComparisonResult,
)
from foe.core.validators import validate_continuous_data
from foe.frequentist.confidence import compute_interval_difference


# NB2 dispersion at or below this is treated as "at the lower bound", i.e. the
# counts are equi- or underdispersed and the fit has collapsed onto Poisson.
# In NB2, Var = mu + alpha * mu^2, so alpha = 1e-4 adds 1% to the variance only
# once mu = 100 -- far above the range of a count KPI.
NB_ALPHA_LOWER_BOUND = 1e-4


@dataclass
class _PathOutcome:
    """Internal carrier for what an analysis path produced."""

    test_name: str
    p_value: float
    test_statistic: float | None = None
    test_statistic_name: str | None = None
    effect_size: float | None = None
    effect_size_name: str | None = None
    is_normal: bool | None = None
    is_homogeneous: bool | None = None
    normality_stat: float | None = None
    normality_p: float | None = None
    levene_stat: float | None = None
    levene_p: float | None = None
    posthoc: list[PairwiseComparisonResult] | None = None
    posthoc_test_name: str | None = None
    dispersion_alpha: float | None = None
    variance_mean_ratio: float | None = None
    extra_warnings: list[str] = field(default_factory=list)


def _finite_or_none(x) -> float | None:
    """float(x) if finite, else None (keeps NaN out of the result model)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


class ContinuousMetricEngine:
    """
    Engine for analyzing continuous metrics (Revenue, Profit, Quantity).

    Two analysis paths:
      * Heuristic    -> a count-data gate runs first: discrete, non-negative
                        count KPIs (e.g. items/tickets per buyer) are routed to
                        Negative Binomial regression regardless of what a
                        normality test would report, since count data violates
                        the continuous-data assumptions those tests are built
                        on. Everything else falls through to the normality/
                        variance decision tree, each branch with its own
                        post-hoc for 3+ groups:
                          - normal, equal variance   -> ANOVA        -> Tukey HSD
                          - normal, unequal variance -> Welch ANOVA  -> Games-Howell
                          - non-normal, 3+ groups    -> Kruskal-Wallis -> Dunn
                          - non-normal, 2 groups     -> Mann-Whitney U
      * Gamma family -> likelihood-ratio test on fitted distributions. The
                        analysis unit selects the model:
                          - per_transaction: a two-parameter Gamma per variant.
                          - per_visitor:     a two-part hurdle model per variant
                                             (Bernoulli convert x Gamma spend),
                                             which keeps non-converting zeros.

    Welch's ANOVA and the Tukey / Games-Howell / Dunn post-hocs are
    implemented natively on numpy + scipy.stats distributions (no pingouin /
    scikit-posthocs dependency).

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
            "per visitor" if unit == AnalysisUnit.PER_VISITOR else "per transaction"
        )
        if not is_significant:
            return (
                f"No Significant Difference: Based on the {test_used} (p={p_value:.3f}), "
                f"there is no statistically significant difference in {kpi} "
                f"{unit_phrase} across the variants. The metric performs similarly "
                "across all groups."
            )

        return (
            f"Significant Variance Detected: The {test_used} (p={p_value:.3f}) detected a "
            f"statistically significant difference in {kpi} {unit_phrase} between the "
            "variants. Review the summary statistics to identify the winning group."
        )

    # ------------------------------------------------------------------ #
    # Outlier helpers (caller-orchestrated)
    # ------------------------------------------------------------------ #

    @staticmethod
    def detect_outliers_ols(
        df: pd.DataFrame,
        kpi: str,
        threshold: float = 3.0,
        group_col: str = "experience_variant_label",
    ) -> list[bool]:
        """
        Flags rows whose internally studentized residual from
        ``kpi ~ C(group_col)`` exceeds ``threshold`` in absolute value.
        Returns a boolean list (JSON serializable) instead of the model.

        Only the residual is used. In this one-way group design a row's
        leverage is exactly 1 / n_g (its variant's size), so leverage and
        DFFITS measure group size rather than how unusual the value is:
        at uneven splits they flag most or all of a small variant. The
        studentized residual already accounts for group size through
        sqrt(1 - h).
        """
        if df.empty or len(df[kpi].dropna()) < 3:
            return [False] * len(df)

        model = smf.ols(f"{kpi} ~ C({group_col})", data=df).fit()
        influence = model.get_influence()

        outlier_mask = np.abs(influence.resid_studentized_internal) > threshold
        return outlier_mask.tolist()  # Safe for JSON

    @classmethod
    def detect_outliers(
        cls,
        df: pd.DataFrame,
        kpi: str,
        group_col: str = "experience_variant_label",
        threshold: float = 3.0,
        exclude_zeros: bool = True,
        large_n_threshold: int = 10_000,
    ) -> OutlierDetectionResult:
        """
        Flags unusually extreme KPI values, choosing the method by data size.

        Rows considered: KPI not missing and, with ``exclude_zeros``, not zero.
        Zero rows are non-converting visitors (or not-yet-filtered non-order
        rows), never outliers; including them would skew the mean/std or the
        OLS fit that the real outliers are judged against.

        * More than ``large_n_threshold`` considered rows -> per-variant
          mean +/- threshold * sd (cheap at scale).
        * Otherwise -> |internally studentized residual| > threshold from
          the one-way OLS fit (``detect_outliers_ols``) on the considered
          rows; ``method`` is ``"studentized_residual"``.

        The returned mask is positionally aligned with ``df``; rows not
        considered are always False.
        """
        values = pd.to_numeric(df[kpi], errors="coerce").to_numpy(dtype=float)
        considered = ~np.isnan(values)
        if exclude_zeros:
            considered &= values != 0
        mask = np.zeros(len(df), dtype=bool)
        n_considered = int(considered.sum())

        if n_considered == 0:
            return OutlierDetectionResult(
                mask=mask.tolist(), method="none", n_considered=0, n_outliers=0
            )

        sub = df.loc[considered, [group_col]].copy()
        sub[kpi] = values[considered]

        if n_considered > large_n_threshold:
            method = "per_variant_sd"
            sub_mask = np.zeros(n_considered, dtype=bool)
            for _, idx in sub.groupby(group_col, observed=True).indices.items():
                v = sub[kpi].to_numpy()[idx]
                if v.size < 2:
                    continue
                mean_v, sd_v = v.mean(), v.std(ddof=1)
                sub_mask[idx] = (v < mean_v - threshold * sd_v) | (
                    v > mean_v + threshold * sd_v
                )
        else:
            method = "studentized_residual"
            sub_mask = np.asarray(
                cls.detect_outliers_ols(
                    sub.reset_index(drop=True), kpi, threshold, group_col
                ),
                dtype=bool,
            )

        mask[considered] = sub_mask
        return OutlierDetectionResult(
            mask=mask.tolist(),
            method=method,
            n_considered=n_considered,
            n_outliers=int(mask.sum()),
        )

    @staticmethod
    def winsorize_series(
        series: pd.Series,
        method: str = "Standard Deviation",
        param: float = 3.0,
        exclude_zeros: bool = False,
    ) -> tuple[list[float], float, float]:
        """
        Caps outliers without removing data points. Returns
        (capped values, lower cap, upper cap).

        Methods:
          * "Standard Deviation": mean +/- param * sd.
          * "Percentile": param is the percent trimmed from *each* tail
            (param=1 -> 1st and 99th percentiles).
          * "Central Percentile": param is the central percent *kept*
            (param=95 -> 2.5th and 97.5th percentiles).

        With ``exclude_zeros``, the caps are computed from the non-zero values
        only and only non-zero values are capped, so a positive lower cap
        never lifts a zero (a non-converting visitor) above zero.
        """
        series = pd.Series(series)
        basis_mask = series.notna()
        if exclude_zeros:
            basis_mask &= series != 0
        basis = series[basis_mask]
        if basis.empty:
            return series.tolist(), 0.0, 0.0

        if method == "Standard Deviation":
            mean_val = basis.mean()
            std_val = basis.std()
            lower = mean_val - (param * std_val)
            upper = mean_val + (param * std_val)
        elif method == "Central Percentile":
            if not 0.0 < param <= 100.0:
                raise ValueError(
                    "Central Percentile expects the percent of data kept, in (0, 100]."
                )
            tail = (100.0 - param) / 2.0
            lower, upper = np.percentile(basis, [tail, 100.0 - tail])
        else:  # "Percentile" (e.g., param=1 means 1st and 99th percentile)
            lower_p = param
            upper_p = 100.0 - param
            lower, upper = np.percentile(basis, [lower_p, upper_p])

        clipped = series.copy()
        clipped[basis_mask] = basis.clip(lower, upper)
        return clipped.tolist(), float(lower), float(upper)

    # ------------------------------------------------------------------ #
    # Native one-way tests and post-hocs
    # ------------------------------------------------------------------ #

    @staticmethod
    def welch_anova(groups: list) -> tuple[float, float, float, float]:
        """
        Welch's heteroscedastic one-way ANOVA (Welch, 1951).

        Returns (F, df1, df2, p_value). Matches R's
        ``oneway.test(var.equal = FALSE)``.
        """
        arrs = [np.asarray(g, dtype=float) for g in groups]
        k = len(arrs)
        n = np.array([a.size for a in arrs], dtype=float)
        if k < 2 or np.any(n < 2):
            raise ValueError("Welch's ANOVA needs 2+ groups with 2+ rows each.")
        means = np.array([a.mean() for a in arrs])
        variances = np.array([a.var(ddof=1) for a in arrs])
        if np.any(variances <= 0):
            raise ValueError(
                "Welch's ANOVA is undefined when a group has zero variance."
            )

        w = n / variances
        w_sum = w.sum()
        weighted_mean = (w * means).sum() / w_sum
        lam = (((1.0 - w / w_sum) ** 2) / (n - 1.0)).sum()

        numerator = (w * (means - weighted_mean) ** 2).sum() / (k - 1)
        denominator = 1.0 + 2.0 * (k - 2) / (k ** 2 - 1) * lam
        f_stat = numerator / denominator
        df1 = float(k - 1)
        df2 = float((k ** 2 - 1) / (3.0 * lam))
        p_value = float(stats.f.sf(f_stat, df1, df2))
        return float(f_stat), df1, df2, p_value

    @staticmethod
    def eta_squared(groups: list) -> float:
        """Between-group sum of squares / total sum of squares (one-way)."""
        arrs = [np.asarray(g, dtype=float) for g in groups]
        pooled = np.concatenate(arrs)
        grand = pooled.mean()
        ss_total = ((pooled - grand) ** 2).sum()
        ss_between = sum(a.size * (a.mean() - grand) ** 2 for a in arrs)
        return float(ss_between / ss_total) if ss_total > 0 else float("nan")

    @staticmethod
    def _pairs(labels: list) -> list[tuple[int, int]]:
        """(i, j) index pairs with i < j; reported as labels[j] vs labels[i]."""
        return list(itertools.combinations(range(len(labels)), 2))

    @classmethod
    def tukey_hsd(
        cls, labels: list, groups: list, alpha: float = 0.05
    ) -> list[PairwiseComparisonResult]:
        """
        Tukey's HSD (Tukey-Kramer for unequal n) on all pairs, using the
        pooled within-group variance. Matches R's ``TukeyHSD(aov(...))``.
        """
        arrs = [np.asarray(g, dtype=float) for g in groups]
        k = len(arrs)
        n = np.array([a.size for a in arrs], dtype=float)
        means = np.array([a.mean() for a in arrs])
        df_resid = n.sum() - k
        mse = sum(((a - a.mean()) ** 2).sum() for a in arrs) / df_resid
        q_crit = float(studentized_range.ppf(1.0 - alpha, k, df_resid))

        results = []
        for i, j in cls._pairs(labels):
            diff = means[j] - means[i]
            se = math.sqrt(mse / 2.0 * (1.0 / n[i] + 1.0 / n[j]))
            q = abs(diff) / se if se > 0 else float("inf")
            p = float(np.clip(studentized_range.sf(q, k, df_resid), 0.0, 1.0))
            results.append(PairwiseComparisonResult(
                comparison=f"{labels[j]} vs {labels[i]}",
                group_a=str(labels[j]), group_b=str(labels[i]),
                statistic=float(q), statistic_name="q",
                p_value=p, p_adj=p, adjustment="tukey",
                is_significant=bool(p < alpha),
                mean_diff=float(diff),
                ci_low=float(diff - q_crit * se),
                ci_high=float(diff + q_crit * se),
            ))
        return results

    @classmethod
    def games_howell(
        cls, labels: list, groups: list, alpha: float = 0.05
    ) -> list[PairwiseComparisonResult]:
        """
        Games-Howell all-pairs comparison: Welch t per pair with Welch-
        Satterthwaite df, referred to the studentized range distribution.
        Suited to unequal variances (the post-hoc for Welch's ANOVA).
        """
        arrs = [np.asarray(g, dtype=float) for g in groups]
        k = len(arrs)
        n = np.array([a.size for a in arrs], dtype=float)
        means = np.array([a.mean() for a in arrs])
        v_over_n = np.array([a.var(ddof=1) for a in arrs]) / n

        results = []
        for i, j in cls._pairs(labels):
            diff = means[j] - means[i]
            se = math.sqrt(v_over_n[i] + v_over_n[j])
            df = (v_over_n[i] + v_over_n[j]) ** 2 / (
                v_over_n[i] ** 2 / (n[i] - 1) + v_over_n[j] ** 2 / (n[j] - 1)
            )
            t = diff / se
            p = float(np.clip(
                studentized_range.sf(math.sqrt(2.0) * abs(t), k, df), 0.0, 1.0
            ))
            crit = float(studentized_range.ppf(1.0 - alpha, k, df)) / math.sqrt(2.0)
            results.append(PairwiseComparisonResult(
                comparison=f"{labels[j]} vs {labels[i]}",
                group_a=str(labels[j]), group_b=str(labels[i]),
                statistic=float(t), statistic_name="t",
                p_value=p, p_adj=p, adjustment="games-howell",
                is_significant=bool(p < alpha),
                mean_diff=float(diff),
                ci_low=float(diff - crit * se),
                ci_high=float(diff + crit * se),
            ))
        return results

    @classmethod
    def dunn_test(
        cls, labels: list, groups: list, alpha: float = 0.05
    ) -> list[PairwiseComparisonResult]:
        """
        Dunn's (1964) all-pairs test on mean ranks, with the tie correction,
        Bonferroni-adjusted over k(k-1)/2 comparisons. The post-hoc for
        Kruskal-Wallis; matches ``dunn.test(method = "bonferroni")`` in R.
        """
        arrs = [np.asarray(g, dtype=float) for g in groups]
        pooled = np.concatenate(arrs)
        n_total = pooled.size
        ranks = rankdata(pooled)
        bounds = np.cumsum([0] + [a.size for a in arrs])
        mean_ranks = [ranks[bounds[g]:bounds[g + 1]].mean() for g in range(len(arrs))]

        _, tie_counts = np.unique(pooled, return_counts=True)
        tie_term = ((tie_counts ** 3 - tie_counts).sum()) / (12.0 * (n_total - 1))
        base_var = n_total * (n_total + 1) / 12.0 - tie_term

        pairs = cls._pairs(labels)
        m = len(pairs)
        results = []
        for i, j in pairs:
            se = math.sqrt(base_var * (1.0 / arrs[i].size + 1.0 / arrs[j].size))
            z = (mean_ranks[j] - mean_ranks[i]) / se if se > 0 else 0.0
            p = float(2.0 * stats.norm.sf(abs(z)))
            p_adj = float(min(p * m, 1.0))
            results.append(PairwiseComparisonResult(
                comparison=f"{labels[j]} vs {labels[i]}",
                group_a=str(labels[j]), group_b=str(labels[i]),
                statistic=float(z), statistic_name="z",
                p_value=p, p_adj=p_adj, adjustment="bonferroni",
                is_significant=bool(p_adj < alpha),
            ))
        return results

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
        """
        Fits a two-parameter Gamma (shape k, scale theta) by exact MLE.
        Returns (k, theta, log_likelihood).

        The MLE of theta given k is mean(x) / k. Substituting it into the
        score equation for k leaves the one-dimensional profile equation

            log k - digamma(k) = log(mean(x)) - mean(log(x)) =: s,

        whose left side falls strictly from +inf to 0, so it has exactly one
        root for any s > 0 (s > 0 holds whenever the data are not all equal).
        It is solved with Brent's method to machine precision, starting from
        Minka's (2002) closed-form approximation. A general-purpose 2-D
        optimiser stops short of the maximum on a sizeable share of samples,
        which shifts likelihood-ratio p-values.

        Missing values are dropped. Raises ValueError if fewer than two
        values remain, any value is <= 0, or all values are equal.
        """
        x = pd.Series(data).dropna().to_numpy(dtype=float)
        if x.size < 2:
            raise ValueError("Need at least two observations to fit a Gamma.")
        if np.any(x <= 0):
            raise ValueError("The Gamma is defined for x > 0; got non-positive values.")

        mean_x = float(x.mean())
        s = math.log(mean_x) - float(np.log(x).mean())
        if not s > 0:
            raise ValueError("All values are equal; the Gamma shape is unbounded.")

        def profile(k: float) -> float:
            return math.log(k) - float(digamma(k)) - s

        # Minka's approximation is within ~1.5% of the root; widen to a bracket.
        k0 = (3.0 - s + math.sqrt((s - 3.0) ** 2 + 24.0 * s)) / (12.0 * s)
        lo, hi = k0 / 2.0, k0 * 2.0
        while profile(lo) <= 0:
            lo /= 2.0
        while profile(hi) >= 0:
            hi *= 2.0
        k = brentq(profile, lo, hi, xtol=1e-15, rtol=4 * np.finfo(float).eps, maxiter=200)

        theta = mean_x / k
        log_lik = float(stats.gamma.logpdf(x, a=k, scale=theta).sum())
        return float(k), float(theta), log_lik

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
    def _lrt_posthoc_result(
        variant, control_label, lrt_stat: float, p_val: float,
        num_comparisons: int, alpha: float,
    ) -> PairwiseComparisonResult:
        adj_p = float(min(p_val * num_comparisons, 1.0))
        return PairwiseComparisonResult(
            comparison=f"{variant} vs {control_label}",
            group_a=str(variant), group_b=str(control_label),
            statistic=float(lrt_stat), statistic_name="LR",
            p_value=float(p_val), p_adj=adj_p, adjustment="bonferroni",
            is_significant=bool(adj_p < alpha),
        )

    @staticmethod
    def run_gamma_posthoc(
        df: pd.DataFrame,
        kpi: str,
        group_col: str,
        control_label: str,
        unit: AnalysisUnit = AnalysisUnit.PER_TRANSACTION,
        alpha: float = 0.05,
    ) -> list[PairwiseComparisonResult]:
        """Pairwise LRTs of each treatment against the control, Bonferroni-adjusted.

        Uses the same unit-appropriate model as the global test, so the pairwise
        degrees of freedom equal the per-group parameter count (2 for a Gamma,
        3 for the two-part hurdle).
        """
        variants = [v for v in df[group_col].unique() if v != control_label]
        num_comparisons = len(variants)
        posthoc_results: list[PairwiseComparisonResult] = []

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

            posthoc_results.append(
                ContinuousMetricEngine._lrt_posthoc_result(
                    variant, control_label, lrt_stat, p_val, num_comparisons, alpha
                )
            )

        return posthoc_results

    # ------------------------------------------------------------------ #
    # Negative Binomial fitting (discrete count KPIs)
    # ------------------------------------------------------------------ #

    @staticmethod
    def is_count_kpi(
        series, max_unique_for_check: int = 50, max_unique_ratio: float = 0.05,
    ) -> bool:
        """
        Heuristic gate: True if every non-null value is a non-negative integer
        (or integer-valued float, e.g. 3.0) AND the cardinality looks like a
        genuine count rather than a continuous KPI that happens to have round
        values (e.g. revenue rounded to whole currency units).

        Both thresholds are parameters (rather than hardcoded) because what
        "looks like a count" varies by business context -- a KPI with 40
        distinct values might be a clear count metric for a low-volume B2B
        funnel, or a coincidentally-round continuous KPI for a high-volume
        consumer funnel. Callers should surface both as configurable settings
        rather than relying on one global default.

        Args:
            max_unique_for_check: absolute cap on distinct values to still
                call it a count metric.
            max_unique_ratio: distinct-values-to-row-count ratio below which
                the KPI is still treated as a count metric even if it exceeds
                max_unique_for_check (relevant for large datasets where a
                genuine count metric can have many distinct values in
                absolute terms while still being a tiny fraction of rows).
        """
        s = pd.Series(series).dropna()
        if s.empty:
            return False
        if (s < 0).any():
            return False
        if not np.allclose(s, np.round(s)):
            return False
        n_unique = s.nunique()
        return n_unique <= max_unique_for_check or (n_unique / len(s)) < max_unique_ratio

    @staticmethod
    def fit_negbin(data, exog: np.ndarray | None = None) -> tuple[float, float, object]:
        """
        Fits a Negative Binomial (NB2) model by MLE. If exog is None, fits an
        intercept-only model. Returns (log_likelihood, alpha, fitted_results).

        alpha is the dispersion parameter statsmodels appends as the last
        entry of the fitted params vector, regardless of exog width.
        """
        arr = np.asarray(data, dtype=float)
        n = arr.size
        if exog is None:
            exog = np.ones((n, 1))
        model = NegativeBinomial(arr, exog, loglike_method="nb2")
        try:
            res = model.fit(disp=0, maxiter=200)
        except Exception:
            start = np.zeros(exog.shape[1] + 1)
            start[0] = np.log(np.mean(arr) + 1e-6)
            start[-1] = 1.0
            res = model.fit(disp=0, maxiter=200, start_params=start)
        return float(res.llf), float(res.params[-1]), res

    @staticmethod
    def _run_negbin_lrt(work: pd.DataFrame, kpi: str, group_col: str) -> tuple[float, float, float]:
        """
        Likelihood Ratio Test comparing a Null NB2 model (single mean, shared
        across all variants) against an Alternative model (separate mean per
        variant via group dummies, shared dispersion). Returns
        (p_value, lr_stat, alpha_estimate).

        Group labels are coerced to str before dummy-encoding so a Categorical
        dtype carrying unused categories can't inflate the dummy matrix with
        columns for variants that aren't actually present in `work`.
        """
        data = work[kpi].to_numpy(dtype=float)
        n = data.size
        dummies = pd.get_dummies(work[group_col].astype(str), drop_first=True).to_numpy(dtype=float)

        exog_null = np.ones((n, 1))
        ll_null, _, res_null = ContinuousMetricEngine.fit_negbin(data, exog_null)

        exog_alt = np.column_stack([np.ones(n), dummies]) if dummies.shape[1] > 0 else exog_null
        ll_alt, alpha_alt, res_alt = ContinuousMetricEngine.fit_negbin(data, exog_alt)

        # Degenerate/tiny samples can fail to converge to a finite log-likelihood on some
        # solver versions -- but a tiny sample that perfectly separates the groups (e.g. one
        # group all-zero, another all-one) instead lets the dummy coefficient run toward
        # +/-infinity while the optimizer still reports a finite log-likelihood and
        # "converged": the fit isn't actually identified, just numerically unbounded. The
        # Hessian is singular at that point, so standard errors can't be computed -- bse comes
        # back all-NaN -- which is a solver-version-independent signal of that pathology,
        # unlike checking the log-likelihood's finiteness alone. Both checks are kept: either
        # is grounds to treat this the same as any other unfittable model (p=1.0, with a
        # warning) rather than letting a nan or a spurious near-zero p_value leak out.
        #
        # One NaN-bse case is harmless: equi-/underdispersed counts (variance <= mean, e.g.
        # items per order, always >= 1) push alpha to its lower bound. The Hessian is then
        # singular in alpha alone, so bse is NaN although the fit is fine -- NB2 with
        # alpha -> 0 is Poisson, and the log-likelihoods and LR statistic are valid. For
        # that case identification is checked exactly: in this one-way design (intercept +
        # one dummy per variant) the Poisson MLE is log(group mean), so it exists iff every
        # group modelled has a positive total. A group of all zeros is the separation case.
        # (Refitting a Poisson model and checking its bse is not a version-stable signal:
        # statsmodels 0.15 returns huge but finite bse for a separated Poisson fit.)
        group_totals = pd.Series(data).groupby(work[group_col].astype(str).to_numpy()).sum()

        def _identified(res, group_totals_positive: bool) -> bool:
            if np.all(np.isfinite(res.bse)):
                return True
            return float(res.params[-1]) <= NB_ALPHA_LOWER_BOUND and group_totals_positive

        if (
            not (np.isfinite(ll_null) and np.isfinite(ll_alt))
            or not _identified(res_null, bool(data.sum() > 0))
            or not _identified(res_alt, bool((group_totals > 0).all()))
        ):
            raise ValueError(
                "Negative Binomial model did not converge to a well-identified fit -- "
                "non-finite log-likelihood, or standard errors could not be computed "
                "(a common symptom of quasi/complete separation in a tiny or degenerate sample)."
            )

        # At the lower bound the fitted alpha (~1e-5..1e-7) is optimizer noise; report
        # the limit it stands for (0.0, i.e. Poisson).
        if alpha_alt <= NB_ALPHA_LOWER_BOUND:
            alpha_alt = 0.0

        df_diff = exog_alt.shape[1] - exog_null.shape[1]
        if df_diff <= 0:
            return 1.0, 0.0, float(alpha_alt)

        lr_stat = max(2.0 * (ll_alt - ll_null), 0.0)
        p_value = float(stats.chi2.sf(lr_stat, df=df_diff))
        return p_value, float(lr_stat), float(alpha_alt)

    @classmethod
    def run_negbin_posthoc(
        cls,
        df: pd.DataFrame,
        kpi: str,
        group_col: str,
        control_label: str,
        alpha: float = 0.05,
    ) -> list[PairwiseComparisonResult]:
        """Pairwise Negative Binomial LRTs of each treatment against the control, Bonferroni-adjusted."""
        variants = [v for v in df[group_col].unique() if v != control_label]
        num_comparisons = len(variants)
        posthoc_results: list[PairwiseComparisonResult] = []

        for variant in variants:
            pair_df = df[df[group_col].isin([control_label, variant])]
            try:
                p_val, lrt_stat, _ = cls._run_negbin_lrt(pair_df, kpi, group_col)
            except Exception:
                lrt_stat, p_val = 0.0, 1.0

            posthoc_results.append(
                cls._lrt_posthoc_result(
                    variant, control_label, lrt_stat, p_val, num_comparisons, alpha
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
        winsorized / outlier-removed frame if desired (see ``detect_outliers``
        and ``winsorize_series``). The unit zero-filter composes on top of
        whatever frame is supplied.
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

        grouped = list(work.groupby(group_col, observed=True))
        labels = [str(name) for name, _ in grouped]
        groups = [g[kpi].to_numpy() for _, g in grouped]
        num_groups = len(groups)

        # Summary statistics (computed on the analysis population).
        summary_df = work.groupby(group_col, observed=True)[kpi].agg(
            ["mean", "std", "count"]
        )
        summary_stats = summary_df.reset_index().to_dict(orient="records")

        # 3. Branch once on the approach.
        if approach == ContinuousApproach.GAMMA_GLM:
            outcome = self._run_gamma_path(
                work, kpi, group_col, groups, num_groups, unit, alpha,
                config.control_label,
            )
        else:
            outcome = self._run_heuristic_path(
                work, kpi, group_col, labels, groups, num_groups, alpha,
                config.control_label,
                count_max_unique=config.count_max_unique,
                count_max_unique_ratio=config.count_max_unique_ratio,
            )
        warnings.extend(outcome.extra_warnings)

        p_value = float(min(max(outcome.p_value, 0.0), 1.0))
        is_significant = bool(p_value < alpha)

        conclusion = self.generate_continuous_conclusion(
            kpi=kpi,
            is_significant=is_significant,
            test_used=outcome.test_name,
            p_value=p_value,
            unit=unit,
        )

        return ContinuousMetricResult(
            kpi=kpi,
            approach_used=approach,
            unit=unit,
            test_name=outcome.test_name,
            p_value=p_value,
            is_significant=is_significant,
            test_statistic=_finite_or_none(outcome.test_statistic),
            test_statistic_name=outcome.test_statistic_name,
            effect_size=_finite_or_none(outcome.effect_size),
            effect_size_name=outcome.effect_size_name,
            is_normal=outcome.is_normal,
            is_homogeneous=outcome.is_homogeneous,
            normality_stat=_finite_or_none(outcome.normality_stat),
            normality_p=_finite_or_none(outcome.normality_p),
            levene_stat=_finite_or_none(outcome.levene_stat),
            levene_p=_finite_or_none(outcome.levene_p),
            summary_stats=summary_stats,
            posthoc_results=outcome.posthoc,
            posthoc_test_name=outcome.posthoc_test_name if outcome.posthoc else None,
            conclusion=conclusion,
            warnings=warnings,
            dispersion_alpha=outcome.dispersion_alpha,
            variance_mean_ratio=_finite_or_none(outcome.variance_mean_ratio),
        )

    # ------------------------------------------------------------------ #
    # Path implementations
    # ------------------------------------------------------------------ #

    def _run_gamma_path(
        self, work, kpi, group_col, groups, num_groups, unit, alpha,
        control_label,
    ) -> _PathOutcome:
        test_name = self._gamma_test_name(unit)
        ll_null, n_params = self.fit_unit_model(work[kpi], unit)
        ll_alt = 0.0
        for grp in groups:
            ll_g, _ = self.fit_unit_model(grp, unit)
            ll_alt += ll_g

        if np.isnan(ll_null) or np.isnan(ll_alt):
            # Validation should prevent this; treat as "no detectable difference".
            return _PathOutcome(
                test_name=test_name, p_value=1.0,
                extra_warnings=[
                    "The likelihood model could not be fit on all groups; the global "
                    "test was treated as non-significant."
                ],
            )

        df_diff = n_params * (num_groups - 1)
        lr_stat = max(2.0 * (ll_alt - ll_null), 0.0)
        p_value = float(stats.chi2.sf(lr_stat, df=df_diff))
        outcome = _PathOutcome(
            test_name=test_name, p_value=p_value,
            test_statistic=lr_stat, test_statistic_name="LR",
        )

        if p_value < alpha and num_groups > 2:
            if control_label:
                outcome.posthoc = self.run_gamma_posthoc(
                    work, kpi, group_col, control_label, unit, alpha
                )
                outcome.posthoc_test_name = (
                    "Two-part hurdle LRT vs control (Bonferroni)"
                    if unit == AnalysisUnit.PER_VISITOR
                    else "Gamma LRT vs control (Bonferroni)"
                )
            else:
                outcome.extra_warnings.append(
                    "Global Gamma test is significant with 3+ groups, but no "
                    "control_label was provided for post-hoc comparisons."
                )

        return outcome

    @staticmethod
    def _gamma_test_name(unit: AnalysisUnit) -> str:
        return (
            "Two-Part Hurdle (Likelihood Ratio Test)"
            if unit == AnalysisUnit.PER_VISITOR
            else "Gamma GLM (Likelihood Ratio Test)"
        )

    def _run_heuristic_path(
        self, work, kpi, group_col, labels, groups, num_groups, alpha,
        control_label,
        count_max_unique: int = 50,
        count_max_unique_ratio: float = 0.05,
    ) -> _PathOutcome:
        # Count-data gate runs first: a discrete, non-negative count KPI
        # (e.g. items/tickets per buyer) violates the continuous-data
        # assumptions the normality/variance tree below is built on, no
        # matter what a normality test reports on a large sample.
        if self.is_count_kpi(
            work[kpi],
            max_unique_for_check=count_max_unique,
            max_unique_ratio=count_max_unique_ratio,
        ):
            return self._run_negbin_path(
                work, kpi, group_col, num_groups, alpha, control_label,
            )

        model = smf.ols(f"{kpi} ~ C({group_col})", data=work).fit()

        # Normality of residuals via D'Agostino's K^2 omnibus test (skewness +
        # kurtosis): reliable for large samples and no N > 5000 warning, so the
        # full residual vector is used. The kurtosis component needs N >= 20;
        # below that, fall back to non-parametric handling.
        resid = model.resid.dropna()
        normality_stat = normality_p = None
        if len(resid) >= 20:
            normality_stat, normality_p = (float(x) for x in normaltest(resid))
            is_normal = bool(normality_p >= 0.05)
        else:
            is_normal = False

        # Homogeneity of variance.
        levene_stat, levene_p = (float(x) for x in levene(*groups))
        is_homogeneous = bool(levene_p >= 0.05)

        outcome = _PathOutcome(
            test_name="", p_value=1.0,
            is_normal=is_normal, is_homogeneous=is_homogeneous,
            normality_stat=normality_stat, normality_p=normality_p,
            levene_stat=levene_stat, levene_p=levene_p,
        )
        posthoc_fn = posthoc_name = None

        if is_normal and is_homogeneous:
            anova_results = sm.stats.anova_lm(model, typ=2)
            ss_between = float(anova_results["sum_sq"].iloc[0])
            ss_resid = float(anova_results["sum_sq"].iloc[1])
            outcome.test_name = "Standard ANOVA"
            outcome.p_value = float(anova_results["PR(>F)"].iloc[0])
            outcome.test_statistic = float(anova_results["F"].iloc[0])
            outcome.test_statistic_name = "F"
            outcome.effect_size = ss_between / (ss_between + ss_resid)
            outcome.effect_size_name = "eta_squared"
            posthoc_fn, posthoc_name = self.tukey_hsd, "Tukey HSD"

        elif is_normal:
            f_stat, _, _, p = self.welch_anova(groups)
            outcome.test_name = "Welch's ANOVA"
            outcome.p_value = p
            outcome.test_statistic = f_stat
            outcome.test_statistic_name = "F"
            outcome.effect_size = self.eta_squared(groups)
            outcome.effect_size_name = "partial_eta_squared"
            posthoc_fn, posthoc_name = self.games_howell, "Games-Howell"

        elif num_groups > 2:
            # Non-normal -> non-parametric.
            h_stat, p = kruskal(*groups)
            n_total = sum(len(g) for g in groups)
            outcome.test_name = "Kruskal-Wallis"
            outcome.p_value = float(p)
            outcome.test_statistic = float(h_stat)
            outcome.test_statistic_name = "H"
            if n_total > num_groups:
                # Clamped to [0, 1] like rstatix::kruskal_effsize: the raw
                # formula dips below 0 when H < k - 1 (no effect at all).
                eta_h = (h_stat - num_groups + 1) / (n_total - num_groups)
                outcome.effect_size = float(min(max(eta_h, 0.0), 1.0))
                outcome.effect_size_name = "eta_squared_h"
            posthoc_fn, posthoc_name = self.dunn_test, "Dunn (Bonferroni)"

        else:
            u_stat, p = mannwhitneyu(groups[0], groups[1], alternative="two-sided")
            outcome.test_name = "Mann-Whitney U"
            outcome.p_value = float(p)
            outcome.test_statistic = float(u_stat)
            outcome.test_statistic_name = "U"
            outcome.effect_size = 1.0 - 2.0 * u_stat / (len(groups[0]) * len(groups[1]))
            outcome.effect_size_name = "rank_biserial"

        if posthoc_fn is not None and outcome.p_value < alpha and num_groups > 2:
            try:
                outcome.posthoc = posthoc_fn(labels, groups, alpha)
                outcome.posthoc_test_name = posthoc_name
            except Exception as exc:  # keep the global result usable
                outcome.extra_warnings.append(
                    f"The {posthoc_name} post-hoc could not be computed: {exc}"
                )

        return outcome

    def _run_negbin_path(
        self, work, kpi, group_col, num_groups, alpha, control_label,
    ) -> _PathOutcome:
        test_name = "Negative Binomial Regression (LRT)"
        mean_val = float(work[kpi].mean())
        var_mean_ratio = float(work[kpi].var()) / mean_val if mean_val > 0 else None
        try:
            p_value, lr_stat, dispersion_alpha = self._run_negbin_lrt(
                work, kpi, group_col
            )
        except Exception:
            return _PathOutcome(
                test_name=test_name, p_value=1.0,
                variance_mean_ratio=var_mean_ratio,
                extra_warnings=[
                    "The Negative Binomial model could not be fit on all groups; "
                    "the global test was treated as non-significant."
                ],
            )

        outcome = _PathOutcome(
            test_name=test_name, p_value=p_value,
            test_statistic=lr_stat, test_statistic_name="LR",
            dispersion_alpha=dispersion_alpha,
            variance_mean_ratio=var_mean_ratio,
        )
        if p_value < alpha and num_groups > 2:
            if control_label:
                outcome.posthoc = self.run_negbin_posthoc(
                    work, kpi, group_col, control_label, alpha
                )
                outcome.posthoc_test_name = (
                    "Negative Binomial LRT vs control (Bonferroni)"
                )
            else:
                outcome.extra_warnings.append(
                    "Global Negative Binomial test is significant with 3+ groups, "
                    "but no control_label was provided for post-hoc comparisons."
                )

        return outcome

    # ------------------------------------------------------------------ #
    # Business case (user-level monetary projection)
    # ------------------------------------------------------------------ #

    @staticmethod
    def estimate_monetary_impact_per_variant(
        mean_ctrl: float,
        std_ctrl: float,
        n_ctrl: int,
        mean_chal: float,
        std_chal: float,
        n_chal: int,
        unit: AnalysisUnit,
        daily_visitors: float,
        visitors_ctrl: int | None = None,
        visitors_chal: int | None = None,
        alpha: float = 0.05,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
        projection_period: int = 183,
    ) -> dict:
        """
        Projects the monetary impact of a continuous, user-level KPI (revenue,
        profit, ...) between a control and a challenger variant over a future
        period, via the same delta-method construction as
        ``FrequentistEngine.estimate_monetary_impact_per_variant`` -- the KPI
        mean here plays the role AOV plays there, and an order rate plays the
        role conversion rate plays there.

        unit = PER_VISITOR: the group means already average over every visitor
            (non-buyers included as 0), so they *are* value-per-visitor; the
            order rate is fixed at 1 for both arms.
        unit = PER_TRANSACTION: the group means are value-per-order (positive
            rows only); ``visitors_ctrl``/``visitors_chal`` (total per-variant
            visitor counts) are required so each arm's order rate (n / visitors)
            can be derived and folded into the projection -- without it, a
            per-order lift can't be translated into a per-visitor (and
            therefore daily-traffic-scaled) monetary impact.
        """
        if unit == AnalysisUnit.PER_VISITOR:
            rate_ctrl = rate_chal = 1.0
            se_rate_ctrl = se_rate_chal = 0.0
        else:
            if not visitors_ctrl or not visitors_chal:
                raise ValueError(
                    "visitors_ctrl and visitors_chal (total per-variant visitor "
                    "counts) are required to project a per-transaction business "
                    "case; they are needed to derive each variant's order rate."
                )
            rate_ctrl = n_ctrl / visitors_ctrl
            rate_chal = n_chal / visitors_chal
            se_rate_ctrl = (
                math.sqrt(rate_ctrl * (1.0 - rate_ctrl) / visitors_ctrl)
                if 0.0 < rate_ctrl < 1.0 else 0.0
            )
            se_rate_chal = (
                math.sqrt(rate_chal * (1.0 - rate_chal) / visitors_chal)
                if 0.0 < rate_chal < 1.0 else 0.0
            )

        se_mean_ctrl = (std_ctrl / math.sqrt(n_ctrl)) if n_ctrl > 0 else 0.0
        se_mean_chal = (std_chal / math.sqrt(n_chal)) if n_chal > 0 else 0.0

        diff_value = (rate_chal * mean_chal) - (rate_ctrl * mean_ctrl)
        var_chal = (mean_chal ** 2) * (se_rate_chal ** 2) + (rate_chal ** 2) * (se_mean_chal ** 2)
        var_ctrl = (mean_ctrl ** 2) * (se_rate_ctrl ** 2) + (rate_ctrl ** 2) * (se_mean_ctrl ** 2)
        se_diff_value = math.sqrt(var_chal + var_ctrl)

        ci_diff_value = compute_interval_difference(
            diff_value, se_diff_value, alpha=alpha, alternative=alternative
        )

        def revenue(x: float) -> float:
            return x * daily_visitors * projection_period

        return {
            "point_estimate": revenue(diff_value),
            "ci_low": revenue(ci_diff_value[0]),
            "ci_high": revenue(ci_diff_value[1]),
            "daily_visitors": daily_visitors,
            "mean_ctrl": mean_ctrl,
            "mean_chal": mean_chal,
            "rate_ctrl": rate_ctrl,
            "rate_chal": rate_chal,
            "projection_period": projection_period,
            "unit": unit.value,
        }

    @staticmethod
    def generate_monetary_conclusion(
        variant_name: str,
        monetary_result: dict,
        is_significant: bool,
    ) -> str:
        """UI-agnostic narrative summary of estimate_monetary_impact_per_variant's output."""
        point = monetary_result["point_estimate"]
        low, high = monetary_result["ci_low"], monetary_result["ci_high"]
        period = monetary_result["projection_period"]

        direction = "gain" if point >= 0 else "loss"
        qualifier = "" if is_significant else " (not statistically significant -- treat as directional)"

        return (
            f"Over the next {period} days, '{variant_name}' is projected to produce a "
            f"{direction} of {point:,.0f} versus the control, with a plausible range of "
            f"{low:,.0f} to {high:,.0f}{qualifier}."
        )

    def run_business_case(
        self,
        group_stats: dict,
        control_label: str,
        unit: AnalysisUnit,
        daily_visitors: float,
        visitor_counts: dict | None = None,
        alpha: float = 0.05,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
        projection_period: int = 183,
        significance_by_variant: dict | None = None,
    ) -> list[dict]:
        """
        Runs estimate_monetary_impact_per_variant for every non-control variant
        against the control, using per-variant {"mean", "std", "count"} stats
        (the same shape run_comparison_suite's summary_stats already produces).
        """
        ctrl = group_stats[control_label]
        visitor_counts = visitor_counts or {}
        significance_by_variant = significance_by_variant or {}

        results = []
        for label, stats_ in group_stats.items():
            if label == control_label:
                continue
            monetary = self.estimate_monetary_impact_per_variant(
                mean_ctrl=ctrl["mean"], std_ctrl=ctrl["std"], n_ctrl=int(ctrl["count"]),
                mean_chal=stats_["mean"], std_chal=stats_["std"], n_chal=int(stats_["count"]),
                unit=unit,
                daily_visitors=daily_visitors,
                visitors_ctrl=visitor_counts.get(control_label),
                visitors_chal=visitor_counts.get(label),
                alpha=alpha,
                alternative=alternative,
                projection_period=projection_period,
            )
            conclusion = self.generate_monetary_conclusion(
                label, monetary, significance_by_variant.get(label, False),
            )
            results.append({"variant": label, "conclusion": conclusion, **monetary})

        return results
