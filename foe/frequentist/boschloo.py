"""
Boschloo's exact unconditional test for two independent proportions.

Intended for small samples (e.g. email A/B tests on small lists or segments,
low-rate micro-conversions) where the normal approximation behind
FrequentistEngine.run_synthesis is unreliable: few conversions, 0% or 100%
arms, or very unequal allocation. Boschloo's test is uniformly more powerful
than Fisher's exact test and controls type I error at any sample size.

Kept separate from FrequentistEngine on purpose: it shares the input contract
(ExperimentInput) and the Šidák policy, but has its own result model
(BoschlooResult) so it can be exposed as a standalone tool.

Limitations:
  * Runtime grows steeply with sample size (scipy evaluates the full grid of
    possible outcomes): roughly 0.05s at 100 per arm, 1s at 300, 20s at 1,000.
    `max_visitors_per_arm` guards against requests that would hang a worker.
    Above that size the z-test in run_synthesis is adequate unless conversion
    counts are tiny.
  * Assumes pure binomial sampling, one outcome per independent unit. A
    reduction_factor (φ) other than 1.0 cannot be applied to an exact test
    and is rejected.
  * Tests H0: p_challenger = p_control only. It does not do non-inferiority
    (shifted-null) testing.
  * The reported interval is Newcombe's hybrid score interval, not an exact
    interval obtained by inverting this test.
"""

import math
from typing import List, Optional, Tuple

from scipy.stats import boschloo_exact
from statsmodels.stats.proportion import confint_proportions_2indep

from foe.core.models import AlternativeHypothesis, BoschlooResult, ExperimentInput
from foe.frequentist.operations import apply_sidak

DEFAULT_MAX_VISITORS_PER_ARM = 1000

# With the challenger in the first column of the 2x2 table, scipy's
# alternatives read as statements about p_challenger vs p_control.
_SCIPY_ALTERNATIVE = {
    AlternativeHypothesis.GREATER: "greater",
    AlternativeHypothesis.LESS: "less",
    AlternativeHypothesis.TWO_SIDED: "two-sided",
}


class BoschlooEngine:
    """Stateless Boschloo exact test for A/B/n conversion data."""

    def __init__(
        self,
        max_visitors_per_arm: int = DEFAULT_MAX_VISITORS_PER_ARM,
        sampling_points: int = 32,
    ):
        """
        Args:
            max_visitors_per_arm: Refuse inputs with more visitors than this in
                any arm (see module docstring on runtime). Raise it deliberately
                if you accept longer runtimes.
            sampling_points: Number of nuisance-parameter grid points passed to
                scipy.stats.boschloo_exact (its `n`). Higher is more precise
                and slower.
        """
        self.max_visitors_per_arm = max_visitors_per_arm
        self.sampling_points = sampling_points

    # ------------------------------------------------------------------ #
    #  Primitives
    # ------------------------------------------------------------------ #

    def exact_p_value(
        self,
        ctrl_conv: int,
        ctrl_n: int,
        chal_conv: int,
        chal_n: int,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
    ) -> float:
        """
        Boschloo p-value for H0: p_chal = p_ctrl against `alternative`
        (GREATER = challenger converts better than control).
        """
        table = [
            [chal_conv, ctrl_conv],
            [chal_n - chal_conv, ctrl_n - ctrl_conv],
        ]
        p = boschloo_exact(
            table, alternative=_SCIPY_ALTERNATIVE[alternative], n=self.sampling_points
        ).pvalue
        # scipy can return NaN for degenerate tables; both arms identical at
        # 0% or 100% carry no evidence of a difference.
        if math.isnan(p):
            return 1.0
        return float(min(max(p, 0.0), 1.0))

    @staticmethod
    def newcombe_interval(
        ctrl_conv: int,
        ctrl_n: int,
        chal_conv: int,
        chal_n: int,
        alpha: float,
        alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
    ) -> Tuple[float, float]:
        """
        Newcombe hybrid score interval for p_chal - p_ctrl. Stays within
        [-1, 1] and behaves well with zero counts, unlike the Wald interval.
        One-sided alternatives return a single bound at level 1 - alpha with
        the other side unbounded, matching compute_interval_difference.
        """
        level_alpha = alpha if alternative == AlternativeHypothesis.TWO_SIDED else 2 * alpha
        low, high = confint_proportions_2indep(
            chal_conv, chal_n, ctrl_conv, ctrl_n,
            method="newcomb", compare="diff", alpha=level_alpha,
        )
        low, high = float(max(low, -1.0)), float(min(high, 1.0))
        if alternative == AlternativeHypothesis.GREATER:
            return (low, float("inf"))
        if alternative == AlternativeHypothesis.LESS:
            return (float("-inf"), high)
        return (low, high)

    @staticmethod
    def generate_conclusion_statement(
        variant_name: str,
        is_significant: bool,
        absolute_diff: float,
        uplift: Optional[float],
    ) -> str:
        """
        UI-agnostic summary. Direction comes from the absolute difference, so
        a zero-conversion control still reads correctly.
        """
        if not is_significant:
            return (
                f"Inconclusive / Flat: Boschloo's exact test finds no statistically "
                f"significant difference between '{variant_name}' and the control."
            )
        lift = (
            f"a relative impact of {uplift:+.2%}"
            if uplift is not None
            else "an undefined relative impact (control had zero conversions)"
        )
        if absolute_diff > 0:
            return (
                f"Significant Positive Impact: '{variant_name}' converts better than "
                f"control by {absolute_diff:+.2%} points ({lift}), "
                "per Boschloo's exact test."
            )
        return (
            f"Significant Negative Impact: '{variant_name}' converts worse than "
            f"control by {absolute_diff:+.2%} points ({lift}), "
            "per Boschloo's exact test."
        )

    # ------------------------------------------------------------------ #
    #  Orchestration
    # ------------------------------------------------------------------ #

    def run(self, data: ExperimentInput) -> List[BoschlooResult]:
        """
        Compares each challenger against the control (index 0). For three or
        more variants alpha is Šidák-adjusted, as in run_synthesis.

        Raises ValueError if data.reduction_factor != 1.0 or any arm exceeds
        max_visitors_per_arm.
        """
        if data.reduction_factor != 1.0:
            raise ValueError(
                "Boschloo's exact test assumes pure binomial sampling and cannot "
                f"apply a reduction_factor; got {data.reduction_factor}. Use 1.0."
            )
        largest = max(data.visitors)
        if largest > self.max_visitors_per_arm:
            raise ValueError(
                f"An arm has {largest} visitors, above max_visitors_per_arm="
                f"{self.max_visitors_per_arm}. Boschloo's runtime grows steeply "
                "with sample size; use FrequentistEngine.run_synthesis at this size "
                "or raise the limit deliberately."
            )

        labels = data.labels or [f"Variant {i}" for i in range(len(data.visitors))]
        alpha = apply_sidak(1.0 - data.confidence_level, len(data.visitors))
        ctrl_n, ctrl_conv = data.visitors[0], data.conversions[0]
        p_ctrl = ctrl_conv / ctrl_n

        results = []
        for i in range(1, len(data.visitors)):
            chal_n, chal_conv = data.visitors[i], data.conversions[i]
            p_chal = chal_conv / chal_n
            diff = p_chal - p_ctrl
            uplift = diff / p_ctrl if p_ctrl > 0 else None

            p_value = self.exact_p_value(
                ctrl_conv, ctrl_n, chal_conv, chal_n, data.alternative
            )
            is_sig = bool(p_value < alpha)
            ci = self.newcombe_interval(
                ctrl_conv, ctrl_n, chal_conv, chal_n, alpha, data.alternative
            )

            results.append(
                BoschlooResult(
                    variant_label=labels[i],
                    control_label=labels[0],
                    control_rate=p_ctrl,
                    conversion_rate=p_chal,
                    absolute_diff=diff,
                    uplift=uplift,
                    p_value=p_value,
                    alpha=alpha,
                    is_significant=is_sig,
                    ci_diff=ci,
                    conclusion=self.generate_conclusion_statement(
                        labels[i], is_sig, diff, uplift
                    ),
                )
            )
        return results
