"""
Decision risk: false-positive risk (FPR) and false-negative discovery rate (FNDR).

How likely is a significant result to be a false positive, or a flat result to
have missed a real effect? Both follow from Bayes' rule given alpha, the
planned power and a prior P(H1).

Power must be the *planned* power at a pre-specified MDE, never power at the
observed effect: observed power is a function of the p-value (exactly 50% at
p = alpha) and would make these risks circular.

Two FPR flavours are reported for a significant result:
  - fpr       ('p <= alpha'): the same for every significant result at a given
              alpha, power and prior. Context only.
  - fpr_at_p  ('p-equals', Colquhoun 2017): uses the observed p-value, so it
              reflects how strong the evidence actually is.

Not in scope here: warning thresholds, UI copy, and zero-variance
("insufficient data") checks. Those belong to the caller.
"""

from typing import Dict, Optional

import numpy as np
from scipy.stats import norm

from foe.core.models import AlternativeHypothesis, DecisionRiskResult

# Prior P(H1) presets. Roughly 10-30% of tests across the industry have a real
# positive effect, so skeptical is the default.
PRIOR_PRESETS: Dict[str, float] = {
    "skeptical": 0.10,
    "neutral": 0.50,
    "optimistic": 0.90,
}
DEFAULT_SENSITIVITY_MODE = "skeptical"
DEFAULT_PRIOR = PRIOR_PRESETS[DEFAULT_SENSITIVITY_MODE]

# exp() overflows a float64 just above 709.
_LOG_ODDS_CLIP = 700.0


def _check_open_unit(name: str, value: float) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1, got {value}.")


def _check_closed_unit(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}.")


def _validate(alpha: float, power: float, prior: float) -> None:
    _check_open_unit("alpha", alpha)
    _check_open_unit("power", power)
    _check_closed_unit("prior", prior)


def resolve_prior(
    sensitivity_mode: str = DEFAULT_SENSITIVITY_MODE,
    custom_prior: Optional[float] = None,
) -> float:
    """
    Prior P(H1) from a preset ('skeptical' 0.10, 'neutral' 0.50, 'optimistic'
    0.90) or, with sensitivity_mode='custom', from `custom_prior` in [0, 1]
    (e.g. a historical win rate as a fraction, not a percentage).
    """
    if sensitivity_mode == "custom":
        if custom_prior is None:
            raise ValueError(
                "custom_prior must be provided when sensitivity_mode is 'custom'."
            )
        _check_closed_unit("custom_prior", custom_prior)
        return float(custom_prior)
    if sensitivity_mode not in PRIOR_PRESETS:
        raise ValueError(
            f"Unknown sensitivity_mode '{sensitivity_mode}'. "
            f"Expected one of {list(PRIOR_PRESETS) + ['custom']}."
        )
    return PRIOR_PRESETS[sensitivity_mode]


def false_positive_risk(alpha: float, power: float, prior: float) -> float:
    """
    Standard FPR, P(H0 | significant):

        FPR = alpha*P(H0) / [ alpha*P(H0) + power*P(H1) ]
    """
    _validate(alpha, power, prior)
    p1, p0 = prior, 1.0 - prior
    denominator = alpha * p0 + power * p1
    return float(alpha * p0 / denominator)


def false_negative_discovery_rate(alpha: float, power: float, prior: float) -> float:
    """
    FNDR, P(H1 | not significant), with beta = 1 - power:

        FNDR = beta*P(H1) / [ beta*P(H1) + (1-alpha)*P(H0) ]

    A high FNDR on a flat result means underpowered, not necessarily inert.
    """
    _validate(alpha, power, prior)
    p1, p0 = prior, 1.0 - prior
    beta = 1.0 - power
    denominator = beta * p1 + (1.0 - alpha) * p0
    return float(beta * p1 / denominator)


def _fpr_at_z(
    z: float,
    alpha: float,
    power: float,
    prior: float,
    alternative: AlternativeHypothesis,
) -> float:
    """
    FPR at an observed z. For one-sided tests `z` is the statistic in the
    tested direction; for two-sided tests only |z| matters.

    H1 is the effect at which the test reaches `power` at `alpha`, i.e. a mean
    of delta = z_crit + z_power on the z scale. The likelihood ratio is
    computed in log space so extreme z cannot overflow.
    """
    if prior <= 0.0:
        return 1.0
    if prior >= 1.0:
        return 0.0
    z_power = norm.ppf(power)
    if alternative == AlternativeHypothesis.TWO_SIDED:
        delta = norm.isf(alpha / 2) + z_power
        a = abs(z)
        # H1 is +/-delta with equal weight; density of |z| under H0 is 2*phi(z).
        log_lr = -(delta**2) / 2 + np.logaddexp(a * delta, -a * delta) - np.log(2)
    else:
        delta = norm.isf(alpha) + z_power
        log_lr = z * delta - delta**2 / 2
    log_odds_h1 = np.log(prior / (1.0 - prior)) + log_lr
    return float(1.0 / (1.0 + np.exp(np.clip(log_odds_h1, -_LOG_ODDS_CLIP, _LOG_ODDS_CLIP))))


def fpr_at_p_value(
    p_value: float,
    alpha: float,
    power: float,
    prior: float,
    alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
) -> float:
    """
    FPR at the observed p-value (Colquhoun 2017, 'p-equals' approach).

    The z statistic is rebuilt from the p-value, so this works for any test
    that reports one (z-test, Welch, Mann-Whitney; for the latter two it is a
    normal approximation, adequate for a risk indicator):
      - one-sided:  z = Phi^-1(1 - p). The direction is already in a one-sided
                    p-value, so GREATER and LESS give the same result.
      - two-sided: |z| = Phi^-1(1 - p/2).

    Pass the raw per-comparison p-value with the per-comparison alpha (e.g.
    Šidák-adjusted alpha), not a multiplicity-adjusted p-value.
    """
    _validate(alpha, power, prior)
    _check_closed_unit("p_value", p_value)
    alternative = AlternativeHypothesis(alternative)
    # isf(p) instead of ppf(1 - p) keeps precision for tiny p-values.
    if alternative == AlternativeHypothesis.TWO_SIDED:
        z = norm.isf(p_value / 2)
    else:
        z = norm.isf(p_value)
    return _fpr_at_z(z, alpha, power, prior, alternative)


def assess_risk(
    p_value: float,
    alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
    alpha: float = 0.05,
    power: float = 0.80,
    prior: float = DEFAULT_PRIOR,
) -> DecisionRiskResult:
    """
    All decision-risk numbers for one comparison in a single call.

    Args:
        p_value:     Raw per-comparison p-value of the test.
        alternative: The tested hypothesis.
        alpha:       Per-comparison alpha used for the significance decision
                     (Šidák-adjusted for A/B/n).
        power:       Planned power at the pre-specified MDE, never observed power.
        prior:       P(H1). Use resolve_prior for presets. Defaults to skeptical.

    The result is significant when p_value < alpha, as in FOE's engines.
    fpr and fndr are always filled; fpr_at_p only for significant results.
    """
    alternative = AlternativeHypothesis(alternative)
    _validate(alpha, power, prior)
    _check_closed_unit("p_value", p_value)
    is_significant = bool(p_value < alpha)
    return DecisionRiskResult(
        p_value=float(p_value),
        alternative=alternative,
        alpha=float(alpha),
        power=float(power),
        prior=float(prior),
        is_significant=is_significant,
        fpr=false_positive_risk(alpha, power, prior),
        fndr=false_negative_discovery_rate(alpha, power, prior),
        fpr_at_p=(
            fpr_at_p_value(p_value, alpha, power, prior, alternative)
            if is_significant
            else None
        ),
    )
