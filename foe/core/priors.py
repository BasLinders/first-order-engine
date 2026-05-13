from typing import Tuple


def get_informed_beta_prior(
    hist_conversions: int,
    hist_visitors: int,
    weight: float = 0.1,
) -> Tuple[float, float]:
    """
    Calculates Alpha and Beta parameters for a Beta distribution based on
    historical performance and a confidence weight.

    The weight controls what *fraction* of the historical data is used as
    pseudo-observations in the prior. Prior strength therefore scales with
    both weight and the magnitude of hist_visitors:

        alpha = (hist_conversions * weight) + 1
        beta  = ((hist_visitors - hist_conversions) * weight) + 1

    Boundary behaviour:
        weight = 0.0  ->  Beta(1, 1)  — fully uninformative prior.
        weight = 1.0  ->  Beta(hist_conversions + 1, hist_non_conversions + 1)
                          — entire history used as pseudo-observations.

    Note: a weight of 1.0 with hist_visitors=50 contributes ~50 pseudo-
    observations (a weak prior). The same weight with hist_visitors=50,000
    contributes ~50,000 (an extremely strong prior). Adjust weight and the
    historical window together to control prior strength.

    Args:
        hist_conversions: Total conversions observed in the historical period.
                          Must be >= 0 and <= hist_visitors.
        hist_visitors:    Total visitors observed in the historical period.
                          Must be >= 0.
        weight:           Fraction of historical data to use as prior
                          pseudo-observations. Clamped to [0.0, 1.0].

    Returns:
        Tuple of (alpha, beta) parameters for a Beta distribution.

    Raises:
        ValueError: If hist_visitors or hist_conversions are negative, or if
                    hist_conversions exceeds hist_visitors.
    """
    if hist_visitors < 0:
        raise ValueError(
            f"hist_visitors must be >= 0, got {hist_visitors}."
        )
    if hist_conversions < 0:
        raise ValueError(
            f"hist_conversions must be >= 0, got {hist_conversions}."
        )
    if hist_conversions > hist_visitors:
        raise ValueError(
            f"hist_conversions ({hist_conversions}) cannot exceed "
            f"hist_visitors ({hist_visitors})."
        )

    weight = float(max(0.0, min(1.0, weight)))

    alpha = (hist_conversions * weight) + 1
    beta = ((hist_visitors - hist_conversions) * weight) + 1

    return float(alpha), float(beta)
