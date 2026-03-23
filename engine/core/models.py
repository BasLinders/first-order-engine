from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

# --- Frequentist ---

class AlternativeHypothesis(Enum):
    """Matches the 'tail' logic in your original experiment_analysis.py."""
    TWO_SIDED = "Two-sided"
    GREATER = "Greater"
    LESS = "Less"

@dataclass(frozen=True)
class ExperimentInput:
    """The raw data and settings for the Axiom Synthesis Engine."""
    visitors: List[int]
    conversions: List[int]
    alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED
    confidence_level: float = 0.95
    reduction_factor: float = 1.0  # CUPED adjustment
    labels: Optional[List[str]] = None

@dataclass(frozen=True)
class FrequentistResult:
    """
    Standardized output for a single variant comparison.
    The Engine will return a List[FrequentistResult] for multi-variant tests.
    """
    variant_label: str
    control_label: str
    conversion_rate: float
    standard_error: float
    p_value: float
    uplift: float
    is_significant: bool
    # CI for the difference (diff_cr - moe, diff_cr + moe)
    ci_diff: Tuple[float, float]
    observed_power: float
    # These match your NI logic
    lower_bound_diff: Optional[float] = None
    is_non_inferior: Optional[bool] = None

# --- Bayesian ---

@dataclass(frozen=True)
class BusinessCaseInput:
    aovs: List[float]
    runtime_days: int
    projection_period: int = 183  # 6 months default
    alpha_prior: float = 1.0
    beta_prior: float = 1.0
