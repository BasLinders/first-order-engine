from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

@dataclass(frozen=True)
class ExperimentInput:
    """The raw data required to initialize any analysis."""
    visitors: List[int]
    conversions: List[int]
    labels: Optional[List[str]] = None

@dataclass(frozen=True)
class FrequentistResult:
    """Standardized output for frequentist analysis results."""
    variant_label: str
    control_label: str
    p_value: float
    uplift: float
    is_significant: bool
    confidence_interval: Tuple[float, float]
    adjusted_alpha: float
    lower_bound_diff: Optional[float] = None
    is_non_inferior: Optional[bool] = None
