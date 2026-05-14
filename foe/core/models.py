from enum import Enum
from datetime import date
from typing import List, Optional, Tuple, Dict
from pydantic import BaseModel, Field, ConfigDict, model_validator

from foe.core.validators import validate_experiment_data


# --- Frequentist ---

class AlternativeHypothesis(str, Enum):
    """Matches the 'tail' logic; string-based for clean JSON serialization."""

    TWO_SIDED = "two-sided"
    GREATER = "greater"
    LESS = "less"


# --- Bayesian ---
# BusinessCaseInput is defined before BayesianResult so that BayesianResult
# can reference it directly without requiring a forward reference or model_rebuild().


class BusinessCaseInput(BaseModel):
    """Inputs for Bayesian risk and business case assessment."""

    model_config = ConfigDict(frozen=True)

    # Dict safely maps variant labels to their AOVs over an API boundary.
    aovs: Dict[str, float] = Field(
        ..., description="Mapping of variant labels to Average Order Value"
    )
    runtime_days: int = Field(..., gt=0)
    projection_period: int = Field(
        183, gt=0, description="Projection period in days (default: 6 months)"
    )
    alpha_prior: float = Field(1.0, gt=0.0)
    beta_prior: float = Field(1.0, gt=0.0)


class ExperimentInput(BaseModel):
    """The raw data and settings for the Axiom Synthesis Engine."""

    model_config = ConfigDict(frozen=True)

    visitors: List[int] = Field(
        ..., description="List of visitor counts per variant", min_length=2
    )
    conversions: List[int] = Field(
        ..., description="List of conversion counts per variant", min_length=2
    )
    alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED
    confidence_level: float = Field(0.95, gt=0.0, lt=1.0)
    reduction_factor: float = Field(
        1.0,
        description=(
            "Variance scaling factor φ. Values below 1.0 shrink standard errors "
            "(stable process); above 1.0 inflate them (overdispersion). Produced by "
            "apply_cuped, run_lin_adjustment, or calculate_aggregate_variance_factor."
        ),
    )
    labels: Optional[List[str]] = Field(
        None, description="Names of the variants (e.g., ['Control', 'Treatment'])"
    )
    biz_case: Optional[BusinessCaseInput] = Field(
        None, description="Optional business case for monetary risk projections (Bayesian only)"
    )

    @model_validator(mode="after")
    def check_statistical_soundness(self) -> "ExperimentInput":
        # ValueError is converted to a clean 422 API error.
        validate_experiment_data(self.visitors, self.conversions)
        return self


class FrequentistResult(BaseModel):
    """
    Standardized output for a single variant comparison.
    The Engine returns a List[FrequentistResult] for multi-variant tests.
    """

    model_config = ConfigDict(frozen=True)

    variant_label: str
    control_label: str
    conversion_rate: float
    standard_error: float
    p_value: float = Field(..., ge=0.0, le=1.0)
    uplift: float
    is_significant: bool
    ci_diff: Tuple[float, float] = Field(
        ...,
        description=(
            "Confidence interval bounds for the difference (lower, upper). "
            "One-sided tests produce an unbounded interval: GREATER yields "
            "(lower, inf) and LESS yields (-inf, upper)."
        ),
    )
    conclusion: str = Field(
        ..., description="Definitive, UI-agnostic summary of the result"
    )
    lower_bound_diff: Optional[float] = None
    is_non_inferior: Optional[bool] = None


class BayesianResult(BaseModel):
    """Standardized output for a single Bayesian variant comparison."""

    model_config = ConfigDict(frozen=True)

    variant_label: str
    control_label: str
    prob_being_best: float
    prob_beat_control: float
    expected_uplift: float
    expected_loss: float
    conclusion: str
    prior_alphas: Optional[List[float]] = None
    prior_betas: Optional[List[float]] = None
    biz_case: Optional[BusinessCaseInput] = None


# --- Sequential ---


class SequentialDataPoint(BaseModel):
    """A single time-series data point for sequential testing algorithms."""

    model_config = ConfigDict(frozen=True)

    date: date
    variant_name: str
    visitors: int = Field(..., ge=0)
    conversions: int = Field(..., ge=0)

    @model_validator(mode="after")
    def check_conversions_within_visitors(self) -> "SequentialDataPoint":
        if self.conversions > self.visitors:
            raise ValueError(
                f"conversions ({self.conversions}) cannot exceed "
                f"visitors ({self.visitors}) for variant '{self.variant_name}' "
                f"on {self.date}."
            )
        return self


class SequentialConfig(BaseModel):
    """Configuration for sequential boundary calculations."""

    model_config = ConfigDict(frozen=True)

    alpha: float = Field(0.05, gt=0.0, lt=1.0, description="Type I error rate.")
    # Renamed from 'beta' to avoid ambiguity with the Beta distribution
    # parameter (beta_prior) used elsewhere in the codebase.
    type_ii_error: float = Field(
        0.20, gt=0.0, lt=1.0, description="Type II error rate (1 - power)."
    )
    tau: float = Field(0.01, gt=0.0, description="Expected difference or threshold")
    num_variants: int = Field(1, ge=1)
    max_visitors: int = Field(10000, gt=0)
    p0: Optional[float] = Field(None, description="Base rate for one-sample tests")
