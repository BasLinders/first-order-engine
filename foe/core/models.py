from enum import Enum
from datetime import date  # noqa: F401
from typing import List, Optional, Tuple, Dict
from pydantic import BaseModel, Field, ConfigDict, model_validator
from foe.core.validators import validate_experiment_data

# --- Frequentist ---


class AlternativeHypothesis(str, Enum):
    """Matches the 'tail' logic; string-based for clean JSON serialization."""

    TWO_SIDED = "two-sided"
    GREATER = "greater"
    LESS = "less"


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
    confidence_level: float = Field(0.95, ge=0.0, lt=1.0)
    reduction_factor: float = Field(1.0, description="CUPED adjustment factor")
    labels: Optional[List[str]] = Field(
        None, description="Names of the variants (e.g., ['Control', 'Treatment'])"
    )

    @model_validator(mode="after")
    def check_statistical_soundness(self) -> "ExperimentInput":
        # ValueError is converted to clean 422 API error.
        validate_experiment_data(self.visitors, self.conversions)
        return self


class FrequentistResult(BaseModel):
    """
    Standardized output for a single variant comparison.
    The Engine will return a List[FrequentistResult] for multi-variant tests.
    """

    model_config = ConfigDict(frozen=True)

    variant_label: str
    control_label: str
    conversion_rate: float
    standard_error: float
    p_value: float = Field(..., ge=0.0, le=1.0)
    uplift: float
    is_significant: bool
    # CI for the difference (diff_cr - moe, diff_cr + moe)
    ci_diff: Tuple[float, float] = Field(
        ..., description="Confidence interval bounds: (lower, upper)"
    )
    conclusion: str = Field(
        ..., description="Definitive, UI-agnostic summary of the result"
    )

    # These match your NI logic
    lower_bound_diff: Optional[float] = None
    is_non_inferior: Optional[bool] = None


# --- Bayesian ---


class BayesianResult(BaseModel):
    """Standardized output for a single Bayesian variant comparison."""

    model_config = ConfigDict(frozen=True)

    variant_label: str
    control_label: str
    prob_being_best: float
    expected_loss: float
    conclusion: str


class BusinessCaseInput(BaseModel):
    """Inputs for Bayesian risk and business case assessment."""

    model_config = ConfigDict(frozen=True)

    # Updated to a Dict to safely map variant labels to their AOVs over an API
    aovs: Dict[str, float] = Field(
        ..., description="Mapping of variant labels to Average Order Value"
    )
    runtime_days: int = Field(..., gt=0)
    projection_period: int = Field(
        183, gt=0, description="Projection period in days (6 months default)"
    )
    alpha_prior: float = Field(1.0, gt=0.0)
    beta_prior: float = Field(1.0, gt=0.0)


# --- Sequential ---


class SequentialDataPoint(BaseModel):
    """A single time-series data point for sequential testing algorithms."""

    model_config = ConfigDict(frozen=True)

    date: date
    variant_name: str
    visitors: int = Field(..., ge=0)
    conversions: int = Field(..., ge=0)


class SequentialConfig(BaseModel):
    """Configuration for sequential boundary calculations."""

    model_config = ConfigDict(frozen=True)

    alpha: float = Field(0.05, gt=0.0, lt=1.0)
    beta: float = Field(0.20, gt=0.0, lt=1.0)
    tau: float = Field(0.01, gt=0.0, description="Expected difference or threshold")
    num_variants: int = Field(1, ge=1)
    max_visitors: int = Field(10000, gt=0)
    p0: Optional[float] = Field(None, description="Base rate for one-sample tests")
