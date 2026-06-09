from enum import Enum
from datetime import date
from typing import List, Optional, Tuple, Dict, Any
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


# --- Continuous metric analysis ---


class AnalysisUnit(str, Enum):
    """
    The unit one row of the metric represents. Determines zero handling and,
    for the Gamma family, which likelihood model is used.

    PER_VISITOR
        Rows are visitors; non-buyers count as 0 and are kept. Captures both
        conversion-rate and spend effects. The Gamma path uses a two-part
        (hurdle) model: Bernoulli(convert) x Gamma(spend | convert).
    PER_TRANSACTION
        Rows are orders; zero-value rows are excluded. Measures order value
        among buyers only. The Gamma path uses a single Gamma per variant.
    """

    PER_VISITOR = "per_visitor"
    PER_TRANSACTION = "per_transaction"


class ContinuousApproach(str, Enum):
    """Which analysis path the engine should take."""

    HEURISTIC = "heuristic"          # normality/variance decision tree
    GAMMA_GLM = "gamma"              # Gamma / two-part likelihood-ratio test


class VarianceScaling(str, Enum):
    """
    How the treatment group's variance is assumed to relate to the control's
    when planning a continuous-metric test (sample size / power).

    EQUAL
        Homoscedastic: treatment variance == control variance (the standard
        two-sample default).
    CV_CONSTANT
        The lift scales the mean and holds the coefficient of variation fixed,
        so the treatment variance grows as sigma**2 * (1 + r)**2. For a positive
        lift this is the more conservative choice (more sample / less power).
    """

    EQUAL = "equal"
    CV_CONSTANT = "cv_constant"


class ContinuousMetricConfig(BaseModel):
    """
    Settings envelope for a continuous-metric comparison.

    The row-level data itself is passed to the engine as a ``pandas.DataFrame``
    (a SQL/CSV export), not through this model: encoding hundreds of thousands
    of rows as JSON would be wasteful and is the wrong shape for distributional
    analysis. This frozen model is the JSON-serializable contract carrying only
    the settings that select and parameterize the test.
    """

    model_config = ConfigDict(frozen=True)

    kpi: str = Field(..., description="Name of the numeric metric column to test.")
    group_col: str = Field(
        "experience_variant_label",
        description="Name of the categorical variant column.",
    )
    approach: ContinuousApproach = ContinuousApproach.HEURISTIC
    unit: AnalysisUnit = AnalysisUnit.PER_TRANSACTION
    control_label: Optional[str] = Field(
        None,
        description="Control variant for Gamma post-hoc pairwise comparisons (3+ groups).",
    )
    alpha: float = Field(0.05, gt=0.0, lt=1.0, description="Significance threshold.")


class GammaPosthocResult(BaseModel):
    """A single pairwise Gamma / two-part LRT comparison against the control."""

    model_config = ConfigDict(frozen=True)

    comparison: str
    lrt_stat: float
    p_value: float = Field(..., ge=0.0, le=1.0)
    p_adj_bonferroni: float = Field(..., ge=0.0, le=1.0)
    is_significant: bool


class ContinuousMetricResult(BaseModel):
    """
    Standardized, JSON-serializable output for a continuous-metric comparison.
    Replaces the previous loose result dict.
    """

    model_config = ConfigDict(frozen=True)

    kpi: str
    approach_used: ContinuousApproach
    unit: AnalysisUnit
    test_name: str
    p_value: float = Field(..., ge=0.0, le=1.0)
    is_significant: bool
    # Diagnostics are only populated for the heuristic path; None under Gamma.
    is_normal: Optional[bool] = None
    is_homogeneous: Optional[bool] = None
    summary_stats: List[Dict[str, Any]] = Field(default_factory=list)
    posthoc_results: Optional[List[GammaPosthocResult]] = None
    conclusion: str = ""
    # Non-fatal interpretive notices (e.g. dropped zero rows, no zeros found).
    warnings: List[str] = Field(default_factory=list)


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
