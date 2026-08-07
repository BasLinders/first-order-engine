from enum import Enum
from datetime import date
from typing import List, Optional, Tuple, Dict, Any, Union
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
        description="Control variant for Gamma/Negative-Binomial post-hoc pairwise comparisons (3+ groups).",
    )
    alpha: float = Field(0.05, gt=0.0, lt=1.0, description="Significance threshold.")
    count_max_unique: int = Field(
        50,
        gt=0,
        description=(
            "Heuristic path only: a KPI with at most this many distinct "
            "non-negative-integer values is treated as a discrete count metric "
            "(e.g. items/tickets per buyer) and routed to Negative Binomial "
            "regression instead of the normality/variance decision tree."
        ),
    )
    count_max_unique_ratio: float = Field(
        0.05,
        gt=0.0,
        lt=1.0,
        description=(
            "Heuristic path only: a KPI is still treated as a count metric if "
            "its distinct-value-to-row-count ratio falls below this threshold, "
            "even when it exceeds count_max_unique (relevant for large datasets "
            "where a genuine count metric can have many distinct values in "
            "absolute terms while remaining a tiny fraction of total rows)."
        ),
    )


class GammaPosthocResult(BaseModel):
    """
    A single pairwise LRT comparison against the control.

    Shared by both the Gamma/two-part path and the Negative Binomial path --
    the two are structurally identical (a likelihood ratio test statistic,
    its p-value, and a Bonferroni-adjusted p-value), so no separate model is
    needed for the count-data case.
    """

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
    # Diagnostics are only populated for the normality/variance heuristic
    # branch; None under Gamma and under the Negative Binomial count-data path.
    is_normal: Optional[bool] = None
    is_homogeneous: Optional[bool] = None
    summary_stats: List[Dict[str, Any]] = Field(default_factory=list)
    posthoc_results: Optional[List[GammaPosthocResult]] = None
    conclusion: str = ""
    # Negative-Binomial dispersion parameter (alpha); populated only when the
    # heuristic path's count-data gate routes to Negative Binomial regression.
    dispersion_alpha: Optional[float] = None
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


# --- Forecasting ---


class ForecastGranularity(str, Enum):
    """Resampling frequency applied to the data before Prophet ever sees it."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class GrowthMode(str, Enum):
    """
    Prophet's trend curve. LOGISTIC is an available option, not a default:
    it requires a concrete capacity ceiling (`cap`) and should only be used
    when one genuinely exists (e.g. a venue's physical capacity).
    """

    LINEAR = "linear"
    LOGISTIC = "logistic"


class SeasonalityMode(str, Enum):
    """
    ADDITIVE: seasonal swings stay a constant absolute size regardless of
    trend. MULTIPLICATIVE: seasonal swings scale with the trend (e.g. a
    growing business's summer peak grows too).
    """

    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"


class CustomHoliday(BaseModel):
    """
    A single user-defined holiday or event row, merged with any other
    custom rows into the one dataframe Prophet expects via its `holidays`
    parameter. Lets Prophet treat this date as an exception to normal
    seasonality (e.g. a park's own event day, a closure, a promotion).
    """

    model_config = ConfigDict(frozen=True)

    holiday: str = Field(..., min_length=1, description="Name of the holiday/event.")
    ds: date
    lower_window: int = Field(
        0, le=0, description="Days before `ds` also treated as part of the event (<= 0)."
    )
    upper_window: int = Field(
        0, ge=0, description="Days after `ds` also treated as part of the event (>= 0)."
    )


class ForecastingEngineConfig(BaseModel):
    """
    Settings envelope for ForecastingEngine.fit(). Fully validated at
    construction time so a caller building this from a UI form gets
    immediate, specific feedback instead of a failure deep inside Prophet.

    The row-level history itself is passed to the engine as a
    ``pandas.DataFrame``, not through this model, for the same reason as
    ContinuousMetricConfig: it is the wrong shape for a JSON settings
    contract.
    """

    model_config = ConfigDict(frozen=True)

    date_col: str = Field(..., description="Name of the date column in the input data.")
    conversions_col: Optional[str] = Field(
        None, description="Column to forecast as 'conversions'."
    )
    revenue_col: Optional[str] = Field(
        None, description="Column to forecast as 'revenue'."
    )

    granularity: ForecastGranularity = ForecastGranularity.DAILY
    periods: int = Field(
        ...,
        gt=0,
        description="Future steps to forecast, in the chosen granularity's units.",
    )

    growth: GrowthMode = GrowthMode.LINEAR
    cap: Optional[float] = Field(
        None, description="Required saturating maximum when growth='logistic'."
    )
    floor: Optional[float] = Field(
        None, description="Optional saturating minimum when growth='logistic'."
    )

    seasonality_mode: SeasonalityMode = SeasonalityMode.MULTIPLICATIVE

    holidays: List[CustomHoliday] = Field(
        default_factory=list,
        description="Custom holidays/events merged into Prophet's holidays dataframe.",
    )

    regressors: List[str] = Field(
        default_factory=list,
        description=(
            "Column names in the input data to add as Prophet regressors. "
            "Fully opt-in -- nothing is added automatically, including weather."
        ),
    )

    interval_width: float = Field(0.95, gt=0.0, lt=1.0)

    cv_horizon_periods: Optional[int] = Field(
        None,
        gt=0,
        description=(
            "Cross-validation horizon, in the chosen granularity's units. "
            "Defaults to `periods` when omitted; cross-validation is skipped "
            "entirely (not run and not faked) when history is too short for "
            "it to mean anything."
        ),
    )

    @model_validator(mode="after")
    def check_targets_and_growth(self) -> "ForecastingEngineConfig":
        if not self.conversions_col and not self.revenue_col:
            raise ValueError(
                "At least one of conversions_col or revenue_col must be set."
            )
        if self.conversions_col and self.conversions_col == self.revenue_col:
            raise ValueError("conversions_col and revenue_col must be different columns.")
        if self.growth == GrowthMode.LOGISTIC:
            if self.cap is None:
                raise ValueError("cap is required when growth='logistic'.")
            if self.floor is not None and self.floor >= self.cap:
                raise ValueError(f"floor ({self.floor}) must be less than cap ({self.cap}).")
        return self


class ForecastPoint(BaseModel):
    """One row of a Prophet forecast, JSON-serializable."""

    model_config = ConfigDict(frozen=True)

    ds: date
    yhat: float
    yhat_lower: float
    yhat_upper: float


class ForecastCVMetrics(BaseModel):
    """Aggregate cross-validation metrics (Prophet's performance_metrics, averaged across cutoffs)."""

    model_config = ConfigDict(frozen=True)

    mape: float = Field(..., ge=0.0)
    rmse: float = Field(..., ge=0.0)
    horizon_periods: int = Field(..., gt=0)


class SingleTargetForecast(BaseModel):
    """Forecast output for one target series (conversions or revenue)."""

    model_config = ConfigDict(frozen=True)

    target: str
    forecast: List[ForecastPoint]
    components: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict,
        description=(
            "Component breakdown records (trend/seasonalities/holidays/"
            "regressors), keyed by component name, for the 'why' behind "
            "the forecast shape."
        ),
    )
    cv_metrics: Optional[ForecastCVMetrics] = None
    warnings: List[str] = Field(default_factory=list)
    conclusion: str = ""


class ForecastingResult(BaseModel):
    """
    Standardized output of ForecastingEngine.fit(). Holds one
    SingleTargetForecast per requested target: conversions and revenue are
    fit as two independent Prophet models rather than one derived from the
    other (see foe.forecasting.operations module docstring for why).
    """

    model_config = ConfigDict(frozen=True)

    granularity: ForecastGranularity
    targets: Dict[str, SingleTargetForecast]
    conclusion: str


# --- Data extraction (BigQuery / GA4) ---
#
# foe.data is a deliberate, opt-in exception to this package's "no I/O"
# design (see README): it is gated behind the `foe[bigquery]` extra and
# isolated from every stats engine above. The models below describe *what*
# to extract and *how to shape it*; like ContinuousMetricConfig and
# ForecastingEngineConfig, they never carry row-level data themselves --
# that stays a pandas.DataFrame, produced by DataEngine and handed to the
# relevant engine (PretestEngine, ForecastingEngine, or an external
# process-mining tool) exactly like a hand-built CSV would be.


class BQConnectionConfig(BaseModel):
    """Identifies which BigQuery project/dataset a DataEngine reads from."""

    model_config = ConfigDict(frozen=True)

    project: str = Field(..., min_length=1)
    dataset: str = Field(
        ..., min_length=1, description="GA4 export dataset, e.g. 'analytics_123456789'."
    )
    location: Optional[str] = Field(
        None,
        description="BigQuery job location (e.g. 'EU', 'US'). Auto-detected from the dataset when omitted.",
    )


class DateRange(BaseModel):
    """An inclusive start/end date pair for a GA4 events_* extraction."""

    model_config = ConfigDict(frozen=True)

    start_date: date
    end_date: date

    @model_validator(mode="after")
    def check_order(self) -> "DateRange":
        if self.start_date > self.end_date:
            raise ValueError(
                f"start_date ({self.start_date}) must be on or before end_date ({self.end_date})."
            )
        return self


class QueryCostEstimate(BaseModel):
    """Result of a BigQuery dry run: a bytes/cost estimate, no rows returned."""

    model_config = ConfigDict(frozen=True)

    bytes_processed: int = Field(..., ge=0)
    gb_processed: float = Field(..., ge=0.0)
    display: str = Field(..., description="Human-readable size, e.g. '482.3 MB'.")
    free_tier_pct: float = Field(
        ..., ge=0.0, description="Percent of the 1 TB monthly free tier this query would consume."
    )
    is_dml: bool = Field(
        False,
        description=(
            "True if the dry run failed because the SQL is a DDL/DML script "
            "(dry run only estimates pure SELECT queries) -- see SequentialExtractionParams."
        ),
    )
    error: Optional[str] = None


class UsageReport(BaseModel):
    """Monthly BigQuery free-tier usage, from INFORMATION_SCHEMA.JOBS_BY_PROJECT."""

    model_config = ConfigDict(frozen=True)

    used_bytes: int = Field(..., ge=0)
    used_gb: float = Field(..., ge=0.0)
    used_display: str
    remaining_bytes: int = Field(..., ge=0)
    remaining_gb: float = Field(..., ge=0.0)
    remaining_display: str
    used_pct: float = Field(..., ge=0.0, le=100.0)
    free_tier_bytes: int = Field(..., gt=0)
    permission_denied: bool = Field(
        False,
        description="True if the caller lacks bigquery.jobs.list on the project (usage could not be read).",
    )
    error: Optional[str] = None


class MatchStrategy(str, Enum):
    """How a raw GA4 event-param string is matched against a known variant string."""

    EXACT = "exact"
    LIKE = "like"


class UserFilterType(str, Enum):
    """
    Scopes an extraction to a subset of users. CONTAINS/REGEX match a
    page_view's page_location; EVENT matches users who fired a named event
    at least once during the date range -- more robust than a URL match
    whenever the URL itself can't differentiate page types.
    """

    CONTAINS = "contains"
    REGEX = "regex"
    EVENT = "event"


class VariantPair(BaseModel):
    """Maps a human-readable variant label ('A', 'B', ...) to its raw GA4 variant string."""

    model_config = ConfigDict(frozen=True)

    label: str = Field(..., min_length=1)
    string: str = Field(..., min_length=1)


class ExperimentDefinition(BaseModel):
    """One experiment's identity and variant map, used by the *ExtractionParams models below."""

    model_config = ConfigDict(frozen=True)

    experiment_id: str = Field(..., min_length=1)
    prefix: str = Field(
        ...,
        min_length=1,
        description="Substring shared by all this experiment's variant strings, used for LIKE matching.",
    )
    variants: List[VariantPair] = Field(..., min_length=1)


class BaselineOutputType(str, Enum):
    BINOMIAL = "binomial"
    REVENUE = "revenue"


class BaselineOutputShape(str, Enum):
    AGGREGATE = "aggregate"
    DAILY = "daily"
    PER_USER = "per_user"  # revenue only -- one row per order, for pre-test distribution fitting


class BaselineExtractionParams(BaseModel):
    """
    Settings for a site-wide baseline export -- feeds PretestEngine's
    sample-size planning, not a running experiment.
    """

    model_config = ConfigDict(frozen=True)

    connection: BQConnectionConfig
    date_range: DateRange
    output_type: BaselineOutputType = BaselineOutputType.BINOMIAL
    output_shape: BaselineOutputShape = BaselineOutputShape.AGGREGATE
    # Adds add-to-cart conversion counts alongside purchase-based ones.
    # Binomial only -- revenue mode has no "conversion" concept to extend.
    kpi_add_to_cart: bool = False
    filter_type: Optional[UserFilterType] = None
    filter_value: str = ""

    @model_validator(mode="after")
    def check_per_user_is_revenue_only(self) -> "BaselineExtractionParams":
        if (
            self.output_shape == BaselineOutputShape.PER_USER
            and self.output_type != BaselineOutputType.REVENUE
        ):
            raise ValueError("output_shape='per_user' is only meaningful for output_type='revenue'.")
        return self


class BinomialExtractionParams(BaseModel):
    """Settings for a binomial (conversion-rate) experiment export."""

    model_config = ConfigDict(frozen=True)

    connection: BQConnectionConfig
    date_range: DateRange
    param_key: str = Field(
        ...,
        min_length=1,
        description="GA4 event-param key carrying the variant string, e.g. 'exp_variant_string'.",
    )
    match_strategy: MatchStrategy
    # Exactly one -- unlike InteractionExtractionParams, this builder
    # processes a single experiment per call (build_binomial only ever
    # reads experiments[0]). A list of length > 1 would silently have its
    # extra entries ignored, so it's capped at 1 here rather than left to
    # surprise a caller who reasonably assumed a list meant "one or more".
    experiments: List[ExperimentDefinition] = Field(..., min_length=1, max_length=1)
    post_exposure_filter: bool = True
    # KPI toggles -- zero-cost group (no extra table scan)
    kpi_transactions: bool = True
    kpi_add_to_cart: bool = True
    kpi_aov: bool = True
    kpi_ideal: bool = False
    kpi_device_split: bool = True
    # KPI toggles -- cost-warning group (adds a page_view scan)
    kpi_login: bool = False
    kpi_create_account: bool = False
    filter_type: Optional[UserFilterType] = None
    filter_value: str = ""


class ContinuousQueryMode(str, Enum):
    ALL_USERS = "all_users"
    REVENUE_ONLY = "revenue_only"


class DeviceFilter(str, Enum):
    ALL = "all"
    DESKTOP = "desktop"
    MOBILE = "mobile"


class ContinuousExtractionParams(BaseModel):
    """Settings for a continuous-metric (revenue/AOV) experiment export."""

    model_config = ConfigDict(frozen=True)

    connection: BQConnectionConfig
    date_range: DateRange
    param_key: str = Field(..., min_length=1)
    match_strategy: MatchStrategy
    # Exactly one -- see BinomialExtractionParams.experiments for why.
    experiments: List[ExperimentDefinition] = Field(..., min_length=1, max_length=1)
    device_filter: DeviceFilter = DeviceFilter.ALL
    query_mode: ContinuousQueryMode = ContinuousQueryMode.ALL_USERS
    post_exposure_filter: bool = True
    filter_type: Optional[UserFilterType] = None
    filter_value: str = ""


class SequentialExtractionParams(BaseModel):
    """
    Settings for a sequential-test export. Unlike the other extraction
    modes, this compiles to a full DDL/DML script (it creates/reads a
    persistent cumulative table), not a pure SELECT -- BigQuery dry run
    cannot cost it (see QueryCostEstimate.is_dml).
    """

    model_config = ConfigDict(frozen=True)

    connection: BQConnectionConfig
    date_range: DateRange
    param_key: str = Field(..., min_length=1)
    # Exactly one -- see BinomialExtractionParams.experiments for why.
    experiments: List[ExperimentDefinition] = Field(..., min_length=1, max_length=1)
    use_persistence: bool = True
    reset_cumulative_data: bool = False
    cumulative_table: str = Field(
        "",
        description=(
            "Fully-qualified 'project.dataset.table' to persist cumulative results to. "
            "Defaults to a placeholder table name when omitted."
        ),
    )
    kpi_transactions: bool = True
    kpi_add_to_cart: bool = True
    kpi_aov: bool = True
    kpi_ideal: bool = False
    kpi_device_split: bool = True
    kpi_login: bool = False
    kpi_create_account: bool = False


class InteractionExtractionParams(BaseModel):
    """
    Settings for an interaction export -- classifies users by which
    combination of two-or-more experiments' variants they were exposed to.
    """

    model_config = ConfigDict(frozen=True)

    connection: BQConnectionConfig
    date_range: DateRange
    param_key: str = Field(..., min_length=1)
    experiments: List[ExperimentDefinition] = Field(
        ...,
        min_length=2,
        description="Two or more experiments; each experiment's variants must be labeled 'A' and 'B'.",
    )
    kpi_transactions: bool = True
    kpi_add_to_cart: bool = True

    @model_validator(mode="after")
    def check_each_experiment_has_a_and_b_variants(self) -> "InteractionExtractionParams":
        # build_interaction's classification CASE WHEN looks up each
        # experiment's 'A'/'B'-labeled variant strings specifically (falling
        # back to '' when a label is missing) and its final WHERE clause
        # excludes any user whose classification came out ''. A typo'd or
        # differently-labeled variant (e.g. 'Control'/'Treatment' instead
        # of 'A'/'B') wouldn't error -- every user in that experiment would
        # silently classify as '' and be dropped from the result entirely.
        for exp in self.experiments:
            labels = {v.label for v in exp.variants}
            if not {"A", "B"}.issubset(labels):
                raise ValueError(
                    f"Experiment '{exp.experiment_id}' must have variants labeled 'A' and 'B' "
                    f"for interaction classification -- got labels {sorted(labels)}."
                )
        return self


ExperimentExtractionParams = Union[
    BinomialExtractionParams,
    ContinuousExtractionParams,
    SequentialExtractionParams,
    InteractionExtractionParams,
]


# --- Event log extraction (process mining) ---


class EventLogExtractionParams(BaseModel):
    """
    Settings for a raw GA4 event-log export shaped for process mining: one
    row per (case, activity, timestamp) -- the standard XES-style triple --
    plus any caller-requested attribute columns. No aggregation: downstream
    process-mining tools (e.g. pm4py) expect row-level events, not summaries.
    """

    model_config = ConfigDict(frozen=True)

    connection: BQConnectionConfig
    date_range: DateRange
    case_id_col: str = Field(
        "user_pseudo_id", description="GA4 column identifying a case (default: one case per user)."
    )
    activity_col: str = Field(
        "event_name", description="GA4 column identifying an activity. Almost always 'event_name'."
    )
    event_names: List[str] = Field(
        default_factory=list, description="Restrict to these event names. Empty = every event in range."
    )
    attribute_params: List[str] = Field(
        default_factory=list,
        description=(
            "event_params keys to unnest as extra event-attribute columns "
            "(e.g. ['page_location', 'page_title']). Only pulls each key's "
            "string_value -- int/float/double-valued params (e.g. "
            "ga_session_id, value, engagement_time_msec) come back NULL."
        ),
    )
    filter_type: Optional[UserFilterType] = None
    filter_value: str = ""


# --- Time-series extraction (forecasting) ---


class TimeSeriesMetric(str, Enum):
    """A metric extract_timeseries can compute per day."""

    VISITORS = "visitors"
    CONVERSIONS = "conversions"
    REVENUE = "revenue"
    TRANSACTIONS = "transactions"
    EVENT_COUNT = "event_count"  # count of a caller-named custom event


class TimeSeriesExtractionParams(BaseModel):
    """
    Settings for a daily time-series export -- feeds ForecastingEngine's
    date_col/conversions_col/revenue_col contract directly. Always pulled
    at daily grain: ForecastingEngine resamples to weekly/monthly itself
    (see foe.forecasting.operations), so pulling anything coarser here
    would throw away information for no benefit.
    """

    model_config = ConfigDict(frozen=True)

    connection: BQConnectionConfig
    date_range: DateRange
    metrics: List[TimeSeriesMetric] = Field(..., min_length=1)
    conversion_event: str = Field(
        "purchase", description="Event name counted for the 'conversions' metric."
    )
    custom_event_name: Optional[str] = Field(
        None, description="Required when metrics includes EVENT_COUNT -- the event to count."
    )
    segment_col: Optional[str] = Field(
        None,
        description=(
            "Optional column to split rows by (e.g. 'device.category'), producing one row "
            "per date+segment instead of one row per date."
        ),
    )
    filter_type: Optional[UserFilterType] = None
    filter_value: str = ""

    @model_validator(mode="after")
    def check_event_count_has_event_name(self) -> "TimeSeriesExtractionParams":
        if TimeSeriesMetric.EVENT_COUNT in self.metrics and not self.custom_event_name:
            raise ValueError("custom_event_name is required when metrics includes EVENT_COUNT.")
        return self
