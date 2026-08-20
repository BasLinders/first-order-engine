"""
tests/test_data_engine.py

Tests for foe.data.sql (pure SQL-string builders) and the extraction
Pydantic models in foe.core.models. foe/data/engine.py (DataEngine itself)
is intentionally not exercised here -- it is a thin wrapper around
google-cloud-bigquery/google-auth-oauthlib (an optional extra) and is
excluded from coverage for the same reason ForecastingEngine's Prophet
internals and the other I/O-adjacent modules are (see pyproject.toml).
These tests instead lock down the SQL each builder produces: no live or
mocked BigQuery client is needed since nothing here executes a query.
"""

from datetime import date

import pytest
from pydantic import ValidationError

from foe.core.models import (
    BaselineExtractionParams,
    BaselineOutputShape,
    BaselineOutputType,
    BinomialExtractionParams,
    BQConnectionConfig,
    ContinuousExtractionParams,
    ContinuousQueryMode,
    DateRange,
    DeviceFilter,
    EventLogExtractionParams,
    ExperimentDefinition,
    InteractionExtractionParams,
    MatchStrategy,
    SequentialExtractionParams,
    TimeSeriesExtractionParams,
    TimeSeriesMetric,
    UserFilterType,
    VariantPair,
)
from foe.data.sql import event_log as event_log_sql
from foe.data.sql import experiments as ex
from foe.data.sql import ga4
from foe.data.sql import timeseries as ts_sql


# --------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------- #

CONN = BQConnectionConfig(project="my-proj", dataset="analytics_123")
RANGE = DateRange(start_date=date(2026, 1, 1), end_date=date(2026, 1, 31))


def make_experiment(exp_id="exp1", prefix="EXP1", a="EXP1_control", b="EXP1_variant") -> ExperimentDefinition:
    return ExperimentDefinition(
        experiment_id=exp_id,
        prefix=prefix,
        variants=[VariantPair(label="A", string=a), VariantPair(label="B", string=b)],
    )


# --------------------------------------------------------------------- #
#  foe.data.sql.ga4 -- shared primitives
# --------------------------------------------------------------------- #


def test_table_ref_backtick_quotes_wildcard_table():
    assert ga4.table_ref("proj", "ds") == "`proj.ds.events_*`"


def test_suffix_filter_bare_and_aliased():
    bare = ga4.suffix_filter("2026-01-01", "2026-01-31")
    assert bare.startswith("_TABLE_SUFFIX BETWEEN")
    aliased = ga4.suffix_filter("2026-01-01", "2026-01-31", alias="e")
    assert aliased.startswith("e._TABLE_SUFFIX BETWEEN")


def test_escape_literal_escapes_quotes_and_backslashes():
    assert ga4.escape_literal("O'Brien") == "O\\'Brien"
    assert ga4.escape_literal("a\\b") == "a\\\\b"


def test_validate_identifier_accepts_dotted_field_and_rejects_injection():
    assert ga4.validate_identifier("device.category") == "device.category"
    with pytest.raises(ValueError, match="identifier"):
        ga4.validate_identifier("device.category; DROP TABLE x")


def test_validate_table_ref_allows_hyphenated_project_ids():
    assert ga4.validate_table_ref("my-project-123.ds.tbl") == "my-project-123.ds.tbl"
    with pytest.raises(ValueError):
        ga4.validate_table_ref("my project.ds.tbl")


def test_table_ref_rejects_backtick_in_project_or_dataset():
    # An unvalidated project/dataset would let a stray backtick break out of
    # the backtick-quoted table reference -- table_ref is the single
    # most-used builder in the package, so this is a direct injection
    # surface if left unchecked.
    with pytest.raises(ValueError):
        ga4.table_ref("proj`.evil.events_* --", "ds")
    with pytest.raises(ValueError):
        ga4.table_ref("proj", "ds`; DROP TABLE x")


def test_escape_raw_string_literal_escapes_only_the_quote():
    assert ga4.escape_raw_string_literal("O'Brien") == "O\\'Brien"
    # Backslashes must survive untouched -- doubling them would corrupt a
    # caller's intended regex escapes (\d, \., etc.).
    assert ga4.escape_raw_string_literal(r"\d+") == r"\d+"


# --------------------------------------------------------------------- #
#  BaselineExtractionParams / build_baseline
# --------------------------------------------------------------------- #


def test_baseline_per_user_requires_revenue_output_type():
    with pytest.raises(ValidationError, match="per_user"):
        BaselineExtractionParams(
            connection=CONN,
            date_range=RANGE,
            output_type=BaselineOutputType.BINOMIAL,
            output_shape=BaselineOutputShape.PER_USER,
        )


@pytest.mark.parametrize(
    "output_shape,output_type",
    [
        (BaselineOutputShape.AGGREGATE, BaselineOutputType.BINOMIAL),
        (BaselineOutputShape.AGGREGATE, BaselineOutputType.REVENUE),
        (BaselineOutputShape.DAILY, BaselineOutputType.BINOMIAL),
        (BaselineOutputShape.DAILY, BaselineOutputType.REVENUE),
        (BaselineOutputShape.PER_USER, BaselineOutputType.REVENUE),
    ],
)
def test_build_baseline_every_shape_and_type_produces_valid_sql_skeleton(output_shape, output_type):
    params = BaselineExtractionParams(
        connection=CONN, date_range=RANGE, output_shape=output_shape, output_type=output_type
    )
    sql = ex.build_baseline(params)
    assert "`my-proj.analytics_123.events_*`" in sql
    assert "2026-01-01" in sql and "2026-01-31" in sql
    assert sql.rstrip().endswith(";")


def test_build_baseline_with_contains_filter_adds_page_view_cte():
    params = BaselineExtractionParams(
        connection=CONN,
        date_range=RANGE,
        filter_type=UserFilterType.CONTAINS,
        filter_value="checkout",
    )
    sql = ex.build_baseline(params)
    assert "filtered_users AS" in sql
    assert "page_view" in sql
    assert "checkout" in sql


def test_build_baseline_with_event_filter_scopes_to_named_event():
    params = BaselineExtractionParams(
        connection=CONN, date_range=RANGE, filter_type=UserFilterType.EVENT, filter_value="add_to_cart"
    )
    sql = ex.build_baseline(params)
    assert "event_name = 'add_to_cart'" in sql


def test_build_baseline_limit_appended():
    params = BaselineExtractionParams(connection=CONN, date_range=RANGE)
    sql = ex.build_baseline(params, limit=10)
    assert "LIMIT 10" in sql


def test_build_baseline_escapes_quote_in_filter_value():
    params = BaselineExtractionParams(
        connection=CONN, date_range=RANGE, filter_type=UserFilterType.EVENT, filter_value="o'brien"
    )
    sql = ex.build_baseline(params)
    assert "event_name = 'o\\'brien'" in sql


def test_build_baseline_regex_filter_escapes_quote_without_touching_backslashes():
    # A quote in a REGEX filter_value must not be able to close the r'...'
    # literal early; the backslash in \d must survive untouched so the
    # regex still means "one or more digits", not something else.
    params = BaselineExtractionParams(
        connection=CONN, date_range=RANGE, filter_type=UserFilterType.REGEX, filter_value=r"o'brien\d+"
    )
    sql = ex.build_baseline(params)
    assert r"REGEXP_CONTAINS(params.value.string_value, r'o\'brien\d+')" in sql


# --------------------------------------------------------------------- #
#  BinomialExtractionParams / build_binomial
# --------------------------------------------------------------------- #


def test_build_binomial_includes_requested_kpi_ctes_only():
    params = BinomialExtractionParams(
        connection=CONN,
        date_range=RANGE,
        param_key="exp_variant_string",
        match_strategy=MatchStrategy.EXACT,
        experiments=[make_experiment()],
        kpi_login=True,
        kpi_ideal=True,
        kpi_create_account=False,
    )
    sql = ex.build_binomial(params)
    assert "login_data AS" in sql
    assert "ideal_users AS" in sql
    assert "create_account_data AS" not in sql
    assert "EXP1_control" in sql and "EXP1_variant" in sql


def test_build_binomial_like_strategy_uses_prefix_and_like_operator():
    params = BinomialExtractionParams(
        connection=CONN,
        date_range=RANGE,
        param_key="exp_variant_string",
        match_strategy=MatchStrategy.LIKE,
        experiments=[make_experiment(prefix="EXP1")],
    )
    sql = ex.build_binomial(params)
    assert "LIKE '%EXP1%'" in sql
    assert "exp_variant_string LIKE '%EXP1_control%'" in sql


def test_binomial_experiments_rejects_more_than_one():
    # build_binomial only ever reads experiments[0] -- a second entry would
    # silently be dropped rather than processed, so the model caps the list
    # at 1 instead of accepting and quietly ignoring extras.
    with pytest.raises(ValidationError):
        BinomialExtractionParams(
            connection=CONN,
            date_range=RANGE,
            param_key="k",
            match_strategy=MatchStrategy.EXACT,
            experiments=[make_experiment(exp_id="exp1"), make_experiment(
                exp_id="exp2", prefix="EXP2")],
        )


def test_build_binomial_without_post_exposure_filter_omits_timestamp_gate():
    params = BinomialExtractionParams(
        connection=CONN,
        date_range=RANGE,
        param_key="exp_variant_string",
        match_strategy=MatchStrategy.EXACT,
        experiments=[make_experiment()],
        post_exposure_filter=False,
    )
    sql = ex.build_binomial(params)
    assert "first_exposure_timestamp" not in sql


# --------------------------------------------------------------------- #
#  ContinuousExtractionParams / build_continuous
# --------------------------------------------------------------------- #


def test_build_continuous_device_filter_and_query_mode():
    params = ContinuousExtractionParams(
        connection=CONN,
        date_range=RANGE,
        param_key="exp_variant_string",
        match_strategy=MatchStrategy.EXACT,
        experiments=[make_experiment()],
        device_filter=DeviceFilter.MOBILE,
        query_mode=ContinuousQueryMode.REVENUE_ONLY,
    )
    sql = ex.build_continuous(params)
    assert "device_filter STRING DEFAULT 'mobile'" in sql
    assert "INNER JOIN ecommerce_data" in sql


def test_build_continuous_all_users_mode_left_joins_ecommerce():
    params = ContinuousExtractionParams(
        connection=CONN,
        date_range=RANGE,
        param_key="exp_variant_string",
        match_strategy=MatchStrategy.EXACT,
        experiments=[make_experiment()],
        query_mode=ContinuousQueryMode.ALL_USERS,
    )
    sql = ex.build_continuous(params)
    assert "LEFT JOIN ecommerce_data" in sql


# --------------------------------------------------------------------- #
#  Shared scan
# --------------------------------------------------------------------- #


def test_experiment_shared_scan_flags_reflect_cost_warning_kpis_and_filters():
    binomial = BinomialExtractionParams(
        connection=CONN,
        date_range=RANGE,
        param_key="k",
        match_strategy=MatchStrategy.EXACT,
        experiments=[make_experiment()],
        kpi_login=True,
        kpi_ideal=True,
    )
    need_page_location, need_payment_type = ex.experiment_shared_scan_flags(
        binomial, None)
    assert need_page_location is True
    assert need_payment_type is True

    plain_binomial = BinomialExtractionParams(
        connection=CONN,
        date_range=RANGE,
        param_key="k",
        match_strategy=MatchStrategy.EXACT,
        experiments=[make_experiment()],
        kpi_login=False,
        kpi_ideal=False,
    )
    need_page_location, need_payment_type = ex.experiment_shared_scan_flags(
        plain_binomial, None)
    assert need_page_location is False
    assert need_payment_type is False


def test_build_binomial_from_shared_scan_has_no_leading_with_or_trailing_semicolon():
    params = BinomialExtractionParams(
        connection=CONN,
        date_range=RANGE,
        param_key="k",
        match_strategy=MatchStrategy.EXACT,
        experiments=[make_experiment()],
    )
    cte_chain = ex.build_binomial_from_shared_scan(params)
    assert not cte_chain.lstrip().upper().startswith("WITH")
    assert not cte_chain.rstrip().endswith(";")


def test_shared_scan_single_output_wrapper_is_valid_sql_skeleton():
    binomial = BinomialExtractionParams(
        connection=CONN,
        date_range=RANGE,
        param_key="k",
        match_strategy=MatchStrategy.EXACT,
        experiments=[make_experiment()],
    )
    shared_select = ex.build_shared_scan_select(
        CONN.project, CONN.dataset, "2026-01-01", "2026-01-31", "k")
    cte_chain = ex.build_binomial_from_shared_scan(binomial)
    sql = ex.build_experiment_single_output_sql(shared_select, cte_chain)
    assert sql.strip().startswith("--")
    assert "WITH shared_scan AS (" in sql
    assert sql.rstrip().endswith(";")


# --------------------------------------------------------------------- #
#  SequentialExtractionParams / build_sequential
# --------------------------------------------------------------------- #


def test_build_sequential_uses_custom_cumulative_table_when_given():
    params = SequentialExtractionParams(
        connection=CONN,
        date_range=RANGE,
        param_key="k",
        experiments=[make_experiment()],
        cumulative_table="my-proj.ds.cumulative",
    )
    sql = ex.build_sequential(params)
    assert "`my-proj.ds.cumulative`" in sql
    assert "`project.dataset.cumulative_test_data`" not in sql


def test_build_sequential_defaults_cumulative_table_when_omitted():
    params = SequentialExtractionParams(
        connection=CONN, date_range=RANGE, param_key="k", experiments=[make_experiment()]
    )
    sql = ex.build_sequential(params)
    assert "`project.dataset.cumulative_test_data`" in sql


def test_build_sequential_rejects_unsafe_cumulative_table():
    params = SequentialExtractionParams(
        connection=CONN,
        date_range=RANGE,
        param_key="k",
        experiments=[make_experiment()],
        cumulative_table="my proj`; DROP TABLE x --",
    )
    with pytest.raises(ValueError):
        ex.build_sequential(params)


def test_build_sequential_persistence_flags_declared_as_bool_literals():
    params = SequentialExtractionParams(
        connection=CONN,
        date_range=RANGE,
        param_key="k",
        experiments=[make_experiment()],
        use_persistence=False,
        reset_cumulative_data=True,
    )
    sql = ex.build_sequential(params)
    assert "DEFAULT FALSE" in sql
    assert "DEFAULT TRUE" in sql


# --------------------------------------------------------------------- #
#  InteractionExtractionParams / build_interaction
# --------------------------------------------------------------------- #


def test_interaction_requires_at_least_two_experiments():
    with pytest.raises(ValidationError):
        InteractionExtractionParams(
            connection=CONN, date_range=RANGE, param_key="k", experiments=[make_experiment()]
        )


def test_interaction_rejects_experiment_without_a_and_b_labels():
    # Without an 'A'/'B'-labeled variant, build_interaction's CASE WHEN
    # silently classifies every one of that experiment's users as '' and
    # the final WHERE clause drops them all -- no error, just an empty
    # result. The model now catches this at construction time instead.
    mislabeled = ExperimentDefinition(
        experiment_id="exp1",
        prefix="EXP1",
        variants=[VariantPair(label="Control", string="EXP1_control"), VariantPair(
            label="Treatment", string="EXP1_variant")],
    )
    with pytest.raises(ValidationError, match="labeled 'A' and 'B'"):
        InteractionExtractionParams(
            connection=CONN, date_range=RANGE, param_key="k", experiments=[mislabeled, make_experiment(exp_id="exp2")]
        )


def test_build_interaction_two_experiments_builds_concat_classification():
    exp1 = make_experiment(exp_id="exp1", a="EXP1_A", b="EXP1_B")
    exp2 = make_experiment(exp_id="exp2", a="EXP2_A", b="EXP2_B")
    params = InteractionExtractionParams(
        connection=CONN, date_range=RANGE, param_key="k", experiments=[exp1, exp2])
    sql = ex.build_interaction(params)
    assert "CONCAT(" in sql
    assert "EXP1_A" in sql and "EXP2_A" in sql


# --------------------------------------------------------------------- #
#  Auto-detect queries
# --------------------------------------------------------------------- #


def test_autodetect_variants_query_scopes_to_param_key_and_prefix():
    sql = ex.build_autodetect_variants_query(
        CONN.project, CONN.dataset, "2026-01-01", "2026-01-02", "k", "EXP1")
    assert "params.key = 'k'" in sql
    assert "LIKE '%EXP1%'" in sql


def test_autodetect_event_names_query_orders_by_frequency():
    sql = ex.build_autodetect_event_names_query(
        CONN.project, CONN.dataset, "2026-01-01", "2026-01-02", limit=50)
    assert "ORDER BY event_count DESC" in sql
    assert "LIMIT 50" in sql


# --------------------------------------------------------------------- #
#  EventLogExtractionParams / build_event_log (process mining)
# --------------------------------------------------------------------- #


def test_build_event_log_default_shape_is_user_and_event_name():
    params = EventLogExtractionParams(connection=CONN, date_range=RANGE)
    sql = event_log_sql.build_event_log(params)
    assert "user_pseudo_id AS case_id" in sql
    assert "event_name AS activity" in sql
    assert "TIMESTAMP_MICROS(event_timestamp) AS timestamp" in sql


def test_build_event_log_restricts_to_requested_event_names():
    params = EventLogExtractionParams(
        connection=CONN, date_range=RANGE, event_names=["page_view", "purchase"])
    sql = event_log_sql.build_event_log(params)
    # event_filter is applied against the raw activity_col ('event_name'),
    # not the 'activity' alias assigned in the outer SELECT.
    assert "event_name IN ('page_view', 'purchase')" in sql


def test_build_event_log_adds_attribute_columns():
    params = EventLogExtractionParams(
        connection=CONN, date_range=RANGE, attribute_params=[
            "page_location", "page_title"]
    )
    sql = event_log_sql.build_event_log(params)
    assert "AS page_location" in sql
    assert "AS page_title" in sql


def test_build_event_log_dedupes_exact_duplicate_attribute_key():
    params = EventLogExtractionParams(
        connection=CONN, date_range=RANGE, attribute_params=[
            "page_location", "page_location"]
    )
    sql = event_log_sql.build_event_log(params)
    assert sql.count("AS page_location") == 1


def test_build_event_log_rejects_colliding_distinct_attribute_keys():
    # 'page.location' and 'page_location' both sanitize to the same column
    # alias -- silently keeping only the first would drop a column the
    # caller explicitly asked for with no indication why.
    params = EventLogExtractionParams(
        connection=CONN, date_range=RANGE, attribute_params=[
            "page.location", "page_location"]
    )
    with pytest.raises(ValueError, match="disambiguate"):
        event_log_sql.build_event_log(params)


def test_build_event_log_event_filter_scopes_cases_not_rows():
    params = EventLogExtractionParams(
        connection=CONN, date_range=RANGE, filter_type=UserFilterType.EVENT, filter_value="purchase"
    )
    sql = event_log_sql.build_event_log(params)
    assert "filtered_cases AS" in sql
    assert "INNER JOIN filtered_cases fc ON base.case_id = fc.case_id" in sql


def test_build_event_log_contains_filter_matches_page_location():
    params = EventLogExtractionParams(
        connection=CONN, date_range=RANGE, filter_type=UserFilterType.CONTAINS, filter_value="/checkout"
    )
    sql = event_log_sql.build_event_log(params)
    assert "params.value.string_value LIKE '%/checkout%'" in sql
    assert "params.key = 'page_location'" in sql


def test_build_event_log_regex_filter_matches_page_location():
    params = EventLogExtractionParams(
        connection=CONN, date_range=RANGE, filter_type=UserFilterType.REGEX, filter_value=r"/product/\d+"
    )
    sql = event_log_sql.build_event_log(params)
    assert r"REGEXP_CONTAINS(params.value.string_value, r'/product/\d+')" in sql


def test_build_event_log_custom_case_and_activity_columns():
    params = EventLogExtractionParams(
        connection=CONN, date_range=RANGE, case_id_col="ga_session_id", activity_col="event_name"
    )
    sql = event_log_sql.build_event_log(params)
    assert "ga_session_id AS case_id" in sql


def test_build_event_log_rejects_unsafe_case_id_column():
    params = EventLogExtractionParams(
        connection=CONN, date_range=RANGE, case_id_col="user_id; DROP TABLE x")
    with pytest.raises(ValueError):
        event_log_sql.build_event_log(params)


def test_build_event_log_always_emits_user_id_by_default():
    params = EventLogExtractionParams(connection=CONN, date_range=RANGE)
    sql = event_log_sql.build_event_log(params)
    assert "user_pseudo_id AS user_id" in sql


def test_build_event_log_include_user_id_false_omits_the_column():
    params = EventLogExtractionParams(
        connection=CONN, date_range=RANGE, include_user_id=False)
    sql = event_log_sql.build_event_log(params)
    assert "AS user_id" not in sql


def test_build_event_log_session_id_param_gives_session_level_case_id():
    # ga_session_id isn't a flat column -- it must come from the nested
    # event_params array, not a plain `{col} AS case_id` reference.
    params = EventLogExtractionParams(
        connection=CONN, date_range=RANGE, session_id_param="ga_session_id"
    )
    sql = event_log_sql.build_event_log(params)
    assert "user_pseudo_id AS case_id" not in sql
    assert "WHERE ep.key = 'ga_session_id'" in sql
    assert "ep.value.int_value" in sql
    assert "AS case_id" in sql
    # user_id is still emitted separately, so a caller can build a
    # composite (user_id + session) key downstream, the way PRoX does.
    assert "user_pseudo_id AS user_id" in sql


def test_build_event_log_session_id_param_and_custom_case_id_col_conflict():
    with pytest.raises(ValidationError, match="mutually exclusive"):
        EventLogExtractionParams(
            connection=CONN, date_range=RANGE, session_id_param="ga_session_id", case_id_col="client_id"
        )


def test_build_event_log_session_id_param_still_used_in_case_filter_cte():
    params = EventLogExtractionParams(
        connection=CONN,
        date_range=RANGE,
        session_id_param="ga_session_id",
        filter_type=UserFilterType.EVENT,
        filter_value="purchase",
    )
    sql = event_log_sql.build_event_log(params)
    assert "filtered_cases AS" in sql
    # the filtered_cases CTE must key off the same session-level expression,
    # not silently fall back to a flat user_pseudo_id reference.
    assert "WHERE ep.key = 'ga_session_id'" in sql.split("filtered_cases AS")[
                                                         1].split("base AS")[0]


def test_build_event_log_include_purchase_revenue_adds_typed_revenue_column():
    params = EventLogExtractionParams(
        connection=CONN, date_range=RANGE, include_purchase_revenue=True)
    sql = event_log_sql.build_event_log(params)
    assert "ecommerce.purchase_revenue AS revenue" in sql


def test_build_event_log_numeric_attribute_params_coalesces_typed_value_slots():
    params = EventLogExtractionParams(
        connection=CONN, date_range=RANGE, numeric_attribute_params=[
            "value", "engagement_time_msec"]
    )
    sql = event_log_sql.build_event_log(params)
    assert "COALESCE(ep.value.double_value, ep.value.float_value, CAST(ep.value.int_value AS FLOAT64))" in sql
    assert "WHERE ep.key = 'value'" in sql
    assert "AS value" in sql
    assert "AS engagement_time_msec" in sql
    # numeric params must not go through the string_value-only reader.
    assert "ep.value.string_value FROM UNNEST(event_params) AS ep WHERE ep.key = 'value'" not in sql


def test_build_event_log_rejects_same_key_as_both_string_and_numeric():
    params = EventLogExtractionParams(
        connection=CONN, date_range=RANGE, attribute_params=["value"], numeric_attribute_params=["value"]
    )
    with pytest.raises(ValueError, match="disambiguate"):
        event_log_sql.build_event_log(params)


def test_build_event_log_string_and_numeric_attribute_params_coexist():
    params = EventLogExtractionParams(
        connection=CONN,
        date_range=RANGE,
        attribute_params=["page_location"],
        numeric_attribute_params=["value"],
    )
    sql = event_log_sql.build_event_log(params)
    assert "AS page_location" in sql
    assert "AS value" in sql


# --------------------------------------------------------------------- #
#  TimeSeriesExtractionParams / build_timeseries (forecasting input)
# --------------------------------------------------------------------- #


def test_timeseries_event_count_requires_custom_event_name():
    with pytest.raises(ValidationError, match="custom_event_name"):
        TimeSeriesExtractionParams(
            connection=CONN, date_range=RANGE, metrics=[TimeSeriesMetric.EVENT_COUNT]
        )


def test_build_timeseries_default_metrics_shape_matches_forecasting_engine_contract():
    params = TimeSeriesExtractionParams(
        connection=CONN,
        date_range=RANGE,
        metrics=[TimeSeriesMetric.VISITORS,
            TimeSeriesMetric.CONVERSIONS, TimeSeriesMetric.REVENUE],
    )
    sql = ts_sql.build_timeseries(params)
    assert "AS date" in sql
    assert "AS visitors" in sql
    assert "AS conversions" in sql
    assert "AS revenue" in sql
    assert "GROUP BY date" in sql
    assert "segment" not in sql


def test_build_timeseries_segment_col_groups_by_segment_too():
    params = TimeSeriesExtractionParams(
        connection=CONN,
        date_range=RANGE,
        metrics=[TimeSeriesMetric.VISITORS],
        segment_col="device.category",
    )
    sql = ts_sql.build_timeseries(params)
    assert "main.device.category AS segment" in sql
    assert "GROUP BY date, segment" in sql


def test_build_timeseries_custom_conversion_event():
    params = TimeSeriesExtractionParams(
        connection=CONN,
        date_range=RANGE,
        metrics=[TimeSeriesMetric.CONVERSIONS],
        conversion_event="sign_up",
    )
    sql = ts_sql.build_timeseries(params)
    assert "event_name = 'sign_up'" in sql


def test_build_timeseries_event_count_metric():
    params = TimeSeriesExtractionParams(
        connection=CONN,
        date_range=RANGE,
        metrics=[TimeSeriesMetric.EVENT_COUNT],
        custom_event_name="add_to_cart",
    )
    sql = ts_sql.build_timeseries(params)
    assert "event_name = 'add_to_cart'" in sql
    assert "AS custom_event_count" in sql


def test_build_timeseries_rejects_unsafe_segment_col():
    params = TimeSeriesExtractionParams(
        connection=CONN,
        date_range=RANGE,
        metrics=[TimeSeriesMetric.VISITORS],
        segment_col="device.category; DROP TABLE x",
    )
    with pytest.raises(ValueError):
        ts_sql.build_timeseries(params)


# --------------------------------------------------------------------- #
#  DateRange validation
# --------------------------------------------------------------------- #


def test_date_range_rejects_start_after_end():
    with pytest.raises(ValidationError, match="start_date"):
        DateRange(start_date=date(2026, 2, 1), end_date=date(2026, 1, 1))
