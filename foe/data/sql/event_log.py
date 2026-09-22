"""
foe/data/sql/event_log.py

Builds a raw GA4 event-log export shaped for process mining: one row per
(case_id, activity, timestamp) -- the standard XES-style event-log triple
-- plus any caller-requested attribute columns. No aggregation: downstream
process-mining tools (e.g. pm4py) expect row-level events, not summaries.

case_id defaults to user_pseudo_id (one case per user); activity defaults
to event_name. case_id_col accepts any top-level or dotted struct-field
column in the GA4 export schema (e.g. 'device.category'). For genuine
session-level traces, set session_id_param instead: GA4's session
identifier (ga_session_id) lives inside event_params as a nested,
INT-valued key, not as a flat column, so it can't be passed via
case_id_col -- session_id_param builds the correlated event_params lookup
that case_id_col alone can't express. Session numbers reset per user and
aren't globally unique on their own, so user_id (emitted by default -- see
include_user_id) is meant to be combined with case_id downstream into a
composite key, the same way PRoX does (user_id + '_' + case_id).

Revenue and other numeric event params: attribute_params only ever reads
event_params' string_value, so numeric params (e.g. 'value', GA4's
event-level monetary value) come back NULL through it. Use
numeric_attribute_params for those, or include_purchase_revenue for the
common case of purchase revenue specifically (a flat, already-typed
FLOAT64 column that doesn't touch event_params at all). NULL revenue on
non-purchase rows is expected GA4 behavior, not a bug -- GA4 only ever
populates ecommerce.purchase_revenue on 'purchase' events.

Standard GA4 segment dimensions (device, traffic source, item category)
live outside event_params entirely -- top-level structs and a repeated
field, not attribute keys -- so they get their own include_* flags rather
than going through attribute_params/numeric_attribute_params:
include_device, include_traffic_source, include_item_category.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Dict, Tuple

from foe.core.models import DateRange, EventLogExtractionParams, UserFilterType
from .ga4 import escape_literal, escape_raw_string_literal, suffix_filter, table_ref, validate_identifier

_esc = escape_literal
_esc_re = escape_raw_string_literal


def _case_filter_cte(p: EventLogExtractionParams, table: str, suffix: str, case_id_expr: str) -> Tuple[str, str]:
    """
    Optional filtered_cases CTE + join, mirroring
    foe.data.sql.experiments._baseline_user_filter but scoped to whatever
    expression is being used as the case identifier (a flat column, or a
    session_id_param correlated subquery) rather than always user_pseudo_id.
    """
    if not (p.filter_type and p.filter_value):
        return "", ""
    if p.filter_type == UserFilterType.EVENT:
        cte = f"""
filtered_cases AS (
  SELECT DISTINCT {case_id_expr} AS case_id
  FROM {table}
  WHERE {suffix}
    AND event_name = '{_esc(p.filter_value)}'
),"""
    else:
        if p.filter_type == UserFilterType.REGEX:
            condition = f"AND REGEXP_CONTAINS(params.value.string_value, r'{_esc_re(p.filter_value)}')"
        else:
            condition = f"AND params.value.string_value LIKE '%{_esc(p.filter_value)}%'"
        cte = f"""
filtered_cases AS (
  SELECT DISTINCT {case_id_expr} AS case_id
  FROM {table}, UNNEST(event_params) AS params
  WHERE {suffix}
    AND event_name = 'page_view'
    AND params.key = 'page_location'
    {condition}
),"""
    join = "INNER JOIN filtered_cases fc ON base.case_id = fc.case_id"
    return cte, join


def _attribute_columns(seen_aliases: Dict[str, Tuple[str, str]], keys: list, kind: str) -> str:
    """
    Builds unnest-and-select column expressions for `keys` (either
    attribute_params or numeric_attribute_params), tracking aliases across
    both lists via the shared `seen_aliases` dict so a key colliding on
    alias -- or the same key requested as both string and numeric -- is
    caught with an actionable error instead of silently dropped or
    silently overwritten.
    """
    field_label = "numeric_attribute_params" if kind == "numeric" else "attribute_params"
    select = ""
    for key in keys:
        alias = validate_identifier(key.replace(".", "_"), field_name=f"{field_label} entry")
        if alias in seen_aliases:
            prev_key, prev_kind = seen_aliases[alias]
            if prev_key == key and prev_kind == kind:
                continue  # exact duplicate request, harmless no-op
            raise ValueError(
                f"'{prev_key}' ({prev_kind}) and '{key}' ({kind}) both map to column alias "
                f"'{alias}' -- rename one to disambiguate (don't request the same event_params "
                "key as both a string and a numeric attribute)."
            )
        seen_aliases[alias] = (key, kind)
        if kind == "numeric":
            select += f""",
    (SELECT COALESCE(ep.value.double_value, ep.value.float_value, CAST(ep.value.int_value AS FLOAT64))
     FROM UNNEST(event_params) AS ep WHERE ep.key = '{_esc(key)}' LIMIT 1) AS {alias}"""
        else:
            select += f""",
    (SELECT ep.value.string_value FROM UNNEST(event_params) AS ep WHERE ep.key = '{_esc(key)}' LIMIT 1) AS {alias}"""
    return select


def build_event_log(p: EventLogExtractionParams, limit: int = 0) -> str:
    table = table_ref(p.connection.project, p.connection.dataset)
    suffix = suffix_filter(p.date_range.start_date.isoformat(), p.date_range.end_date.isoformat())
    activity_col = validate_identifier(p.activity_col, field_name="activity_col")

    if p.session_id_param:
        # Genuine session-level case identifier: ga_session_id (or any other
        # INT-valued event_params key) lives in the nested array, not as a
        # flat column, so case_id_col can't express this -- CAST to STRING
        # so case_id is consistently typed regardless of which branch built
        # it (case_id_col could point at a STRING column just as easily).
        case_id_expr = (
            "CAST((SELECT ep.value.int_value FROM UNNEST(event_params) AS ep "
            f"WHERE ep.key = '{_esc(p.session_id_param)}' LIMIT 1) AS STRING)"
        )
    else:
        case_id_expr = validate_identifier(p.case_id_col, field_name="case_id_col")

    event_filter = ""
    if p.event_names:
        names = ", ".join(f"'{_esc(n)}'" for n in p.event_names)
        event_filter = f"\n    AND {activity_col} IN ({names})"

    user_id_select = ",\n    user_pseudo_id AS user_id" if p.include_user_id else ""
    revenue_select = ",\n    ecommerce.purchase_revenue AS revenue" if p.include_purchase_revenue else ""
    device_select = ",\n    device.category AS device_category" if p.include_device else ""
    traffic_source_select = (
        ",\n    COALESCE(session_traffic_source_last_click.manual_campaign.source, traffic_source.source) "
        "AS traffic_source"
        ",\n    COALESCE(session_traffic_source_last_click.manual_campaign.medium, traffic_source.medium) "
        "AS traffic_medium"
        if p.include_traffic_source
        else ""
    )
    item_category_select = (
        ",\n    (SELECT item_category FROM UNNEST(items) LIMIT 1) AS category"
        if p.include_item_category
        else ""
    )

    seen_aliases: Dict[str, Tuple[str, str]] = {}
    attribute_select = _attribute_columns(seen_aliases, p.attribute_params, "string")
    attribute_select += _attribute_columns(seen_aliases, p.numeric_attribute_params, "numeric")

    filter_cte, filter_join = _case_filter_cte(p, table, suffix, case_id_expr)
    limit_clause = f"\nLIMIT {limit}" if limit else ""

    return f"""-- Event log export (process mining) — one row per (case, activity, timestamp)
WITH{filter_cte}
base AS (
  SELECT
    {case_id_expr} AS case_id,
    {activity_col} AS activity,
    TIMESTAMP_MICROS(event_timestamp) AS timestamp{user_id_select}{revenue_select}{device_select}{traffic_source_select}{item_category_select}{attribute_select}
  FROM {table}
  WHERE {suffix}{event_filter}
)

SELECT base.*
FROM base
{filter_join}
WHERE base.case_id IS NOT NULL
ORDER BY base.case_id, base.timestamp{limit_clause};
"""


def build_event_log_preview(p: EventLogExtractionParams, sample_rows: int = 20, sample_days: int = 1) -> str:
    """
    The same SQL build_event_log() would build for `p`'s requested columns, capped to a cheap
    preview: narrowed to the last `sample_days` day(s) of p.date_range and LIMITed to
    `sample_rows`, both hard-capped regardless of caller input -- this is meant to run
    automatically/eagerly in a UI (DataEngine.preview_columns), not on an explicit "Extract
    Data" click.

    Narrowing the date window, not just the row count, is the actual cost lever here: BigQuery
    bills a wildcard-table scan (`events_*`) by bytes read across every matched
    events_YYYYMMDD shard regardless of a trailing LIMIT, so LIMIT alone wouldn't shrink cost
    for a wide date range the way it would on a single non-partitioned table. TABLESAMPLE
    SYSTEM would reduce it further, but BigQuery does not support TABLESAMPLE on wildcard
    tables, so narrowing _TABLE_SUFFIX (the same recent-window trick DataEngine's
    autodetect_variants/autodetect_event_names/autodetect_kpis already use) is the only lever
    actually available against this table shape.
    """
    sample_rows = max(1, min(sample_rows, 50))
    sample_start = max(p.date_range.end_date - timedelta(days=sample_days), p.date_range.start_date)
    narrowed = p.model_copy(update={"date_range": DateRange(start_date=sample_start, end_date=p.date_range.end_date)})
    return build_event_log(narrowed, limit=sample_rows)
