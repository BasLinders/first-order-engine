"""
foe/data/sql/event_log.py

Builds a raw GA4 event-log export shaped for process mining: one row per
(case_id, activity, timestamp) -- the standard XES-style event-log triple
-- plus any caller-requested attribute columns. No aggregation: downstream
process-mining tools (e.g. pm4py) expect row-level events, not summaries.

case_id defaults to user_pseudo_id (one case per user); activity defaults
to event_name. Both accept any top-level or dotted struct-field column in
the GA4 export schema (e.g. 'device.category'). Note this does NOT cover
session-level grouping: GA4's session identifier (ga_session_id) lives
inside event_params as a nested, INT-valued key, not as a flat column --
it can't be passed as case_id_col directly, and attribute_params can't
carry it through either (see attribute_params' docstring: it only extracts
event_params' string_value). Pass user_pseudo_id (the default) and derive
session boundaries downstream in your process-mining tool instead.
"""

from __future__ import annotations

from typing import Tuple

from foe.core.models import EventLogExtractionParams, UserFilterType
from .ga4 import escape_literal, escape_raw_string_literal, suffix_filter, table_ref, validate_identifier

_esc = escape_literal
_esc_re = escape_raw_string_literal


def _case_filter_cte(p: EventLogExtractionParams, table: str, suffix: str, case_id_col: str) -> Tuple[str, str]:
    """
    Optional filtered_cases CTE + join, mirroring
    foe.data.sql.experiments._baseline_user_filter but scoped to whatever
    column is being used as the case identifier rather than always
    user_pseudo_id.
    """
    if not (p.filter_type and p.filter_value):
        return "", ""
    if p.filter_type == UserFilterType.EVENT:
        cte = f"""
filtered_cases AS (
  SELECT DISTINCT {case_id_col} AS case_id
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
  SELECT DISTINCT {case_id_col} AS case_id
  FROM {table}, UNNEST(event_params) AS params
  WHERE {suffix}
    AND event_name = 'page_view'
    AND params.key = 'page_location'
    {condition}
),"""
    join = "INNER JOIN filtered_cases fc ON base.case_id = fc.case_id"
    return cte, join


def build_event_log(p: EventLogExtractionParams, limit: int = 0) -> str:
    table = table_ref(p.connection.project, p.connection.dataset)
    suffix = suffix_filter(p.date_range.start_date.isoformat(), p.date_range.end_date.isoformat())
    case_id_col = validate_identifier(p.case_id_col, field_name="case_id_col")
    activity_col = validate_identifier(p.activity_col, field_name="activity_col")

    event_filter = ""
    if p.event_names:
        names = ", ".join(f"'{_esc(n)}'" for n in p.event_names)
        event_filter = f"\n    AND {activity_col} IN ({names})"

    attribute_select = ""
    seen_aliases = {}  # alias -> the attribute_params key that claimed it
    for key in p.attribute_params:
        alias = validate_identifier(key.replace(".", "_"), field_name="attribute_params entry")
        if alias in seen_aliases:
            if seen_aliases[alias] != key:
                # Two DIFFERENT keys collapsed to the same alias (e.g.
                # 'page.location' and 'page_location' both -> 'page_location')
                # -- silently keeping only the first would drop a column the
                # caller explicitly asked for with no indication why.
                raise ValueError(
                    f"attribute_params {seen_aliases[alias]!r} and {key!r} both map to column "
                    f"alias '{alias}' -- rename one to disambiguate."
                )
            continue  # exact duplicate key requested twice: harmless no-op
        seen_aliases[alias] = key
        attribute_select += f""",
    (SELECT ep.value.string_value FROM UNNEST(event_params) AS ep WHERE ep.key = '{_esc(key)}' LIMIT 1) AS {alias}"""

    filter_cte, filter_join = _case_filter_cte(p, table, suffix, case_id_col)
    limit_clause = f"\nLIMIT {limit}" if limit else ""

    return f"""-- Event log export (process mining) — one row per (case, activity, timestamp)
WITH{filter_cte}
base AS (
  SELECT
    {case_id_col} AS case_id,
    {activity_col} AS activity,
    TIMESTAMP_MICROS(event_timestamp) AS timestamp{attribute_select}
  FROM {table}
  WHERE {suffix}{event_filter}
    AND {case_id_col} IS NOT NULL
)

SELECT base.*
FROM base
{filter_join}
ORDER BY base.case_id, base.timestamp{limit_clause};
"""
