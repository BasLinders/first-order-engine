"""
foe/data/sql/timeseries.py

Builds a daily time-series export from GA4 events_* -- feeds
ForecastingEngine's date_col/conversions_col/revenue_col contract directly
(see foe.forecasting.operations). Always pulled at daily grain:
ForecastingEngine resamples to weekly/monthly itself, so pulling anything
coarser here would throw away information for no benefit.

Structurally this is _build_baseline_daily from foe.data.sql.experiments,
generalized: any subset of {visitors, conversions, revenue, transactions,
a caller-named custom event count}, optionally split by a segment column
(e.g. device.category) instead of always the fixed conversion event
'purchase'.
"""

from __future__ import annotations

from typing import Tuple

from foe.core.models import TimeSeriesExtractionParams, TimeSeriesMetric, UserFilterType
from .ga4 import escape_literal, escape_raw_string_literal, suffix_filter, table_ref, validate_identifier

_esc = escape_literal
_esc_re = escape_raw_string_literal


def _segment_user_filter(p: TimeSeriesExtractionParams, table: str, suffix: str) -> Tuple[str, str]:
    """Mirrors foe.data.sql.experiments._baseline_user_filter exactly -- same
    CONTAINS/REGEX/EVENT semantics, scoped to user_pseudo_id."""
    if not (p.filter_type and p.filter_value):
        return "", ""
    if p.filter_type == UserFilterType.EVENT:
        cte = f"""
filtered_users AS (
  SELECT DISTINCT user_pseudo_id
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
filtered_users AS (
  SELECT DISTINCT user_pseudo_id
  FROM {table}, UNNEST(event_params) AS params
  WHERE {suffix}
    AND event_name = 'page_view'
    AND params.key = 'page_location'
    {condition}
),"""
    join = "INNER JOIN filtered_users ON main.user_pseudo_id = filtered_users.user_pseudo_id"
    return cte, join


def build_timeseries(p: TimeSeriesExtractionParams, limit: int = 0) -> str:
    table = table_ref(p.connection.project, p.connection.dataset)
    suffix = suffix_filter(p.date_range.start_date.isoformat(), p.date_range.end_date.isoformat())
    user_filter_cte, join_clause = _segment_user_filter(p, table, suffix)

    segment_select = ""
    group_extra = ""
    order_extra = ""
    if p.segment_col:
        segment_expr = validate_identifier(p.segment_col, field_name="segment_col")
        segment_select = f"  main.{segment_expr} AS segment,\n"
        group_extra = ", segment"
        order_extra = ", segment"

    metric_cols = []
    if TimeSeriesMetric.VISITORS in p.metrics:
        metric_cols.append("  COUNT(DISTINCT main.user_pseudo_id) AS visitors")
    if TimeSeriesMetric.CONVERSIONS in p.metrics:
        metric_cols.append(
            f"  COUNT(DISTINCT CASE WHEN main.event_name = '{_esc(p.conversion_event)}' "
            f"THEN main.user_pseudo_id END) AS conversions"
        )
    if TimeSeriesMetric.TRANSACTIONS in p.metrics:
        metric_cols.append(
            "  COUNT(DISTINCT CASE WHEN main.event_name = 'purchase' "
            "THEN main.ecommerce.transaction_id END) AS transactions"
        )
    if TimeSeriesMetric.REVENUE in p.metrics:
        metric_cols.append(
            "  SUM(CASE WHEN main.event_name = 'purchase' "
            "THEN main.ecommerce.purchase_revenue ELSE 0 END) AS revenue"
        )
    if TimeSeriesMetric.EVENT_COUNT in p.metrics:
        metric_cols.append(
            f"  COUNT(DISTINCT CASE WHEN main.event_name = '{_esc(p.custom_event_name)}' "
            f"THEN main.user_pseudo_id END) AS custom_event_count"
        )

    select_block = ",\n".join(metric_cols)
    limit_clause = f"\nLIMIT {limit}" if limit else ""

    return f"""-- Time-series export (daily) — forecasting input
DECLARE start_date STRING DEFAULT '{p.date_range.start_date.isoformat()}';
DECLARE end_date   STRING DEFAULT '{p.date_range.end_date.isoformat()}';

WITH{user_filter_cte}
dummy AS (SELECT 1)  -- placeholder when no user filter

SELECT
  PARSE_DATE('%Y%m%d', main.event_date) AS date,
{segment_select}{select_block}
FROM {table} AS main
{join_clause}
WHERE main.{suffix}
GROUP BY date{group_extra}
ORDER BY date{order_extra}{limit_clause};
"""
