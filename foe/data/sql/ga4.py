"""
foe/data/sql/ga4.py

Shared primitives for building SQL against a GA4 BigQuery export
(`events_*` wildcard tables). Every extraction module in foe.data.sql
composes queries from these building blocks plus its own domain logic.

Trust boundary: nothing here is a parameterized query. GA4's `events_*`
wildcard table name and its `_TABLE_SUFFIX` pseudo-column can't be bound
as query parameters in the standard BigQuery client libraries, so this
module follows the same string-composition approach as the tool it was
ported from (hexkit's utility/sql_builder.py). Two helpers narrow that
gap: `escape_literal` guards against an unescaped quote in a
caller-supplied string (a filter value, a variant string, an event name)
breaking out of its SQL string literal; `validate_identifier` guards
column/struct-field names that get spliced in unquoted, where an
unescaped value is a direct injection surface rather than a broken query.
Callers are still responsible for treating extraction params as
configuration, not raw untrusted end-user input.
"""

from __future__ import annotations

import re

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
_TABLE_REF_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
# GCP project IDs: lowercase letters/digits/hyphens (uppercase tolerated here
# too -- some legacy numeric/alias projects deviate slightly). BigQuery
# dataset IDs: letters/digits/underscores only, no hyphens or dots.
_PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-]*$")
_DATASET_ID_RE = re.compile(r"^[A-Za-z0-9_]+$")


def table_ref(project: str, dataset: str) -> str:
    """
    Fully-qualified, backtick-quoted GA4 events_* wildcard table. Validates
    project/dataset against BigQuery's own ID charset -- this is the single
    most-used builder in the package (every extraction query goes through
    it), and project/dataset are spliced in unquoted inside backticks, so a
    stray backtick or quote here is a direct injection surface rather than
    just a broken query.
    """
    if not _PROJECT_ID_RE.match(project):
        raise ValueError(f"project must be a valid GCP project ID -- got {project!r}.")
    if not _DATASET_ID_RE.match(dataset):
        raise ValueError(f"dataset must be a valid BigQuery dataset ID (letters, digits, underscore) -- got {dataset!r}.")
    return f"`{project}.{dataset}.events_*`"


def suffix_filter(start: str, end: str, alias: str = "") -> str:
    """
    A _TABLE_SUFFIX BETWEEN clause scoping a wildcard-table scan to
    [start, end] (both 'YYYY-MM-DD'). `alias`, if given, qualifies the
    pseudo-column (e.g. "e" -> "e._TABLE_SUFFIX"); an empty alias leaves it
    bare so a caller can prefix its own alias directly (e.g. "main.{suffix}").
    """
    col = f"{alias}._TABLE_SUFFIX" if alias else "_TABLE_SUFFIX"
    return (
        f"{col} BETWEEN FORMAT_DATE('%Y%m%d', PARSE_DATE('%Y-%m-%d', '{escape_literal(start)}'))\n"
        f"    AND FORMAT_DATE('%Y%m%d', PARSE_DATE('%Y-%m-%d', '{escape_literal(end)}'))"
    )


def escape_literal(value: str) -> str:
    """Escapes backslashes and single quotes so `value` is safe inside a '...' SQL string literal."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


def escape_raw_string_literal(value: str) -> str:
    """
    Escapes only the single-quote delimiter, for use inside a raw string
    literal (r'...') -- e.g. the pattern argument to REGEXP_CONTAINS.
    Deliberately does NOT touch backslashes: a raw string literal doesn't
    treat them as SQL escapes, so a caller's regex is free to use `\\d`,
    `\\.`, etc. -- doubling those backslashes (escape_literal's behavior)
    would silently change the pattern's meaning. Only the quote needs
    neutralizing so a caller-supplied value can't close the literal early.
    """
    return value.replace("'", "\\'")


def validate_identifier(value: str, *, field_name: str = "identifier") -> str:
    """
    Validates that `value` is safe to splice into SQL unquoted as a column
    or struct-field reference (e.g. 'device.category'). Raises ValueError
    on anything else -- these values are never wrapped in quotes, so an
    unescaped literal here is a direct SQL-injection surface, not just a
    broken query.
    """
    if not _IDENTIFIER_RE.match(value):
        raise ValueError(
            f"{field_name} must be a valid column/field identifier (letters, digits, "
            f"underscore, dot) -- got {value!r}."
        )
    return value


def validate_table_ref(value: str, *, field_name: str = "table reference") -> str:
    """
    Validates that `value` is safe to splice into SQL unquoted inside a
    backtick-quoted table reference (e.g. 'my-project.my_dataset.my_table').
    Looser than validate_identifier: GCP project IDs commonly contain
    hyphens.
    """
    if not _TABLE_REF_RE.match(value):
        raise ValueError(
            f"{field_name} must look like 'project.dataset.table' (letters, digits, "
            f"underscore, dot, hyphen) -- got {value!r}."
        )
    return value
