"""
foe/data/sql: pure SQL-string builders for GA4 BigQuery exports.

No BigQuery client, no network access -- every build_* function here takes
a Pydantic params model (foe.core.models) and returns a SQL string. See
foe/data/engine.py's DataEngine for execution.
"""
