"""
foe/data/engine.py

DataEngine: a wrapper around BigQuery connectivity, execution, and GA4
extraction recipes. This is the one deliberate I/O exception in foe (see
README and the "Data extraction" section of foe/core/models.py) -- gated
behind the `foe[bigquery]` extra so importing foe itself never requires
google-cloud-bigquery.

What DataEngine owns:
  * OAuth (framework-free -- no session/cookie/secrets-manager assumptions;
    see build_auth_url/exchange_code below for how state survives the
    redirect without server-side storage).
  * A bigquery.Client factory, project/dataset discovery, dry-run cost
    estimation, monthly free-tier usage, and query execution (including
    session-scoped shared scans).
  * Delegating extraction requests to foe.data.sql.* builders and running
    the resulting SQL.

What it deliberately does NOT own: any UI, any session/cookie storage, any
secrets-manager integration. A caller's web framework (Streamlit, Flask,
FastAPI, a bare CLI) decides how build_auth_url's returned URL is served
and how exchange_code's returned Credentials are persisted between
requests -- DataEngine only knows about Google's OAuth/BigQuery APIs.
"""

from __future__ import annotations

import base64
import json
import urllib.parse
from datetime import date, timedelta
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from foe.core.models import (
    BaselineExtractionParams,
    BinomialExtractionParams,
    BQConnectionConfig,
    ContinuousExtractionParams,
    EventLogExtractionParams,
    InteractionExtractionParams,
    QueryCostEstimate,
    SequentialExtractionParams,
    TimeSeriesExtractionParams,
    UsageReport,
)
from foe.data.sql import event_log as _event_log_sql
from foe.data.sql import experiments as _experiments_sql
from foe.data.sql import timeseries as _timeseries_sql

if TYPE_CHECKING:
    import pandas as pd
    from google.cloud import bigquery
    from google.oauth2.credentials import Credentials

try:
    from google.auth.transport.requests import Request as _GoogleAuthRequest
    from google.oauth2.credentials import Credentials as _Credentials
    from google_auth_oauthlib.flow import Flow as _Flow
    from google.cloud import bigquery as _bigquery
    from google.cloud import resourcemanager_v3 as _resourcemanager_v3

    _BIGQUERY_AVAILABLE = True
except ImportError:
    _BIGQUERY_AVAILABLE = False


DEFAULT_SCOPES = [
    "https://www.googleapis.com/auth/bigquery",
    "https://www.googleapis.com/auth/cloud-platform.read-only",
]

_FREE_TIER_BYTES = 1_000_000_000_000  # 1 TB


def _require_bigquery_deps() -> None:
    if not _BIGQUERY_AVAILABLE:
        raise ImportError(
            "DataEngine's BigQuery/OAuth features require the optional 'bigquery' extra. "
            "Install with: pip install 'foe[bigquery]'"
        )


def _format_bytes(n: int) -> str:
    if n < 1_000:
        return f"{n} B"
    if n < 1_000_000:
        return f"{n / 1_000:.1f} KB"
    if n < 1_000_000_000:
        return f"{n / 1_000_000:.1f} MB"
    return f"{n / 1_000_000_000:.2f} GB"


class DataEngine:
    """
    BigQuery connectivity + GA4 extraction, decoupled from any UI framework.
    Construct via from_credentials()/from_client() once a caller has
    obtained credentials however fits its own auth flow (build_auth_url/
    exchange_code below, or any other means -- a service account, an
    already-built bigquery.Client, etc.).
    """

    def __init__(self, client: bigquery.Client, credentials: Optional[Credentials] = None):
        _require_bigquery_deps()
        self._client = client
        self._credentials = credentials

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_credentials(cls, credentials: Credentials, project: Optional[str] = None) -> "DataEngine":
        _require_bigquery_deps()
        client = _bigquery.Client(credentials=credentials, project=project)
        return cls(client, credentials=credentials)

    @classmethod
    def from_client(cls, client: bigquery.Client) -> "DataEngine":
        """
        Wraps an already-built bigquery.Client (e.g. from a service account
        or Application Default Credentials). list_projects()/cross-project
        list_datasets() are unavailable this way -- they need the raw
        Credentials object, not just a Client scoped to one project; use
        from_credentials() if you need those.
        """
        return cls(client)

    # ------------------------------------------------------------------ #
    # OAuth -- framework-free
    # ------------------------------------------------------------------ #
    # The PKCE verifier and any caller-supplied extra_state are encoded into
    # the OAuth `state` parameter itself, which Google echoes back on the
    # callback verbatim. This means NO server-side session storage is
    # required to survive the redirect -- useful for a framework whose
    # redirect can start a fresh session with no prior state available
    # (e.g. Streamlit), and for a stateless serverless callback handler.
    #
    # client_id/client_secret/redirect_uri are deliberately NOT put in
    # state: state travels through the browser (URL bar, history, proxy and
    # access logs, a Referer header if the callback page later navigates
    # elsewhere), which is not where a confidential OAuth client secret
    # should ever appear. The caller already has these values -- it passed
    # them to build_auth_url -- so exchange_code asks for them again
    # directly rather than round-tripping them through the browser.
    #
    # Caveat: the verifier itself still travels in that same URL, alongside
    # `code` on the callback request. Standard PKCE keeps the verifier out
    # of any URL entirely (held in server-side session/app memory, sent
    # only in the token-exchange POST body), specifically so that someone
    # who can observe the redirect chain doesn't also get the verifier.
    # Putting it in `state` narrows that protection -- accepted here as the
    # default cost of needing no server-side session at all.
    #
    # Hashing the verifier before putting it in state does NOT fix this: at
    # exchange time Google's token endpoint needs the RAW verifier (it
    # hashes it itself and compares to the code_challenge sent earlier), so
    # a caller holding only a hash could never complete the exchange.
    # Hashing is a mitigation for values you only ever compare, not for a
    # secret a third party must receive verbatim.
    #
    # A caller with real session storage and a stricter threat model should
    # instead: (1) keep the `verifier` this function returns entirely out of
    # `extra_state`, storing it server-side themselves; (2) pass it to
    # exchange_code(..., verifier=...) directly, which always overrides
    # whatever (if anything) is in `state`. This keeps the actual secret out
    # of the browser while state still optionally carries it for callers
    # who have nowhere else to put it.
    #
    # extra_state is entirely caller-controlled and travels through the
    # browser the same way -- never put a secret in it either.

    @staticmethod
    def build_auth_url(
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        scopes: Optional[List[str]] = None,
        extra_state: Optional[dict] = None,
    ) -> Tuple[str, str]:
        """
        Builds the Google OAuth consent URL. Returns (auth_url, verifier).
        `verifier` is embedded in auth_url's `state` param by default (see
        module note above) so a caller with no session storage can ignore
        the returned value entirely -- exchange_code() will find it there.
        A caller that wants the verifier off the browser round-trip should
        store the returned value itself and pass it to
        exchange_code(..., verifier=...) instead.

        `extra_state` is any caller-defined dict that should survive the
        redirect -- decoded back out by exchange_code(). Never put secrets
        in it; like the verifier, it travels through the browser.
        """
        _require_bigquery_deps()
        config = _client_config(client_id, client_secret, redirect_uri)
        flow = _Flow.from_client_config(
            config, scopes=scopes or DEFAULT_SCOPES, redirect_uri=redirect_uri)

        auth_url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
        )

        verifier: str = getattr(flow, "code_verifier", None) or ""
        state_data = json.dumps({"verifier": verifier, "extra": extra_state or {}})
        encoded = base64.urlsafe_b64encode(state_data.encode()).decode().rstrip("=")

        parsed = urllib.parse.urlparse(auth_url)
        qs = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        qs["state"] = [encoded]
        new_query = urllib.parse.urlencode({k: v[0] for k, v in qs.items()})
        return urllib.parse.urlunparse(parsed._replace(query=new_query)), verifier

    @staticmethod
    def exchange_code(
        code: str,
        state: str,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        verifier: Optional[str] = None,
    ) -> Tuple[Credentials, dict]:
        """
        Exchanges an OAuth callback `code` for Credentials. client_id/
        client_secret/redirect_uri must be the same values passed to
        build_auth_url -- they are read from the caller's own config, not
        from `state` (state carries only the PKCE verifier and extra_state;
        see the module note above for why the secret never travels through
        the browser).

        `verifier`, if given, overrides whatever `state` carries -- pass
        the value returned by build_auth_url here, from your own
        server-side storage, to keep it off the browser round-trip entirely.
        In that case `state` no longer needs to decode to anything
        meaningful (a caller doing its own CSRF/state handling can pass
        state="" here) -- a malformed/empty state is treated as carrying no
        verifier and no extra_state rather than raising, so the override
        path is fully independent of state's format. Omit `verifier` to
        fall back to state's copy (the default, session-free path) --
        there it must be Google's untouched echo of build_auth_url's state.
        Returns (credentials, extra_state) -- extra_state is whatever dict
        was passed to build_auth_url's extra_state, decoded back out (or
        {} when state didn't decode).
        """
        _require_bigquery_deps()
        try:
            padding = 4 - len(state) % 4
            padded = state + ("=" * (padding % 4))
            state_data = json.loads(base64.urlsafe_b64decode(padded).decode())
        except Exception:
            state_data = {}
        verifier = verifier or state_data.get("verifier") or None
        extra_state = state_data.get("extra", {}) or {}

        config = _client_config(client_id, client_secret, redirect_uri)
        flow = _Flow.from_client_config(
            config, scopes=DEFAULT_SCOPES, redirect_uri=redirect_uri)
        fetch_kwargs = {"code": code}
        if verifier:
            fetch_kwargs["code_verifier"] = verifier
        flow.fetch_token(**fetch_kwargs)
        return flow.credentials, extra_state

    @staticmethod
    def refresh_if_expired(credentials: Credentials) -> Credentials:
        """Refreshes `credentials` in place if expired and a refresh_token is available."""
        _require_bigquery_deps()
        if credentials.expired and credentials.refresh_token:
            credentials.refresh(_GoogleAuthRequest())
        return credentials

    @staticmethod
    def credentials_to_dict(credentials: Credentials) -> dict:
        """JSON-safe representation of Credentials, for a caller's own session/token storage."""
        return {
            "token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "token_uri": credentials.token_uri,
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "scopes": list(credentials.scopes) if credentials.scopes else [],
        }

    @staticmethod
    def credentials_from_dict(data: dict) -> Credentials:
        """Rebuilds Credentials from credentials_to_dict's output."""
        _require_bigquery_deps()
        return _Credentials(
            token=data["token"],
            refresh_token=data.get("refresh_token"),
            token_uri=data.get("token_uri"),
            client_id=data.get("client_id"),
            client_secret=data.get("client_secret"),
            scopes=data.get("scopes"),
        )

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #

    def list_projects(self) -> Dict[str, str]:
        """Projects accessible to the current credentials. {project_id: display_name}."""
        if self._credentials is None:
            raise RuntimeError(
                "list_projects() needs the raw Credentials object -- construct DataEngine "
                "via from_credentials() rather than from_client()."
            )
        rm_client = _resourcemanager_v3.ProjectsClient(credentials=self._credentials)
        result = {p.project_id: p.display_name or "" for p in rm_client.search_projects()}
        return dict(sorted(result.items()))

    def list_datasets(self, project: Optional[str] = None) -> Dict[str, str]:
        """Datasets in `project` (default: the engine's own project). {dataset_id: friendly_name}."""
        if project and project != self._client.project:
            if self._credentials is None:
                raise RuntimeError(
                    "Listing datasets in a different project needs the raw Credentials object "
                    "-- construct DataEngine via from_credentials() rather than from_client()."
                )
            client = _bigquery.Client(credentials=self._credentials, project=project)
        else:
            client = self._client
        result = {d.dataset_id: d.friendly_name or "" for d in client.list_datasets()}
        return dict(sorted(result.items()))

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    def dry_run(self, sql: str) -> QueryCostEstimate:
        """Estimated bytes scanned via a BigQuery dry run. Does NOT work for
        scripts containing DDL/DML (e.g. SequentialExtractionParams output) --
        see QueryCostEstimate.is_dml."""
        try:
            job_config = _bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
            job = self._client.query(sql, job_config=job_config)
            bytes_processed = job.total_bytes_processed or 0
            gb = bytes_processed / 1e9
            return QueryCostEstimate(
                bytes_processed=bytes_processed,
                gb_processed=round(gb, 2),
                display=_format_bytes(bytes_processed),
                free_tier_pct=round((gb / (_FREE_TIER_BYTES / 1e9)) * 100, 3),
                is_dml=False,
                error=None,
            )
        except Exception as e:
            err_str = str(e)
            is_dml = any(kw in err_str.upper()
                         for kw in ("DDL", "DML", "SCRIPT", "CREATE", "INSERT"))
            return QueryCostEstimate(
                bytes_processed=0,
                gb_processed=0.0,
                display="N/A",
                free_tier_pct=0.0,
                is_dml=is_dml,
                error=err_str,
            )

    def run(self, sql: str) -> pd.DataFrame:
        """Executes SQL and returns a DataFrame."""
        return self._client.query(sql).result().to_dataframe()

    def preview(self, sql: str, limit: int = 25) -> pd.DataFrame:
        """Strips a trailing semicolon, appends LIMIT if not already present,
        runs the query. Only valid for pure SELECT queries (not DDL/DML scripts)."""
        clean = sql.rstrip().rstrip(";")
        last_line = clean.upper().rsplit("\n", 1)[-1]
        if "LIMIT" not in last_line:
            clean = clean + f"\nLIMIT {limit}"
        return self.run(clean)

    def create_scan_session(self, create_temp_table_sql: str) -> str:
        """Runs a `CREATE TEMP TABLE ... AS SELECT ...` with a BigQuery session
        enabled and returns the session_id, so later queries can read that temp
        table via run_in_session."""
        job_config = _bigquery.QueryJobConfig(create_session=True)
        job = self._client.query(create_temp_table_sql, job_config=job_config)
        job.result()
        return job.session_info.session_id

    def run_in_session(self, sql: str, session_id: str) -> pd.DataFrame:
        """Runs `sql` against an existing BigQuery session (e.g. to read a TEMP
        TABLE created by create_scan_session)."""
        job_config = _bigquery.QueryJobConfig(
            connection_properties=[_bigquery.ConnectionProperty(
                key="session_id", value=session_id)]
        )
        job = self._client.query(sql, job_config=job_config)
        return job.result().to_dataframe()

    def run_shared_scan(self, create_temp_sql: str, select_sqls: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """Materializes a shared scan once (via a BigQuery session), then runs
        each labeled SELECT against it in that session -- e.g.
        {"binomial": df, "continuous": df} -- billing the big scan only once."""
        session_id = self.create_scan_session(create_temp_sql)
        return {label: self.run_in_session(sql, session_id) for label, sql in select_sqls.items()}

    def monthly_usage(self, dataset: str, project: Optional[str] = None) -> UsageReport:
        """
        Queries INFORMATION_SCHEMA.JOBS_BY_PROJECT for bytes processed this
        calendar month and derives remaining 1TB free-tier budget.
        INFORMATION_SCHEMA queries are themselves free.
        """
        project = project or self._client.project
        result = dict(
            used_bytes=0,
            used_gb=0.0,
            used_display="0 B",
            remaining_bytes=_FREE_TIER_BYTES,
            remaining_gb=round(_FREE_TIER_BYTES / 1e9, 2),
            remaining_display=_format_bytes(_FREE_TIER_BYTES),
            used_pct=0.0,
            free_tier_bytes=_FREE_TIER_BYTES,
            permission_denied=False,
            error=None,
        )
        try:
            region = self._dataset_region(dataset, project)
            sql = f"""
SELECT IFNULL(SUM(total_bytes_processed), 0) AS bytes_used
FROM `{region}`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE DATE(creation_time) >= DATE_TRUNC(CURRENT_DATE(), MONTH)
  AND job_type = 'QUERY'
  AND state = 'DONE'
  AND cache_hit = FALSE
"""
            rows = list(self._client.query(sql).result())
            used = int(rows[0]["bytes_used"]) if rows else 0
            remaining = max(_FREE_TIER_BYTES - used, 0)
            result.update(
                used_bytes=used,
                used_gb=round(used / 1e9, 2),
                used_display=_format_bytes(used),
                remaining_bytes=remaining,
                remaining_gb=round(remaining / 1e9, 2),
                remaining_display=_format_bytes(remaining),
                used_pct=min(round((used / _FREE_TIER_BYTES) * 100, 2), 100.0),
            )
        except Exception as e:
            err = str(e)
            result["error"] = err
            result["permission_denied"] = (
                "ACCESS_DENIED" in err.upper()
                or "PERMISSION_DENIED" in err.upper()
                or "does not have bigquery.jobs.list" in err
            )
        return UsageReport(**result)

    def _dataset_region(self, dataset_id: str, project: Optional[str] = None) -> str:
        """INFORMATION_SCHEMA region prefix for a dataset's location, e.g. 'EU' -> 'region-eu'.
        Falls back to 'region-eu' on error."""
        try:
            ds = self._client.get_dataset(
                f"{project or self._client.project}.{dataset_id}")
            return f"region-{ds.location.lower()}"
        except Exception:
            return "region-eu"

    # ------------------------------------------------------------------ #
    # Extraction recipes
    # ------------------------------------------------------------------ #

    def extract_baseline(self, params: BaselineExtractionParams, limit: int = 0) -> pd.DataFrame:
        return self.run(_experiments_sql.build_baseline(params, limit=limit))

    def extract_binomial(self, params: BinomialExtractionParams, limit: int = 0) -> pd.DataFrame:
        return self.run(_experiments_sql.build_binomial(params, limit=limit))

    def extract_continuous(self, params: ContinuousExtractionParams, limit: int = 0) -> pd.DataFrame:
        return self.run(_experiments_sql.build_continuous(params, limit=limit))

    def extract_interaction(self, params: InteractionExtractionParams, limit: int = 0) -> pd.DataFrame:
        return self.run(_experiments_sql.build_interaction(params, limit=limit))

    def extract_sequential(self, params: SequentialExtractionParams) -> pd.DataFrame:
        """
        Runs the sequential-test script (creates/updates a persistent
        cumulative table as a side effect) and returns its final aggregation.
        This is a multi-statement script, not a pure SELECT -- dry_run() will
        report is_dml=True rather than a cost estimate; there is no
        preview()-safe variant.
        """
        return self.run(_experiments_sql.build_sequential(params))

    def extract_combined_experiment(
        self,
        binomial: Optional[BinomialExtractionParams] = None,
        continuous: Optional[ContinuousExtractionParams] = None,
        limit: int = 0,
    ) -> Dict[str, pd.DataFrame]:
        """
        Extracts binomial and/or continuous experiment data. When both are
        given, they must share date_range and param_key -- the underlying
        events_* scan is materialized once (via a BigQuery session) and both
        outputs are read from it, billing the scan a single time instead of
        twice.
        """
        if binomial is None and continuous is None:
            raise ValueError("At least one of binomial or continuous must be given.")
        if binomial is not None and continuous is not None:
            if (
                binomial.connection != continuous.connection
                or binomial.date_range != continuous.date_range
                or binomial.param_key != continuous.param_key
            ):
                raise ValueError(
                    "binomial and continuous must share the same connection, date_range, and "
                    "param_key to be extracted from a single shared scan (the scan is built "
                    "from whichever one is passed as `binomial` -- a mismatched `continuous` "
                    "would otherwise be silently queried against the wrong project/dataset)."
                )

        anchor = binomial or continuous
        need_page_location, need_payment_type = _experiments_sql.experiment_shared_scan_flags(
            binomial, continuous)
        shared_select = _experiments_sql.build_shared_scan_select(
            anchor.connection.project,
            anchor.connection.dataset,
            anchor.date_range.start_date.isoformat(),
            anchor.date_range.end_date.isoformat(),
            anchor.param_key,
            need_page_location=need_page_location,
            need_payment_type=need_payment_type,
        )

        if binomial is not None and continuous is not None:
            temp_table_sql = _experiments_sql.build_experiment_shared_scan_temp_table_sql(
                shared_select)
            select_sqls = {
                "binomial": _experiments_sql.build_experiment_session_output_sql(
                    _experiments_sql.build_binomial_from_shared_scan(binomial), limit=limit
                ),
                "continuous": _experiments_sql.build_experiment_session_output_sql(
                    _experiments_sql.build_continuous_from_shared_scan(continuous), limit=limit
                ),
            }
            return self.run_shared_scan(temp_table_sql, select_sqls)

        cte_chain = (
            _experiments_sql.build_binomial_from_shared_scan(binomial)
            if binomial is not None
            else _experiments_sql.build_continuous_from_shared_scan(continuous)
        )
        sql = _experiments_sql.build_experiment_single_output_sql(
            shared_select, cte_chain, limit=limit)
        label = "binomial" if binomial is not None else "continuous"
        return {label: self.run(sql)}

    def extract_event_log(self, params: EventLogExtractionParams, limit: int = 0) -> pd.DataFrame:
        """Raw GA4 event rows shaped for process mining (one row per case/activity/timestamp)."""
        return self.run(_event_log_sql.build_event_log(params, limit=limit))

    def extract_timeseries(self, params: TimeSeriesExtractionParams, limit: int = 0) -> pd.DataFrame:
        """Daily time series for ForecastingEngine's date_col/conversions_col/revenue_col contract."""
        return self.run(_timeseries_sql.build_timeseries(params, limit=limit))

    # ------------------------------------------------------------------ #
    # Auto-detect helpers
    # ------------------------------------------------------------------ #
    # All three sample a short recent window rather than the full (possibly
    # months-long) date range -- what they're looking for (a variant string,
    # a known KPI event, an event name) is stable over the course of an
    # experiment, so scanning the whole range buys no extra accuracy, only
    # extra cost.

    def autodetect_variants(
        self,
        connection: BQConnectionConfig,
        start_date: date,
        end_date: date,
        param_key: str,
        prefix: str,
        sample_days: int = 1,
    ) -> List[str]:
        sample_start, sample_end = _sample_recent_window(
            start_date, end_date, sample_days)
        sql = _experiments_sql.build_autodetect_variants_query(
            connection.project, connection.dataset, sample_start.isoformat(
            ), sample_end.isoformat(), param_key, prefix
        )
        df = self.run(sql)
        return df["variant_string"].tolist() if not df.empty else []

    def autodetect_event_names(
        self,
        connection: BQConnectionConfig,
        start_date: date,
        end_date: date,
        limit: int = 100,
        sample_days: int = 2,
    ) -> List[str]:
        sample_start, sample_end = _sample_recent_window(
            start_date, end_date, sample_days)
        sql = _experiments_sql.build_autodetect_event_names_query(
            connection.project, connection.dataset, sample_start.isoformat(), sample_end.isoformat(), limit
        )
        df = self.run(sql)
        return df["event_name"].tolist() if not df.empty else []

    def autodetect_kpis(
        self, connection: BQConnectionConfig, start_date: date, end_date: date, sample_days: int = 2
    ) -> List[str]:
        sample_start, sample_end = _sample_recent_window(
            start_date, end_date, sample_days)
        sql = _experiments_sql.build_autodetect_kpi_query(
            connection.project, connection.dataset, sample_start.isoformat(), sample_end.isoformat()
        )
        df = self.run(sql)
        return df["event_name"].tolist() if not df.empty else []


def _client_config(client_id: str, client_secret: str, redirect_uri: str) -> dict:
    return {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [redirect_uri],
        }
    }


def _sample_recent_window(start_date: date, end_date: date, days: int) -> Tuple[date, date]:
    """Clamps to the last `days` day(s) ending at end_date, staying within
    [start_date, end_date]. The full range is left untouched for the actual
    extraction query -- only auto-detect probes use this narrower window."""
    start_dt = max(end_date - timedelta(days=days), start_date)
    return start_dt, end_date
