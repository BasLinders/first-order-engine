import json
from unittest.mock import MagicMock

import pytest
from flask import Flask

# All tests in this file require the gcp package to be installed.
# They are excluded from the standard unit test run via -m "not integration".
# Run explicitly with: pytest -m integration
pytestmark = pytest.mark.integration

from gcp.functions.frequentist.main import frequentist_handler  # noqa: E402
from gcp.functions.bayesian.main import bayesian_handler        # noqa: E402

# A minimal Flask app is needed only to provide an application context.
# The handlers themselves are GCP Cloud Function entrypoints that use Flask
# internally (e.g. flask.jsonify), so any Flask context-dependent call
# requires this wrapper.  All tests must run inside `with _app.app_context()`.
_app = Flask(__name__)


def _make_request(method: str, data: dict) -> MagicMock:
    """
    Builds a minimal mock of a Flask/GCP request object.

    get_json() returns the supplied dict regardless of kwargs (force, silent),
    which is correct for happy-path and known-invalid-data tests.  To simulate
    a missing or malformed body, override get_json.return_value = None in the
    individual test.
    """
    mock = MagicMock()
    mock.method = method
    mock.get_json.return_value = data
    return mock


class TestAPIHandlers:

    # ------------------------------------------------------------------
    # Frequentist handler
    # ------------------------------------------------------------------

    def test_frequentist_handler_success(self):
        """
        Valid POST → 200 with one FrequentistResult for the challenger.

        10% vs 15% CR at n=1000 is a clear winner (z ≈ 3.4, p ≈ 0.0007),
        so is_significant must be True and the conclusion must name a
        positive impact.
        """
        data = {
            "visitors": [1000, 1000],
            "conversions": [100, 150],
            "labels": ["Control", "Test"],
            "confidence_level": 0.95,
        }
        with _app.app_context():
            response, status_code, headers = frequentist_handler(_make_request("POST", data))

        assert status_code == 200
        response_data = json.loads(response.data)
        assert len(response_data) == 1
        assert response_data[0]["variant_label"] == "Test"
        assert response_data[0]["is_significant"] is True
        assert "Significant Positive Impact" in response_data[0]["conclusion"]

    def test_frequentist_handler_invalid_data(self):
        """
        Invalid POST (conversions exceed visitors) → 422 with error detail.
        """
        data = {"visitors": [100, 100], "conversions": [150, 20]}
        with _app.app_context():
            response, status_code, headers = frequentist_handler(_make_request("POST", data))

        assert status_code == 422
        response_data = json.loads(response.data)
        assert response_data["error"] == "Validation Error"
        assert "conversions" in response_data.get("details", "")

    def test_frequentist_handler_method_not_allowed(self):
        """Non-POST/OPTIONS methods → 405."""
        with _app.app_context():
            response, status_code, headers = frequentist_handler(_make_request("GET", {}))

        assert status_code == 405

    def test_cors_preflight_frequentist(self):
        """OPTIONS request → 204 with the correct CORS headers."""
        with _app.app_context():
            response, status_code, headers = frequentist_handler(_make_request("OPTIONS", {}))

        assert status_code == 204
        assert headers["Access-Control-Allow-Methods"] == "POST"

    # ------------------------------------------------------------------
    # Bayesian handler
    # ------------------------------------------------------------------

    def test_bayesian_handler_invalid_data(self):
        """Invalid POST → 422 with index-aware error detail."""
        data = {"visitors": [100, 100], "conversions": [150, 20]}
        with _app.app_context():
            response, status_code, headers = bayesian_handler(_make_request("POST", data))

        assert status_code == 422
        response_data = json.loads(response.data)
        assert response_data["error"] == "Validation Error"
        assert "conversions" in response_data.get("details", "")

    def test_bayesian_handler_method_not_allowed(self):
        """Non-POST/OPTIONS methods → 405."""
        with _app.app_context():
            response, status_code, headers = bayesian_handler(_make_request("GET", {}))

        assert status_code == 405

    def test_cors_preflight_bayesian(self):
        """OPTIONS request on the Bayesian handler → 204 with CORS headers."""
        with _app.app_context():
            response, status_code, headers = bayesian_handler(_make_request("OPTIONS", {}))

        assert status_code == 204
        assert headers["Access-Control-Allow-Methods"] == "POST"
