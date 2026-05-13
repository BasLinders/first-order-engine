import json
from unittest.mock import MagicMock

from flask import Flask

from gcp.functions.frequentist.main import frequentist_handler
from gcp.functions.bayesian.main import bayesian_handler

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
            # Handlers return (flask.Response, status_code, headers).
            # response.data contains the JSON bytes produced by flask.jsonify.
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

        validators.py now includes the variant index and values in the message,
        e.g. 'Variant at index 0: conversions (150) exceed visitors (100).'
        The handler must propagate this detail in the response body.
        """
        data = {"visitors": [100, 100], "conversions": [150, 20]}
        with _app.app_context():
            response, status_code, headers = frequentist_handler(_make_request("POST", data))

        assert status_code == 422
        response_data = json.loads(response.data)
        assert response_data["error"] == "Validation Error"
        assert "conversions" in response_data.get("details", "")

    def test_frequentist_handler_method_not_allowed(self):
        """
        Non-POST/OPTIONS methods → 405 Method Not Allowed.
        """
        with _app.app_context():
            response, status_code, headers = frequentist_handler(_make_request("GET", {}))

        assert status_code == 405

    def test_cors_preflight_frequentist(self):
        """
        OPTIONS request → 204 with the correct CORS headers.
        Must run inside app_context for consistency with other handler tests.
        """
        with _app.app_context():
            response, status_code, headers = frequentist_handler(_make_request("OPTIONS", {}))

        assert status_code == 204
        assert headers["Access-Control-Allow-Methods"] == "POST"

    # ------------------------------------------------------------------
    # Bayesian handler
    # ------------------------------------------------------------------

    def test_bayesian_handler_invalid_data(self):
        """
        Invalid POST (conversions exceed visitors on variant 0) → 422.

        Checks both the top-level error key and that the detail message
        surfaces the index-aware text from our updated validators.py.
        """
        data = {"visitors": [100, 100], "conversions": [150, 20]}
        with _app.app_context():
            response, status_code, headers = bayesian_handler(_make_request("POST", data))

        assert status_code == 422
        response_data = json.loads(response.data)
        assert response_data["error"] == "Validation Error"
        assert "conversions" in response_data.get("details", "")

    def test_bayesian_handler_method_not_allowed(self):
        """
        Non-POST/OPTIONS methods → 405 Method Not Allowed.
        """
        with _app.app_context():
            response, status_code, headers = bayesian_handler(_make_request("GET", {}))

        assert status_code == 405

    def test_cors_preflight_bayesian(self):
        """
        OPTIONS request on the Bayesian handler → 204 with CORS headers.
        """
        with _app.app_context():
            response, status_code, headers = bayesian_handler(_make_request("OPTIONS", {}))

        assert status_code == 204
        assert headers["Access-Control-Allow-Methods"] == "POST"
