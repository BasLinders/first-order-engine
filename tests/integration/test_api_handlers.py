from unittest.mock import MagicMock
import json
from flask import Flask

# Import the handlers
# Note: Ensure your PYTHONPATH includes the project root
from gcp.functions.frequentist.main import frequentist_handler
from gcp.functions.bayesian.main import bayesian_handler

_app = Flask(__name__)


class TestAPIHandlers:

    def test_frequentist_handler_success(self):
        """Tests a valid POST request to the Frequentist API."""
        data = {
            "visitors": [1000, 1000],
            "conversions": [100, 150],
            "labels": ["Control", "Test"],
            "confidence_level": 0.95
        }

        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.get_json.return_value = data

        with _app.app_context():
            response, status_code, headers = frequentist_handler(mock_request)

        assert status_code == 200
        response_data = json.loads(response.data)

        assert len(response_data) == 1
        assert response_data[0]["variant_label"] == "Test"
        assert response_data[0]["is_significant"] is True
        assert "Significant Positive Impact" in response_data[0]["conclusion"]

    def test_bayesian_handler_invalid_data(self):
        """Tests that the Bayesian handler correctly returns a 422 on invalid data."""
        data = {"visitors": [100, 100], "conversions": [150, 20]}

        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.get_json.return_value = data

        with _app.app_context():
            response, status_code, headers = bayesian_handler(mock_request)

        assert status_code == 422
        response_data = json.loads(response.data)
        assert response_data["error"] == "Validation Error"

    def test_cors_preflight(self):
        """Ensures the Frequentist handler supports CORS OPTIONS requests."""
        mock_request = MagicMock()
        mock_request.method = "OPTIONS"

        response, status_code, headers = frequentist_handler(mock_request)

        assert status_code == 204
        assert headers["Access-Control-Allow-Methods"] == "POST"
