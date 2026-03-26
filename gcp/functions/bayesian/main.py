import functions_framework
from flask import jsonify
from pydantic import ValidationError
from foe.core.models import ExperimentInput
from foe.bayesian.operations import BayesianEngine

@functions_framework.http
def bayesian_handler(request):
    """
    HTTP Cloud Function entry point for Bayesian A/B Analysis.
    Expects a JSON payload matching the ExperimentInput schema.
    Computes Probability of Being Best, Expected Loss, and Lift Distributions.
    """
    
    # Handle CORS (Essential for Streamlit or Looker integrations)
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Max-Age': '3600'
        }
        return ('', 204, headers)

    headers = {'Access-Control-Allow-Origin': '*'}

    # Parse JSON payload
    request_json = request.get_json(silent=True)
    if not request_json:
        return (jsonify({"error": "Bad Request", "message": "Missing or invalid JSON payload"}), 400, headers)

    try:
        # Model Validation
        input_data = ExperimentInput(**request_json)

        # Engine Execution
        engine = BayesianEngine()
        results = engine.run_probability_analysis(input_data)

        # Serialization
        payload = [r.model_dump() for r in results]

        return (jsonify(payload), 200, headers)

    except ValidationError as e:
        # 422 Unprocessable Entity: The schema is wrong or the test data is logically invalid
        return (jsonify({
            "error": "Validation Error",
            "details": e.errors(include_url=False, include_context=False)
        }), 422, headers)
        
    except Exception as e:
        # 500 Internal Server Error: Something went wrong deep in the math engine
        return (jsonify({
            "error": "Internal Engine Error",
            "message": str(e)
        }), 500, headers)
