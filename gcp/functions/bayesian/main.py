import functions_framework
from flask import jsonify
from pydantic import ValidationError
from foe.core.models import ExperimentInput
from foe.bayesian.operations import BayesianEngine


@functions_framework.http
def bayesian_handler(request):
    """
    HTTP Cloud Function entry point for Bayesian A/B Analysis.
    Supports informed priors and optional business case projections.
    """
    
    # 1. Handle CORS
    if request.method == 'OPTIONS':
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Max-Age": "3600"
        }
        return ("", 204, headers)

    headers = {"Access-Control-Allow-Origin": "*"}

    # 2. Parse JSON payload
    request_json = request.get_json(silent=True)
    if not request_json:
        return (
            jsonify(
                {"error": "Bad Request", "message": "Missing or invalid JSON payload"}
            ),
            400,
            headers
        )

    try:
        # 3. Model Validation
        input_data = ExperimentInput(**request_json)
        engine = BayesianEngine()

        # 4. Step One: Run Probability Analysis
        # We use getattr to safely grab optional prior fields if they exist in your Pydantic model
        prob_results = engine.run_probability_analysis(
            visitors=input_data.visitors,
            conversions=input_data.conversions,
            prior_alphas=getattr(input_data, 'prior_alphas', None),
            prior_betas=getattr(input_data, 'prior_betas', None)
        )

        # 5. Step Two: Check for Business Case
        # If the user provided AOV and Projection data, run the monetary engine
        if hasattr(input_data, 'biz_case') and input_data.biz_case:
            full_results = engine.run_monetary_projection(
                visitors=input_data.visitors,
                conversions=input_data.conversions,
                biz_case=input_data.biz_case,
                prob_best_overall=prob_results['prob_being_best'],
                variant_labels=input_data.labels
            )
            return (jsonify(full_results), 200, headers)

        # 6. Fallback: Return raw probabilities if no business case provided
        return (jsonify(prob_results), 200, headers)

    except ValidationError as e:
        # 422 Unprocessable Entity: The schema is wrong or the test data is logically invalid
        return (
            jsonify(
                {
                    "error": "Validation Error",
                    "details": e.errors(include_url=False, include_context=False)
                }
            ),
            422,
            headers
        )

    except Exception as e:
        # 500 Internal Server Error: Something went wrong deep in the math engine
        return (
            jsonify({"error": "Internal Engine Error", "message": str(e)}),
            500,
            headers
        )
