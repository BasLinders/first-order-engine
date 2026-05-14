import functions_framework
from flask import jsonify
from pydantic import ValidationError
from foe.core.models import ExperimentInput
from foe.bayesian.operations import BayesianEngine, BetaPrior


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

        # 4. Run Probability Analysis
        prob_results = engine.run_probability_analysis(data=input_data)

        # 5. If a business case was provided, run the monetary projection
        if input_data.biz_case:
            labels = input_data.labels or [
                f"Variant {i}" for i in range(len(input_data.visitors))
            ]
            control_pbb = 1.0 - sum(r.prob_being_best for r in prob_results)
            prob_best_overall = [control_pbb] + [r.prob_being_best for r in prob_results]

            beta_prior = BetaPrior(
                alpha=input_data.biz_case.alpha_prior,
                beta=input_data.biz_case.beta_prior,
            )
            full_results = engine.run_monetary_projection(
                visitors=input_data.visitors,
                conversions=input_data.conversions,
                biz_case=input_data.biz_case,
                prob_best_overall=prob_best_overall,
                variant_labels=labels,
                beta_prior=beta_prior,
            )
            return (jsonify(full_results), 200, headers)

        # 6. Fallback: return probability results
        payload = [r.model_dump() for r in prob_results]
        return (jsonify(payload), 200, headers)

    except ValidationError as e:
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
        return (
            jsonify({"error": "Internal Engine Error", "message": str(e)}),
            500,
            headers
        )
