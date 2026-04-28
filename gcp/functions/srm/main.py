import functions_framework
from flask import jsonify
from foe.srm.operations import SRMEngine


@functions_framework.http
def srm_handler(request):
    if request.method == "OPTIONS":
        return (
            "",
            204,
            {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        )

    headers = {"Access-Control-Allow-Origin": "*"}
    request_json = request.get_json(silent=True)

    observed = request_json.get("observed")
    expected = request_json.get("expected")

    if not observed or not expected:
        return (
            jsonify({"error": "Missing required fields: observed, expected"}),
            400,
            headers
        )

    try:
        engine = SRMEngine()
        results = engine.calculate_chi_squared(observed, expected)
        return (jsonify(results), 200, headers)
    except Exception as e:
        return (
            jsonify({"error": "Internal Engine Error", "message": str(e)}),
            500,
            headers
        )
