import functions_framework
import pandas as pd
from flask import jsonify
from foe.continuous.operations import ContinuousMetricEngine


@functions_framework.http
def continuous_handler(request):
    if request.method == "OPTIONS":
        return (
            "",
            204,
            {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST",
                "Access-Control-Allow-Headers": "Content-Type",
            },
        )

    headers = {"Access-Control-Allow-Origin": "*"}
    request_json = request.get_json(silent=True)

    data = request_json.get("data")  # List of dicts
    kpi = request_json.get("kpi")

    if not data or not kpi:
        return (jsonify({"error": "Missing required fields: data, kpi"}), 400, headers)

    try:
        df = pd.DataFrame(data)
        engine = ContinuousMetricEngine()
        results = engine.run_comparison_suite(df, kpi)
        return (jsonify(results), 200, headers)
    except Exception as e:
        return (
            jsonify({"error": "Internal Engine Error", "message": str(e)}),
            500,
            headers,
        )
