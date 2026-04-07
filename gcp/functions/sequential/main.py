import functions_framework
import pandas as pd
from flask import jsonify
from pydantic import ValidationError
from foe.core.models import SequentialDataPoint, SequentialConfig
from foe.sequential.operations import SequentialEngine


@functions_framework.http
def sequential_handler(request):
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

    try:
        # Validate data points and config
        data_points = [SequentialDataPoint(**dp) for dp in request_json.get("data", [])]
        config = SequentialConfig(**request_json.get("config", {}))

        # Convert to DataFrame for processing
        df = pd.DataFrame([dp.model_dump() for dp in data_points])

        engine = SequentialEngine()
        results = engine.process_test_trajectory(df, config)

        return (jsonify(results), 200, headers)

    except ValidationError as e:
        return (
            jsonify({"error": "Validation Error", "details": e.errors()}),
            422,
            headers,
        )
    except Exception as e:
        return (
            jsonify({"error": "Internal Engine Error", "message": str(e)}),
            500,
            headers,
        )
