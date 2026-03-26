import functions_framework
import pandas as pd
from flask import jsonify
from foe.interaction.operations import InteractionEngine

@functions_framework.http
def interaction_handler(request):
    if request.method == 'OPTIONS':
        return ('', 204, {'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'POST', 'Access-Control-Allow-Headers': 'Content-Type'})

    headers = {'Access-Control-Allow-Origin': '*'}
    request_json = request.get_json(silent=True)
    
    data = request_json.get('data')
    kpi = request_json.get('kpi')
    factors = request_json.get('factors') # List of column names
    
    try:
        df = pd.DataFrame(data)
        engine = InteractionEngine()
        results = engine.run_interaction_analysis(df, test_cols)
        return (jsonify(results), 200, headers)
    except Exception as e:
        return (jsonify({"error": "Internal Engine Error", "message": str(e)}), 500, headers)
