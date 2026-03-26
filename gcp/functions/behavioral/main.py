import functions_framework
import pandas as pd
from flask import jsonify
from foe.behavioral.operations import BehavioralEngine

@functions_framework.http
def behavioral_handler(request):
    if request.method == 'OPTIONS':
        return ('', 204, {'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'POST', 'Access-Control-Allow-Headers': 'Content-Type'})

    headers = {'Access-Control-Allow-Origin': '*'}
    request_json = request.get_json(silent=True)
    
    data = request_json.get('data')
    kpi = request_json.get('kpi')
    control = request_json.get('control_label')
    
    try:
        df = pd.DataFrame(data)
        engine = BehavioralEngine()
        
        # Optional: add logic here to call detect_outliers_mask if params are provided
        results = engine.run_welch_inference(df, kpi, control, alpha)
        return (jsonify(results), 200, headers)
    except Exception as e:
        return (jsonify({"error": "Internal Engine Error", "message": str(e)}), 500, headers)
