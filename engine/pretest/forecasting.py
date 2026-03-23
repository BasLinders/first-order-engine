import pandas as pd
from prophet import Prophet

# --- Prophet Seasonal Forecast ---

def run_seasonal_forecast(df: pd.DataFrame, periods: int = 42, interval: float = 0.95) -> pd.DataFrame:
    """Prophet logic for seasonal traffic prediction."""
    # Ensure columns: ds, visitors, conversions
    results = {}
    for col in ['visitors', 'conversions']:
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, interval_width=interval)
        m.fit(df[['ds', col]].rename(columns={col: 'y'}))
        future = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)
        results[col] = forecast[forecast['ds'] > df['ds'].max()][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    # Merge and clip
    final = pd.merge(results['visitors'], results['conversions'], on='ds', suffixes=('_vis', '_conv'))
    for c in final.columns:
        if 'yhat' in c: final[c] = final[c].clip(lower=0)
    return final
