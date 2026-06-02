import pandas as pd
from prophet import Prophet
from typing import Dict, Any


class ForecastingEngine:
    """
    Handles time-series forecasting for A/B testing traffic and conversions.
    Isolated from the core engine to manage heavy dependencies (Prophet/Stan).
    """

    @staticmethod
    def generate_forecast_conclusion(
        total_expected_visitors: float,
        total_expected_conversions: float,
        periods: int,
        interval: float
    ) -> str:
        """
        Generates a UI-friendly summary of the forecasted traffic.
        """
        conf_pct = interval * 100
        if total_expected_visitors <= 0:
            return "Forecast indicates no significant expected traffic for the specified period."

        expected_cr = (total_expected_conversions / total_expected_visitors) * 100

        return (
            f"Traffic Projection: Over the next {periods} days, the model projects approximately "
            f"{int(total_expected_visitors):,} visitors and {int(total_expected_conversions):,} conversions "
            f"(an estimated baseline conversion rate of {expected_cr:.2f}%). "
            f"These estimates represent the median forecast with a {conf_pct:.0f}% confidence interval."
        )

    @classmethod
    def run_seasonal_forecast(
        cls, df: pd.DataFrame, periods: int = 42, interval: float = 0.95
    ) -> Dict[str, Any]:
        """
        Prophet logic for seasonal traffic prediction.
        Returns JSON-safe primitives suitable for Cloud APIs.
        """
        if (
            df.empty
            or "ds" not in df.columns
            or "visitors" not in df.columns
            or "conversions" not in df.columns
        ):
            return {
                "forecast": [],
                "conclusion": "Invalid input data: Missing required columns (ds, visitors, conversions).",
            }

        # Prophet typically needs at least a few weeks of data to detect seasonality
        if len(df) < 14:
            return {
                "forecast": [],
                "conclusion": "Insufficient historical data. At least 14 days are recommended for seasonal forecasting.",
            }

        results = {}
        max_date = df["ds"].max()

        for col in ["visitors", "conversions"]:
            # Suppressing daily seasonality as A/B test data is usually aggregated by day, not by hour
            m = Prophet(
                yearly_seasonality=True, # type: ignore[arg-type]
                weekly_seasonality=True, # type: ignore[arg-type]
                daily_seasonality=False, # type: ignore[arg-type]
                interval_width=interval
            )

            train_df = df[["ds", col]].rename(columns={col: "y"})
            m.fit(train_df)

            future = m.make_future_dataframe(periods=periods)
            forecast = m.predict(future)

            # Filter to only future dates
            future_forecast = forecast[forecast["ds"] > max_date][
                ["ds", "yhat", "yhat_lower", "yhat_upper"]
            ]
            results[col] = future_forecast

        # Merge the two forecasts
        final = pd.merge(
            results["visitors"],
            results["conversions"],
            on="ds",
            suffixes=("_vis", "_conv")
        )

        # Clip negative predictions (you can't have negative visitors)
        for c in final.columns:
            if "yhat" in c:
                final[c] = final[c].clip(lower=0)

        # Calculate totals for the conclusion string
        total_vis = float(final["yhat_vis"].sum())
        total_conv = float(final["yhat_conv"].sum())

        # JSON Serialization Fix: Convert datetime to string format (YYYY-MM-DD)
        final["ds"] = final["ds"].dt.strftime("%Y-%m-%d")

        # Convert DataFrame to a list of dictionaries for the API response
        forecast_records = final.to_dict(orient="records")

        return {
            "forecast": forecast_records,
            "conclusion": cls.generate_forecast_conclusion(
                total_expected_visitors=total_vis,
                total_expected_conversions=total_conv,
                periods=periods,
                interval=interval
            ),
        }
