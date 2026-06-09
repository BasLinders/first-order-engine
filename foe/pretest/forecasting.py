import pandas as pd
from prophet import Prophet
from typing import Dict, Any, Optional

from foe.core.models import AnalysisUnit


class ForecastingEngine:
    """
    Handles time-series forecasting for A/B testing traffic and metrics.
    Isolated from the core engine to manage heavy dependencies (Prophet/Stan);
    imports only pandas, Prophet, and the lightweight AnalysisUnit enum.
    """

    @staticmethod
    def generate_forecast_conclusion(
        total_count: float,
        total_value: float,
        periods: int,
        interval: float,
        is_rate: bool,
        unit_noun: str,
    ) -> str:
        """UI-friendly summary of the forecast. `is_rate` formats the per-unit
        mean as a conversion-rate percentage; otherwise as a mean value."""
        conf_pct = interval * 100
        if total_count <= 0:
            return "Forecast indicates no significant expected volume for the specified period."

        per_unit = total_value / total_count
        if is_rate:
            metric_phrase = f"an estimated baseline conversion rate of {per_unit * 100:.2f}%"
        else:
            metric_phrase = f"an estimated mean of {per_unit:,.2f} per {unit_noun}"

        return (
            f"Traffic Projection: Over the next {periods} days, the model projects approximately "
            f"{int(total_count):,} {unit_noun} and {int(total_value):,} total "
            f"({metric_phrase}). These estimates represent the median forecast with a "
            f"{conf_pct:.0f}% confidence interval."
        )

    @staticmethod
    def _resolve_columns(
        columns: list[str],
        unit: AnalysisUnit,
        count_col: Optional[str],
        value_col: Optional[str],
    ) -> tuple[str, str, bool]:
        """Resolve which count and value columns to forecast.

        Returns (count_col, value_col, is_rate). `is_rate` is True for the
        binomial (conversions) path so the conclusion formats a percentage.
        Defaults: count_col is 'visitors' for per-visitor and 'transactions'
        for per-transaction; value_col is 'conversions' if present (binomial),
        else 'kpi_total' (continuous). Either may be overridden by argument.
        """
        if count_col is None:
            count_col = "visitors" if unit == AnalysisUnit.PER_VISITOR else "transactions"
        if value_col is None:
            value_col = "conversions" if "conversions" in columns else "kpi_total"
        is_rate = value_col == "conversions"
        return count_col, value_col, is_rate

    @classmethod
    def run_seasonal_forecast(
        cls,
        df: pd.DataFrame,
        periods: int = 42,
        interval: float = 0.95,
        *,
        unit: AnalysisUnit = AnalysisUnit.PER_VISITOR,
        count_col: Optional[str] = None,
        value_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Forecast daily volume and a metric total with Prophet, for the chosen
        analysis unit. Returns JSON-safe primitives.

        Args:
            df:        Daily history with a 'ds' date column plus the count and
                       value columns (see below). At least 14 rows recommended.
            periods:   Days to forecast (default 42 = 6 weeks).
            interval:  Prophet prediction-interval width (default 0.95).
            unit:      AnalysisUnit. Selects the default count column:
                       PER_VISITOR -> 'visitors', PER_TRANSACTION -> 'transactions'.
            count_col: Override the denominator/sample-size count column by name.
                       Defaults per `unit`. For per-transaction, supply the actual
                       transaction count column (do not derive it from visitors).
            value_col: Override the metric column to forecast. Defaults to
                       'conversions' if that column is present (binomial), else
                       'kpi_total' (continuous sum-of-metric).

        Returns:
            dict with:
              'forecast'   : list of records, each {ds, pred_count, count_lower,
                             count_upper, pred_value, value_lower, value_upper}.
              'count_col'  : the forecasted count column name.
              'value_col'  : the forecasted value column name.
              'unit'       : the analysis unit used.
              'conclusion' : UI summary string.
        """
        cols = list(df.columns)
        resolved_count, resolved_value, is_rate = cls._resolve_columns(
            cols, unit, count_col, value_col
        )

        if df.empty or "ds" not in cols:
            return {"forecast": [], "conclusion": "Invalid input data: missing 'ds' date column."}
        for needed in (resolved_count, resolved_value):
            if needed not in cols:
                return {
                    "forecast": [],
                    "conclusion": f"Invalid input data: missing required column '{needed}'.",
                }

        if len(df) < 14:
            return {
                "forecast": [],
                "conclusion": "Insufficient historical data. At least 14 days are recommended for seasonal forecasting.",
            }

        max_date = df["ds"].max()
        predictions: dict[str, pd.DataFrame] = {}

        # Branch cleanly: forecast the resolved count and value series the same
        # way, labelling outputs generically as 'count' and 'value'.
        for role, source_col in (("count", resolved_count), ("value", resolved_value)):
            m = Prophet(
                yearly_seasonality=True,   # type: ignore[arg-type]
                weekly_seasonality=True,   # type: ignore[arg-type]
                daily_seasonality=False,   # type: ignore[arg-type]
                interval_width=interval,
            )
            train_df = df[["ds", source_col]].rename(columns={source_col: "y"})
            m.fit(train_df)

            future = m.make_future_dataframe(periods=periods)
            forecast = m.predict(future)
            future_only = forecast[forecast["ds"] > max_date][
                ["ds", "yhat", "yhat_lower", "yhat_upper"]
            ].rename(
                columns={
                    "yhat": f"pred_{role}",
                    "yhat_lower": f"{role}_lower",
                    "yhat_upper": f"{role}_upper",
                }
            )
            predictions[role] = future_only

        final = pd.merge(predictions["count"], predictions["value"], on="ds")

        # Counts and metric sums are non-negative.
        for c in final.columns:
            if c != "ds":
                final[c] = final[c].clip(lower=0)

        total_count = float(final["pred_count"].sum())
        total_value = float(final["pred_value"].sum())

        final["ds"] = final["ds"].dt.strftime("%Y-%m-%d")
        forecast_records = final.to_dict(orient="records")

        unit_noun = "visitors" if unit == AnalysisUnit.PER_VISITOR else "transactions"
        return {
            "forecast": forecast_records,
            "count_col": resolved_count,
            "value_col": resolved_value,
            "unit": unit.value,
            "conclusion": cls.generate_forecast_conclusion(
                total_count=total_count,
                total_value=total_value,
                periods=periods,
                interval=interval,
                is_rate=is_rate,
                unit_noun=unit_noun,
            ),
        }
