import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from foe.core.models import (
    CustomHoliday,
    ForecastGranularity,
    ForecastingEngineConfig,
    GrowthMode,
    SeasonalityMode,
)
from foe.forecasting.operations import ForecastingEngine


# --------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------- #


def make_daily_df(
    n_days: int = 120,
    start: str = "2023-01-01",
    conversions_col: str = "conversions",
    revenue_col: str = None,
    regressor_col: str = None,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    trend = np.arange(n_days) * 0.3
    conversions = 50 + trend + rng.normal(0, 3, n_days)

    data = {"date": dates, conversions_col: conversions}
    if revenue_col:
        data[revenue_col] = conversions * (25 + rng.normal(0, 2, n_days))
    if regressor_col:
        data[regressor_col] = 15 + 5 * \
            np.sin(np.arange(n_days) / 15) + rng.normal(0, 1, n_days)
    return pd.DataFrame(data)


# --------------------------------------------------------------------- #
#  ForecastingEngineConfig validation
# --------------------------------------------------------------------- #


def test_config_requires_at_least_one_target():
    with pytest.raises(ValidationError, match="conversions_col or revenue_col"):
        ForecastingEngineConfig(date_col="date", periods=7)


def test_config_conversions_and_revenue_must_differ():
    with pytest.raises(ValidationError, match="must be different columns"):
        ForecastingEngineConfig(
            date_col="date", conversions_col="kpi", revenue_col="kpi", periods=7
        )


def test_config_logistic_requires_cap():
    with pytest.raises(ValidationError, match="cap is required"):
        ForecastingEngineConfig(
            date_col="date",
            conversions_col="conversions",
            periods=7,
            growth=GrowthMode.LOGISTIC,
        )


def test_config_floor_must_be_less_than_cap():
    with pytest.raises(ValidationError, match="floor"):
        ForecastingEngineConfig(
            date_col="date",
            conversions_col="conversions",
            periods=7,
            growth=GrowthMode.LOGISTIC,
            cap=100.0,
            floor=100.0,
        )


def test_config_valid_defaults():
    config = ForecastingEngineConfig(
        date_col="date", conversions_col="conversions", periods=14
    )
    assert config.granularity == ForecastGranularity.DAILY
    assert config.growth == GrowthMode.LINEAR
    assert config.seasonality_mode == SeasonalityMode.MULTIPLICATIVE
    assert config.holidays == []
    assert config.regressors == []


def test_custom_holiday_window_signs():
    holiday = CustomHoliday(holiday="Park Event", ds="2023-06-01",
                            lower_window=-2, upper_window=1)
    assert holiday.lower_window == -2
    assert holiday.upper_window == 1
    with pytest.raises(ValidationError):
        CustomHoliday(holiday="Bad", ds="2023-06-01", lower_window=1)
    with pytest.raises(ValidationError):
        CustomHoliday(holiday="Bad", ds="2023-06-01", upper_window=-1)


# --------------------------------------------------------------------- #
#  Resampling (pure pandas, no Prophet involved)
# --------------------------------------------------------------------- #


def test_resample_daily_sums_targets_and_averages_regressors():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=3, freq="D"),
            "conversions": [10, 20, 30],
            "price": [5.0, 6.0, 7.0],
        }
    )
    out = ForecastingEngine._resample(
        df, "date", ["conversions"], ["price"], ForecastGranularity.DAILY
    )
    assert list(out["conversions"]) == [10, 20, 30]
    assert list(out["price"]) == [5.0, 6.0, 7.0]
    assert list(out.columns) == ["ds", "conversions", "price"]


def test_resample_weekly_sums_targets():
    # W-MON bins are labeled by their right edge, so a run starting on a
    # Monday still splits unevenly across the first/last bin boundaries;
    # only the total across all bins is guaranteed.
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-02", periods=14, freq="D"),
            "conversions": [1] * 14,
        }
    )
    out = ForecastingEngine._resample(
        df, "date", ["conversions"], [], ForecastGranularity.WEEKLY
    )
    assert out["conversions"].sum() == 14


def test_resample_monthly_sums_targets():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=60, freq="D"),
            "conversions": [2] * 60,
        }
    )
    out = ForecastingEngine._resample(
        df, "date", ["conversions"], [], ForecastGranularity.MONTHLY
    )
    assert len(out) == 3  # Jan, Feb, Mar
    assert out["conversions"].sum() == 120


# --------------------------------------------------------------------- #
#  fit(): input validation
# --------------------------------------------------------------------- #


def test_fit_missing_column_raises():
    df = make_daily_df()
    config = ForecastingEngineConfig(
        date_col="date", conversions_col="does_not_exist", periods=7
    )
    with pytest.raises(ValueError, match="missing required column"):
        ForecastingEngine.fit(df, config)


def test_fit_empty_data_raises():
    df = make_daily_df(n_days=0)
    config = ForecastingEngineConfig(
        date_col="date", conversions_col="conversions", periods=7)
    with pytest.raises(ValueError, match="empty"):
        ForecastingEngine.fit(df, config)


def test_fit_regressor_gap_in_history_raises():
    df = make_daily_df(n_days=30, regressor_col="temp")
    df = df.drop(index=5)  # creates a gap in the daily date index
    config = ForecastingEngineConfig(
        date_col="date",
        conversions_col="conversions",
        periods=5,
        regressors=["temp"],
    )
    with pytest.raises(ValueError, match="missing period"):
        ForecastingEngine.fit(df, config)


# --------------------------------------------------------------------- #
#  fit(): end-to-end (real Prophet fits, kept small for speed)
# --------------------------------------------------------------------- #


def test_fit_conversions_only_returns_forecast():
    df = make_daily_df(n_days=90)
    config = ForecastingEngineConfig(
        date_col="date", conversions_col="conversions", periods=7
    )
    result = ForecastingEngine.fit(df, config)

    assert set(result.targets.keys()) == {"conversions"}
    target = result.targets["conversions"]
    assert len(target.forecast) == 90 + 7
    assert all(p.yhat_lower <= p.yhat <= p.yhat_upper for p in target.forecast)
    assert "trend" in target.components
    assert target.cv_metrics is not None
    assert target.cv_metrics.horizon_periods == 7
    assert "Cross-validated MAPE" in target.conclusion
    assert result.conclusion


def test_fit_both_targets_are_independent_models():
    df = make_daily_df(n_days=90, revenue_col="revenue")
    config = ForecastingEngineConfig(
        date_col="date",
        conversions_col="conversions",
        revenue_col="revenue",
        periods=5,
    )
    result = ForecastingEngine.fit(df, config)

    assert set(result.targets.keys()) == {"conversions", "revenue"}
    # Independently fit series should not produce identical forecasts.
    conv_yhat = [p.yhat for p in result.targets["conversions"].forecast]
    rev_yhat = [p.yhat for p in result.targets["revenue"].forecast]
    assert conv_yhat != rev_yhat


def test_fit_regressor_missing_future_coverage_raises():
    df = make_daily_df(n_days=60, regressor_col="temp")
    config = ForecastingEngineConfig(
        date_col="date",
        conversions_col="conversions",
        periods=7,
        regressors=["temp"],
    )
    with pytest.raises(ValueError, match="Regressor 'temp' is missing values"):
        ForecastingEngine.fit(df, config)


def test_fit_regressor_with_future_values_succeeds():
    df = make_daily_df(n_days=60, regressor_col="temp")
    config = ForecastingEngineConfig(
        date_col="date",
        conversions_col="conversions",
        periods=7,
        regressors=["temp"],
    )
    future_dates = pd.date_range(
        df["date"].max() + pd.Timedelta(days=1), periods=7, freq="D")
    future_regressors = pd.DataFrame({"ds": future_dates, "temp": 18.0})

    result = ForecastingEngine.fit(df, config, future_regressors=future_regressors)
    assert len(result.targets["conversions"].forecast) == 60 + 7


def test_fit_logistic_growth_applies_cap():
    df = make_daily_df(n_days=60)
    config = ForecastingEngineConfig(
        date_col="date",
        conversions_col="conversions",
        periods=7,
        growth=GrowthMode.LOGISTIC,
        cap=200.0,
        floor=0.0,
    )
    result = ForecastingEngine.fit(df, config)
    assert all(p.yhat <= 200.0 + 1e-6 for p in result.targets["conversions"].forecast)


def test_fit_monthly_short_history_warns():
    df = make_daily_df(n_days=200)  # ~6-7 months, well under 24 months
    config = ForecastingEngineConfig(
        date_col="date",
        conversions_col="conversions",
        granularity=ForecastGranularity.MONTHLY,
        periods=2,
    )
    result = ForecastingEngine.fit(df, config)
    warnings = result.targets["conversions"].warnings
    assert any("yearly seasonality" in w for w in warnings)


def test_fit_holidays_on_weekly_granularity_warns():
    df = make_daily_df(n_days=90)
    config = ForecastingEngineConfig(
        date_col="date",
        conversions_col="conversions",
        granularity=ForecastGranularity.WEEKLY,
        periods=3,
        holidays=[CustomHoliday(holiday="Launch Day", ds="2023-02-01")],
    )
    result = ForecastingEngine.fit(df, config)
    warnings = result.targets["conversions"].warnings
    assert any("exact date" in w for w in warnings)
