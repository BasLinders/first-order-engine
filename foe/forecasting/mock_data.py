"""
foe/forecasting/mock_data.py

Synthetic demo data for ForecastingEngine. Exists purely to give a caller
(typically a Streamlit "generate demo data" button) a one-click way to show
off the engine: trend + weekly/yearly seasonality, holiday spikes (both
well-known retail dates and a fictional company event, to show off custom
events distinct from generic holidays), and a marketing-spend covariate
whose effect on conversions is real rather than merely correlated, so
selecting it as a regressor visibly improves the fit. Nothing here is used
by ForecastingEngine itself.
"""

from dataclasses import dataclass
from datetime import date as date_type
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from foe.core.models import (
    CustomHoliday,
    ForecastGranularity,
    ForecastingEngineConfig,
    SeasonalityMode,
)

_DATE_COL = "date"
_CONVERSIONS_COL = "conversions"
_REVENUE_COL = "revenue"
_REGRESSOR_COL = "marketing_spend"

# (month, day, name, lower_window, upper_window, conversions_bump)
# "Founders Day" is a fictional company event (not a public holiday) --
# included to demo custom, business-specific events alongside generic ones.
_HOLIDAY_EVENTS: List[Tuple[int, int, str, int, int, float]] = [
    (11, 25, "Black Friday", -1, 2, 45.0),
    (12, 25, "Christmas", -2, 1, 35.0),
    (1, 1, "New Year's Day", 0, 1, 20.0),
    (7, 15, "Summer Sale", -1, 1, 18.0),
    (5, 10, "Founders Day", 0, 0, 22.0),
]

# Neighboring days inside a holiday's window get this fraction of the bump.
_HOLIDAY_WINDOW_DECAY = 0.4


@dataclass(frozen=True)
class MockForecastDataset:
    """
    A self-contained demo bundle: synthetic history, future regressor
    values covering the forecast horizon, and a suggested config wired to
    both -- so a caller can go straight to
    ``ForecastingEngine.fit(dataset.data, dataset.suggested_config, dataset.future_regressors)``.
    Custom holidays are already inside ``suggested_config.holidays``.
    """

    data: pd.DataFrame
    future_regressors: pd.DataFrame
    suggested_config: ForecastingEngineConfig


def _generate_marketing_spend(n_days: int, rng: np.random.Generator) -> np.ndarray:
    """Baseline spend with occasional campaign bursts (~5% of days)."""
    campaign_day = rng.random(n_days) < 0.05
    spend = 200 + campaign_day * \
        rng.uniform(800, 1500, n_days) + rng.normal(0, 20, n_days)
    return np.clip(spend, 50, None)


def _apply_holidays(
    dates: pd.DatetimeIndex,
) -> Tuple[np.ndarray, List[CustomHoliday]]:
    """
    Bakes each event in `_HOLIDAY_EVENTS` into every year it falls inside
    `dates`, returning both the additive bump array and the matching
    CustomHoliday rows (one per occurrence per year, mirroring how a real
    multi-year holidays dataframe is shaped).
    """
    bump = np.zeros(len(dates))
    holidays: List[CustomHoliday] = []
    date_index = pd.Index(dates)

    for month, day, name, lower_window, upper_window, magnitude in _HOLIDAY_EVENTS:
        for year in range(dates[0].year, dates[-1].year + 1):
            try:
                event_date = pd.Timestamp(year=year, month=month, day=day)
            except ValueError:
                continue
            if event_date < dates[0] or event_date > dates[-1]:
                continue

            holidays.append(
                CustomHoliday(
                    holiday=name,
                    ds=event_date.date(),
                    lower_window=lower_window,
                    upper_window=upper_window,
                )
            )
            for offset in range(lower_window, upper_window + 1):
                pos = date_index.get_indexer(
                    [event_date + pd.Timedelta(days=offset)])[0]
                if pos != -1:
                    bump[pos] += magnitude * \
                        (1.0 if offset == 0 else _HOLIDAY_WINDOW_DECAY)

    return bump, holidays


def generate_mock_forecast_data(
    n_days: int = 730,
    periods: int = 60,
    end_date: Optional[date_type] = None,
    seed: Optional[int] = None,
) -> MockForecastDataset:
    """
    Generates a synthetic retail-style dataset built to exercise every
    ForecastingEngine feature at once: an upward trend, weekly seasonality
    (weekend dip), yearly seasonality (a Nov/Dec peak), holiday spikes, and
    a marketing-spend covariate with a genuine effect on conversions.

    Two years of history is the default: long enough that cross-validation
    has room to run and that a monthly-granularity demo doesn't immediately
    trip the "short history" warning.

    Args:
        n_days:    Days of synthetic history to generate.
        periods:   Forecast horizon (days) the accompanying
                   `future_regressors` and `suggested_config` are built for.
        end_date:  Last date of history. Defaults to today.
        seed:      Optional RNG seed for reproducible demo data.

    Returns:
        A MockForecastDataset ready to pass straight into
        ``ForecastingEngine.fit(data, config, future_regressors)``.
    """
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.today().normalize()
    dates = pd.date_range(end=end, periods=n_days, freq="D")
    day_index = np.arange(n_days)

    trend = 40 + day_index * 0.05

    weekday = dates.weekday.values
    weekly = np.where(weekday >= 5, -8.0, 3.0)  # weekend dip

    day_of_year = dates.dayofyear.values
    yearly = 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25) + 10 * np.exp(
        -((day_of_year - 330) ** 2) / (2 * 20.0 ** 2)
    )  # broad summer/winter swing, plus a late-November shopping-season bump

    spend = _generate_marketing_spend(n_days, rng)
    spend_effect = 0.01 * (spend - 200)  # genuine, not just correlated

    holiday_bump, holidays = _apply_holidays(dates)

    noise = rng.normal(0, 4, n_days)
    conversions = np.clip(
        trend + weekly + yearly + spend_effect + holiday_bump + noise, 1, None
    )

    aov = 20 + 3 * np.sin(2 * np.pi * (day_of_year - 200) / \
                          365.25) + rng.normal(0, 1.5, n_days)
    revenue = conversions * np.clip(aov, 5, None)

    data = pd.DataFrame(
        {
            _DATE_COL: dates,
            _CONVERSIONS_COL: conversions,
            _REVENUE_COL: revenue,
            _REGRESSOR_COL: spend,
        }
    )

    future_dates = pd.date_range(
        start=dates[-1] + pd.Timedelta(days=1), periods=periods, freq="D")
    future_spend = _generate_marketing_spend(periods, rng)
    future_regressors = pd.DataFrame({"ds": future_dates, _REGRESSOR_COL: future_spend})

    suggested_config = ForecastingEngineConfig(
        date_col=_DATE_COL,
        conversions_col=_CONVERSIONS_COL,
        revenue_col=_REVENUE_COL,
        granularity=ForecastGranularity.DAILY,
        periods=periods,
        seasonality_mode=SeasonalityMode.MULTIPLICATIVE,
        holidays=holidays,
        regressors=[_REGRESSOR_COL],
    )

    return MockForecastDataset(
        data=data,
        future_regressors=future_regressors,
        suggested_config=suggested_config,
    )
