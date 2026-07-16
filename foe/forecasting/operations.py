"""
foe/forecasting/operations.py

Prophet-based forecasting for conversions and/or revenue at daily/weekly/
monthly granularity, with optional custom holidays/events and user-supplied
covariates (regressors).

Design decisions worth calling out:

* Two independent Prophet models, not one derived series. When both
  conversions and revenue are selected, this engine fits Prophet twice --
  once per target -- rather than forecasting revenue as a forecasted
  conversion count times a separately-forecast average order value.
  Multiplying two independently-forecast series compounds their
  uncertainty in a way that is hard to reason about and rarely more
  accurate in practice; two independent fits are simpler, each with its
  own honest prediction interval.
* Granularity-driven seasonality. Yearly/weekly seasonality terms are
  switched on or off by data granularity (see `_seasonality_flags`), not
  left to Prophet's own defaults, because weekly seasonality is
  meaningless once the data has already been aggregated to one point per
  week or month.
* No implicit covariates. Regressors are only ever what the caller lists
  in `config.regressors` -- nothing (including weather) is added
  automatically. This module has no network access; a caller wanting a
  weather regressor fetches it elsewhere and passes it in like any other
  covariate, via the input data and/or `future_regressors`.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

from foe.core.models import (
    CustomHoliday,
    ForecastCVMetrics,
    ForecastGranularity,
    ForecastingEngineConfig,
    ForecastingResult,
    ForecastPoint,
    GrowthMode,
    SingleTargetForecast,
)

_GRANULARITY_FREQ = {
    ForecastGranularity.DAILY: "D",
    ForecastGranularity.WEEKLY: "W-MON",
    ForecastGranularity.MONTHLY: "MS",
}

_GRANULARITY_NOUN = {
    ForecastGranularity.DAILY: ("day", "days"),
    ForecastGranularity.WEEKLY: ("week", "weeks"),
    ForecastGranularity.MONTHLY: ("month", "months"),
}

# Below this many aggregated points, yearly seasonality on monthly data is
# flagged as unreliable (roughly 2 years of monthly history).
_MONTHLY_YEARLY_SEASONALITY_MIN_PERIODS = 24

# Cross-validation is skipped below this multiple of the horizon -- Prophet's
# rolling-origin CV needs several non-overlapping windows to mean anything.
_CV_MIN_HISTORY_MULTIPLE = 3

_DAYS_PER_PERIOD = {
    ForecastGranularity.DAILY: 1,
    ForecastGranularity.WEEKLY: 7,
    ForecastGranularity.MONTHLY: 30,
}


class ForecastingEngine:
    """
    Fits Prophet models for conversions and/or revenue on daily/weekly/
    monthly aggregated data, with custom holidays, opt-in covariates, and
    growth/seasonality controls exposed via ForecastingEngineConfig.
    """

    # ------------------------------------------------------------------ #
    # Data layer
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resample(
        df: pd.DataFrame,
        date_col: str,
        target_cols: List[str],
        regressor_cols: List[str],
        granularity: ForecastGranularity,
    ) -> pd.DataFrame:
        """
        Aggregates to the chosen granularity before Prophet ever sees the
        data, rather than relying on Prophet's own frequency handling.
        Targets (conversions/revenue) are summed; regressors are averaged,
        since covariates like price or temperature are rates/levels rather
        than counts that should compound under resampling.
        """
        work = df[[date_col] + target_cols + regressor_cols].copy()
        work[date_col] = pd.to_datetime(work[date_col])
        work = work.set_index(date_col).sort_index()

        freq = _GRANULARITY_FREQ[granularity]
        agg = {col: "sum" for col in target_cols}
        agg.update({col: "mean" for col in regressor_cols})
        resampled = work.resample(freq).agg(agg).reset_index()
        return resampled.rename(columns={date_col: "ds"})

    # ------------------------------------------------------------------ #
    # Seasonality / holidays
    # ------------------------------------------------------------------ #

    @staticmethod
    def _seasonality_flags(granularity: ForecastGranularity) -> Dict[str, bool]:
        if granularity == ForecastGranularity.DAILY:
            return {"yearly_seasonality": True, "weekly_seasonality": True}
        # WEEKLY and MONTHLY: weekly seasonality is meaningless once data is
        # aggregated above daily resolution.
        return {"yearly_seasonality": True, "weekly_seasonality": False}

    @staticmethod
    def _build_holidays_df(holidays: List[CustomHoliday]) -> Optional[pd.DataFrame]:
        if not holidays:
            return None
        return pd.DataFrame(
            [
                {
                    "holiday": h.holiday,
                    "ds": pd.Timestamp(h.ds),
                    "lower_window": h.lower_window,
                    "upper_window": h.upper_window,
                }
                for h in holidays
            ]
        )

    # ------------------------------------------------------------------ #
    # Covariates
    # ------------------------------------------------------------------ #

    @staticmethod
    def _standardize(series: pd.Series) -> Tuple[pd.Series, float, float]:
        mean = float(series.mean())
        std = float(series.std())
        if std == 0 or np.isnan(std):
            std = 1.0
        return (series - mean) / std, mean, std

    @staticmethod
    def _future_index(
        last_ds: pd.Timestamp, periods: int, granularity: ForecastGranularity
    ) -> pd.DatetimeIndex:
        freq = _GRANULARITY_FREQ[granularity]
        full = pd.date_range(start=last_ds, periods=periods + 1, freq=freq)
        return full[1:]

    @classmethod
    def _validate_regressor_horizon_coverage(
        cls,
        name: str,
        future_regressors: Optional[pd.DataFrame],
        future_index: pd.DatetimeIndex,
    ) -> None:
        """
        Confirms a regressor has known values across the entire forecast
        horizon (either uploaded future values or an auto-fetched series
        such as weather, both supplied via `future_regressors`). Raises
        rather than truncating the horizon or fitting silently without
        them -- Prophet cannot predict through a NaN regressor anyway, so
        this turns an opaque failure into an actionable one.
        """
        if future_regressors is not None and name in future_regressors.columns:
            fr = future_regressors.set_index(
                pd.to_datetime(future_regressors["ds"])
            )[name]
            available = fr.reindex(future_index)
        else:
            available = pd.Series(index=future_index, dtype=float)

        missing = int(available.isna().sum())
        if missing:
            raise ValueError(
                f"Regressor '{name}' is missing values for {missing} of "
                f"{len(future_index)} dates in the forecast horizon. Pass "
                f"future_regressors covering the full horizon for '{name}' "
                "before fitting."
            )

    # ------------------------------------------------------------------ #
    # Cross-validation
    # ------------------------------------------------------------------ #

    @classmethod
    def _cross_validate(
        cls,
        model: Prophet,
        train_df: pd.DataFrame,
        config: ForecastingEngineConfig,
    ) -> Optional[ForecastCVMetrics]:
        span_days = (train_df["ds"].max() - train_df["ds"].min()).days
        horizon_periods = config.cv_horizon_periods or config.periods
        days_per_period = _DAYS_PER_PERIOD[config.granularity]
        horizon_days = horizon_periods * days_per_period

        if span_days < horizon_days * _CV_MIN_HISTORY_MULTIPLE:
            return None

        initial_days = max(span_days - 2 * horizon_days, horizon_days)
        period_days = max(horizon_days // 2, 1)

        try:
            df_cv = cross_validation(
                model,
                initial=f"{initial_days} days",
                period=f"{period_days} days",
                horizon=f"{horizon_days} days",
                disable_tqdm=True,
            )
            df_perf = performance_metrics(df_cv)
        except Exception:
            # Prophet's rolling-origin CV can fail on data that technically
            # meets the span check above but is too irregular for even one
            # clean cutoff (e.g. sparse history). Diagnostics are a nicety,
            # not something that should crash a forecast either way.
            return None

        return ForecastCVMetrics(
            mape=float(df_perf["mape"].mean()),
            rmse=float(df_perf["rmse"].mean()),
            horizon_periods=horizon_periods,
        )

    # ------------------------------------------------------------------ #
    # Components / conclusion
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_components(
        forecast: pd.DataFrame, regressor_cols: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        candidate_cols = ["trend", "yearly", "weekly", "holidays"] + regressor_cols
        components: Dict[str, List[Dict[str, Any]]] = {}
        for col in candidate_cols:
            if col in forecast.columns:
                sub = forecast[["ds", col]].copy()
                sub["ds"] = sub["ds"].dt.strftime("%Y-%m-%d")
                components[col] = sub.rename(columns={col: "value"}).to_dict(
                    orient="records"
                )
        return components

    @staticmethod
    def _generate_conclusion(
        target_name: str,
        forecast_points: List[ForecastPoint],
        config: ForecastingEngineConfig,
        cv_metrics: Optional[ForecastCVMetrics],
    ) -> str:
        if not forecast_points:
            return f"No forecast could be generated for '{target_name}'."

        singular, plural = _GRANULARITY_NOUN[config.granularity]
        future_total = sum(p.yhat for p in forecast_points[-config.periods:])

        if cv_metrics:
            quality = (
                f" Cross-validated MAPE is {cv_metrics.mape:.1%} over a "
                f"{cv_metrics.horizon_periods}-{singular} horizon."
            )
        else:
            quality = (
                " Cross-validation metrics unavailable (insufficient history "
                "for a reliable rolling-origin evaluation)."
            )

        return (
            f"Forecast ({config.granularity.value}): over the next "
            f"{config.periods} {plural}, '{target_name}' is projected to "
            f"total {future_total:,.0f} ({config.interval_width:.0%} "
            f"interval).{quality}"
        )

    # ------------------------------------------------------------------ #
    # Per-target fit
    # ------------------------------------------------------------------ #

    @classmethod
    def _fit_one_target(
        cls,
        target_name: str,
        train_df: pd.DataFrame,
        config: ForecastingEngineConfig,
        regressor_cols: List[str],
        future_regressors: Optional[pd.DataFrame],
    ) -> SingleTargetForecast:
        warnings: List[str] = []

        if (
            config.granularity == ForecastGranularity.MONTHLY
            and len(train_df) < _MONTHLY_YEARLY_SEASONALITY_MIN_PERIODS
        ):
            warnings.append(
                "History shorter than ~2 years of monthly data: yearly "
                "seasonality estimates may be unreliable."
            )

        if config.holidays and config.granularity != ForecastGranularity.DAILY:
            warnings.append(
                "Custom holidays are matched by exact date; on weekly/monthly "
                "granularity, widen lower_window/upper_window so the event "
                "overlaps a period boundary, or it may have no effect."
            )

        holidays_df = cls._build_holidays_df(config.holidays)
        model_kwargs: Dict[str, Any] = dict(
            growth=config.growth.value,
            seasonality_mode=config.seasonality_mode.value,
            interval_width=config.interval_width,
            daily_seasonality=False,
            holidays=holidays_df,
            **cls._seasonality_flags(config.granularity),
        )
        m = Prophet(**model_kwargs)

        fit_df = train_df[["ds", target_name]].rename(columns={target_name: "y"})

        if config.growth == GrowthMode.LOGISTIC:
            fit_df["cap"] = config.cap
            fit_df["floor"] = config.floor if config.floor is not None else 0.0

        regressor_scaling: Dict[str, Tuple[float, float]] = {}
        for reg in regressor_cols:
            standardized, mean, std = cls._standardize(train_df[reg])
            fit_df[reg] = standardized.values
            regressor_scaling[reg] = (mean, std)
            m.add_regressor(reg)

        m.fit(fit_df)

        last_ds = train_df["ds"].max()
        future_index = cls._future_index(last_ds, config.periods, config.granularity)
        for reg in regressor_cols:
            cls._validate_regressor_horizon_coverage(reg, future_regressors, future_index)

        future = m.make_future_dataframe(
            periods=config.periods, freq=_GRANULARITY_FREQ[config.granularity]
        )
        if config.growth == GrowthMode.LOGISTIC:
            future["cap"] = config.cap
            future["floor"] = config.floor if config.floor is not None else 0.0

        for reg in regressor_cols:
            mean, std = regressor_scaling[reg]
            raw_values = pd.Series(index=future["ds"], dtype=float)
            raw_values.update(train_df.set_index("ds")[reg])
            if future_regressors is not None and reg in future_regressors.columns:
                fr = future_regressors.set_index(
                    pd.to_datetime(future_regressors["ds"])
                )[reg]
                raw_values.update(fr)
            future[reg] = ((raw_values.values - mean) / std)

        forecast = m.predict(future)

        forecast_points = [
            ForecastPoint(
                ds=row.ds.date(),
                yhat=float(row.yhat),
                yhat_lower=float(row.yhat_lower),
                yhat_upper=float(row.yhat_upper),
            )
            for row in forecast.itertuples()
        ]

        cv_metrics = cls._cross_validate(m, train_df, config)
        components = cls._extract_components(forecast, regressor_cols)
        conclusion = cls._generate_conclusion(
            target_name, forecast_points, config, cv_metrics
        )

        return SingleTargetForecast(
            target=target_name,
            forecast=forecast_points,
            components=components,
            cv_metrics=cv_metrics,
            warnings=warnings,
            conclusion=conclusion,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @classmethod
    def fit(
        cls,
        data: pd.DataFrame,
        config: ForecastingEngineConfig,
        future_regressors: Optional[pd.DataFrame] = None,
    ) -> ForecastingResult:
        """
        Fits one Prophet model per requested target (conversions and/or
        revenue -- see module docstring for why two independent models
        rather than one derived from the other).

        Args:
            data: Raw history with `config.date_col`, the requested target
                column(s), and any columns named in `config.regressors`.
                Rows need not already be at the chosen granularity --
                resampling happens here, summing targets and averaging
                regressors.
            config: A validated ForecastingEngineConfig.
            future_regressors: Optional dataframe of known covariate values
                for the forecast horizon (`ds` plus one column per name in
                `config.regressors`), e.g. an uploaded marketing spend plan
                or a weather forecast fetched by the caller. Required
                whenever a regressor's historical values don't already
                extend across the whole future horizon.

        Returns:
            A ForecastingResult with one SingleTargetForecast per requested
            target, keyed "conversions" / "revenue".
        """
        target_cols = [c for c in (config.conversions_col, config.revenue_col) if c]
        required_cols = [config.date_col, *target_cols, *config.regressors]
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Input data is missing required column(s): {missing}.")
        if data.empty:
            raise ValueError("Input data is empty.")

        resampled = cls._resample(
            data, config.date_col, target_cols, config.regressors, config.granularity
        )

        if len(resampled) < 2:
            raise ValueError(
                f"Not enough history after resampling to '{config.granularity.value}' "
                f"granularity ({len(resampled)} row(s)); need at least 2."
            )

        for reg in config.regressors:
            n_missing = int(resampled[reg].isna().sum())
            if n_missing:
                raise ValueError(
                    f"Regressor '{reg}' has {n_missing} missing period(s) after "
                    f"resampling to '{config.granularity.value}' granularity. Fill "
                    "these gaps in the input data before fitting."
                )

        targets: Dict[str, SingleTargetForecast] = {}
        if config.conversions_col:
            targets["conversions"] = cls._fit_one_target(
                config.conversions_col,
                resampled,
                config,
                config.regressors,
                future_regressors,
            )
        if config.revenue_col:
            targets["revenue"] = cls._fit_one_target(
                config.revenue_col,
                resampled,
                config,
                config.regressors,
                future_regressors,
            )

        conclusion = " ".join(t.conclusion for t in targets.values())

        return ForecastingResult(
            granularity=config.granularity,
            targets=targets,
            conclusion=conclusion,
        )
