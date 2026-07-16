import pandas as pd

from foe.forecasting.mock_data import generate_mock_forecast_data
from foe.forecasting.operations import ForecastingEngine


def test_generate_mock_forecast_data_shape():
    dataset = generate_mock_forecast_data(n_days=400, periods=30, seed=42)

    assert list(dataset.data.columns) == [
        "date",
        "conversions",
        "revenue",
        "marketing_spend",
    ]
    assert len(dataset.data) == 400
    assert (dataset.data["conversions"] > 0).all()
    assert (dataset.data["revenue"] > 0).all()


def test_generate_mock_forecast_data_future_regressors_cover_horizon():
    dataset = generate_mock_forecast_data(n_days=400, periods=30, seed=42)

    last_history_date = dataset.data["date"].max()
    assert len(dataset.future_regressors) == 30
    assert dataset.future_regressors["ds"].min() == last_history_date + pd.Timedelta(days=1)
    assert dataset.future_regressors["marketing_spend"].notna().all()


def test_generate_mock_forecast_data_holidays_fall_within_history():
    dataset = generate_mock_forecast_data(n_days=400, periods=30, seed=42)

    holidays = dataset.suggested_config.holidays
    assert holidays  # at least one event should fall in ~13 months of history
    min_date = dataset.data["date"].min().date()
    max_date = dataset.data["date"].max().date()
    assert all(min_date <= h.ds <= max_date for h in holidays)
    assert any(h.holiday == "Founders Day" for h in holidays)


def test_generate_mock_forecast_data_reproducible_with_seed():
    a = generate_mock_forecast_data(n_days=200, periods=14, seed=7)
    b = generate_mock_forecast_data(n_days=200, periods=14, seed=7)
    pd.testing.assert_frame_equal(a.data, b.data)

    c = generate_mock_forecast_data(n_days=200, periods=14, seed=8)
    assert not a.data["conversions"].equals(c.data["conversions"])


def test_generate_mock_forecast_data_end_to_end_fit():
    """The whole point: this bundle should be usable straight out of the box."""
    dataset = generate_mock_forecast_data(n_days=400, periods=14, seed=1)

    result = ForecastingEngine.fit(
        dataset.data, dataset.suggested_config, dataset.future_regressors
    )

    assert set(result.targets.keys()) == {"conversions", "revenue"}
    for target in result.targets.values():
        assert len(target.forecast) == 400 + 14
        assert "holidays" in target.components
