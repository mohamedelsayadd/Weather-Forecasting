import pandas as pd
import pytest

from src.services.weather_preprocessing import build_chronos_context, preprocess_hourly_weather


def test_preprocess_hourly_weather_sorts_interpolates_and_limits_context() -> None:
    raw_df = pd.DataFrame(
        {
            "time": ["2026-06-01 02:00", "2026-06-01 00:00", "2026-06-01 01:00"],
            "temperature_2m": [12.0, 10.0, None],
            "relative_humidity_2m": [120.0, 50.0, 75.0],
            "surface_pressure": [1001.0, 999.0, None],
            "wind_speed_10m": [3.0, -1.0, 1.0],
            "wind_direction_10m": [370.0, 90.0, None],
        }
    )

    result = preprocess_hourly_weather(raw_df, context_hours=3)

    assert result["timestamp"].tolist() == [
        pd.Timestamp("2026-06-01 00:00"),
        pd.Timestamp("2026-06-01 01:00"),
        pd.Timestamp("2026-06-01 02:00"),
    ]
    assert result.isna().sum().sum() == 0
    assert result.loc[1, "temperature_2m"] == 11.0
    assert result.loc[2, "relative_humidity_2m"] == 100.0
    assert result.loc[0, "wind_speed_10m"] == 0.0
    assert result.loc[2, "wind_direction_10m"] == 10.0


def test_preprocess_hourly_weather_rejects_missing_columns() -> None:
    raw_df = pd.DataFrame({"time": ["2026-06-01 00:00"], "temperature_2m": [10.0]})

    with pytest.raises(ValueError, match="Missing expected weather columns"):
        preprocess_hourly_weather(raw_df, context_hours=1)


def test_build_chronos_context_uses_selected_parameter_as_target() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-01", periods=2, freq="h"),
            "temperature_2m": [10.0, 11.0],
            "wind_speed_10m": [2.0, 3.0],
        }
    )

    result = build_chronos_context(df, ["temperature", "wind_speed"])


    assert result.columns.tolist() == ["item_id", "timestamp", "temperature", "wind_speed"]
    assert result["item_id"].tolist() == ["weather_series", "weather_series"]
    assert result["temperature"].tolist() == [10.0, 11.0]
    assert result["wind_speed"].tolist() == [2.0, 3.0]
