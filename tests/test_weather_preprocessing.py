import pandas as pd
import pytest

from src.services.weather_preprocessing import (
    build_chronos_context,
    preprocess_dynamic_hourly_weather,
    preprocess_hourly_weather,
)

SENSORS = ["SO2", "NO2", "ambient_temp"]


def _sensor_frame(timestamps: list[pd.Timestamp]) -> pd.DataFrame:
    frame = pd.DataFrame({"timestamp": timestamps})
    for offset, sensor in enumerate(SENSORS):
        frame[sensor] = [float(index + offset) for index in range(len(timestamps))]
    return frame


def test_preprocess_hourly_weather_sorts_interpolates_and_limits_context() -> None:
    raw_df = pd.DataFrame(
        {
            "time": ["2026-06-01 02:00", "2026-06-01 00:00", "2026-06-01 01:00"],
            "temperature_2m": [12.0, 10.0, None],
            "relative_humidity_2m": [120.0, 50.0, 75.0],
            "wind_speed_10m": [3.0, -1.0, 1.0],
            "wind_direction_10m": [370.0, 90.0, None],
            "cloud_cover": [110.0, 50.0, None],
            "shortwave_radiation": [100.0, -5.0, None],
            "precipitation": [1.0, -1.0, None],
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
    assert result.loc[2, "cloud_cover"] == 100.0
    assert result.loc[0, "shortwave_radiation"] == 0.0
    assert result.loc[0, "precipitation"] == 0.0


def test_preprocess_hourly_weather_rejects_missing_columns() -> None:
    raw_df = pd.DataFrame({"time": ["2026-06-01 00:00"], "temperature_2m": [10.0]})

    with pytest.raises(ValueError, match="Missing expected weather columns"):
        preprocess_hourly_weather(raw_df, context_hours=1)


def test_preprocess_dynamic_hourly_weather_keeps_end_of_hour_readings() -> None:
    # ReNile-IOT labels each hourly bucket by its final second, e.g. 11:59:59 for hour 11.
    # Before flooring, asfreq("h") reindexed onto a HH:00:00 grid and destroyed every reading.
    timestamps = [pd.Timestamp("2026-07-26 16:00:00") + pd.Timedelta(hours=hour, seconds=3599) for hour in range(404)]
    raw_df = _sensor_frame(timestamps)

    result = preprocess_dynamic_hourly_weather(raw_df, context_hours=336, min_context_coverage=0.5)

    assert len(result) == 336
    assert result["timestamp"].tolist() == pd.date_range("2026-07-29 12:00:00", periods=336, freq="h").tolist()
    assert result[SENSORS].isna().sum().sum() == 0
    # Values survived intact rather than being interpolated away.
    assert result["SO2"].tolist() == [float(index) for index in range(68, 404)]


def test_preprocess_dynamic_hourly_weather_leaves_on_the_hour_labels_unchanged() -> None:
    timestamps = list(pd.date_range("2026-07-26 16:00:00", periods=10, freq="h"))
    raw_df = _sensor_frame(timestamps)

    result = preprocess_dynamic_hourly_weather(raw_df, context_hours=10, min_context_coverage=0.5)

    assert result["timestamp"].tolist() == timestamps
    assert result["SO2"].tolist() == [float(index) for index in range(10)]


def test_preprocess_dynamic_hourly_weather_collapses_readings_within_one_hour() -> None:
    timestamps = [
        pd.Timestamp("2026-07-26 16:00:00"),
        pd.Timestamp("2026-07-26 16:30:00"),
        pd.Timestamp("2026-07-26 17:00:00"),
    ]
    raw_df = _sensor_frame(timestamps)

    result = preprocess_dynamic_hourly_weather(raw_df, context_hours=2, min_context_coverage=0.5)

    assert result["timestamp"].tolist() == [pd.Timestamp("2026-07-26 16:00:00"), pd.Timestamp("2026-07-26 17:00:00")]
    assert result["SO2"].tolist() == [1.0, 2.0]


def test_preprocess_dynamic_hourly_weather_rejects_mostly_interpolated_context() -> None:
    # Real readings only in the first 20 hours; the 336-hour context window is pure ffill.
    timestamps = list(pd.date_range("2026-07-26 16:00:00", periods=20, freq="h"))
    timestamps.append(pd.Timestamp("2026-08-12 11:00:00"))
    raw_df = _sensor_frame(timestamps)

    with pytest.raises(ValueError, match="mostly interpolated"):
        preprocess_dynamic_hourly_weather(raw_df, context_hours=336, min_context_coverage=0.5)


def test_preprocess_dynamic_hourly_weather_allows_gaps_above_coverage_threshold() -> None:
    timestamps = [
        timestamp
        for index, timestamp in enumerate(pd.date_range("2026-07-26 16:00:00", periods=336, freq="h"))
        if index % 4 or index == 0  # drop every 4th hour, keeping the endpoints -> ~75% coverage
    ]
    raw_df = _sensor_frame(timestamps)

    result = preprocess_dynamic_hourly_weather(raw_df, context_hours=336, min_context_coverage=0.5)

    assert len(result) == 336
    assert result[SENSORS].isna().sum().sum() == 0


def test_build_chronos_context_uses_selected_parameter_as_target() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-01", periods=2, freq="h"),
            "temperature_2m": [10.0, 11.0],
            "wind_speed_10m": [2.0, 3.0],
            "shortwave_radiation": [400.0, 500.0],
        }
    )

    result = build_chronos_context(df, ["temperature", "windSpeed", "solarRadiation"])


    assert result.columns.tolist() == ["item_id", "timestamp", "temperature", "windSpeed", "solarRadiation"]
    assert result["item_id"].tolist() == ["weather_series", "weather_series"]
    assert result["temperature"].tolist() == [10.0, 11.0]
    assert result["windSpeed"].tolist() == [2.0, 3.0]
    assert result["solarRadiation"].tolist() == [400.0, 500.0]
