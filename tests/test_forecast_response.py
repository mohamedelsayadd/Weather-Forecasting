import pandas as pd

from src.services.chronos_forecasting import forecast_dataframe_to_daily_ranges, forecast_dataframe_to_hourly_rows


def test_forecast_dataframe_to_hourly_rows_pivots_chronos_output() -> None:
    pred_df = _prediction_dataframe(hours=3, targets=["SO2", "ambient_temp"])

    assert forecast_dataframe_to_hourly_rows(pred_df, ["SO2", "ambient_temp"], hours=2) == [
        {"time": "2026-06-02T00:00:00", "SO2": 0.0, "ambient_temp": 100.0},
        {"time": "2026-06-02T01:00:00", "SO2": 1.0, "ambient_temp": 101.0},
    ]


def test_forecast_dataframe_to_hourly_rows_clamps_negative_values_to_zero() -> None:
    pred_df = _prediction_dataframe(hours=2, targets=["SO2"])
    pred_df.loc[pred_df["timestamp"] == pd.Timestamp("2026-06-02T00:00:00"), "predictions"] = -5.0

    assert forecast_dataframe_to_hourly_rows(pred_df, ["SO2"], hours=2) == [
        {"time": "2026-06-02T00:00:00", "SO2": 0.0},
        {"time": "2026-06-02T01:00:00", "SO2": 1.0},
    ]


def test_forecast_dataframe_to_daily_ranges_calculates_min_max_per_target() -> None:
    pred_df = _prediction_dataframe(hours=168, targets=["SO2", "ambient_temp"])

    result = forecast_dataframe_to_daily_ranges(pred_df, ["SO2", "ambient_temp"], days=7)

    assert len(result) == 7
    assert result[0] == {
        "time": "2026-06-02",
        "SO2": {"min": 0.0, "max": 23.0},
        "ambient_temp": {"min": 100.0, "max": 123.0},
    }
    assert result[-1] == {
        "time": "2026-06-08",
        "SO2": {"min": 144.0, "max": 167.0},
        "ambient_temp": {"min": 244.0, "max": 267.0},
    }


def test_forecast_dataframe_to_daily_ranges_clamps_negative_values_to_zero() -> None:
    pred_df = _prediction_dataframe(hours=168, targets=["SO2"])
    pred_df.loc[pred_df["timestamp"] < pd.Timestamp("2026-06-02T03:00:00"), "predictions"] = [-8.0, -3.0, 2.0]

    result = forecast_dataframe_to_daily_ranges(pred_df, ["SO2"], days=7)

    assert result[0] == {
        "time": "2026-06-02",
        "SO2": {"min": 0.0, "max": 23.0},
    }


def _prediction_dataframe(hours: int, targets: list[str]) -> pd.DataFrame:
    timestamps = pd.date_range("2026-06-02", periods=hours, freq="h")
    rows = []
    for target in targets:
        offset = 100 if target == "ambient_temp" else 0
        for index, timestamp in enumerate(timestamps):
            rows.append(
                {
                    "item_id": "weather_series",
                    "timestamp": timestamp,
                    "target_name": target,
                    "predictions": float(index + offset),
                    "0.1": float(index + offset - 1),
                    "0.5": float(index + offset),
                    "0.9": float(index + offset + 1),
                }
            )
    return pd.DataFrame(rows)
