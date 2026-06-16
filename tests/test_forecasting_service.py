import asyncio

import pandas as pd

from src.core.config import Settings
from src.models.schemas.forecast import ForecastRequest, ForecastResponse
from src.services.forecasting_service import ForecastingService


class FakeWeatherProvider:
    def __init__(self) -> None:
        self.jwt: str | None = None
        self.device_id: str | None = None

    async def fetch_recent_weather(self, jwt: str, device_id: str) -> pd.DataFrame:
        self.jwt = jwt
        self.device_id = device_id
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-06-01", periods=336, freq="h"),
                "SO2": [float(value) for value in range(336)],
                "ambient_temp": [float(value) for value in range(100, 436)],
            }
        )


class FakeAIModel:
    def __init__(self) -> None:
        self.context_df: pd.DataFrame | None = None
        self.targets: list[str] | None = None

    def forecast(self, context_df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
        self.context_df = context_df
        self.targets = targets
        timestamps = pd.date_range("2026-06-15", periods=168, freq="h")
        rows = []
        for target in targets:
            offset = 100 if target == "ambient_temp" else 0
            for index, timestamp in enumerate(timestamps):
                rows.append({"target_name": target, "timestamp": timestamp, "predictions": float(index + offset)})
        return pd.DataFrame(rows)

    def forecast_to_hourly_rows(self, pred_df: pd.DataFrame, targets: list[str], hours: int) -> list[dict[str, object]]:
        from src.services.chronos_forecasting import forecast_dataframe_to_hourly_rows

        return forecast_dataframe_to_hourly_rows(pred_df, targets, hours)

    def forecast_to_daily_ranges(self, pred_df: pd.DataFrame, targets: list[str], days: int) -> list[dict[str, object]]:
        from src.services.chronos_forecasting import forecast_dataframe_to_daily_ranges

        return forecast_dataframe_to_daily_ranges(pred_df, targets, days)


def make_settings() -> Settings:
    return Settings(
        app_name="test",
        logging_level="INFO",
        open_meteo_url="https://example.com",
        open_meteo_timeout_seconds=1.0,
        open_meteo_history_days=7,
        open_meteo_timezone_fallback="UTC",
        chronos_model_id="test-model",
        chronos_device_map="cpu",
        prediction_length=168,
        context_hours=336,
    )


def test_create_forecast_uses_all_renile_iot_columns() -> None:
    response, ai_model, weather_provider = asyncio.run(create_forecast_with_renile_iot_values())

    assert weather_provider.jwt == "test-token"
    assert weather_provider.device_id == "test-device"
    assert response.current == {
        "time": "2026-06-14T23:00:00",
        "SO2": 335.0,
        "ambient_temp": 435.0,
    }
    assert len(response.hourly24) == 24
    assert response.hourly24[0] == {"time": "2026-06-15T00:00:00", "SO2": 0.0, "ambient_temp": 100.0}
    assert len(response.daily7) == 7
    assert response.daily7[0] == {
        "time": "2026-06-15",
        "SO2": {"min": 0.0, "max": 23.0},
        "ambient_temp": {"min": 100.0, "max": 123.0},
    }
    assert ai_model.context_df is not None
    assert ai_model.targets == ["SO2", "ambient_temp"]
    assert ai_model.context_df["SO2"].tolist() == [float(value) for value in range(336)]
    assert ai_model.context_df["ambient_temp"].tolist() == [float(value) for value in range(100, 436)]


async def create_forecast_with_renile_iot_values() -> tuple[ForecastResponse, FakeAIModel, FakeWeatherProvider]:
    ai_model = FakeAIModel()
    weather_provider = FakeWeatherProvider()
    service = ForecastingService(
        weather_provider=weather_provider,
        ai_model=ai_model,
        settings=make_settings(),
    )
    request = ForecastRequest(
        JWT="test-token",
        device_id="test-device",
    )

    response = await service.create_forecast(request)
    return response, ai_model, weather_provider
