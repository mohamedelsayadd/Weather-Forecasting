import asyncio

import pandas as pd

from src.core.config import Settings
from src.models.schemas.forecast import ForecastRequest, ForecastResponse
from src.services.forecasting_service import ForecastingService


class FailingWeatherProvider:
    async def fetch_recent_weather(self, latitude: float, longitude: float) -> pd.DataFrame:
        raise AssertionError("Weather provider should not be called when past_weather_values is provided.")


class FakeAIModel:
    def __init__(self) -> None:
        self.context_df: pd.DataFrame | None = None

    def forecast(self, context_df: pd.DataFrame) -> pd.DataFrame:
        self.context_df = context_df
        return pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-06-02 00:00")],
                "predictions": [20.5],
                "0.1": [19.0],
                "0.5": [20.5],
                "0.9": [22.0],
            }
        )

    def forecast_to_records(self, pred_df: pd.DataFrame) -> list[dict[str, float | str]]:
        return [
            {
                "timestamp": pd.Timestamp(pred_df.loc[0, "timestamp"]).isoformat(),
                "prediction": float(pred_df.loc[0, "predictions"]),
                "q10": float(pred_df.loc[0, "0.1"]),
                "q50": float(pred_df.loc[0, "0.5"]),
                "q90": float(pred_df.loc[0, "0.9"]),
            }
        ]


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
        prediction_length=24,
        context_hours=168,
    )


def test_create_forecast_uses_past_weather_values_without_weather_provider() -> None:
    response, ai_model = asyncio.run(create_forecast_with_past_weather_values())

    assert response.latitude is None
    assert response.longitude is None
    assert response.weather_parameter == "temperature_2m"
    assert response.prediction_length == 24
    assert len(response.forecast) == 1
    assert ai_model.context_df is not None
    assert ai_model.context_df["target"].tolist() == [float(value) for value in range(2, 170)]


async def create_forecast_with_past_weather_values() -> tuple[ForecastResponse, FakeAIModel]:
    ai_model = FakeAIModel()
    service = ForecastingService(
        weather_provider=FailingWeatherProvider(),
        ai_model=ai_model,
        settings=make_settings(),
    )
    request = ForecastRequest(
        weather_parameter="temperature_2m",
        past_weather_values={"temperature_2m": [float(value) for value in range(170)]},
    )

    response = await service.create_forecast(request)
    return response, ai_model
