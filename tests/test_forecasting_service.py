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
        self.targets: list[str] | None = None

    def forecast(self, context_df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
        self.context_df = context_df
        self.targets = targets
        return pd.DataFrame(
            {
                "target_name": targets,
                "timestamp": [pd.Timestamp("2026-06-02 00:00")] * len(targets),
                "predictions": [20.5] * len(targets),
                "0.1": [19.0] * len(targets),
                "0.5": [20.5] * len(targets),
                "0.9": [22.0] * len(targets),
            }
        )

    def forecast_to_records(self, pred_df: pd.DataFrame, targets: list[str]) -> dict[str, list[dict[str, float | str]]]:
        forecasts = {target: [] for target in targets}
        for _, row in pred_df.iterrows():
            forecasts[str(row["target_name"])].append(
                {
                    "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                    "prediction": float(row["predictions"]),
                    "q10": float(row["0.1"]),
                    "q50": float(row["0.5"]),
                    "q90": float(row["0.9"]),
                }
            )
        return forecasts


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
    assert response.weather_parameters == ["temperature", "wind_speed"]
    assert response.prediction_length == 24
    assert len(response.forecasts["temperature"]) == 1
    assert len(response.forecasts["wind_speed"]) == 1
    assert ai_model.context_df is not None
    assert ai_model.targets == ["temperature", "wind_speed"]
    assert ai_model.context_df["temperature"].tolist() == [float(value) for value in range(2, 170)]
    assert ai_model.context_df["wind_speed"].tolist() == [float(value) for value in range(102, 270)]


async def create_forecast_with_past_weather_values() -> tuple[ForecastResponse, FakeAIModel]:
    ai_model = FakeAIModel()
    service = ForecastingService(
        weather_provider=FailingWeatherProvider(),
        ai_model=ai_model,
        settings=make_settings(),
    )
    request = ForecastRequest(
        weather_parameter=["temperature", "wind_speed"],
        past_weather_values={
            "temperature": [float(value) for value in range(170)],
            "wind_speed": [float(value) for value in range(100, 270)],
        },
    )

    response = await service.create_forecast(request)
    return response, ai_model
