from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.core.config import Settings
from src.models.ai_model.interface import AIModel
from src.models.schemas.forecast import ForecastRequest, ForecastResponse
from src.services.weather.interface import WeatherProvider
from src.utils.preprocessing import build_dynamic_chronos_context

logger = logging.getLogger(__name__)


class ForecastingService:
    def __init__(self, weather_provider: WeatherProvider, ai_model: AIModel, settings: Settings) -> None:
        self.weather_provider = weather_provider
        self.ai_model = ai_model
        self.settings = settings

    async def create_forecast(self, request_body: ForecastRequest) -> ForecastResponse:
        logger.info("Forecast service started source=renile_iot device_id=%s", request_body.device_id)
        weather_df = await self.weather_provider.fetch_recent_weather(request_body.JWT, request_body.device_id)
        logger.info(
            "Forecast service received processed weather data device_id=%s rows=%s columns=%s preview=%s",
            request_body.device_id,
            len(weather_df),
            weather_df.columns.tolist(),
            weather_df.head(3).to_dict(orient="records"),
        )
        context_df, weather_parameters = build_dynamic_chronos_context(weather_df)
        current = _current_weather_record(weather_df, weather_parameters)
        logger.info("Forecast service built current weather record device_id=%s current=%s", request_body.device_id, current)
        logger.info(
            "Forecast service prepared model input device_id=%s targets=%s rows=%s columns=%s preview=%s",
            request_body.device_id,
            weather_parameters,
            len(context_df),
            context_df.columns.tolist(),
            context_df.head(3).to_dict(orient="records"),
        )

        logger.info("Forecast model inference started device_id=%s targets=%s", request_body.device_id, weather_parameters)
        pred_df = self.ai_model.forecast(context_df, weather_parameters)
        logger.info(
            "Forecast model inference completed device_id=%s prediction_rows=%s prediction_columns=%s prediction_preview=%s",
            request_body.device_id,
            len(pred_df),
            pred_df.columns.tolist(),
            pred_df.head(3).to_dict(orient="records"),
        )
        hourly24 = self.ai_model.forecast_to_hourly_rows(pred_df, weather_parameters, hours=24)
        daily7 = self.ai_model.forecast_to_daily_ranges(pred_df, weather_parameters, days=7)
        logger.info(
            "Forecast response sections built device_id=%s hourly24_rows=%s daily7_rows=%s hourly24_preview=%s daily7_preview=%s",
            request_body.device_id,
            len(hourly24),
            len(daily7),
            hourly24[:3],
            daily7[:3],
        )

        logger.info(
            "Forecast request completed source=renile_iot device_id=%s weather_parameters=%s hourly24_rows=%s daily7_rows=%s",
            request_body.device_id,
            weather_parameters,
            len(hourly24),
            len(daily7),
        )

        return ForecastResponse(
            current=current,
            hourly24=hourly24,
            daily7=daily7,
        )


def _current_weather_record(weather_df: pd.DataFrame, weather_parameters: list[str]) -> dict[str, Any]:
    if weather_df.empty:
        raise ValueError("Weather dataframe is empty.")

    latest = weather_df.sort_values("timestamp").iloc[-1]
    record: dict[str, Any] = {"time": pd.Timestamp(latest["timestamp"]).isoformat()}
    for parameter in weather_parameters:
        value = latest[parameter]
        record[parameter] = None if pd.isna(value) else float(value)
    return record
