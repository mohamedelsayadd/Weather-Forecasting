from __future__ import annotations

import logging

from src.core.config import Settings
from src.models.ai_model.interface import AIModel
from src.models.schemas.forecast import ForecastRequest, ForecastResponse
from src.services.weather.interface import WeatherProvider
from src.utils.preprocessing import build_chronos_context

logger = logging.getLogger(__name__)


class ForecastingService:
    def __init__(self, weather_provider: WeatherProvider, ai_model: AIModel, settings: Settings) -> None:
        self.weather_provider = weather_provider
        self.ai_model = ai_model
        self.settings = settings

    async def create_forecast(self, request_body: ForecastRequest) -> ForecastResponse:
        weather_df = await self.weather_provider.fetch_recent_weather(request_body.latitude, request_body.longitude)
        context_df = build_chronos_context(weather_df, request_body.weather_parameter)
        pred_df = self.ai_model.forecast(context_df)
        forecast = self.ai_model.forecast_to_records(pred_df)

        logger.info(
            "Forecast request completed weather_parameter=%s forecast_points=%s",
            request_body.weather_parameter,
            len(forecast),
        )

        return ForecastResponse(
            latitude=request_body.latitude,
            longitude=request_body.longitude,
            weather_parameter=request_body.weather_parameter,
            prediction_length=self.settings.prediction_length,
            forecast=forecast,
        )
