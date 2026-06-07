from __future__ import annotations

import logging

from src.core.config import Settings
from src.models.ai_model.interface import AIModel
from src.models.schemas.forecast import ForecastRequest, ForecastResponse
from src.services.weather.interface import WeatherProvider
from src.utils.preprocessing import build_chronos_context, build_chronos_context_from_values

logger = logging.getLogger(__name__)


class ForecastingService:
    def __init__(self, weather_provider: WeatherProvider, ai_model: AIModel, settings: Settings) -> None:
        self.weather_provider = weather_provider
        self.ai_model = ai_model
        self.settings = settings

    async def create_forecast(self, request_body: ForecastRequest) -> ForecastResponse:
        weather_parameters = request_body.selected_weather_parameters
        if request_body.past_weather_values is not None:
            context_df = build_chronos_context_from_values(
                request_body.past_weather_values,
                weather_parameters,
                self.settings.context_hours,
            )
            source = "past_weather_values"
        else:
            if request_body.latitude is None or request_body.longitude is None:
                raise ValueError("latitude and longitude are required when past_weather_values is not provided.")
            weather_df = await self.weather_provider.fetch_recent_weather(request_body.latitude, request_body.longitude)
            context_df = build_chronos_context(weather_df, weather_parameters)
            source = "open_meteo"

        pred_df = self.ai_model.forecast(context_df, weather_parameters)
        forecasts = self.ai_model.forecast_to_records(pred_df, weather_parameters)

        logger.info(
            "Forecast request completed source=%s weather_parameters=%s forecast_points=%s",
            source,
            weather_parameters,
            sum(len(rows) for rows in forecasts.values()),
        )

        return ForecastResponse(
            latitude=request_body.latitude,
            longitude=request_body.longitude,
            weather_parameters=weather_parameters,
            prediction_length=self.settings.prediction_length,
            forecasts=forecasts,
        )
