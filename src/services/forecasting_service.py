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
        logger.info(
            "Forecast service started source=renile_iot device_id=%s start_date=%s",
            request_body.device_id,
            request_body.start_date,
        )
        weather = await self.weather_provider.fetch_recent_weather(
            request_body.JWT,
            request_body.device_id,
            request_body.start_date,
        )
        weather_df = weather.df
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
        last12hour = _last_hours_weather_records(weather_df, weather_parameters, hours=12)
        logger.info("Forecast service built last12hour records device_id=%s rows=%s", request_body.device_id, len(last12hour))
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
        hourly168 = self.ai_model.forecast_to_hourly_rows(pred_df, weather_parameters, hours=168)
        daily7 = self.ai_model.forecast_to_daily_ranges(pred_df, weather_parameters, days=7)
        logger.info(
            "Forecast response sections built device_id=%s hourly168_rows=%s daily7_rows=%s hourly168_preview=%s daily7_preview=%s",
            request_body.device_id,
            len(hourly168),
            len(daily7),
            hourly168[:3],
            daily7[:3],
        )

        logger.info(
            "Forecast request completed source=renile_iot device_id=%s weather_parameters=%s hourly168_rows=%s daily7_rows=%s",
            request_body.device_id,
            weather_parameters,
            len(hourly168),
            len(daily7),
        )

        return ForecastResponse(
            last12hour=last12hour,
            current=current,
            hourly168=hourly168,
            daily7=daily7,
            presence=weather.presence,
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


def _last_hours_weather_records(weather_df: pd.DataFrame, weather_parameters: list[str], hours: int) -> list[dict[str, Any]]:
    if weather_df.empty:
        raise ValueError("Weather dataframe is empty.")

    sorted_df = weather_df.sort_values("timestamp")
    latest = pd.Timestamp(sorted_df["timestamp"].iloc[-1])
    window = sorted_df[pd.to_datetime(sorted_df["timestamp"]) >= latest - pd.Timedelta(hours=hours)]
    records: list[dict[str, Any]] = []
    for _, row in window.iterrows():
        record: dict[str, Any] = {"time": pd.Timestamp(row["timestamp"]).isoformat()}
        for parameter in weather_parameters:
            value = row[parameter]
            record[parameter] = None if pd.isna(value) else float(value)
        records.append(record)
    return records
