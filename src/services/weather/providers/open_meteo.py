from __future__ import annotations

from datetime import datetime, timedelta
import logging
import time
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from src.core.config import Settings
from src.services.weather.interface import WeatherProvider
from src.utils.preprocessing import WEATHER_COLUMNS, preprocess_hourly_weather
from src.utils.timezone import resolve_timezone

logger = logging.getLogger(__name__)


class OpenMeteoWeatherProvider(WeatherProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def fetch_recent_weather(self, latitude: float, longitude: float) -> pd.DataFrame:
        return await fetch_recent_weather(latitude, longitude, self.settings)


async def fetch_recent_weather(latitude: float, longitude: float, settings: Settings) -> pd.DataFrame:
    timezone_name = resolve_timezone(latitude, longitude, settings)
    timezone = ZoneInfo(timezone_name)
    end_date = datetime.now(timezone).date()
    start_date = end_date - timedelta(days=settings.open_meteo_history_days)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(WEATHER_COLUMNS),
        "timezone": timezone_name,
    }

    logger.info(
        "Fetching Open-Meteo archive weather latitude=%.4f longitude=%.4f start_date=%s end_date=%s history_days=%s timezone=%s columns=%s",
        latitude,
        longitude,
        start_date,
        end_date,
        settings.open_meteo_history_days,
        timezone_name,
        len(WEATHER_COLUMNS),
    )
    start_time = time.perf_counter()
    async with httpx.AsyncClient(timeout=settings.open_meteo_timeout_seconds) as client:
        response = await client.get(settings.open_meteo_url, params=params)
        response.raise_for_status()
    latency_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "Open-Meteo archive weather fetched status_code=%s latency_ms=%.2f",
        response.status_code,
        latency_ms,
    )

    payload = response.json()
    if "hourly" not in payload:
        reason = payload.get("reason", "No reason returned by Open-Meteo")
        logger.warning("Open-Meteo response missing hourly data reason=%s", reason)
        raise ValueError(f"Open-Meteo response does not contain hourly data: {reason}")

    raw_df = pd.DataFrame(payload["hourly"])
    if raw_df.empty:
        logger.warning("Open-Meteo returned empty hourly dataframe")
        raise ValueError("Open-Meteo returned an empty hourly dataframe.")

    raw_df["time"] = pd.to_datetime(raw_df["time"], errors="coerce")
    current_hour = pd.Timestamp(datetime.now(timezone)).tz_localize(None).floor("h")
    raw_df = raw_df[raw_df["time"] <= current_hour]

    logger.info("Open-Meteo raw hourly rows available rows=%s current_hour=%s", len(raw_df), current_hour)
    return preprocess_hourly_weather(raw_df, context_hours=settings.context_hours)
