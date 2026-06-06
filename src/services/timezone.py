from __future__ import annotations

import logging
from functools import lru_cache

from timezonefinder import TimezoneFinder

from src.core.config import Settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_timezone_finder() -> TimezoneFinder:
    return TimezoneFinder()


def resolve_timezone(latitude: float, longitude: float, settings: Settings) -> str:
    timezone_name = get_timezone_finder().timezone_at(lat=latitude, lng=longitude)
    if timezone_name:
        logger.info(
            "Resolved timezone from coordinates latitude=%.4f longitude=%.4f timezone=%s",
            latitude,
            longitude,
            timezone_name,
        )
        return timezone_name

    logger.warning(
        "Could not resolve timezone from coordinates latitude=%.4f longitude=%.4f fallback_timezone=%s",
        latitude,
        longitude,
        settings.open_meteo_timezone_fallback,
    )
    return settings.open_meteo_timezone_fallback
