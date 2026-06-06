from __future__ import annotations

from src.core.config import Settings
from src.services.weather.interface import WeatherProvider
from src.services.weather.providers.open_meteo import OpenMeteoWeatherProvider


def create_weather_provider(settings: Settings, provider: str = "open_meteo") -> WeatherProvider:
    if provider == "open_meteo":
        return OpenMeteoWeatherProvider(settings)
    raise ValueError(f"Unsupported weather provider: {provider}")
