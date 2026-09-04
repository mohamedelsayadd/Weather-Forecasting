from __future__ import annotations

from src.core.config import Settings
from src.services.weather.interface import WeatherProvider
from src.services.weather.providers.renile_iot import ReNileIOTWeatherProvider


def create_weather_provider(settings: Settings, provider: str = "renile_iot") -> WeatherProvider:
    if provider == "renile_iot":
        return ReNileIOTWeatherProvider(settings)
    raise ValueError(f"Unsupported weather provider: {provider}")
