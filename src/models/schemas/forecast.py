from typing import Literal

from pydantic import BaseModel, Field


WeatherParameter = Literal[
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
]


class ForecastRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    weather_parameter: WeatherParameter


class ForecastPoint(BaseModel):
    timestamp: str
    prediction: float
    q10: float
    q50: float
    q90: float


class ForecastResponse(BaseModel):
    latitude: float
    longitude: float
    weather_parameter: WeatherParameter
    prediction_length: int
    forecast: list[ForecastPoint]
