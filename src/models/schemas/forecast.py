from typing import Literal

from pydantic import BaseModel, Field, model_validator


WeatherParameter = Literal[
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
]


class ForecastRequest(BaseModel):
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    weather_parameter: WeatherParameter
    past_weather_values: dict[WeatherParameter, list[float]] | None = None

    @model_validator(mode="after")
    def validate_location_or_past_weather_values(self) -> "ForecastRequest":
        if self.past_weather_values is None:
            if self.latitude is None or self.longitude is None:
                raise ValueError("latitude and longitude are required when past_weather_values is not provided.")
            return self

        if self.weather_parameter not in self.past_weather_values:
            raise ValueError("past_weather_values must include the requested weather_parameter.")

        short_parameters = [
            parameter
            for parameter, values in self.past_weather_values.items()
            if len(values) < 168
        ]
        if short_parameters:
            raise ValueError(
                f"Each provided past_weather_values parameter must contain at least 168 readings: {short_parameters}"
            )

        return self


class ForecastPoint(BaseModel):
    timestamp: str
    prediction: float
    q10: float
    q50: float
    q90: float


class ForecastResponse(BaseModel):
    latitude: float | None
    longitude: float | None
    weather_parameter: WeatherParameter
    prediction_length: int
    forecast: list[ForecastPoint]
