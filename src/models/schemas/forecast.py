from typing import Literal

from pydantic import BaseModel, Field, model_validator


WeatherParameter = Literal[
    "temperature",
    "relative_humidity",
    "surface_pressure",
    "wind_speed",
    "wind_direction",
]
WeatherParameterSelection = WeatherParameter | Literal["all"] | list[WeatherParameter]
ALL_WEATHER_PARAMETERS: tuple[WeatherParameter, ...] = (
    "temperature",
    "relative_humidity",
    "surface_pressure",
    "wind_speed",
    "wind_direction",
)


class ForecastRequest(BaseModel):
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    weather_parameter: WeatherParameterSelection
    past_weather_values: dict[WeatherParameter, list[float]] | None = None

    @model_validator(mode="after")
    def validate_location_or_past_weather_values(self) -> "ForecastRequest":
        selected_parameters = self.selected_weather_parameters

        if self.past_weather_values is None:
            if self.latitude is None or self.longitude is None:
                raise ValueError("latitude and longitude are required when past_weather_values is not provided.")
            return self

        missing_parameters = [parameter for parameter in selected_parameters if parameter not in self.past_weather_values]
        if missing_parameters:
            raise ValueError(f"past_weather_values must include the requested weather parameters: {missing_parameters}")

        short_parameters = [
            parameter
            for parameter, values in self.past_weather_values.items()
            if parameter in selected_parameters
            if len(values) < 168
        ]
        if short_parameters:
            raise ValueError(
                f"Each provided past_weather_values parameter must contain at least 168 readings: {short_parameters}"
            )

        return self

    @property
    def selected_weather_parameters(self) -> list[WeatherParameter]:
        if self.weather_parameter == "all":
            return list(ALL_WEATHER_PARAMETERS)
        if isinstance(self.weather_parameter, list):
            if not 2 <= len(self.weather_parameter) <= 4:
                raise ValueError("weather_parameter list must contain 2 to 4 parameters. Use a string for one parameter or 'all' for all parameters.")
            if len(self.weather_parameter) != len(set(self.weather_parameter)):
                raise ValueError("weather_parameter list must not contain duplicates.")
            return self.weather_parameter
        return [self.weather_parameter]


class ForecastPoint(BaseModel):
    timestamp: str
    prediction: float
    q10: float
    q50: float
    q90: float


class ForecastResponse(BaseModel):
    latitude: float | None
    longitude: float | None
    weather_parameters: list[WeatherParameter]
    prediction_length: int
    forecasts: dict[WeatherParameter, list[ForecastPoint]]
