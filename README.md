## Weather Forecasting API

FastAPI service that forecasts the next 24 hours for one requested weather parameter using Chronos-2. It can either fetch the latest 7 days / 168 hourly weather records from Open-Meteo by location, or use direct historical values supplied in the request.

### Run

```bash
uv run uvicorn src.main:app --reload
```

### Endpoint

`POST /api/v1/forecast`

```json
{
  "latitude": 30.0551,
  "longitude": 31.3570,
  "weather_parameter": "temperature_2m"
}
```

Or provide historical values directly and omit latitude/longitude. Shortened example:

```json
{
  "weather_parameter": "temperature_2m",
  "past_weather_values": {
    "temperature_2m": [20.1, 20.4, 20.0]
  }
}
```

Each provided `past_weather_values` parameter must contain at least 168 hourly readings. Only the requested `weather_parameter` is required, and the service uses the latest 168 readings from that parameter as the Chronos context.

Supported `weather_parameter` values:

```text
temperature_2m
relative_humidity_2m
surface_pressure
wind_speed_10m
wind_direction_10m
```

When `past_weather_values` is provided, Open-Meteo is not called. Otherwise, the service fetches all five weather columns from Open-Meteo, preprocesses the hourly data, then uses only the requested parameter as the Chronos `target`.
