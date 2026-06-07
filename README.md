## Weather Forecasting API

FastAPI service that forecasts the next 24 hours for one or more requested weather parameters using Chronos-2. It can either fetch the latest 7 days / 168 hourly weather records from Open-Meteo by location, or use direct historical values supplied in the request.

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
  "weather_parameter": "temperature"
}
```

Request 2 to 4 parameters:

```json
{
  "latitude": 30.0551,
  "longitude": 31.3570,
  "weather_parameter": ["temperature", "wind_speed"]
}
```

Request all supported parameters:

```json
{
  "latitude": 30.0551,
  "longitude": 31.3570,
  "weather_parameter": "all"
}
```

Or provide historical values directly and omit latitude/longitude. Shortened example:

```json
{
  "weather_parameter": ["temperature", "wind_speed"],
  "past_weather_values": {
    "temperature": [20.1, 20.4, 20.0],
    "wind_speed": [3.1, 3.4, 3.0]
  }
}
```

Each selected `past_weather_values` parameter must contain at least 168 hourly readings. Only the selected parameters are required, and the service uses the latest 168 readings from each selected parameter as the Chronos context.

Supported `weather_parameter` values:

```text
temperature
relative_humidity
surface_pressure
wind_speed
wind_direction
```

When `past_weather_values` is provided, Open-Meteo is not called. Otherwise, the service fetches all five weather columns from Open-Meteo, preprocesses the hourly data, then passes the selected parameters to Chronos as the `target` list.
