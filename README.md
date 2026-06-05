## Weather Forecasting API

FastAPI service that fetches the latest 7 days / 168 hourly weather records from the Open-Meteo archive API and forecasts the next 24 hours for one requested weather parameter using Chronos-2.

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

Supported `weather_parameter` values:

```text
temperature_2m
relative_humidity_2m
surface_pressure
wind_speed_10m
wind_direction_10m
```

The service always fetches all five weather columns from Open-Meteo, preprocesses the hourly data, then uses only the requested parameter as the Chronos `target`.
