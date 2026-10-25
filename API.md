# Forecast API

## `POST /api/v1/forecast`

Fetches the device's recent sensor readings from ReNile-IOT and returns a 7-day forecast for every sensor.

### Request

`Content-Type: application/json`

| Field        | Type     | Required | Description                                                                                       |
|--------------|----------|----------|---------------------------------------------------------------------------------------------------|
| `JWT`        | string   | yes      | ReNile-IOT token, sent upstream as `Authorization: JWT <token>`.                                  |
| `device_id`  | string   | yes      | ReNile-IOT device ID.                                                                             |
| `start_date` | datetime | no       | ISO datetime (`2026-09-01T12:00`, treated as UTC if it has no offset). Forecasts from this point instead of now. Must not be in the future. |

Any other fields are rejected.

```json
{
  "JWT": "eyJhbGciOi...",
  "device_id": "681740b2b2b422389cd7831e",
  "start_date": "2026-09-01T12:00"
}
```

### Response `200 OK`

Sensor keys (e.g. `SO2`, `ambient_temp`) are dynamic and match the ones the device reports.

Each sensor is validated on its own. A sensor is rejected when it has no readings, less than 14 days of history, or real readings covering less than 50% of the 336-hour window. A rejected sensor keeps its key, with `null` as the value in `last24hour`, `current`, `hourly168` and `daily7`. The other sensors are forecast normally. The request fails only when every sensor is rejected.

| Field        | Description                                                         |
|--------------|---------------------------------------------------------------------|
| `last24hour` | Observed hourly readings for the last 24 hours, including the latest one. |
| `current`    | Latest observed reading.                                            |
| `hourly168`  | Hourly forecast for the next 168 hours.                             |
| `daily7`     | Daily `min`/`max` forecast for the next 7 days.                     |
| `presence`   | Device presence from ReNile-IOT, unchanged. `last_online_at` may be `null`. |

```json
{
  "last24hour": [
    {"time": "2026-06-15T22:00:00", "SO2": 22.4, "ambient_temp": 30.1}
  ],
  "current": {"time": "2026-06-16T10:00:00", "SO2": 24.0, "ambient_temp": 33.2},
  "hourly168": [
    {"time": "2026-06-16T11:00:00", "SO2": 25.1, "ambient_temp": 34.0}
  ],
  "daily7": [
    {
      "time": "2026-06-17",
      "SO2": {"min": 20.0, "max": 41.0},
      "ambient_temp": {"min": 27.5, "max": 40.2}
    }
  ],
  "presence": {"is_online": true, "last_online_at": "2026-10-18T07:06:04.865Z"}
}
```

### Errors

Every error response has this shape:

```json
{"detail": "<message>"}
```

For request validation errors, `detail` is a list of FastAPI validation errors instead of a string.

The insufficient-history `409` also includes structured fields, so clients don't need to parse the message:

```json
{
  "detail": "Insufficient history: at least 14 days of data are required, but the device only has 10 days of data.",
  "available_days": 10,
  "required_days": 14
}
```

- `available_days` (int): full days of data the device has, rounded down (`0` means less than one day).
- `required_days` (int): days of data needed for a forecast.

| Status | When                                                                                                  | Example `detail`                                  |
|--------|-------------------------------------------------------------------------------------------------------|---------------------------------------------------|
| `422`  | The request body is invalid: a field is missing or empty, there is an extra field, or `start_date` is malformed or in the future. | `[{"loc": ["body", "device_id"], "msg": "Field required", ...}]` |
| `422`  | The data can't be used for a forecast: an empty or invalid ReNile-IOT payload, missing `presence`, or no data before `start_date`. | `"ReNile-IOT response is empty."`    |
| `409`  | Not enough history: every sensor has less than 14 days (336 hours) of data.                            | `"Insufficient history: at least 14 days of data are required, but the device only has 10 days of data."` |
| `409`  | Every sensor is rejected and at least one is empty or covers less than 50% of the 336-hour window.     | `"Insufficient readings: real sensor readings cover less than 50% of the required 336-hour window."` |
| `502`  | ReNile-IOT returned an error status, for example because of an invalid JWT.                            | `"ReNile-IOT returned an error: 401"`             |
| `502`  | ReNile-IOT couldn't be reached (network error or timeout).                                            | `"Failed to fetch weather data from ReNile-IOT."` |
| `503`  | The service hasn't finished loading the model.                                                        | `"Forecasting dependencies are not loaded."`      |

### Example

```bash
curl -X POST http://localhost:8000/api/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{"JWT": "eyJhbGciOi...", "device_id": "681740b2b2b422389cd7831e"}'
```

Every response includes an `x-request-id` header. If you send your own `x-request-id`, it is echoed back.
