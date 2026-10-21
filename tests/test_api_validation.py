from datetime import datetime

from fastapi.testclient import TestClient
import httpx
import pandas as pd
import pytest

import src.main as app_module
from src.main import app
from src.models.schemas.forecast import ForecastRequest
from src.services.weather_preprocessing import preprocess_dynamic_hourly_weather


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_lifespan_loads_chronos_pipeline(monkeypatch) -> None:
    fake_pipeline = object()

    def fake_loader(settings):
        return fake_pipeline

    monkeypatch.setattr(app_module, "load_chronos_pipeline", fake_loader)
    test_app = app_module.create_app()

    with TestClient(test_app):
        assert test_app.state.chronos_pipeline is fake_pipeline


def test_forecast_rejects_invalid_weather_parameter() -> None:
    response = client.post(
        "/api/v1/forecast",
        json={
            "JWT": "test-token",
            "device_id": "test-device",
            "latitude": 30.0,
            "longitude": 31.0,
            "weather_parameter": "rain",
        },
    )

    assert response.status_code == 422


def test_forecast_rejects_extra_body_fields() -> None:
    response = client.post(
        "/api/v1/forecast",
        json={"JWT": "test-token", "device_id": "test-device", "weather_parameter": "temperature"},
    )

    assert response.status_code == 422


def test_forecast_requires_jwt() -> None:
    response = client.post(
        "/api/v1/forecast",
        json={
            "device_id": "test-device",
        },
    )

    assert response.status_code == 422


def test_forecast_requires_device_id() -> None:
    response = client.post(
        "/api/v1/forecast",
        json={
            "JWT": "test-token",
        },
    )

    assert response.status_code == 422


def test_forecast_rejects_empty_jwt() -> None:
    response = client.post(
        "/api/v1/forecast",
        json={"JWT": "", "device_id": "test-device"},
    )

    assert response.status_code == 422


def test_forecast_rejects_empty_device_id() -> None:
    response = client.post(
        "/api/v1/forecast",
        json={"JWT": "test-token", "device_id": ""},
    )

    assert response.status_code == 422


def test_forecast_rejects_future_start_date() -> None:
    response = client.post(
        "/api/v1/forecast",
        json={"JWT": "test-token", "device_id": "test-device", "start_date": "2999-01-01T00:00"},
    )

    assert response.status_code == 422


def test_forecast_request_normalizes_start_date_to_naive_utc() -> None:
    request = ForecastRequest(JWT="test-token", device_id="test-device", start_date="2026-09-01T14:00+02:00")

    assert request.start_date == datetime(2026, 9, 1, 12, 0)


class FakeWeatherProvider:
    def __init__(self, timestamps: list[pd.Timestamp] | None = None, error: Exception | None = None) -> None:
        self.timestamps = timestamps
        self.error = error

    async def fetch_recent_weather(self, jwt, device_id, end_time=None):
        if self.error is not None:
            raise self.error
        raw_df = pd.DataFrame({"timestamp": self.timestamps, "SO2": [float(i) for i in range(len(self.timestamps))]})
        preprocess_dynamic_hourly_weather(raw_df, context_hours=336, min_context_coverage=0.5)
        raise AssertionError("Expected preprocessing to reject the data.")


def _post_forecast_with_provider(monkeypatch, provider: FakeWeatherProvider):
    monkeypatch.setattr(app.state, "weather_provider", provider, raising=False)
    monkeypatch.setattr(app.state, "ai_model", object(), raising=False)
    return client.post("/api/v1/forecast", json={"JWT": "test-token", "device_id": "test-device"})


def test_forecast_returns_409_for_insufficient_history(monkeypatch) -> None:
    timestamps = list(pd.date_range("2026-07-01", periods=13 * 24, freq="h"))

    response = _post_forecast_with_provider(monkeypatch, FakeWeatherProvider(timestamps))

    assert response.status_code == 409
    assert response.json() == {"detail": "Insufficient history: at least 14 days of data are required."}


def test_forecast_returns_409_for_insufficient_readings(monkeypatch) -> None:
    # 14-day span, but only the first 100 hours have real readings (<50% of 336).
    timestamps = list(pd.date_range("2026-07-01", periods=100, freq="h"))
    timestamps.append(pd.Timestamp("2026-07-01") + pd.Timedelta(hours=335))

    response = _post_forecast_with_provider(monkeypatch, FakeWeatherProvider(timestamps))

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Insufficient readings: real sensor readings cover less than 50% of the required 336-hour window."
    }


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (ValueError("ReNile-IOT response is empty."), 422),
        (
            httpx.HTTPStatusError(
                "unauthorized",
                request=httpx.Request("GET", "https://renile-iot.com"),
                response=httpx.Response(401),
            ),
            502,
        ),
    ],
)
def test_forecast_keeps_status_codes_for_unrelated_errors(monkeypatch, error: Exception, expected_status: int) -> None:
    response = _post_forecast_with_provider(monkeypatch, FakeWeatherProvider(error=error))

    assert response.status_code == expected_status
