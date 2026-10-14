from datetime import datetime

from fastapi.testclient import TestClient

import src.main as app_module
from src.main import app
from src.models.schemas.forecast import ForecastRequest


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
