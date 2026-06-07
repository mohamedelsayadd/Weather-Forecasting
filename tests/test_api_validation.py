from fastapi.testclient import TestClient

import src.main as app_module
from src.main import app


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
            "latitude": 30.0,
            "longitude": 31.0,
            "weather_parameter": "rain",
        },
    )

    assert response.status_code == 422


def test_forecast_requires_location_without_past_weather_values() -> None:
    response = client.post(
        "/api/v1/forecast",
        json={"weather_parameter": "temperature_2m"},
    )

    assert response.status_code == 422


def test_forecast_rejects_short_past_weather_values() -> None:
    response = client.post(
        "/api/v1/forecast",
        json={
            "weather_parameter": "temperature_2m",
            "past_weather_values": {"temperature_2m": [20.0] * 167},
        },
    )

    assert response.status_code == 422


def test_forecast_requires_requested_parameter_in_past_weather_values() -> None:
    response = client.post(
        "/api/v1/forecast",
        json={
            "weather_parameter": "temperature_2m",
            "past_weather_values": {"wind_speed_10m": [5.0] * 168},
        },
    )

    assert response.status_code == 422
