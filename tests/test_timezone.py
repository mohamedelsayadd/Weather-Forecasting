from src.core.config import Settings
from src.services import timezone as timezone_service
from src.services.timezone import resolve_timezone


def make_settings() -> Settings:
    return Settings(
        app_name="test",
        logging_level="INFO",
        open_meteo_url="https://archive-api.open-meteo.com/v1/archive",
        open_meteo_timeout_seconds=30,
        open_meteo_history_days=7,
        open_meteo_timezone_fallback="Africa/Cairo",
        renile_iot_url="https://renile-iot.com/api/v1/data/",
        renile_iot_timeout_seconds=30,
        renile_iot_history_days=20,
        renile_iot_data_type="month_hours",
        chronos_model_id="amazon/chronos-2",
        chronos_device_map="cpu",
        prediction_length=24,
        context_hours=168,
    )


def test_resolve_timezone_from_cairo_coordinates() -> None:
    timezone_name = resolve_timezone(30.0551, 31.3570, make_settings())

    assert timezone_name == "Africa/Cairo"


def test_resolve_timezone_uses_fallback_when_lookup_fails(monkeypatch) -> None:
    class FakeTimezoneFinder:
        def timezone_at(self, lat: float, lng: float):
            return None

    monkeypatch.setattr(timezone_service, "get_timezone_finder", lambda: FakeTimezoneFinder())

    timezone_name = resolve_timezone(0.0, 0.0, make_settings())

    assert timezone_name == "Africa/Cairo"
