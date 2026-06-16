from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str
    logging_level: str
    open_meteo_url: str = "https://archive-api.open-meteo.com/v1/archive"
    open_meteo_timeout_seconds: float = 30.0
    open_meteo_history_days: int = 7
    open_meteo_timezone_fallback: str = "UTC"
    renile_iot_url: str = "https://renile-iot.com/api/v1/data/"
    renile_iot_timeout_seconds: float = 30.0
    renile_iot_history_days: int = 20
    renile_iot_data_type: str = "month_hours"
    chronos_model_id: str
    chronos_device_map: str
    prediction_length: int
    context_hours: int

    model_config = SettingsConfigDict(env_file="src/.env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
