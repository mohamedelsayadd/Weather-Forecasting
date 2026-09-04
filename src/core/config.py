from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str
    logging_level: str
    open_meteo_url: str
    open_meteo_timeout_seconds: float
    open_meteo_history_days: int
    open_meteo_timezone_fallback: str
    renile_iot_url: str
    renile_iot_timeout_seconds: float
    renile_iot_history_days: int
    renile_iot_data_type: str
    chronos_model_id: str
    chronos_device_map: str
    prediction_length: int
    context_hours: int
    min_context_coverage: float = 0.5

    model_config = SettingsConfigDict(env_file="src/.env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
