from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str
    logging_level: str
    open_meteo_url: str
    open_meteo_timeout_seconds: float
    open_meteo_history_days: int
    chronos_model_id: str
    chronos_device_map: str
    prediction_length: int
    context_hours: int

    model_config = SettingsConfigDict(env_file="src/.env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
