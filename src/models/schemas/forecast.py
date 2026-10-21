from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid") # to prevent any extra field

    JWT: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    start_date: datetime | None = None

    @field_validator("start_date")
    @classmethod
    def _normalize_start_date(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        if value > datetime.now(timezone.utc).replace(tzinfo=None):
            raise ValueError("start_date must not be in the future.")
        return value


class Presence(BaseModel):
    is_online: bool
    last_online_at: str | None


class ForecastResponse(BaseModel):
    last12hour: list[dict[str, Any]]
    current: dict[str, Any]
    hourly168: list[dict[str, Any]]
    daily7: list[dict[str, Any]]
    presence: Presence
