from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid") # to prevent any extra field

    JWT: str = Field(min_length=1)
    device_id: str = Field(min_length=1)


class ForecastResponse(BaseModel):
    current: dict[str, Any]
    hourly24: list[dict[str, Any]]
    daily7: list[dict[str, Any]]
