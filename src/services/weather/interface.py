from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.models.schemas.forecast import Presence


@dataclass
class WeatherData:
    df: pd.DataFrame
    presence: Presence


class WeatherProvider(ABC):
    @abstractmethod
    async def fetch_recent_weather(self, jwt: str, device_id: str, end_time: datetime | None = None) -> WeatherData:
        raise NotImplementedError
