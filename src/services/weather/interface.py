from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class WeatherProvider(ABC):
    @abstractmethod
    async def fetch_recent_weather(self, jwt: str, device_id: str, end_time: datetime | None = None) -> pd.DataFrame:
        raise NotImplementedError
