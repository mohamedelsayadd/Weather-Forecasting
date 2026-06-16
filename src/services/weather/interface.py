from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class WeatherProvider(ABC):
    @abstractmethod
    async def fetch_recent_weather(self, jwt: str, device_id: str) -> pd.DataFrame:
        raise NotImplementedError
