from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class WeatherProvider(ABC):
    @abstractmethod
    async def fetch_recent_weather(self, latitude: float, longitude: float) -> pd.DataFrame:
        raise NotImplementedError
