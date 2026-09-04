from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class AIModel(ABC):
    @abstractmethod
    def forecast(self, context_df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def forecast_to_hourly_rows(self, pred_df: pd.DataFrame, targets: list[str], hours: int) -> list[dict[str, object]]:
        raise NotImplementedError

    @abstractmethod
    def forecast_to_daily_ranges(self, pred_df: pd.DataFrame, targets: list[str], days: int) -> list[dict[str, object]]:
        raise NotImplementedError
