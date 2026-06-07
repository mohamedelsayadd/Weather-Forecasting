from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class AIModel(ABC):
    @abstractmethod
    def forecast(self, context_df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def forecast_to_records(self, pred_df: pd.DataFrame, targets: list[str]) -> dict[str, list[dict[str, float | str]]]:
        raise NotImplementedError
