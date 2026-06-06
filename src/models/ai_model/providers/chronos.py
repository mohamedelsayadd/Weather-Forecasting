from __future__ import annotations

import logging
import time

import pandas as pd
from chronos import BaseChronosPipeline, Chronos2Pipeline

from src.core.config import Settings
from src.models.ai_model.interface import AIModel

logger = logging.getLogger(__name__)


def load_chronos_pipeline(settings: Settings) -> Chronos2Pipeline:
    logger.info(
        "Loading Chronos model model_id=%s device_map=%s",
        settings.chronos_model_id,
        settings.chronos_device_map,
    )
    start_time = time.perf_counter()
    pipeline = BaseChronosPipeline.from_pretrained(
        settings.chronos_model_id,
        device_map=settings.chronos_device_map,
    )
    latency_ms = (time.perf_counter() - start_time) * 1000
    logger.info("Chronos model loaded latency_ms=%.2f", latency_ms)
    return pipeline


class ChronosAIModel(AIModel):
    def __init__(self, settings: Settings, pipeline: Chronos2Pipeline | None = None) -> None:
        self.settings = settings
        self.pipeline = pipeline if pipeline is not None else load_chronos_pipeline(settings)

    def forecast(self, context_df: pd.DataFrame) -> pd.DataFrame:
        return forecast_next_24_hours(context_df, self.pipeline, self.settings)

    def forecast_to_records(self, pred_df: pd.DataFrame) -> list[dict[str, float | str]]:
        return forecast_dataframe_to_records(pred_df)


def forecast_next_24_hours(
    context_df: pd.DataFrame,
    pipeline: Chronos2Pipeline,
    settings: Settings,
) -> pd.DataFrame:
    logger.info(
        "Chronos forecast started context_rows=%s prediction_length=%s",
        len(context_df),
        settings.prediction_length,
    )
    start_time = time.perf_counter()
    pred_df = pipeline.predict_df(
        context_df,
        prediction_length=settings.prediction_length,
        quantile_levels=[0.1, 0.5, 0.9],
        timestamp_column="timestamp",
        target="target",
    )
    latency_ms = (time.perf_counter() - start_time) * 1000
    logger.info("Chronos forecast completed prediction_rows=%s latency_ms=%.2f", len(pred_df), latency_ms)
    return pred_df


def forecast_dataframe_to_records(pred_df: pd.DataFrame) -> list[dict[str, float | str]]:
    logger.info("Converting Chronos forecast dataframe rows=%s columns=%s", len(pred_df), list(pred_df.columns))
    q10_column = _find_column(pred_df, "0.1", 0.1)
    q50_column = _find_column(pred_df, "0.5", 0.5)
    q90_column = _find_column(pred_df, "0.9", 0.9)
    required_columns = {"timestamp", "predictions"}
    missing_columns = required_columns - set(pred_df.columns)
    if missing_columns:
        logger.warning("Chronos forecast dataframe missing columns=%s", sorted(missing_columns))
        raise ValueError(f"Chronos prediction dataframe is missing columns: {sorted(missing_columns)}")

    records = []
    for _, row in pred_df.sort_values("timestamp").iterrows():
        records.append(
            {
                "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                "prediction": float(row["predictions"]),
                "q10": float(row[q10_column]),
                "q50": float(row[q50_column]),
                "q90": float(row[q90_column]),
            }
        )
    logger.info("Chronos forecast dataframe converted records=%s", len(records))
    return records


def _find_column(df: pd.DataFrame, *candidates: str | float) -> str | float:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    logger.warning("Chronos forecast dataframe missing quantile columns candidates=%s", list(candidates))
    raise ValueError(f"Chronos prediction dataframe is missing columns: {list(candidates)}")
