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

    def forecast(self, context_df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
        return forecast_next_horizon(context_df, targets, self.pipeline, self.settings)

    def forecast_to_hourly_rows(self, pred_df: pd.DataFrame, targets: list[str], hours: int) -> list[dict[str, object]]:
        return forecast_dataframe_to_hourly_rows(pred_df, targets, hours)

    def forecast_to_daily_ranges(self, pred_df: pd.DataFrame, targets: list[str], days: int) -> list[dict[str, object]]:
        return forecast_dataframe_to_daily_ranges(pred_df, targets, days)


def forecast_next_horizon(
    context_df: pd.DataFrame,
    targets: list[str],
    pipeline: Chronos2Pipeline,
    settings: Settings,
) -> pd.DataFrame:
    logger.info(
        "Chronos forecast started context_rows=%s targets=%s prediction_length=%s",
        len(context_df),
        targets,
        settings.prediction_length,
    )
    start_time = time.perf_counter()
    pred_df = pipeline.predict_df(
        context_df,
        prediction_length=settings.prediction_length,
        quantile_levels=[0.1, 0.5, 0.9],
        timestamp_column="timestamp",
        target=targets,
    )
    latency_ms = (time.perf_counter() - start_time) * 1000
    logger.info("Chronos forecast completed prediction_rows=%s latency_ms=%.2f", len(pred_df), latency_ms)
    return pred_df


def forecast_next_24_hours(
    context_df: pd.DataFrame,
    targets: list[str],
    pipeline: Chronos2Pipeline,
    settings: Settings,
) -> pd.DataFrame:
    return forecast_next_horizon(context_df, targets, pipeline, settings)


def forecast_dataframe_to_hourly_rows(pred_df: pd.DataFrame, targets: list[str], hours: int = 24) -> list[dict[str, object]]:
    normalized = _normalize_forecast_dataframe(pred_df, targets)
    wide_df = _forecast_dataframe_to_wide(normalized, targets)
    hourly_df = wide_df.head(hours)
    rows = [_row_to_time_record(row, targets) for _, row in hourly_df.iterrows()]
    logger.info("Chronos forecast dataframe converted to hourly rows rows=%s preview=%s", len(rows), rows[:3])
    return rows


def forecast_dataframe_to_daily_ranges(pred_df: pd.DataFrame, targets: list[str], days: int = 7) -> list[dict[str, object]]:
    normalized = _normalize_forecast_dataframe(pred_df, targets)
    wide_df = _forecast_dataframe_to_wide(normalized, targets)
    wide_df["date"] = pd.to_datetime(wide_df["timestamp"]).dt.date.astype(str)

    daily_rows: list[dict[str, object]] = []
    for date, group in wide_df.groupby("date", sort=True):
        row: dict[str, object] = {"time": date}
        for target in targets:
            values = pd.to_numeric(group[target], errors="coerce").dropna().clip(lower=0)
            if values.empty:
                row[target] = {"min": None, "max": None}
            else:
                row[target] = {"min": float(values.min()), "max": float(values.max())}
        daily_rows.append(row)
        if len(daily_rows) == days:
            break

    if len(daily_rows) < days:
        logger.warning("Chronos daily forecast has fewer days than expected days=%s rows=%s", days, len(daily_rows))
        raise ValueError(f"Chronos prediction dataframe is missing daily forecast rows: expected {days}, got {len(daily_rows)}")

    logger.info("Chronos forecast dataframe converted to daily ranges rows=%s preview=%s", len(daily_rows), daily_rows[:3])
    return daily_rows


def forecast_dataframe_to_records(pred_df: pd.DataFrame, targets: list[str]) -> dict[str, list[dict[str, float | str]]]:
    logger.info("Converting Chronos forecast dataframe rows=%s columns=%s", len(pred_df), list(pred_df.columns))
    q10_column = _find_column(pred_df, "0.1", 0.1)
    q50_column = _find_column(pred_df, "0.5", 0.5)
    q90_column = _find_column(pred_df, "0.9", 0.9)
    required_columns = {"timestamp", "predictions"}
    missing_columns = required_columns - set(pred_df.columns)
    if missing_columns:
        logger.warning("Chronos forecast dataframe missing columns=%s", sorted(missing_columns))
        raise ValueError(f"Chronos prediction dataframe is missing columns: {sorted(missing_columns)}")

    if "target_name" not in pred_df.columns:
        if len(targets) != 1:
            logger.warning("Chronos forecast dataframe missing target_name for multi-target output")
            raise ValueError("Chronos prediction dataframe is missing target_name for multi-target output.")
        pred_df = pred_df.copy()
        pred_df["target_name"] = targets[0]

    forecasts = {target: [] for target in targets}
    for _, row in pred_df.sort_values(["target_name", "timestamp"]).iterrows():
        target_name = str(row["target_name"])
        if target_name not in forecasts:
            logger.warning("Chronos forecast dataframe returned unexpected target_name=%s", target_name)
            continue
        forecasts[target_name].append(
            {
                "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                "prediction": _forecast_response_value(row["predictions"]),
                "q10": _forecast_response_value(row[q10_column]),
                "q50": _forecast_response_value(row[q50_column]),
                "q90": _forecast_response_value(row[q90_column]),
            }
        )

    missing_targets = [target for target, records in forecasts.items() if not records]
    if missing_targets:
        logger.warning("Chronos forecast dataframe missing forecast rows for targets=%s", missing_targets)
        raise ValueError(f"Chronos prediction dataframe is missing forecast rows for targets: {missing_targets}")

    logger.info("Chronos forecast dataframe converted targets=%s records=%s", targets, sum(len(rows) for rows in forecasts.values()))
    return forecasts


def _normalize_forecast_dataframe(pred_df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    logger.info("Normalizing Chronos forecast dataframe rows=%s columns=%s targets=%s", len(pred_df), list(pred_df.columns), targets)
    required_columns = {"timestamp", "predictions"}
    missing_columns = required_columns - set(pred_df.columns)
    if missing_columns:
        logger.warning("Chronos forecast dataframe missing columns=%s", sorted(missing_columns))
        raise ValueError(f"Chronos prediction dataframe is missing columns: {sorted(missing_columns)}")

    normalized = pred_df.copy()
    if "target_name" not in normalized.columns:
        if len(targets) != 1:
            logger.warning("Chronos forecast dataframe missing target_name for multi-target output")
            raise ValueError("Chronos prediction dataframe is missing target_name for multi-target output.")
        normalized["target_name"] = targets[0]

    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="coerce")
    normalized["predictions"] = pd.to_numeric(normalized["predictions"], errors="coerce")
    normalized = normalized.dropna(subset=["timestamp", "predictions"])

    missing_targets = sorted(set(targets) - set(normalized["target_name"].astype(str)))
    if missing_targets:
        logger.warning("Chronos forecast dataframe missing forecast rows for targets=%s", missing_targets)
        raise ValueError(f"Chronos prediction dataframe is missing forecast rows for targets: {missing_targets}")

    return normalized


def _forecast_dataframe_to_wide(pred_df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    wide_df = (
        pred_df[pred_df["target_name"].astype(str).isin(targets)]
        .pivot_table(index="timestamp", columns="target_name", values="predictions", aggfunc="last")
        .reset_index()
        .sort_values("timestamp")
    )
    missing_targets = sorted(set(targets) - set(wide_df.columns))
    if missing_targets:
        logger.warning("Chronos wide forecast missing targets=%s", missing_targets)
        raise ValueError(f"Chronos prediction dataframe is missing forecast rows for targets: {missing_targets}")
    return wide_df[["timestamp", *targets]]


def _row_to_time_record(row: pd.Series, targets: list[str]) -> dict[str, object]:
    record: dict[str, object] = {"time": pd.Timestamp(row["timestamp"]).isoformat()}
    for target in targets:
        record[target] = _forecast_response_value(row[target])
    return record


def _forecast_response_value(value: object) -> float | None:
    if pd.isna(value):
        return None
    return max(float(value), 0.0)


def _find_column(df: pd.DataFrame, *candidates: str | float) -> str | float:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    logger.warning("Chronos forecast dataframe missing quantile columns candidates=%s", list(candidates))
    raise ValueError(f"Chronos prediction dataframe is missing columns: {list(candidates)}")
