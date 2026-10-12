from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import date, datetime, timedelta, timezone
import logging
import math
import time
from typing import Any

import httpx
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.core.config import Settings
from src.models.ai_model.interface import AIModel
from src.services.weather.providers.renile_iot import parse_renile_iot_payload
from src.utils.preprocessing import build_dynamic_chronos_context, preprocess_dynamic_hourly_weather

logger = logging.getLogger(__name__)

EVALUATION_HOURS = 168
EVALUATION_DAYS = 7
MAX_TRAILING_GAP_HOURS = 24

HistoryFetcher = Callable[[str, str, datetime, Settings], Awaitable[pd.DataFrame]]


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    JWT: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    start_date: date


class EvaluationResponse(BaseModel):
    window: dict[str, str]
    metrics: dict[str, dict[str, float | int | None]]
    daily: list[dict[str, Any]]
    hourly: list[dict[str, Any]]


async def fetch_renile_iot_history(jwt: str, device_id: str, start_time: datetime, settings: Settings) -> pd.DataFrame:
    params = {
        "data_type": settings.renile_iot_data_type,
        "start_time": start_time.strftime("%Y-%m-%d %H:%M"),
        "device_id": device_id,
    }
    headers = {"Authorization": f"JWT {jwt}", "Accept": "application/json"}
    logger.info(
        "ReNile-IOT evaluation history call prepared method=GET url=%s device_id=%s params=%s authorization=redacted",
        settings.renile_iot_url,
        device_id,
        params,
    )
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=settings.renile_iot_timeout_seconds) as client:
        response = await client.get(settings.renile_iot_url, params=params, headers=headers)
        response.raise_for_status()
    latency_ms = (time.perf_counter() - start) * 1000

    raw_df = parse_renile_iot_payload(response.json())
    logger.info(
        "ReNile-IOT evaluation history received device_id=%s rows=%s start=%s end=%s latency_ms=%.2f",
        device_id,
        len(raw_df),
        raw_df["timestamp"].min(),
        raw_df["timestamp"].max(),
        latency_ms,
    )
    return raw_df


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class EvaluationService:
    def __init__(
        self,
        ai_model: AIModel,
        settings: Settings,
        fetch_history: HistoryFetcher = fetch_renile_iot_history,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.ai_model = ai_model
        self.settings = settings
        self.fetch_history = fetch_history
        self.now = now

    async def evaluate(self, request_body: EvaluationRequest) -> EvaluationResponse:
        cutoff = pd.Timestamp(request_body.start_date)
        window_end = cutoff + pd.Timedelta(hours=EVALUATION_HOURS)
        if window_end > pd.Timestamp(self.now()):
            raise ValueError(f"Evaluation window has not fully elapsed: it ends at {window_end.isoformat()} UTC.")

        logger.info("Evaluation started device_id=%s cutoff=%s window_end=%s", request_body.device_id, cutoff, window_end)
        fetch_start = cutoff.to_pydatetime() - timedelta(days=self.settings.renile_iot_history_days)
        raw_df = await self.fetch_history(request_body.JWT, request_body.device_id, fetch_start, self.settings)

        context_df, targets = self._build_context(raw_df, cutoff)
        pred_df = self.ai_model.forecast(context_df, targets)
        hourly_predictions = self.ai_model.forecast_to_hourly_rows(pred_df, targets, hours=EVALUATION_HOURS)
        predicted_df = _predicted_frame(hourly_predictions, targets, cutoff)
        actual_df = _actual_frame(raw_df, targets, cutoff)

        metrics = _compute_metrics(predicted_df, actual_df, targets)
        logger.info("Evaluation completed device_id=%s targets=%s metrics=%s", request_body.device_id, targets, metrics)
        return EvaluationResponse(
            window={
                "start": cutoff.isoformat(),
                "end": (window_end - pd.Timedelta(hours=1)).isoformat(),
            },
            metrics=metrics,
            daily=_daily_comparison(predicted_df, actual_df, targets),
            hourly=_hourly_comparison(predicted_df, actual_df, targets),
        )

    def _build_context(self, raw_df: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, list[str]]:
        history_df = raw_df[pd.to_datetime(raw_df["timestamp"]) < cutoff]
        if history_df.empty:
            raise ValueError(f"No readings before cutoff {cutoff.isoformat()}.")
        history_df = _pad_trailing_gap(history_df, cutoff)

        processed_df = preprocess_dynamic_hourly_weather(
            history_df,
            context_hours=self.settings.context_hours,
            min_context_coverage=self.settings.min_context_coverage,
        )
        last_hour = pd.Timestamp(processed_df["timestamp"].iloc[-1])
        expected_last_hour = cutoff - pd.Timedelta(hours=1)
        if last_hour != expected_last_hour:
            raise ValueError(
                f"History must end at {expected_last_hour.isoformat()}, but the last reading before cutoff is at {last_hour.isoformat()}."
            )
        return build_dynamic_chronos_context(processed_df)


def _pad_trailing_gap(history_df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    expected_last_hour = cutoff - pd.Timedelta(hours=1)
    last_hour = pd.to_datetime(history_df["timestamp"]).max().floor("h")
    gap_hours = int((expected_last_hour - last_hour) / pd.Timedelta(hours=1))
    if gap_hours <= 0:
        return history_df
    if gap_hours > MAX_TRAILING_GAP_HOURS:
        raise ValueError(
            f"Last reading before cutoff is at {last_hour.isoformat()} ({gap_hours} hours gap, "
            f"max {MAX_TRAILING_GAP_HOURS}). Choose another start_date."
        )

    logger.warning(
        "Evaluation history has trailing gap, padding to cutoff last_reading=%s gap_hours=%s",
        last_hour,
        gap_hours,
    )
    padding = pd.DataFrame({"timestamp": [expected_last_hour]})
    return pd.concat([history_df, padding], ignore_index=True)


def _window_index(cutoff: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(cutoff, periods=EVALUATION_HOURS, freq="h", name="timestamp")


def _predicted_frame(hourly_rows: list[dict[str, Any]], targets: list[str], cutoff: pd.Timestamp) -> pd.DataFrame:
    predicted_df = pd.DataFrame(hourly_rows)
    predicted_df["timestamp"] = pd.to_datetime(predicted_df.pop("time"))
    predicted_df = predicted_df.set_index("timestamp")
    if predicted_df.index[0] != cutoff:
        raise ValueError(f"Forecast starts at {predicted_df.index[0].isoformat()}, expected {cutoff.isoformat()}.")
    return predicted_df.reindex(_window_index(cutoff))[targets].astype(float)


def _actual_frame(raw_df: pd.DataFrame, targets: list[str], cutoff: pd.Timestamp) -> pd.DataFrame:
    actual_df = raw_df.copy()
    actual_df["timestamp"] = pd.to_datetime(actual_df["timestamp"], errors="coerce").dt.floor("h")
    window = _window_index(cutoff)
    actual_df = actual_df[actual_df["timestamp"].isin(window)]
    actual_df = actual_df.reindex(columns=["timestamp", *targets])
    actual_df[targets] = actual_df[targets].apply(pd.to_numeric, errors="coerce")
    hourly_last = actual_df.sort_values("timestamp", kind="stable").groupby("timestamp")[targets].last()
    return hourly_last.reindex(window)


def _compute_metrics(predicted_df: pd.DataFrame, actual_df: pd.DataFrame, targets: list[str]) -> dict[str, dict[str, float | int | None]]:
    metrics: dict[str, dict[str, float | int | None]] = {}
    for target in targets:
        pairs = pd.DataFrame({"predicted": predicted_df[target], "actual": actual_df[target]}).dropna()
        errors = pairs["predicted"] - pairs["actual"]
        nonzero = pairs[pairs["actual"] != 0]
        metrics[target] = {
            "mae": float(errors.abs().mean()) if not pairs.empty else None,
            "rmse": math.sqrt(float((errors**2).mean())) if not pairs.empty else None,
            "mape": float(((nonzero["predicted"] - nonzero["actual"]).abs() / nonzero["actual"].abs()).mean() * 100)
            if not nonzero.empty
            else None,
            "bias": float(errors.mean()) if not pairs.empty else None,
            "actual_hours": len(pairs),
        }
    return metrics


def _hourly_comparison(predicted_df: pd.DataFrame, actual_df: pd.DataFrame, targets: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for timestamp in predicted_df.index:
        row: dict[str, Any] = {"time": timestamp.isoformat()}
        for target in targets:
            row[target] = {
                "predicted": _optional_float(predicted_df.at[timestamp, target]),
                "actual": _optional_float(actual_df.at[timestamp, target]),
            }
        rows.append(row)
    return rows


def _daily_comparison(predicted_df: pd.DataFrame, actual_df: pd.DataFrame, targets: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for day, predicted_day in predicted_df.groupby(predicted_df.index.date):
        actual_day = actual_df.loc[predicted_day.index]
        row: dict[str, Any] = {"time": str(day)}
        for target in targets:
            row[target] = {
                "pred_min": _optional_float(predicted_day[target].min()),
                "pred_max": _optional_float(predicted_day[target].max()),
                "actual_min": _optional_float(actual_day[target].min()),
                "actual_max": _optional_float(actual_day[target].max()),
            }
        rows.append(row)
    return rows[:EVALUATION_DAYS]


def _optional_float(value: object) -> float | None:
    return None if pd.isna(value) else float(value)
