from __future__ import annotations

from datetime import datetime, timedelta
import logging
import time
from typing import Any

import httpx
import pandas as pd

from pydantic import ValidationError

from src.core.config import Settings
from src.models.schemas.forecast import Presence
from src.services.weather.interface import WeatherData, WeatherProvider
from src.utils.preprocessing import preprocess_dynamic_hourly_weather

logger = logging.getLogger(__name__)


class ReNileIOTWeatherProvider(WeatherProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def fetch_recent_weather(self, jwt: str, device_id: str, end_time: datetime | None = None) -> WeatherData:
        return await fetch_recent_weather(jwt, device_id, self.settings, end_time)


async def fetch_recent_weather(
    jwt: str,
    device_id: str,
    settings: Settings,
    end_time: datetime | None = None,
) -> WeatherData:
    start_time = history_start_time(end_time or datetime.now(), settings.renile_iot_history_days)
    params = {
        "data_type": settings.renile_iot_data_type,
        "start_time": start_time.strftime("%Y-%m-%d %H:%M"),
        "device_id": device_id,
        "include_presence": "true",
    }
    headers = {
        "Authorization": f"JWT {jwt}",
        "Accept": "application/json",
    }

    logger.info(
        "ReNile-IOT API call prepared method=GET url=%s device_id=%s params=%s timeout_seconds=%s authorization=redacted",
        settings.renile_iot_url,
        device_id,
        params,
        settings.renile_iot_timeout_seconds,
    )
    logger.info(
        "Fetching ReNile-IOT weather data device_id=%s data_type=%s start_time=%s end_time=%s history_days=%s",
        device_id,
        params["data_type"],
        params["start_time"],
        end_time,
        settings.renile_iot_history_days,
    )
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=settings.renile_iot_timeout_seconds) as client:
        response = await client.get(settings.renile_iot_url, params=params, headers=headers)
        response.raise_for_status()
    latency_ms = (time.perf_counter() - start) * 1000
    payload = response.json()
    logger.info(
        "ReNile-IOT API response received device_id=%s status_code=%s latency_ms=%.2f payload_preview=%s",
        device_id,
        response.status_code,
        latency_ms,
        _preview_payload(payload),
    )

    presence, sensor_payload = extract_presence(payload)
    logger.info(
        "ReNile-IOT presence received device_id=%s is_online=%s last_online_at=%s",
        device_id,
        presence.is_online,
        presence.last_online_at,
    )
    raw_df = parse_renile_iot_payload(sensor_payload)
    logger.info(
        "ReNile-IOT raw dataframe parsed rows=%s columns=%s preview=%s",
        len(raw_df),
        raw_df.columns.tolist(),
        _preview_dataframe(raw_df),
    )
    if end_time is not None:
        raw_df = filter_before(raw_df, end_time)
    processed_df = preprocess_dynamic_hourly_weather(
        raw_df,
        context_hours=settings.context_hours,
        min_context_coverage=settings.min_context_coverage,
    )
    logger.info(
        "ReNile-IOT processed dataframe ready rows=%s columns=%s preview=%s",
        len(processed_df),
        processed_df.columns.tolist(),
        _preview_dataframe(processed_df),
    )
    return WeatherData(df=processed_df, presence=presence)


def extract_presence(payload: dict[str, Any]) -> tuple[Presence, dict[str, Any]]:
    sensor_payload = dict(payload)
    raw_presence = sensor_payload.pop("presence", None)
    if raw_presence is None:
        logger.warning("ReNile-IOT response is missing presence")
        raise ValueError("ReNile-IOT response is missing presence.")
    try:
        presence = Presence.model_validate(raw_presence)
    except ValidationError as exc:
        logger.warning("ReNile-IOT presence is invalid presence=%s", raw_presence)
        raise ValueError("ReNile-IOT presence is invalid.") from exc
    return presence, sensor_payload


def history_start_time(end_time: datetime, history_days: int) -> datetime:
    return end_time - timedelta(days=history_days)


def filter_before(df: pd.DataFrame, end_time: datetime) -> pd.DataFrame:
    filtered = df[df["timestamp"] < pd.Timestamp(end_time)].reset_index(drop=True)
    logger.info("ReNile-IOT data filtered before end_time=%s rows_before=%s rows_after=%s", end_time, len(df), len(filtered))
    if filtered.empty:
        logger.warning("ReNile-IOT has no data before end_time=%s", end_time)
        raise ValueError(f"ReNile-IOT returned no data before start_date {end_time.isoformat()}.")
    return filtered


def parse_renile_iot_payload(payload: dict[str, Any]) -> pd.DataFrame:
    if not payload:
        logger.warning("ReNile-IOT response is empty")
        raise ValueError("ReNile-IOT response is empty.")

    logger.info("Parsing ReNile-IOT payload sensors=%s", list(payload))

    sensor_frames: list[pd.DataFrame] = []

    for sensor_name, sensor_data in payload.items():
        sensor_frames.append(_parse_sensor_payload(sensor_name, sensor_data))

    merged_df = sensor_frames[0]
    for sensor_df in sensor_frames[1:]:
        merged_df = pd.merge(merged_df, sensor_df, on="timestamp", how="outer")

    merged_df = merged_df.sort_values("timestamp").reset_index(drop=True)
    logger.info(
        "ReNile-IOT payload parsed and merged rows=%s sensors=%s start=%s end=%s preview=%s",
        len(merged_df),
        len(merged_df.columns) - 1,
        merged_df["timestamp"].min(),
        merged_df["timestamp"].max(),
        _preview_dataframe(merged_df),
    )
    return merged_df


def _parse_sensor_payload(sensor_name: str, sensor_data: Any) -> pd.DataFrame:
    if not isinstance(sensor_data, dict):
        logger.warning("ReNile-IOT sensor payload is invalid sensor=%s payload_type=%s", sensor_name, type(sensor_data).__name__)
        raise ValueError(f"ReNile-IOT sensor payload is invalid: {sensor_name}")

    values = sensor_data.get("data")
    labels = sensor_data.get("labels")
    if values is None or labels is None:
        logger.warning("ReNile-IOT sensor payload missing labels or data sensor=%s", sensor_name)
        raise ValueError(f"ReNile-IOT sensor payload must include labels and data: {sensor_name}")
    if len(labels) != len(values):
        logger.warning(
            "ReNile-IOT sensor payload internal length mismatch sensor=%s labels=%s values=%s",
            sensor_name,
            len(labels),
            len(values),
        )
        raise ValueError(f"ReNile-IOT sensor payload length mismatch: {sensor_name}")

    sensor_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(labels, utc=True, errors="coerce").tz_localize(None),
            sensor_name: [_parse_decimal_value(value) for value in values],
        }
    )
    logger.info(
        "ReNile-IOT sensor parsed sensor=%s readings=%s valid_timestamps=%s start=%s end=%s preview=%s",
        sensor_name,
        len(values),
        int(sensor_df["timestamp"].notna().sum()),
        sensor_df["timestamp"].min(),
        sensor_df["timestamp"].max(),
        _preview_dataframe(sensor_df),
    )
    return sensor_df


def _parse_decimal_value(value: Any) -> float:
    if isinstance(value, dict) and "$numberDecimal" in value:
        value = value["$numberDecimal"]
    return float(value)


def _preview_payload(payload: dict[str, Any], max_sensors: int = 5) -> dict[str, Any]:
    preview: dict[str, Any] = {"sensor_count": len(payload), "sensors": []}
    for sensor_name, sensor_data in list(payload.items())[:max_sensors]:
        if isinstance(sensor_data, dict):
            labels = sensor_data.get("labels") or []
            data = sensor_data.get("data") or []
            preview["sensors"].append(
                {
                    "name": sensor_name,
                    "label_count": len(labels),
                    "data_count": len(data),
                    "first_label": labels[0] if labels else None,
                    "last_label": labels[-1] if labels else None,
                    "first_value": data[0] if data else None,
                }
            )
        else:
            preview["sensors"].append({"name": sensor_name, "payload_type": type(sensor_data).__name__})
    return preview


def _preview_dataframe(df: pd.DataFrame, rows: int = 3) -> dict[str, Any]:
    return {
        "head": df.head(rows).to_dict(orient="records"),
        "tail": df.tail(rows).to_dict(orient="records"),
    }
