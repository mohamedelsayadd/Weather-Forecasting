from __future__ import annotations

from datetime import datetime, timedelta
import logging
import time
from typing import Any

import httpx
import pandas as pd

from src.core.config import Settings
from src.services.weather.interface import WeatherProvider
from src.utils.preprocessing import preprocess_dynamic_hourly_weather

logger = logging.getLogger(__name__)


class ReNileIOTWeatherProvider(WeatherProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def fetch_recent_weather(self, jwt: str, device_id: str) -> pd.DataFrame:
        return await fetch_recent_weather(jwt, device_id, self.settings)


async def fetch_recent_weather(jwt: str, device_id: str, settings: Settings) -> pd.DataFrame:
    start_time = datetime.now() - timedelta(days=settings.renile_iot_history_days)
    params = {
        "data_type": settings.renile_iot_data_type,
        "start_time": start_time.strftime("%Y-%m-%d %H:%M"),
        "device_id": device_id,
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
        "Fetching ReNile-IOT weather data device_id=%s data_type=%s start_time=%s history_days=%s",
        device_id,
        params["data_type"],
        params["start_time"],
        settings.renile_iot_history_days,
    )
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=settings.renile_iot_timeout_seconds) as client:
        response = await client.get(settings.renile_iot_url, params=params, headers=headers)
        response.raise_for_status()
    latency_ms = (time.perf_counter() - start) * 1000
    payload = response.json()
    logger.info(
        "ReNile-IOT API response received status_code=%s latency_ms=%.2f payload_preview=%s",
        response.status_code,
        latency_ms,
        _preview_payload(payload),
    )

    raw_df = parse_renile_iot_payload(payload)
    logger.info(
        "ReNile-IOT raw dataframe parsed rows=%s columns=%s preview=%s",
        len(raw_df),
        raw_df.columns.tolist(),
        _preview_dataframe(raw_df),
    )
    processed_df = preprocess_dynamic_hourly_weather(raw_df, context_hours=settings.context_hours)
    logger.info(
        "ReNile-IOT processed dataframe ready rows=%s columns=%s preview=%s",
        len(processed_df),
        processed_df.columns.tolist(),
        _preview_dataframe(processed_df),
    )
    return processed_df


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
