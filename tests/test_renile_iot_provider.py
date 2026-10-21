from datetime import datetime

import pandas as pd
import pytest

from src.services.weather.providers.renile_iot import (
    extract_presence,
    filter_before,
    history_start_time,
    parse_renile_iot_payload,
)


def test_extract_presence_separates_presence_from_sensors() -> None:
    sensor = {"labels": ["2026-06-14T00:00:00.000Z"], "data": [{"$numberDecimal": "24"}]}
    payload = {
        "SO2": sensor,
        "presence": {"is_online": True, "last_online_at": "2026-10-18T07:06:04.865Z"},
    }

    presence, sensor_payload = extract_presence(payload)

    assert presence.model_dump() == {"is_online": True, "last_online_at": "2026-10-18T07:06:04.865Z"}
    assert sensor_payload == {"SO2": sensor}


def test_extract_presence_rejects_missing_presence() -> None:
    with pytest.raises(ValueError, match="missing presence"):
        extract_presence({"SO2": {"labels": [], "data": []}})


def test_parse_renile_iot_payload_preserves_dynamic_sensor_columns() -> None:
    payload = {
        "SO2": {
            "labels": ["2026-06-14T00:00:00.000Z", "2026-06-14T01:00:00.000Z"],
            "data": [{"$numberDecimal": "24"}, {"$numberDecimal": "25.5"}],
        },
        "ambient_temp": {
            "labels": ["2026-06-14T00:00:00.000Z", "2026-06-14T01:00:00.000Z"],
            "data": [{"$numberDecimal": "30.1"}, {"$numberDecimal": "31.2"}],
        },
    }

    result = parse_renile_iot_payload(payload)

    assert result.columns.tolist() == ["timestamp", "SO2", "ambient_temp"]
    assert result["timestamp"].tolist() == [pd.Timestamp("2026-06-14 00:00:00"), pd.Timestamp("2026-06-14 01:00:00")]
    assert result["SO2"].tolist() == [24.0, 25.5]
    assert result["ambient_temp"].tolist() == [30.1, 31.2]


def test_parse_renile_iot_payload_outer_joins_misaligned_sensors() -> None:
    payload = {
        "SO2": {
            "labels": ["2026-06-14T00:00:00.000Z", "2026-06-14T02:00:00.000Z"],
            "data": [{"$numberDecimal": "24"}, {"$numberDecimal": "26"}],
        },
        "ambient_temp": {
            "labels": [
                "2026-06-14T01:00:00.000Z",
                "2026-06-14T02:00:00.000Z",
                "2026-06-14T03:00:00.000Z",
            ],
            "data": [{"$numberDecimal": "30.1"}, {"$numberDecimal": "31.2"}, {"$numberDecimal": "32.3"}],
        },
    }

    result = parse_renile_iot_payload(payload)

    assert result.columns.tolist() == ["timestamp", "SO2", "ambient_temp"]
    assert result["timestamp"].tolist() == [
        pd.Timestamp("2026-06-14 00:00:00"),
        pd.Timestamp("2026-06-14 01:00:00"),
        pd.Timestamp("2026-06-14 02:00:00"),
        pd.Timestamp("2026-06-14 03:00:00"),
    ]
    assert result["SO2"].tolist()[:1] == [24.0]
    assert pd.isna(result.loc[1, "SO2"])
    assert result.loc[2, "SO2"] == 26.0
    assert pd.isna(result.loc[0, "ambient_temp"])
    assert result["ambient_temp"].tolist()[1:] == [30.1, 31.2, 32.3]


def test_parse_renile_iot_payload_rejects_sensor_internal_length_mismatch() -> None:
    payload = {
        "SO2": {
            "labels": ["2026-06-14T00:00:00.000Z", "2026-06-14T01:00:00.000Z"],
            "data": [{"$numberDecimal": "24"}],
        },
    }

    with pytest.raises(ValueError, match="length mismatch: SO2"):
        parse_renile_iot_payload(payload)


def test_history_start_time_is_relative_to_end_time() -> None:
    assert history_start_time(datetime(2026, 9, 1, 12, 0), 20) == datetime(2026, 8, 12, 12, 0)


def test_filter_before_drops_rows_at_or_after_end_time() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-09-01 10:00", periods=4, freq="h"),
            "SO2": [1.0, 2.0, 3.0, 4.0],
        }
    )

    result = filter_before(df, datetime(2026, 9, 1, 12, 0))

    assert result["timestamp"].tolist() == [pd.Timestamp("2026-09-01 10:00"), pd.Timestamp("2026-09-01 11:00")]


def test_filter_before_rejects_when_no_data_before_end_time() -> None:
    df = pd.DataFrame({"timestamp": [pd.Timestamp("2026-09-01 12:00")], "SO2": [1.0]})

    with pytest.raises(ValueError, match="no data before start_date"):
        filter_before(df, datetime(2026, 9, 1, 12, 0))
