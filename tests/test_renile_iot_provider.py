import pandas as pd
import pytest

from src.services.weather.providers.renile_iot import parse_renile_iot_payload


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
