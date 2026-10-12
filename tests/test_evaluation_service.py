import asyncio
from datetime import date, datetime
import math

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pandas as pd
import pytest

from src.api.v1.endpoints.evaluation import router as evaluation_router
from src.core.config import Settings, get_settings
from src.models.ai_model.providers.chronos import forecast_dataframe_to_daily_ranges, forecast_dataframe_to_hourly_rows
from src.services.evaluation_service import EvaluationRequest, EvaluationResponse, EvaluationService

CUTOFF = pd.Timestamp("2026-10-05")
PREDICTED_SO2 = 10.0


class FakeAIModel:
    def __init__(self) -> None:
        self.context_df: pd.DataFrame | None = None

    def forecast(self, context_df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
        self.context_df = context_df
        start = pd.Timestamp(context_df["timestamp"].max()) + pd.Timedelta(hours=1)
        timestamps = pd.date_range(start, periods=168, freq="h")
        values = {"SO2": PREDICTED_SO2, "ambient_temp": -3.0}
        rows = [
            {"target_name": target, "timestamp": timestamp, "predictions": values[target]}
            for target in targets
            for timestamp in timestamps
        ]
        return pd.DataFrame(rows)

    def forecast_to_hourly_rows(self, pred_df: pd.DataFrame, targets: list[str], hours: int) -> list[dict[str, object]]:
        return forecast_dataframe_to_hourly_rows(pred_df, targets, hours)

    def forecast_to_daily_ranges(self, pred_df: pd.DataFrame, targets: list[str], days: int) -> list[dict[str, object]]:
        return forecast_dataframe_to_daily_ranges(pred_df, targets, days)


def make_settings() -> Settings:
    return Settings(
        app_name="test",
        logging_level="INFO",
        open_meteo_url="https://example.com",
        open_meteo_timeout_seconds=1.0,
        open_meteo_history_days=7,
        open_meteo_timezone_fallback="UTC",
        renile_iot_url="https://renile-iot.com/api/v1/data/",
        renile_iot_timeout_seconds=30,
        renile_iot_history_days=20,
        renile_iot_data_type="month_hours",
        chronos_model_id="test-model",
        chronos_device_map="cpu",
        prediction_length=168,
        context_hours=336,
    )


def make_raw_df() -> pd.DataFrame:
    history = pd.DataFrame(
        {
            "timestamp": pd.date_range(CUTOFF - pd.Timedelta(hours=336), periods=336, freq="h"),
            "SO2": [float(value) for value in range(336)],
            "ambient_temp": 25.0,
        }
    )
    window_hours = pd.date_range(CUTOFF, periods=168, freq="h")
    actual_so2 = [8.0] * 168
    actual_so2[2] = 0.0
    actual_so2[3] = 14.0
    actual = pd.DataFrame({"timestamp": window_hours, "SO2": actual_so2, "ambient_temp": float("nan")})
    actual = actual.drop(index=1)
    early_duplicate = pd.DataFrame({"timestamp": [CUTOFF + pd.Timedelta(minutes=10)], "SO2": [999.0], "ambient_temp": [float("nan")]})
    late_duplicate = pd.DataFrame({"timestamp": [CUTOFF + pd.Timedelta(minutes=40)], "SO2": [8.0], "ambient_temp": [float("nan")]})
    after_window = pd.DataFrame({"timestamp": [CUTOFF + pd.Timedelta(hours=170)], "SO2": [5000.0], "ambient_temp": [5000.0]})
    raw_df = pd.concat([history, actual, early_duplicate, late_duplicate, after_window], ignore_index=True)
    return raw_df.sort_values("timestamp", kind="stable").reset_index(drop=True)


def run_evaluation(
    raw_df: pd.DataFrame,
    now: datetime = datetime(2026, 10, 12, 12),
) -> tuple[EvaluationResponse, FakeAIModel, list[datetime]]:
    ai_model = FakeAIModel()
    fetch_starts: list[datetime] = []

    async def fake_fetch(jwt: str, device_id: str, start_time: datetime, settings: Settings) -> pd.DataFrame:
        fetch_starts.append(start_time)
        return raw_df

    service = EvaluationService(ai_model=ai_model, settings=make_settings(), fetch_history=fake_fetch, now=lambda: now)
    request = EvaluationRequest(JWT="test-token", device_id="test-device", start_date=date(2026, 10, 5))
    response = asyncio.run(service.evaluate(request))
    return response, ai_model, fetch_starts


def test_context_uses_only_readings_before_cutoff() -> None:
    response, ai_model, fetch_starts = run_evaluation(make_raw_df())

    assert fetch_starts == [datetime(2026, 9, 15)]
    assert ai_model.context_df is not None
    assert len(ai_model.context_df) == 336
    assert ai_model.context_df["timestamp"].max() == CUTOFF - pd.Timedelta(hours=1)
    assert ai_model.context_df["SO2"].tolist() == [float(value) for value in range(336)]
    assert response.window == {"start": "2026-10-05T00:00:00", "end": "2026-10-11T23:00:00"}


def test_missing_actuals_are_null_and_duplicates_keep_last_reading() -> None:
    response, _, _ = run_evaluation(make_raw_df())

    assert len(response.hourly) == 168
    assert response.hourly[0] == {
        "time": "2026-10-05T00:00:00",
        "SO2": {"predicted": 10.0, "actual": 8.0},
        "ambient_temp": {"predicted": 0.0, "actual": None},
    }
    assert response.hourly[1]["SO2"] == {"predicted": 10.0, "actual": None}
    assert response.hourly[-1]["time"] == "2026-10-11T23:00:00"
    assert response.metrics["ambient_temp"] == {"mae": None, "rmse": None, "mape": None, "bias": None, "actual_hours": 0}
    assert len(response.daily) == 7
    assert response.daily[0] == {
        "time": "2026-10-05",
        "SO2": {"pred_min": 10.0, "pred_max": 10.0, "actual_min": 0.0, "actual_max": 14.0},
        "ambient_temp": {"pred_min": 0.0, "pred_max": 0.0, "actual_min": None, "actual_max": None},
    }


def test_metrics_exclude_missing_actuals_and_zero_from_mape() -> None:
    response, _, _ = run_evaluation(make_raw_df())

    errors = [2.0] * 165 + [10.0, -4.0]
    mape_terms = [0.25] * 165 + [4.0 / 14.0]
    metrics = response.metrics["SO2"]
    assert metrics["actual_hours"] == 167
    assert metrics["mae"] == pytest.approx(sum(abs(error) for error in errors) / 167)
    assert metrics["rmse"] == pytest.approx(math.sqrt(sum(error**2 for error in errors) / 167))
    assert metrics["bias"] == pytest.approx(sum(errors) / 167)
    assert metrics["mape"] == pytest.approx(sum(mape_terms) / 166 * 100)


def drop_hours_before_cutoff(raw_df: pd.DataFrame, hours: int) -> pd.DataFrame:
    gap = raw_df["timestamp"].between(CUTOFF - pd.Timedelta(hours=hours), CUTOFF - pd.Timedelta(minutes=1))
    return raw_df[~gap]


def test_pads_short_trailing_gap_with_last_reading() -> None:
    _, ai_model, _ = run_evaluation(drop_hours_before_cutoff(make_raw_df(), hours=6))

    assert ai_model.context_df is not None
    assert len(ai_model.context_df) == 336
    assert ai_model.context_df["timestamp"].max() == CUTOFF - pd.Timedelta(hours=1)
    assert ai_model.context_df["SO2"].tolist()[-7:] == [329.0] * 7


def test_rejects_trailing_gap_longer_than_limit() -> None:
    with pytest.raises(ValueError, match="25 hours gap, max 24"):
        run_evaluation(drop_hours_before_cutoff(make_raw_df(), hours=25))


def test_rejects_window_that_has_not_elapsed() -> None:
    with pytest.raises(ValueError, match="has not fully elapsed"):
        run_evaluation(make_raw_df(), now=datetime(2026, 10, 11, 23, 30))


def make_client() -> TestClient:
    app = FastAPI()
    app.include_router(evaluation_router, prefix="/api/v1")
    app.state.ai_model = FakeAIModel()
    app.dependency_overrides[get_settings] = make_settings
    return TestClient(app)


def test_endpoint_returns_422_for_future_window() -> None:
    response = make_client().post(
        "/api/v1/evaluate",
        json={"JWT": "test-token", "device_id": "test-device", "start_date": "2099-01-01"},
    )

    assert response.status_code == 422
    assert "has not fully elapsed" in response.json()["detail"]


def test_endpoint_rejects_extra_fields_and_invalid_date() -> None:
    client = make_client()

    extra = client.post(
        "/api/v1/evaluate",
        json={"JWT": "test-token", "device_id": "test-device", "start_date": "2026-10-05", "hours": 24},
    )
    invalid_date = client.post(
        "/api/v1/evaluate",
        json={"JWT": "test-token", "device_id": "test-device", "start_date": "05-10-2026"},
    )

    assert extra.status_code == 422
    assert invalid_date.status_code == 422
