import pandas as pd

from src.services.chronos_forecasting import forecast_dataframe_to_records


def test_forecast_dataframe_to_records_converts_chronos_output() -> None:
    pred_df = pd.DataFrame(
        {
            "item_id": ["weather_series"],
            "timestamp": [pd.Timestamp("2026-06-02 00:00")],
            "target_name": ["target"],
            "predictions": [20.5],
            "0.1": [19.0],
            "0.5": [20.5],
            "0.9": [22.0],
        }
    )

    assert forecast_dataframe_to_records(pred_df) == [
        {
            "timestamp": "2026-06-02T00:00:00",
            "prediction": 20.5,
            "q10": 19.0,
            "q50": 20.5,
            "q90": 22.0,
        }
    ]
