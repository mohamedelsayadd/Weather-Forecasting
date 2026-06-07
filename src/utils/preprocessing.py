from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
]
WEATHER_PARAMETER_TO_COLUMN = {
    "temperature": "temperature_2m",
    "relative_humidity": "relative_humidity_2m",
    "surface_pressure": "surface_pressure",
    "wind_speed": "wind_speed_10m",
    "wind_direction": "wind_direction_10m",
}
WEATHER_PARAMETERS = list(WEATHER_PARAMETER_TO_COLUMN)


def preprocess_hourly_weather(df: pd.DataFrame, context_hours: int) -> pd.DataFrame:
    logger.info("Weather preprocessing started rows=%s context_hours=%s", len(df), context_hours)
    if "time" not in df.columns:
        logger.warning("Weather preprocessing failed: missing time column")
        raise ValueError("Open-Meteo dataframe is missing the time column.")

    missing_columns = sorted(set(WEATHER_COLUMNS) - set(df.columns))
    if missing_columns:
        logger.warning("Weather preprocessing failed: missing_columns=%s", missing_columns)
        raise ValueError(f"Missing expected weather columns: {missing_columns}")

    cleaned = df[["time", *WEATHER_COLUMNS]].copy()
    cleaned["timestamp"] = pd.to_datetime(cleaned.pop("time"), errors="coerce")
    invalid_timestamps = int(cleaned["timestamp"].isna().sum())
    cleaned = cleaned.dropna(subset=["timestamp"])

    if cleaned.empty:
        logger.warning("Weather preprocessing failed: no valid timestamps invalid_timestamps=%s", invalid_timestamps)
        raise ValueError("Open-Meteo returned no valid hourly timestamps.")

    for column in WEATHER_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    missing_before = int(cleaned[WEATHER_COLUMNS].isna().sum().sum())
    cleaned = cleaned.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    cleaned = cleaned.set_index("timestamp").asfreq("h")
    missing_after_frequency = int(cleaned.isna().sum().sum())
    cleaned = cleaned.interpolate(method="time").ffill().bfill()

    if cleaned.isna().any().any():
        logger.warning("Weather preprocessing failed: nulls remain after interpolation")
        raise ValueError("Weather dataframe still contains missing values after interpolation.")

    cleaned["relative_humidity_2m"] = cleaned["relative_humidity_2m"].clip(0, 100)
    cleaned["wind_speed_10m"] = cleaned["wind_speed_10m"].clip(lower=0)
    cleaned["wind_direction_10m"] = cleaned["wind_direction_10m"] % 360

    if len(cleaned) < context_hours:
        logger.warning("Weather preprocessing failed: insufficient rows rows=%s context_hours=%s", len(cleaned), context_hours)
        raise ValueError(f"Need at least {context_hours} hourly records, got {len(cleaned)}.")

    result = cleaned.tail(context_hours).reset_index()
    logger.info(
        "Weather preprocessing completed input_rows=%s output_rows=%s missing_before=%s missing_after_frequency=%s invalid_timestamps=%s start=%s end=%s",
        len(df),
        len(result),
        missing_before,
        missing_after_frequency,
        invalid_timestamps,
        result["timestamp"].min(),
        result["timestamp"].max(),
    )
    return result


def build_chronos_context(df: pd.DataFrame, weather_parameters: list[str]) -> pd.DataFrame:
    logger.info("Building Chronos context weather_parameters=%s rows=%s", weather_parameters, len(df))
    _validate_weather_parameters(weather_parameters)
    column_mapping = {WEATHER_PARAMETER_TO_COLUMN[parameter]: parameter for parameter in weather_parameters}

    missing_columns = sorted(set(column_mapping) - set(df.columns))
    if missing_columns:
        logger.warning("Chronos context failed: missing_columns=%s", missing_columns)
        raise ValueError(f"Missing expected weather columns: {missing_columns}")

    context_df = pd.DataFrame({"item_id": "weather_series", "timestamp": pd.to_datetime(df["timestamp"])})
    for source_column, target_column in column_mapping.items():
        context_df[target_column] = pd.to_numeric(df[source_column], errors="raise")

    logger.info(
        "Chronos context built weather_parameters=%s rows=%s start=%s end=%s",
        weather_parameters,
        len(context_df),
        context_df["timestamp"].min(),
        context_df["timestamp"].max(),
    )
    return context_df


def build_chronos_context_from_values(
    past_weather_values: dict[str, list[float]],
    weather_parameters: list[str],
    context_hours: int,
) -> pd.DataFrame:
    logger.info("Building Chronos context from past values weather_parameters=%s", weather_parameters)
    _validate_weather_parameters(weather_parameters)
    timestamps = pd.date_range(
        end=pd.Timestamp.utcnow().floor("h").tz_localize(None),
        periods=context_hours,
        freq="h",
    )
    context_df = pd.DataFrame({"item_id": "weather_series", "timestamp": timestamps})

    for weather_parameter in weather_parameters:
        if weather_parameter not in past_weather_values:
            logger.warning("Chronos context failed: missing past values weather_parameter=%s", weather_parameter)
            raise ValueError("past_weather_values must include all requested weather parameters.")

        values = pd.to_numeric(pd.Series(past_weather_values[weather_parameter]), errors="coerce")
        if values.isna().any():
            logger.warning("Chronos context failed: non-numeric past values weather_parameter=%s", weather_parameter)
            raise ValueError("past_weather_values must contain only numeric readings.")
        if len(values) < context_hours:
            logger.warning(
                "Chronos context failed: insufficient past values weather_parameter=%s rows=%s context_hours=%s",
                weather_parameter,
                len(values),
                context_hours,
            )
            raise ValueError(f"Need at least {context_hours} hourly records, got {len(values)}.")

        context_df[weather_parameter] = values.tail(context_hours).reset_index(drop=True)

    logger.info(
        "Chronos context from past values built weather_parameters=%s rows=%s start=%s end=%s",
        weather_parameters,
        len(context_df),
        context_df["timestamp"].min(),
        context_df["timestamp"].max(),
    )
    return context_df


def _validate_weather_parameters(weather_parameters: list[str]) -> None:
    unsupported_parameters = sorted(set(weather_parameters) - set(WEATHER_PARAMETERS))
    if unsupported_parameters:
        logger.warning("Chronos context failed: unsupported weather_parameters=%s", unsupported_parameters)
        raise ValueError(f"Unsupported weather parameters: {unsupported_parameters}")
