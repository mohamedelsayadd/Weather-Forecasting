from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
    "shortwave_radiation",
    "precipitation",
]
WEATHER_PARAMETER_TO_COLUMN = {
    "temperature": "temperature_2m",
    "humidity": "relative_humidity_2m",
    "windSpeed": "wind_speed_10m",
    "windDirection": "wind_direction_10m",
    "cloud": "cloud_cover",
    "solarRadiation": "shortwave_radiation",
    "precipitation": "precipitation",
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
    cleaned["cloud_cover"] = cleaned["cloud_cover"].clip(0, 100)
    cleaned["shortwave_radiation"] = cleaned["shortwave_radiation"].clip(lower=0)
    cleaned["precipitation"] = cleaned["precipitation"].clip(lower=0)

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


def preprocess_dynamic_hourly_weather(
    df: pd.DataFrame,
    context_hours: int,
    min_context_coverage: float = 0.0,
) -> pd.DataFrame:
    logger.info(
        "Dynamic weather preprocessing started rows=%s columns=%s context_hours=%s min_context_coverage=%s preview=%s",
        len(df),
        df.columns.tolist(),
        context_hours,
        min_context_coverage,
        _preview_dataframe(df),
    )
    if "timestamp" not in df.columns:
        logger.warning("Dynamic weather preprocessing failed: missing timestamp column")
        raise ValueError("Weather dataframe is missing the timestamp column.")

    sensor_columns = [column for column in df.columns if column != "timestamp"]
    if not sensor_columns:
        logger.warning("Dynamic weather preprocessing failed: no sensor columns")
        raise ValueError("Weather dataframe does not contain any sensor columns.")

    cleaned = df[["timestamp", *sensor_columns]].copy()
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce").dt.floor("h")
    invalid_timestamps = int(cleaned["timestamp"].isna().sum())
    cleaned = cleaned.dropna(subset=["timestamp"])

    if cleaned.empty:
        logger.warning("Dynamic weather preprocessing failed: no valid timestamps invalid_timestamps=%s", invalid_timestamps)
        raise ValueError("Weather dataframe contains no valid timestamps.")

    for column in sensor_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    missing_before = int(cleaned[sensor_columns].isna().sum().sum())
    logger.info(
        "Dynamic weather preprocessing numeric conversion completed rows=%s sensors=%s missing_values=%s preview=%s",
        len(cleaned),
        sensor_columns,
        missing_before,
        _preview_dataframe(cleaned),
    )
    cleaned = cleaned.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    cleaned = cleaned.set_index("timestamp").asfreq("h")
    real_mask = cleaned.notna()
    missing_after_frequency = int((~real_mask).sum().sum())
    logger.info(
        "Dynamic weather preprocessing hourly frequency applied rows=%s missing_values=%s start=%s end=%s preview=%s",
        len(cleaned),
        missing_after_frequency,
        cleaned.index.min(),
        cleaned.index.max(),
        _preview_dataframe(cleaned.reset_index()),
    )
    cleaned = cleaned.interpolate(method="time").ffill().bfill()

    if cleaned.isna().any().any():
        logger.warning("Dynamic weather preprocessing failed: nulls remain after interpolation")
        raise ValueError("Weather dataframe still contains missing values after interpolation.")

    if len(cleaned) < context_hours:
        logger.warning("Dynamic weather preprocessing failed: insufficient rows rows=%s context_hours=%s", len(cleaned), context_hours)
        raise ValueError(f"Need at least {context_hours} hourly records, got {len(cleaned)}.")

    result = cleaned.tail(context_hours).reset_index()

    context_mask = real_mask.tail(context_hours)
    sensor_coverage = {column: float(context_mask[column].mean()) for column in sensor_columns}
    real_coverage = float(context_mask.to_numpy().mean())
    starved_sensors = sorted(column for column, coverage in sensor_coverage.items() if coverage < min_context_coverage)

    if starved_sensors:
        logger.warning(
            "Dynamic weather preprocessing failed: interpolated context context_hours=%s real_coverage=%.4f "
            "min_context_coverage=%s starved_sensors=%s sensor_coverage=%s",
            context_hours,
            real_coverage,
            min_context_coverage,
            starved_sensors,
            sensor_coverage,
        )
        raise ValueError(
            f"Context is mostly interpolated: sensors {starved_sensors} have less than "
            f"{min_context_coverage:.0%} real readings across the {context_hours}-hour context."
        )

    logger.info(
        "Dynamic weather preprocessing completed input_rows=%s output_rows=%s sensors=%s missing_before=%s missing_after_frequency=%s invalid_timestamps=%s real_coverage=%.4f real_rows_in_context=%s sensor_coverage=%s start=%s end=%s processed_preview=%s",
        len(df),
        len(result),
        len(sensor_columns),
        missing_before,
        missing_after_frequency,
        invalid_timestamps,
        real_coverage,
        int(context_mask.all(axis=1).sum()),
        sensor_coverage,
        result["timestamp"].min(),
        result["timestamp"].max(),
        _preview_dataframe(result),
    )
    return result


def build_dynamic_chronos_context(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    logger.info(
        "Building dynamic Chronos context rows=%s columns=%s preview=%s",
        len(df),
        df.columns.tolist(),
        _preview_dataframe(df),
    )
    if "timestamp" not in df.columns:
        logger.warning("Dynamic Chronos context failed: missing timestamp column")
        raise ValueError("Weather dataframe is missing the timestamp column.")

    targets = [column for column in df.columns if column != "timestamp"]
    if not targets:
        logger.warning("Dynamic Chronos context failed: no target columns")
        raise ValueError("Weather dataframe does not contain any target columns.")

    context_df = pd.DataFrame({"item_id": "weather_series", "timestamp": pd.to_datetime(df["timestamp"])})
    for target in targets:
        context_df[target] = pd.to_numeric(df[target], errors="raise")

    logger.info(
        "Dynamic Chronos context built targets=%s rows=%s start=%s end=%s context_preview=%s",
        targets,
        len(context_df),
        context_df["timestamp"].min(),
        context_df["timestamp"].max(),
        _preview_dataframe(context_df),
    )
    return context_df, targets


def _preview_dataframe(df: pd.DataFrame, rows: int = 3) -> list[dict[str, object]]:
    return df.head(rows).to_dict(orient="records")


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
