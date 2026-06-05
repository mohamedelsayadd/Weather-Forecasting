import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
import httpx

from src.core.config import Settings, get_settings
from src.models.schemas.forecast import ForecastRequest, ForecastResponse
from src.services.chronos_forecasting import forecast_dataframe_to_records, forecast_next_24_hours
from src.services.open_meteo import fetch_recent_weather
from src.services.weather_preprocessing import build_chronos_context

router = APIRouter(prefix="/forecast", tags=["forecast"])
logger = logging.getLogger(__name__)


@router.post("", response_model=ForecastResponse)
async def create_forecast(
    request_body: ForecastRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> ForecastResponse:
    logger.info(
        "Forecast request received latitude=%.4f longitude=%.4f weather_parameter=%s",
        request_body.latitude,
        request_body.longitude,
        request_body.weather_parameter,
    )
    try:
        pipeline = request.app.state.chronos_pipeline
        weather_df = await fetch_recent_weather(request_body.latitude, request_body.longitude, settings)
        context_df = build_chronos_context(weather_df, request_body.weather_parameter)
        pred_df = forecast_next_24_hours(context_df, pipeline, settings)
        forecast = forecast_dataframe_to_records(pred_df)
    except AttributeError as exc:
        logger.exception("Forecast request failed because Chronos model is not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chronos model is not loaded.",
        ) from exc
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "Open-Meteo returned an error status_code=%s latitude=%.4f longitude=%.4f",
            exc.response.status_code,
            request_body.latitude,
            request_body.longitude,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Open-Meteo returned an error: {exc.response.status_code}",
        ) from exc
    except httpx.HTTPError as exc:
        logger.exception(
            "Open-Meteo request failed latitude=%.4f longitude=%.4f",
            request_body.latitude,
            request_body.longitude,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to fetch weather data from Open-Meteo.",
        ) from exc
    except ValueError as exc:
        logger.warning(
            "Forecast request validation failed weather_parameter=%s error=%s",
            request_body.weather_parameter,
            exc,
        )
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    logger.info(
        "Forecast request completed weather_parameter=%s forecast_points=%s",
        request_body.weather_parameter,
        len(forecast),
    )

    return ForecastResponse(
        latitude=request_body.latitude,
        longitude=request_body.longitude,
        weather_parameter=request_body.weather_parameter,
        prediction_length=settings.prediction_length,
        forecast=forecast,
    )
