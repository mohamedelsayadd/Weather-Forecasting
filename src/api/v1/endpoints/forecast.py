import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
import httpx

from src.core.config import Settings, get_settings
from src.models.schemas.forecast import ForecastRequest, ForecastResponse
from src.services.forecasting_service import ForecastingService

router = APIRouter(prefix="/forecast", tags=["forecast"])
logger = logging.getLogger(__name__)


def get_forecasting_service(request: Request, settings: Settings) -> ForecastingService:
    return ForecastingService(
        weather_provider=request.app.state.weather_provider,
        ai_model=request.app.state.ai_model,
        settings=settings,
    )

@router.post("", response_model=ForecastResponse)
async def create_forecast(
    request_body: ForecastRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> ForecastResponse:
    source = "past_weather_values" if request_body.past_weather_values is not None else "open_meteo"
    logger.info(
        "Forecast request received source=%s latitude=%s longitude=%s weather_parameter=%s",
        source,
        request_body.latitude,
        request_body.longitude,
        request_body.weather_parameter,
    )
    try:
        forecasting_service = get_forecasting_service(request, settings)
        return await forecasting_service.create_forecast(request_body)
    except AttributeError as exc:
        logger.exception("Forecast request failed because application dependencies are not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Forecasting dependencies are not loaded.",
        ) from exc
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "Open-Meteo returned an error status_code=%s latitude=%s longitude=%s",
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
            "Open-Meteo request failed latitude=%s longitude=%s",
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
