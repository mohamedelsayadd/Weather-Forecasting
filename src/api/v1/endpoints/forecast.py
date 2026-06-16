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
    logger.info(
        "Forecast request received source=renile_iot device_id=%s",
        request_body.device_id,
    )
    try:
        forecasting_service = get_forecasting_service(request, settings)
        response = await forecasting_service.create_forecast(request_body)
        logger.info(
            "Forecast API response ready device_id=%s current_keys=%s hourly24_rows=%s daily7_rows=%s hourly24_preview=%s daily7_preview=%s",
            request_body.device_id,
            list(response.current),
            len(response.hourly24),
            len(response.daily7),
            response.hourly24[:3],
            response.daily7[:3],
        )
        return response
    except AttributeError as exc:
        logger.exception("Forecast request failed because application dependencies are not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Forecasting dependencies are not loaded.",
        ) from exc
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "ReNile-IOT returned an error status_code=%s device_id=%s",
            exc.response.status_code,
            request_body.device_id,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"ReNile-IOT returned an error: {exc.response.status_code}",
        ) from exc
    except httpx.HTTPError as exc:
        logger.exception(
            "ReNile-IOT request failed device_id=%s",
            request_body.device_id,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to fetch weather data from ReNile-IOT.",
        ) from exc
    except ValueError as exc:
        logger.warning(
            "Forecast request validation failed device_id=%s error=%s",
            request_body.device_id,
            exc,
        )
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
