import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
import httpx

from src.core.config import Settings, get_settings
from src.models.schemas.forecast import ForecastRequest, ForecastResponse
from src.services.forecasting_service import ForecastingService
from src.utils.preprocessing import InsufficientHistoryError, InsufficientReadingsError

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
) -> ForecastResponse | JSONResponse:
    logger.info(
        "Forecast request received source=renile_iot device_id=%s start_date=%s",
        request_body.device_id,
        request_body.start_date,
    )
    try:
        forecasting_service = get_forecasting_service(request, settings)
        response = await forecasting_service.create_forecast(request_body)
        logger.info(
            "Forecast API response ready device_id=%s last12hour_rows=%s current_keys=%s hourly168_rows=%s daily7_rows=%s is_online=%s hourly168_preview=%s daily7_preview=%s",
            request_body.device_id,
            len(response.last12hour),
            list(response.current),
            len(response.hourly168),
            len(response.daily7),
            response.presence.is_online,
            response.hourly168[:3],
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
    except InsufficientHistoryError as exc:
        logger.warning(
            "Forecast request has insufficient history device_id=%s available_days=%s required_days=%s",
            request_body.device_id,
            exc.available_days,
            exc.required_days,
        )
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "detail": str(exc),
                "available_days": exc.available_days,
                "required_days": exc.required_days,
            },
        )
    except InsufficientReadingsError as exc:
        logger.warning(
            "Forecast request has insufficient data device_id=%s error=%s",
            request_body.device_id,
            exc,
        )
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        logger.warning(
            "Forecast request validation failed device_id=%s error=%s",
            request_body.device_id,
            exc,
        )
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
