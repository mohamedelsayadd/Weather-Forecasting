import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
import httpx

from src.core.config import Settings, get_settings
from src.services.evaluation_service import EvaluationRequest, EvaluationResponse, EvaluationService

router = APIRouter(prefix="/evaluate", tags=["evaluation"])
logger = logging.getLogger(__name__)


def get_evaluation_service(request: Request, settings: Settings) -> EvaluationService:
    return EvaluationService(ai_model=request.app.state.ai_model, settings=settings)


@router.post("", response_model=EvaluationResponse)
async def create_evaluation(
    request_body: EvaluationRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> EvaluationResponse:
    logger.info(
        "Evaluation request received source=renile_iot device_id=%s start_date=%s",
        request_body.device_id,
        request_body.start_date,
    )
    try:
        evaluation_service = get_evaluation_service(request, settings)
        response = await evaluation_service.evaluate(request_body)
        logger.info(
            "Evaluation API response ready device_id=%s window=%s metrics=%s",
            request_body.device_id,
            response.window,
            response.metrics,
        )
        return response
    except AttributeError as exc:
        logger.exception("Evaluation request failed because application dependencies are not loaded")
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
        logger.exception("ReNile-IOT request failed device_id=%s", request_body.device_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to fetch weather data from ReNile-IOT.",
        ) from exc
    except ValueError as exc:
        logger.warning(
            "Evaluation request validation failed device_id=%s error=%s",
            request_body.device_id,
            exc,
        )
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
