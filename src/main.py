from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
import time
from uuid import uuid4

from fastapi import FastAPI, Request

from src.api.v1.endpoints.forecast import router as forecast_router
from src.core.config import get_settings
from src.core.logging import configure_logging
from src.services.chronos_forecasting import load_chronos_pipeline

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info("Application startup started app_name=%s", settings.app_name)
    app.state.chronos_pipeline = load_chronos_pipeline(settings)
    logger.info("Application startup completed")
    yield
    logger.info("Application shutdown completed")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.logging_level)
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.include_router(forecast_router, prefix="/api/v1")

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid4()))
        start_time = time.perf_counter()
        logger.info(
            "Request started request_id=%s method=%s path=%s client=%s",
            request_id,
            request.method,
            request.url.path,
            request.client.host if request.client else "unknown",
        )

        try:
            response = await call_next(request)
        except Exception:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.exception(
                "Request failed request_id=%s method=%s path=%s latency_ms=%.2f",
                request_id,
                request.method,
                request.url.path,
                latency_ms,
            )
            raise

        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Request completed request_id=%s method=%s path=%s status_code=%s latency_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
        )
        response.headers["x-request-id"] = request_id
        return response

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
