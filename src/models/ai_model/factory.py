from __future__ import annotations

from src.core.config import Settings
from src.models.ai_model.interface import AIModel
from src.models.ai_model.providers.chronos import ChronosAIModel


def create_ai_model(settings: Settings, provider: str = "chronos", **kwargs: object) -> AIModel:
    if provider == "chronos":
        return ChronosAIModel(settings=settings, **kwargs)
    raise ValueError(f"Unsupported AI model provider: {provider}")
