"""Model adapter registry and helper utilities."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Type

from dotenv import load_dotenv

from .base import BaseModelAdapter, ModelInfo

logger = logging.getLogger(__name__)

load_dotenv()


class ModelRegistry:
    """Registry for model adapter implementations."""

    def __init__(self) -> None:
        self._adapters: Dict[str, Type[BaseModelAdapter]] = {}

    def register(self, key: str, adapter_cls: Type[BaseModelAdapter]) -> None:
        key_lower = key.lower()
        if key_lower in self._adapters:
            logger.warning("Overwriting existing adapter registration for key '%s'.", key_lower)
        self._adapters[key_lower] = adapter_cls
        logger.debug("Registered adapter '%s': %s", key_lower, adapter_cls)

    def create(self, key: str, model_name: str, **kwargs) -> BaseModelAdapter:
        adapter_cls = self._adapters.get(key.lower())
        if not adapter_cls:
            raise KeyError(f"No adapter registered with key '{key}'.")
        return adapter_cls(model_name=model_name, **kwargs)

    def keys(self) -> Iterable[str]:
        return self._adapters.keys()

    def available_models(self) -> Dict[str, List[ModelInfo]]:
        models: Dict[str, List[ModelInfo]] = {}
        for key, adapter_cls in self._adapters.items():
            try:
                models[key] = adapter_cls.list_models()
            except Exception as exc:  # pylint: disable=broad-except
                logger.debug("Failed to collect info for adapter %s: %s", key, exc)
        return models


registry = ModelRegistry()

# Late imports to avoid circular references
from .ollama_adapter import OllamaAdapter  # noqa: E402  (import at end for registration)
from .openai_adapter import OpenAIAdapter  # noqa: E402
from .claude_adapter import ClaudeAdapter  # noqa: E402
from .gemini_adapter import GeminiAdapter  # noqa: E402

registry.register("ollama", OllamaAdapter)
registry.register("openai", OpenAIAdapter)
registry.register("claude", ClaudeAdapter)
registry.register("gemini", GeminiAdapter)

__all__ = [
    "BaseModelAdapter",
    "ModelInfo",
    "registry",
    "OllamaAdapter",
    "OpenAIAdapter",
    "ClaudeAdapter",
    "GeminiAdapter",
]
