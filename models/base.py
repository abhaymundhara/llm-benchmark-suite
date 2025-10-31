from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AdapterConfigurationError(RuntimeError):
    """Raised when a model adapter is misconfigured."""


class ModelGenerationError(RuntimeError):
    """Raised when a model fails to generate a response."""


@dataclass(frozen=True)
class ModelInfo:
    """Describes capabilities and metadata for a model adapter."""

    name: str
    provider: str
    supports_stream: bool = False
    modalities: List[str] = field(default_factory=lambda: ["text"])
    context_window: Optional[int] = None
    description: Optional[str] = None
    homepage: Optional[str] = None


@dataclass
class GenerationMetrics:
    """Captures auxiliary metrics returned by a generation call."""

    latency_ms: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class GenerationResult:
    """Wrapper for model generations with metadata."""

    output_text: str
    metrics: GenerationMetrics


class BaseModelAdapter(abc.ABC):
    """Abstract interface for model adapters."""

    def __init__(self, model_name: str, *, timeout: float = 60.0) -> None:
        self.model_name = model_name
        self.timeout = timeout

    @abc.abstractmethod
    def generate(self, prompt: str, temperature: float, max_tokens: int) -> GenerationResult:
        """Generate text from the supplied prompt."""

    @abc.abstractmethod
    def get_info(self) -> ModelInfo:
        """Return metadata describing the model."""

    @classmethod
    def list_models(cls) -> List[ModelInfo]:
        """Enumerate available models for the adapter."""
        raise NotImplementedError

    @staticmethod
    def validate_generation_params(prompt: str, temperature: float, max_tokens: int) -> None:
        """Validate common generation parameters."""
        if not prompt.strip():
            raise ValueError("Prompt must not be empty.")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0.")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")
