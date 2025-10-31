from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List

import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError

from .base import (
    AdapterConfigurationError,
    BaseModelAdapter,
    GenerationMetrics,
    GenerationResult,
    ModelInfo,
    ModelGenerationError,
)

logger = logging.getLogger(__name__)


class GeminiAdapter(BaseModelAdapter):
    """Adapter for Google Gemini models."""

    MODEL_SPECS: Dict[str, Dict[str, Any]] = {
        "gemini-1.5-pro-latest": {
            "context_window": 1_000_000,
            "description": "Gemini 1.5 Pro multimodal flagship model.",
        },
        "gemini-1.5-flash-latest": {
            "context_window": 1_000_000,
            "description": "Gemini 1.5 Flash optimized for speed.",
        },
    }

    def __init__(self, model_name: str, *, timeout: float = 60.0) -> None:
        super().__init__(model_name=model_name, timeout=timeout)
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise AdapterConfigurationError("GOOGLE_API_KEY environment variable is required.")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name)

    @classmethod
    def list_models(cls) -> List[ModelInfo]:
        return [
            ModelInfo(
                name=name,
                provider="google",
                supports_stream=True,
                context_window=spec.get("context_window"),
                description=spec.get("description"),
                homepage="https://ai.google.dev/gemini-api/docs/models",
            )
            for name, spec in cls.MODEL_SPECS.items()
        ]

    def get_info(self) -> ModelInfo:
        specs = self.MODEL_SPECS.get(self.model_name, {})
        return ModelInfo(
            name=self.model_name,
            provider="google",
            supports_stream=True,
            context_window=specs.get("context_window"),
            description=specs.get("description"),
            homepage="https://ai.google.dev/gemini-api/docs/models",
        )

    def generate(self, prompt: str, temperature: float, max_tokens: int) -> GenerationResult:
        self.validate_generation_params(prompt, temperature, max_tokens)
        raw: Dict[str, Any] = {}
        start_time = time.perf_counter()

        try:
            response = self._model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            output_text = response.text or ""
            usage = response.usage_metadata
            raw = {
                "usage_metadata": {
                    "prompt_token_count": usage.prompt_token_count if usage else None,
                    "candidates_token_count": usage.candidates_token_count if usage else None,
                    "total_token_count": usage.total_token_count if usage else None,
                },
                "prompt_feedback": getattr(response, "prompt_feedback", None),
            }
            input_tokens = usage.prompt_token_count if usage else None
            output_tokens = usage.candidates_token_count if usage else None
        except GoogleAPIError as exc:
            raise ModelGenerationError(f"Gemini generation failed: {exc}") from exc
        finally:
            latency_ms = (time.perf_counter() - start_time) * 1000.0

        if not output_text:
            raise ModelGenerationError("Gemini returned an empty response.")

        metrics = GenerationMetrics(
            latency_ms=latency_ms,
            raw_response=raw,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return GenerationResult(output_text=output_text.strip(), metrics=metrics)
