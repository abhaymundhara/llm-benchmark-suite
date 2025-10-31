from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List

from openai import OpenAI, OpenAIError

from .base import (
    AdapterConfigurationError,
    BaseModelAdapter,
    GenerationMetrics,
    GenerationResult,
    ModelInfo,
    ModelGenerationError,
)

logger = logging.getLogger(__name__)


class OpenAIAdapter(BaseModelAdapter):
    """Adapter for OpenAI GPT models."""

    MODEL_SPECS: Dict[str, Dict[str, Any]] = {
        "gpt-4o": {
            "context_window": 128_000,
            "description": "GPT-4o multimodal flagship model.",
        },
        "gpt-4.1": {
            "context_window": 200_000,
            "description": "Next-generation GPT-4.1 reasoning model.",
        },
        "gpt-4o-mini": {
            "context_window": 128_000,
            "description": "GPT-4o lightweight variant for cost-efficient workloads.",
        },
        "gpt-3.5-turbo": {
            "context_window": 16_000,
            "description": "GPT-3.5 Turbo legacy model.",
        },
    }

    def __init__(self, model_name: str, *, timeout: float = 60.0) -> None:
        super().__init__(model_name=model_name, timeout=timeout)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise AdapterConfigurationError("OPENAI_API_KEY environment variable is required.")
        self._client = OpenAI(api_key=api_key, timeout=timeout)

    @classmethod
    def list_models(cls) -> List[ModelInfo]:
        infos: List[ModelInfo] = []
        for name, spec in cls.MODEL_SPECS.items():
            infos.append(
                ModelInfo(
                    name=name,
                    provider="openai",
                    supports_stream=True,
                    context_window=spec.get("context_window"),
                    description=spec.get("description"),
                    homepage="https://platform.openai.com/docs/models",
                )
            )
        return infos

    def get_info(self) -> ModelInfo:
        specs = self.MODEL_SPECS.get(self.model_name, {})
        return ModelInfo(
            name=self.model_name,
            provider="openai",
            supports_stream=True,
            context_window=specs.get("context_window"),
            description=specs.get("description"),
            homepage="https://platform.openai.com/docs/models",
        )

    def generate(self, prompt: str, temperature: float, max_tokens: int) -> GenerationResult:
        self.validate_generation_params(prompt, temperature, max_tokens)

        start_time = time.perf_counter()
        raw: Dict[str, Any] = {}
        try:
            response = self._client.responses.create(
                model=self.model_name,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            output_text = response.output_text or ""
            annotations = response.output[0].annotations if response.output else None
            raw = response.model_dump()
            prompt_tokens = raw.get("usage", {}).get("input_tokens")
            completion_tokens = raw.get("usage", {}).get("output_tokens")
        except OpenAIError as exc:
            raise ModelGenerationError(f"OpenAI generation failed: {exc}") from exc
        finally:
            latency_ms = (time.perf_counter() - start_time) * 1000.0

        if not output_text:
            raise ModelGenerationError("OpenAI returned an empty response.")

        metrics = GenerationMetrics(
            latency_ms=latency_ms,
            raw_response=raw,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
        )
        if annotations:
            metrics.raw_response["annotations"] = annotations
        return GenerationResult(output_text=output_text.strip(), metrics=metrics)
