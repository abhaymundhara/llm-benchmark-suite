from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List

import anthropic
from anthropic import APIError

from .base import (
    AdapterConfigurationError,
    BaseModelAdapter,
    GenerationMetrics,
    GenerationResult,
    ModelInfo,
    ModelGenerationError,
)

logger = logging.getLogger(__name__)


class ClaudeAdapter(BaseModelAdapter):
    """Adapter for Anthropic Claude models."""

    MODEL_SPECS: Dict[str, Dict[str, Any]] = {
        "claude-3-opus-20240229": {
            "context_window": 200_000,
            "description": "Claude 3 Opus - highest intelligence flagship model.",
        },
        "claude-3-sonnet-20240229": {
            "context_window": 200_000,
            "description": "Claude 3 Sonnet - balanced performance and cost.",
        },
        "claude-3-haiku-20240307": {
            "context_window": 200_000,
            "description": "Claude 3 Haiku - fastest and most cost-effective model.",
        },
    }

    def __init__(self, model_name: str, *, timeout: float = 60.0) -> None:
        super().__init__(model_name=model_name, timeout=timeout)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise AdapterConfigurationError("ANTHROPIC_API_KEY environment variable is required.")
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout)

    @classmethod
    def list_models(cls) -> List[ModelInfo]:
        return [
            ModelInfo(
                name=name,
                provider="anthropic",
                supports_stream=True,
                context_window=spec.get("context_window"),
                description=spec.get("description"),
                homepage="https://docs.anthropic.com/claude/docs",
            )
            for name, spec in cls.MODEL_SPECS.items()
        ]

    def get_info(self) -> ModelInfo:
        specs = self.MODEL_SPECS.get(self.model_name, {})
        return ModelInfo(
            name=self.model_name,
            provider="anthropic",
            supports_stream=True,
            context_window=specs.get("context_window"),
            description=specs.get("description"),
            homepage="https://docs.anthropic.com/claude/docs",
        )

    def generate(self, prompt: str, temperature: float, max_tokens: int) -> GenerationResult:
        self.validate_generation_params(prompt, temperature, max_tokens)

        start_time = time.perf_counter()
        raw: Dict[str, Any] = {}
        try:
            response = self._client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            text_parts = [
                block.text for block in response.content if block.type == "text"  # type: ignore[attr-defined]
            ]
            output_text = "\n".join(text_parts)
            raw = response.model_dump()
            usage = raw.get("usage", {})
            prompt_tokens = usage.get("input_tokens")
            completion_tokens = usage.get("output_tokens")
        except APIError as exc:
            raise ModelGenerationError(f"Anthropic generation failed: {exc}") from exc
        finally:
            latency_ms = (time.perf_counter() - start_time) * 1000.0

        if not output_text:
            raise ModelGenerationError("Anthropic returned an empty response.")

        metrics = GenerationMetrics(
            latency_ms=latency_ms,
            raw_response=raw,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
        )
        return GenerationResult(output_text=output_text.strip(), metrics=metrics)
