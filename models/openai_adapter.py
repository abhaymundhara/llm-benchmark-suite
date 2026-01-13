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
        # GPT-5.2 series (latest flagship models - December 2024)
        "gpt-5.2": {
            "context_window": 128_000,
            "description": "GPT-5.2: The best model for coding and agentic tasks across industries.",
        },
        "gpt-5.2-pro": {
            "context_window": 128_000,
            "description": "GPT-5.2 pro: Version of GPT-5.2 that produces smarter and more precise responses.",
        },
        # GPT-5 series
        "gpt-5": {
            "context_window": 128_000,
            "description": "GPT-5: Previous intelligent reasoning model for coding and agentic tasks.",
        },
        "gpt-5-mini": {
            "context_window": 128_000,
            "description": "GPT-5 mini: A faster, cost-efficient version of GPT-5 for well-defined tasks.",
        },
        "gpt-5-nano": {
            "context_window": 128_000,
            "description": "GPT-5 nano: Fastest, most cost-efficient version of GPT-5.",
        },
        "gpt-5-pro": {
            "context_window": 128_000,
            "description": "GPT-5 pro: Version of GPT-5 that produces smarter and more precise responses.",
        },
        # GPT-4.1 series
        "gpt-4.1": {
            "context_window": 200_000,
            "description": "GPT-4.1: Smartest non-reasoning model.",
        },
        "gpt-4.1-mini": {
            "context_window": 200_000,
            "description": "GPT-4.1 mini: Smaller, faster version of GPT-4.1.",
        },
        "gpt-4.1-nano": {
            "context_window": 200_000,
            "description": "GPT-4.1 nano: Smallest, fastest version of GPT-4.1.",
        },
        # o-series reasoning models
        "o3": {
            "context_window": 200_000,
            "description": "o3: Reasoning model for complex tasks, succeeded by GPT-5.",
        },
        "o3-pro": {
            "context_window": 200_000,
            "description": "o3-pro: Version of o3 with more compute for better responses.",
        },
        "o3-mini": {
            "context_window": 128_000,
            "description": "o3-mini: A small model alternative to o3.",
        },
        "o4-mini": {
            "context_window": 128_000,
            "description": "o4-mini: Fast, cost-efficient reasoning model, succeeded by GPT-5 mini.",
        },
        # GPT-4o series
        "gpt-4o": {
            "context_window": 128_000,
            "description": "GPT-4o: Fast, intelligent, flexible GPT model.",
        },
        "gpt-4o-mini": {
            "context_window": 128_000,
            "description": "GPT-4o mini: Fast, affordable small model for focused tasks.",
        },
        # Legacy GPT-4 models
        "gpt-4-turbo": {
            "context_window": 128_000,
            "description": "GPT-4 Turbo: An older high-intelligence GPT model.",
        },
        "gpt-4": {
            "context_window": 8_000,
            "description": "GPT-4: An older high-intelligence GPT model.",
        },
        "gpt-3.5-turbo": {
            "context_window": 16_000,
            "description": "GPT-3.5 Turbo: Legacy GPT model for cheaper chat and non-chat tasks.",
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
            # Determine which parameter to use based on model
            # Reasoning models (o-series, GPT-5.x) use max_completion_tokens
            # Other models use max_tokens
            uses_completion_tokens = (
                self.model_name.startswith("o") or  # o1, o3, o4-mini, etc.
                self.model_name.startswith("gpt-5")  # GPT-5.x series
            )
            
            # Build request parameters
            request_params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            }
            
            # Add appropriate token limit parameter
            if uses_completion_tokens:
                request_params["max_completion_tokens"] = max_tokens
            else:
                request_params["max_tokens"] = max_tokens
            
            # Use Chat Completions API (standard, well-supported)
            response = self._client.chat.completions.create(**request_params)
            
            # Extract the response text
            if not response.choices:
                raise ModelGenerationError("OpenAI returned no choices.")
            
            output_text = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason
            
            raw = response.model_dump()
            prompt_tokens = raw.get("usage", {}).get("prompt_tokens", 0)
            completion_tokens = raw.get("usage", {}).get("completion_tokens", 0)
            
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
            finish_reason=finish_reason,
            max_tokens=max_tokens,
        )
        return GenerationResult(output_text=output_text.strip(), metrics=metrics)
