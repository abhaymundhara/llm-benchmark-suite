from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List

import requests

from .base import (
    AdapterConfigurationError,
    BaseModelAdapter,
    GenerationMetrics,
    GenerationResult,
    ModelInfo,
    ModelGenerationError,
)

logger = logging.getLogger(__name__)


class OllamaAdapter(BaseModelAdapter):
    """Adapter for interacting with local Ollama models."""

    DEFAULT_HOST = "http://localhost:11434"

    def __init__(self, model_name: str, *, timeout: float = 300.0, host: str | None = None) -> None:
        super().__init__(model_name=model_name, timeout=timeout)
        self.host = host or os.environ.get("OLLAMA_HOST", self.DEFAULT_HOST)
        if not self.host.startswith("http"):
            raise AdapterConfigurationError("OLLAMA_HOST must include a scheme (e.g. http://).")

    @classmethod
    def list_models(cls) -> List[ModelInfo]:
        host = os.environ.get("OLLAMA_HOST", cls.DEFAULT_HOST)
        try:
            response = requests.get(f"{host}/api/tags", timeout=5, verify=False)  # type: ignore[arg-type]
            response.raise_for_status()
            data = response.json()
            models = data.get("models", [])
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Unable to fetch Ollama models from %s: %s", host, exc)
            return []

        infos: List[ModelInfo] = []
        for model in models:
            infos.append(
                ModelInfo(
                    name=model.get("name", "unknown"),
                    provider="ollama",
                    description=model.get("details", {}).get("description"),
                    context_window=model.get("details", {}).get("context_length"),
                    supports_stream=True,
                    homepage="https://ollama.com/library",
                )
            )
        return infos

    def get_info(self) -> ModelInfo:
        models = {info.name: info for info in self.list_models()}
        info = models.get(self.model_name)
        if not info:
            info = ModelInfo(
                name=self.model_name,
                provider="ollama",
                supports_stream=True,
                description="Custom Ollama model",
                homepage="https://ollama.com/library",
            )
        return info

    def generate(self, prompt: str, temperature: float, max_tokens: int) -> GenerationResult:
        self.validate_generation_params(prompt, temperature, max_tokens)

        # Set context window size - important for models with long prompts
        # Default Ollama uses 2048, but models like swe-13b support 16K
        num_ctx = 16384  # Use full 16K context window for models that support it
        
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": num_ctx,  # Set context window size to 16K
            },
        }
        url = f"{self.host}/api/generate"
        logger.debug("Sending generation request to Ollama at %s (num_ctx=%d)", url, num_ctx)

        start_time = time.perf_counter()
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise ModelGenerationError(f"Ollama request failed: {exc}") from exc
        finally:
            latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Handle DeepSeek-R1 and other reasoning models that separate thinking from response
        response_text = data.get("response", "")
        thinking_text = data.get("thinking", "")
        
        # Combine thinking and response for reasoning models
        if thinking_text and response_text:
            # Both fields present: concatenate with clear separation
            output = f"<thinking>\n{thinking_text}\n</thinking>\n\n{response_text}"
        elif thinking_text:
            # Only thinking present: use it (DeepSeek-R1 may only generate thinking)
            output = thinking_text
        elif response_text:
            # Only response present: normal case
            output = response_text
        else:
            # Neither field has content
            raise ModelGenerationError("Ollama returned an empty response.")

        metrics = GenerationMetrics(
            latency_ms=latency_ms,
            raw_response=data,
            output_tokens=data.get("eval_count"),
            input_tokens=data.get("prompt_eval_count"),
        )
        return GenerationResult(output_text=output.strip(), metrics=metrics)
