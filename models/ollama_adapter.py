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
    DEFAULT_SUGGESTED_GENERATION_TOKENS = 2048
    MAX_SUGGESTED_GENERATION_TOKENS = 4096
    
    # Model-specific token limits based on known model architectures (FALLBACK ONLY)
    # These are used only if auto-detection from /api/show fails
    MODEL_TOKEN_LIMITS = {
        # Qwen models - most support 128K context
        "qwen2.5": 32768,
        "qwen2.5-coder": 32768,  # Qwen2.5-Coder supports 128K context, but 32K is safer for generation
        "qwen": 8192,
        # DeepSeek models
        "deepseek-r1": 32768,
        "deepseek-coder-v2": 32768,
        "deepseek-coder": 16384,
        "deepseek": 16384,
        # CodeLlama models
        "codellama": 16384,
        # Llama models
        "llama3.3": 32768,
        "llama3.2": 32768,
        "llama3.1": 32768,
        "llama3": 8192,
        # Mistral models
        "mixtral": 32768,
        "mistral": 32768,
        # Other code models
        "starcoder2": 16384,
        "starcoder": 8192,
        "phi3": 32768,
        "phi": 2048,
        "gemma2": 8192,
        "gemma": 8192,
        "dolphin": 8192,
        # SWE-Llama models
        "swe13b": 16384,
        # Nemotron models
        "nemotron": 32768,  # Fallback; actual detection will find 1M for newer versions
    }

    DEFAULT_MIN_NUM_CTX = 16384
    DEFAULT_MAX_NUM_CTX = 2_000_000  # Support large context models (up to 2M tokens)

    def __init__(self, model_name: str, *, timeout: float = 900.0, host: str | None = None) -> None:
        super().__init__(model_name=model_name, timeout=timeout)
        self.host = host or os.environ.get("OLLAMA_HOST", self.DEFAULT_HOST)
        if not self.host.startswith("http"):
            raise AdapterConfigurationError("OLLAMA_HOST must include a scheme (e.g. http://).")

        self._reported_context_window: int | None = None
        self._detected_max_tokens: int | None = None
        
        # Try to auto-detect the model's actual context length
        try:
            self._detected_max_tokens = self._detect_model_context_length()
            if self._detected_max_tokens:
                logger.info(
                    "Auto-detected context length for %s: %d tokens",
                    self.model_name, self._detected_max_tokens
                )
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Unable to auto-detect context length for %s: %s", self.model_name, exc)
        
        # Also get the context window from the model info (used for num_ctx calculation)
        try:
            info = self.get_info()
            if info.context_window:
                self._reported_context_window = int(info.context_window)
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Unable to read Ollama context window for %s: %s", self.model_name, exc)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate for sizing context without a tokenizer."""
        if not text:
            return 0
        # Heuristic: code-ish text averages ~3-4 chars/token; use 4 as a conservative estimate.
        return max(1, (len(text) + 3) // 4)

    def _compute_num_ctx(self, prompt: str, max_tokens: int) -> int:
        """Compute an appropriate num_ctx given prompt + requested generation size."""
        min_ctx = int(os.environ.get("OLLAMA_MIN_NUM_CTX", str(self.DEFAULT_MIN_NUM_CTX)))
        max_ctx_env = os.environ.get("OLLAMA_MAX_NUM_CTX")
        if max_ctx_env:
            max_ctx = int(max_ctx_env)
        elif self._reported_context_window:
            max_ctx = int(self._reported_context_window)
        else:
            max_ctx = self.DEFAULT_MAX_NUM_CTX

        # num_ctx must cover prompt + generation (+ small buffer for system text / formatting).
        prompt_tokens = self._estimate_tokens(prompt)
        desired_ctx = prompt_tokens + int(max_tokens) + 1024

        if max_ctx < min_ctx:
            min_ctx = max_ctx

        if desired_ctx > max_ctx:
            logger.warning(
                "Requested context %d exceeds max_ctx=%d for %s; output may truncate. "
                "Set OLLAMA_MAX_NUM_CTX to raise the cap.",
                desired_ctx,
                max_ctx,
                self.model_name,
            )

        return max(min_ctx, min(desired_ctx, max_ctx))

    def _detect_model_context_length(self) -> int | None:
        """
        Query Ollama's /api/show endpoint to auto-detect the model's actual context length.
        
        Returns:
            int | None: The model's context_length if available, None otherwise.
        """
        try:
            url = f"{self.host}/api/show"
            payload = {"name": self.model_name}
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            # Look for context_length in the model info
            # The key format is typically: {family}.context_length
            model_info = data.get("model_info", {})
            for key, value in model_info.items():
                if "context_length" in key and isinstance(value, (int, float)):
                    context_length = int(value)
                    logger.debug(
                        "Found context_length=%d in model info for %s (key: %s)",
                        context_length, self.model_name, key
                    )
                    return context_length
            
            logger.debug("No context_length found in model info for %s", self.model_name)
            return None
            
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug(
                "Failed to detect context length for %s via /api/show: %s",
                self.model_name, exc
            )
            return None

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

    def _get_model_max_tokens(self) -> int:
        """
        Determine the appropriate max_tokens value for the model.
        
        Priority:
        1. Auto-detected context length from /api/show (most accurate)
        2. Pattern matching against known models (fallback)
        3. Default value for unknown models
        
        Returns:
            int: Maximum tokens the model can generate based on its architecture.
        """
        # First, use auto-detected value if available
        if self._detected_max_tokens:
            logger.debug(
                "Using auto-detected max_tokens=%d for %s",
                self._detected_max_tokens, self.model_name
            )
            return self._detected_max_tokens
        
        # Fallback to pattern matching
        model_lower = self.model_name.lower()
        
        # Check each model pattern in priority order (more specific first)
        for pattern, max_tokens in sorted(
            self.MODEL_TOKEN_LIMITS.items(), 
            key=lambda x: len(x[0]), 
            reverse=True
        ):
            if pattern in model_lower:
                logger.debug(
                    "Detected model pattern '%s' in '%s', setting max_tokens=%d (fallback)",
                    pattern, self.model_name, max_tokens
                )
                return max_tokens
        
        # Default fallback for unknown models
        default_tokens = 2048
        logger.warning(
            "Unknown model '%s', using default max_tokens=%d",
            self.model_name, default_tokens
        )
        return default_tokens

    def get_suggested_generation_tokens(self) -> int:
        """Return a safe default for UI generation length, not the full context window."""
        hard_cap = self._get_model_max_tokens()
        return max(
            self.DEFAULT_SUGGESTED_GENERATION_TOKENS,
            min(hard_cap, self.MAX_SUGGESTED_GENERATION_TOKENS),
        )

    def generate(self, prompt: str, temperature: float, max_tokens: int) -> GenerationResult:
        self.validate_generation_params(prompt, temperature, max_tokens)

        # Respect the model's hard limit but honor the requested size if it is smaller.
        model_max = self._get_model_max_tokens()
        if max_tokens > model_max:
            logger.info(
                "Requesting %d tokens exceeds %s's max (%d); capping to model limit",
                max_tokens, self.model_name, model_max
            )
            print(f"Capping max_tokens to model max: {max_tokens} → {model_max} for {self.model_name}")
            max_tokens = model_max

        # Set context window size - scale with prompt + requested generation.
        # This avoids constraining large outputs (e.g., ~20k tokens) with a fixed 16k window.
        prompt_tokens = self._estimate_tokens(prompt)
        num_ctx = self._compute_num_ctx(prompt, max_tokens)
        
        # Add stop sequences to prevent over-generation.
        # Keep these narrowly targeted to SWE-style prompts so we don't accidentally stop other benchmarks.
        stop_sequences: List[str] | None = None
        if "<patch>" in prompt:
            # Stop once the model closes the patch; extraction already keeps the first patch.
            stop_sequences = ["</patch>"]

        think_enabled = os.environ.get("OLLAMA_THINK", "").strip().lower() in {"1", "true", "yes", "on"}
        
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "think": think_enabled,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": num_ctx,
                **({"stop": stop_sequences} if stop_sequences else {}),
            },
        }
        url = f"{self.host}/api/generate"
        logger.debug("Sending generation request to Ollama at %s (num_ctx=%d, num_predict=%d)", 
                     url, num_ctx, max_tokens)

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

        if "<patch>" in prompt and "<patch>" in output and "</patch>" not in output:
            logger.info("Appending missing </patch> tag to model output for %s", self.model_name)
            output = output.rstrip() + "\n</patch>"

        metrics = GenerationMetrics(
            latency_ms=latency_ms,
            raw_response=data,
            output_tokens=data.get("eval_count"),
            input_tokens=data.get("prompt_eval_count"),
            finish_reason=data.get("done_reason"),
            max_tokens=max_tokens,
            num_ctx=num_ctx,
            prompt_tokens_estimate=prompt_tokens,
        )
        return GenerationResult(output_text=output.strip(), metrics=metrics)
