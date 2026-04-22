from __future__ import annotations

import inspect
import logging
import os
import re
import time
from typing import Any, Dict, List, Tuple

import httpx
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

    LEGACY_COMPLETIONS_MODELS = {
        "gpt-3.5-turbo-instruct",
        "davinci-002",
        "babbage-002",
    }

    MODEL_SPECS: Dict[str, Dict[str, Any]] = {
        # GPT-5.4 series
        "gpt-5.4": {
            "context_window": 1_050_000,
            "description": "GPT-5.4: OpenAI's most capable model for professional work.",
        },
        "gpt-5.4-pro": {
            "context_window": 1_050_000,
            "description": "GPT-5.4 pro: Higher-compute GPT-5.4 variant for tougher professional tasks.",
        },
        # GPT-5.3 / 5.2 Codex series
        "gpt-5.3-codex": {
            "context_window": 400_000,
            "description": "GPT-5.3-Codex: Agentic coding model optimized for Codex-style tasks.",
        },
        "gpt-5.2": {
            "context_window": 400_000,
            "description": "GPT-5.2: The best model for coding and agentic tasks across industries.",
        },
        "gpt-5.2-pro": {
            "context_window": 400_000,
            "description": "GPT-5.2 pro: Version of GPT-5.2 that produces smarter and more precise responses.",
        },
        "gpt-5.2-codex": {
            "context_window": 400_000,
            "description": "GPT-5.2-Codex: Intelligent coding model optimized for long-horizon, agentic coding tasks.",
        },
        # GPT-5 series
        "gpt-5.1": {
            "context_window": 400_000,
            "description": "GPT-5.1: Flagship model for coding and agentic tasks with configurable reasoning effort.",
        },
        "gpt-5.1-codex": {
            "context_window": 400_000,
            "description": "GPT-5.1-Codex: Codex-tuned variant of GPT-5.1 for agentic coding tasks.",
        },
        "gpt-5.1-codex-max": {
            "context_window": 400_000,
            "description": "GPT-5.1-Codex-Max: GPT-5.1 Codex variant optimized for long-running coding tasks.",
        },
        "gpt-5.1-codex-mini": {
            "context_window": 400_000,
            "description": "GPT-5.1 Codex mini: Smaller, more cost-effective Codex variant.",
        },
        "gpt-5": {
            "context_window": 400_000,
            "description": "GPT-5: Previous intelligent reasoning model for coding and agentic tasks.",
        },
        "gpt-5-mini": {
            "context_window": 400_000,
            "description": "GPT-5 mini: A faster, cost-efficient version of GPT-5 for well-defined tasks.",
        },
        "gpt-5-nano": {
            "context_window": 400_000,
            "description": "GPT-5 nano: Fastest, most cost-efficient version of GPT-5.",
        },
        "gpt-5-pro": {
            "context_window": 400_000,
            "description": "GPT-5 pro: Version of GPT-5 that produces smarter and more precise responses.",
        },
        "gpt-5-codex": {
            "context_window": 400_000,
            "description": "GPT-5-Codex: GPT-5 variant optimized for agentic coding in Codex.",
        },
        # GPT-4.1 series
        "gpt-4.1": {
            "context_window": 1_047_576,
            "description": "GPT-4.1: Smartest non-reasoning model.",
        },
        "gpt-4.1-mini": {
            "context_window": 1_047_576,
            "description": "GPT-4.1 mini: Smaller, faster version of GPT-4.1.",
        },
        "gpt-4.1-nano": {
            "context_window": 1_047_576,
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
            "context_window": 200_000,
            "description": "o3-mini: A small model alternative to o3.",
        },
        "o4-mini": {
            "context_window": 200_000,
            "description": "o4-mini: Fast, cost-efficient reasoning model, succeeded by GPT-5 mini.",
        },
        "o1": {
            "context_window": 200_000,
            "description": "o1: Previous full o-series reasoning model.",
        },
        "o1-pro": {
            "context_window": 200_000,
            "description": "o1-pro: Higher-compute version of o1.",
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

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        """Normalize common provider-suffixed aliases to canonical OpenAI model IDs."""
        normalized = model_name.strip()
        lower_normalized = normalized.lower()
        for suffix in ("/openai", ":openai"):
            if lower_normalized.endswith(suffix):
                candidate = normalized[: -len(suffix)].strip()
                if candidate:
                    return candidate
        return normalized

    def __init__(self, model_name: str, *, timeout: float = 60.0) -> None:
        normalized_model_name = self._normalize_model_name(model_name)
        super().__init__(model_name=normalized_model_name, timeout=timeout)
        if normalized_model_name != model_name:
            logger.info(
                "Normalized OpenAI model name '%s' to '%s'.",
                model_name,
                normalized_model_name,
            )
        self._api_key = os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise AdapterConfigurationError("OPENAI_API_KEY environment variable is required.")
        self._base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        try:
            self._client = OpenAI(api_key=self._api_key, timeout=timeout, base_url=self._base_url)
        except TypeError as exc:
            # Compatibility fallback for environments where openai/httpx versions
            # are temporarily mismatched (e.g. unexpected 'proxies' kwarg error).
            if "unexpected keyword argument 'proxies'" not in str(exc):
                raise
            logger.warning(
                "Detected OpenAI/httpx compatibility issue (%s). "
                "Retrying with explicit httpx client.",
                exc,
            )
            http_client = httpx.Client(base_url=self._base_url, timeout=timeout)
            self._client = OpenAI(api_key=self._api_key, timeout=timeout, base_url=self._base_url, http_client=http_client)
        self._supports_max_completion_tokens = self._detect_max_completion_tokens_support()

    def _detect_max_completion_tokens_support(self) -> bool:
        """Detect whether the installed SDK exposes max_completion_tokens in chat.completions.create."""
        try:
            signature = inspect.signature(self._client.chat.completions.create)
        except (TypeError, ValueError):
            return False
        return "max_completion_tokens" in signature.parameters

    @staticmethod
    def _extract_unsupported_token_param(error_text: str) -> str | None:
        """Extract the unsupported token parameter name from SDK/API errors."""
        patterns = (
            r"unsupported parameter:\s*['\"](max(?:_completion)?_tokens)['\"]",
            r"unexpected keyword argument ['\"](max(?:_completion)?_tokens)['\"]",
        )
        for pattern in patterns:
            match = re.search(pattern, error_text)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _is_non_chat_model_error(error_text: str) -> bool:
        return "not a chat model" in error_text and "v1/completions" in error_text

    def _is_legacy_completions_model(self) -> bool:
        model = self.model_name.lower()
        return model in self.LEGACY_COMPLETIONS_MODELS or model.startswith("text-")

    def _generate_with_responses_api(
        self,
        *,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Tuple[str, str | None, Dict[str, Any], int, int]:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "input": prompt,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=self.timeout) as client:
            http_response = client.post(f"{self._base_url}/responses", headers=headers, json=payload)

        if http_response.status_code >= 400:
            message = http_response.text
            try:
                message = str(http_response.json())
            except ValueError:
                pass
            raise ModelGenerationError(
                f"OpenAI generation failed: Responses API error {http_response.status_code} - {message}"
            )

        raw = http_response.json()
        usage = raw.get("usage", {})
        prompt_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)

        output_text = raw.get("output_text", "")
        if not output_text:
            text_chunks: List[str] = []
            for item in raw.get("output", []):
                for content in item.get("content", []):
                    content_type = content.get("type")
                    if content_type in {"output_text", "text"}:
                        text = content.get("text")
                        if text:
                            text_chunks.append(text)
            output_text = "".join(text_chunks)

        finish_reason = raw.get("status")
        return output_text, finish_reason, raw, prompt_tokens, completion_tokens

    def _build_request_params(
        self,
        *,
        prompt: str,
        temperature: float,
        max_tokens: int,
        token_param: str,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if token_param == "max_completion_tokens":
            if self._supports_max_completion_tokens:
                params["max_completion_tokens"] = max_tokens
            else:
                # Older OpenAI SDKs may not expose this kwarg, but still forward extra_body.
                params["extra_body"] = {"max_completion_tokens": max_tokens}
        else:
            params["max_tokens"] = max_tokens
        return params

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
            # Reasoning models and GPT-5 family use max_completion_tokens.
            prefers_completion_tokens = (
                self.model_name.startswith("o") or self.model_name.startswith("gpt-5")
            )
            first_param = "max_completion_tokens" if prefers_completion_tokens else "max_tokens"
            second_param = "max_tokens" if first_param == "max_completion_tokens" else "max_completion_tokens"

            attempts = [first_param, second_param]
            last_exc: TypeError | OpenAIError | None = None
            response = None
            used_legacy_completions_endpoint = False
            used_responses_endpoint = False
            output_text = ""
            finish_reason = None
            prompt_tokens = 0
            completion_tokens = 0

            for index, token_param in enumerate(attempts):
                request_params = self._build_request_params(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    token_param=token_param,
                )
                try:
                    response = self._client.chat.completions.create(**request_params)
                    break
                except (TypeError, OpenAIError) as exc:
                    last_exc = exc
                    error_text = str(exc).lower()
                    if self._is_non_chat_model_error(error_text):
                        if self._is_legacy_completions_model():
                            logger.info(
                                "Model %s rejected chat endpoint; retrying with legacy completions endpoint.",
                                self.model_name,
                            )
                            response = self._client.completions.create(
                                model=self.model_name,
                                prompt=prompt,
                                temperature=temperature,
                                max_tokens=max_tokens,
                            )
                            used_legacy_completions_endpoint = True
                        else:
                            logger.info(
                                "Model %s rejected chat endpoint; retrying with Responses API endpoint.",
                                self.model_name,
                            )
                            (
                                output_text,
                                finish_reason,
                                raw,
                                prompt_tokens,
                                completion_tokens,
                            ) = self._generate_with_responses_api(
                                prompt=prompt,
                                temperature=temperature,
                                max_tokens=max_tokens,
                            )
                            used_responses_endpoint = True
                        break
                    if index == len(attempts) - 1:
                        raise
                    unsupported_param = self._extract_unsupported_token_param(error_text)
                    if unsupported_param != token_param:
                        raise
                    logger.info(
                        "Retrying with '%s' instead of '%s' for model %s",
                        second_param,
                        token_param,
                        self.model_name,
                    )

            if response is None:
                if not used_responses_endpoint:
                    assert last_exc is not None
                    raise last_exc

            # Extract the response text
            if not used_responses_endpoint:
                if not response.choices:
                    raise ModelGenerationError("OpenAI returned no choices.")

                choice = response.choices[0]
                if used_legacy_completions_endpoint:
                    output_text = getattr(choice, "text", "") or ""
                else:
                    output_text = choice.message.content or ""
                finish_reason = choice.finish_reason

                raw = response.model_dump()
                prompt_tokens = raw.get("usage", {}).get("prompt_tokens", 0)
                completion_tokens = raw.get("usage", {}).get("completion_tokens", 0)

        except (OpenAIError, TypeError) as exc:
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
