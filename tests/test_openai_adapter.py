import os
import types
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from models.openai_adapter import OpenAIAdapter


class _FakeResponse:
    def __init__(self, content: str = "ok") -> None:
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content), finish_reason="stop")]

    def model_dump(self) -> Dict[str, Any]:
        return {"usage": {"prompt_tokens": 11, "completion_tokens": 7}}


class _FakeCompletionResponse:
    def __init__(self, text: str = "legacy-completion-ok") -> None:
        self.choices = [types.SimpleNamespace(text=text, finish_reason="stop")]

    def model_dump(self) -> Dict[str, Any]:
        return {"usage": {"prompt_tokens": 13, "completion_tokens": 5}}


class _LegacyCompletions:
    """Simulates an older SDK signature that lacks max_completion_tokens."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int | None = None,
        extra_body: Dict[str, Any] | None = None,
    ) -> _FakeResponse:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "extra_body": extra_body,
            }
        )
        return _FakeResponse("legacy-ok")


class _ModernCompletions:
    """Simulates a newer SDK signature that supports max_completion_tokens."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        extra_body: Dict[str, Any] | None = None,
    ) -> _FakeResponse:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "max_completion_tokens": max_completion_tokens,
                "extra_body": extra_body,
            }
        )
        if max_tokens is not None:
            raise TypeError(
                "Unsupported parameter: 'max_tokens' is not supported with this model. "
                "Use 'max_completion_tokens' instead."
            )
        return _FakeResponse("modern-ok")


class _ChatRejectingCompletions:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        extra_body: Dict[str, Any] | None = None,
    ) -> _FakeResponse:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "max_completion_tokens": max_completion_tokens,
                "extra_body": extra_body,
            }
        )
        raise TypeError(
            "This is not a chat model and thus not supported in the v1/chat/completions endpoint. "
            "Did you mean to use v1/completions?"
        )


class _LegacyTextCompletions:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def create(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> _FakeCompletionResponse:
        self.calls.append(
            {
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return _FakeCompletionResponse("text-completion-ok")


class OpenAIAdapterTokenParamTests(unittest.TestCase):
    def _build_adapter(
        self,
        model_name: str,
        completions: Any,
        text_completions: Any | None = None,
    ) -> OpenAIAdapter:
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=completions),
            completions=text_completions or types.SimpleNamespace(create=lambda **_: None),
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False), patch(
            "models.openai_adapter.OpenAI", return_value=fake_client
        ):
            return OpenAIAdapter(model_name=model_name)

    def test_gpt5_uses_extra_body_on_legacy_sdk(self) -> None:
        completions = _LegacyCompletions()
        adapter = self._build_adapter("gpt-5.4", completions)

        result = adapter.generate("say hi", temperature=0.0, max_tokens=64)

        self.assertEqual(result.output_text, "legacy-ok")
        self.assertEqual(len(completions.calls), 1)
        self.assertIsNone(completions.calls[0]["max_tokens"])
        self.assertEqual(completions.calls[0]["extra_body"], {"max_completion_tokens": 64})

    def test_retries_with_max_completion_tokens_when_max_tokens_rejected(self) -> None:
        completions = _ModernCompletions()
        adapter = self._build_adapter("gpt-4.1", completions)

        result = adapter.generate("say hi", temperature=0.0, max_tokens=32)

        self.assertEqual(result.output_text, "modern-ok")
        self.assertEqual(len(completions.calls), 2)
        self.assertEqual(completions.calls[0]["max_tokens"], 32)
        self.assertIsNone(completions.calls[0]["max_completion_tokens"])
        self.assertEqual(completions.calls[1]["max_completion_tokens"], 32)
        self.assertIsNone(completions.calls[1]["max_tokens"])

    def test_normalizes_provider_suffix_model_name(self) -> None:
        completions = _LegacyCompletions()
        adapter = self._build_adapter("gpt-5.4/openai", completions)

        result = adapter.generate("say hi", temperature=0.0, max_tokens=16)

        self.assertEqual(result.output_text, "legacy-ok")
        self.assertEqual(adapter.model_name, "gpt-5.4")
        self.assertEqual(completions.calls[0]["model"], "gpt-5.4")

    def test_falls_back_to_legacy_completions_for_non_chat_model(self) -> None:
        chat_completions = _ChatRejectingCompletions()
        text_completions = _LegacyTextCompletions()
        adapter = self._build_adapter("gpt-3.5-turbo-instruct", chat_completions, text_completions)

        result = adapter.generate("say hi", temperature=0.1, max_tokens=40)

        self.assertEqual(result.output_text, "text-completion-ok")
        self.assertEqual(len(chat_completions.calls), 1)
        self.assertEqual(len(text_completions.calls), 1)
        self.assertEqual(text_completions.calls[0]["model"], "gpt-3.5-turbo-instruct")
        self.assertEqual(text_completions.calls[0]["prompt"], "say hi")
        self.assertEqual(text_completions.calls[0]["max_tokens"], 40)

    def test_non_legacy_non_chat_model_uses_responses_api_fallback(self) -> None:
        chat_completions = _ChatRejectingCompletions()
        adapter = self._build_adapter("gpt-5.3-codex", chat_completions)

        with patch.object(
            adapter,
            "_generate_with_responses_api",
            return_value=("responses-ok", "completed", {"usage": {"input_tokens": 9, "output_tokens": 4}}, 9, 4),
        ) as responses_mock:
            result = adapter.generate("say hi", temperature=0.2, max_tokens=24)

        self.assertEqual(result.output_text, "responses-ok")
        self.assertEqual(result.metrics.input_tokens, 9)
        self.assertEqual(result.metrics.output_tokens, 4)
        responses_mock.assert_called_once_with(prompt="say hi", temperature=0.2, max_tokens=24)


if __name__ == "__main__":
    unittest.main()
