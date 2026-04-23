import unittest
from unittest.mock import Mock, patch

from models.ollama_adapter import OllamaAdapter


class OllamaThinkingTests(unittest.TestCase):
    def test_generate_disables_thinking_by_default(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "response": "def add(a, b):\n    return a + b",
            "done_reason": "stop",
            "eval_count": 12,
            "prompt_eval_count": 10,
        }

        with patch.object(OllamaAdapter, "_detect_model_context_length", return_value=262144), patch.object(
            OllamaAdapter, "list_models", return_value=[]
        ), patch("models.ollama_adapter.requests.post", return_value=response) as post_mock:
            adapter = OllamaAdapter("qwen3.5:9b")
            adapter.generate("write add", temperature=0.2, max_tokens=512)

        payload = post_mock.call_args.kwargs["json"]
        self.assertIs(payload["think"], False)


if __name__ == "__main__":
    unittest.main()
