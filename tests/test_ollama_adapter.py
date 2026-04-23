import unittest
from unittest.mock import patch

from models.ollama_adapter import OllamaAdapter


class OllamaAdapterDefaultsTests(unittest.TestCase):
    def test_suggested_generation_tokens_do_not_expand_to_full_context_window(self) -> None:
        with patch.object(OllamaAdapter, "_detect_model_context_length", return_value=262144), patch.object(
            OllamaAdapter, "list_models", return_value=[]
        ):
            adapter = OllamaAdapter("qwen3.5:9b")
            self.assertEqual(adapter.get_suggested_generation_tokens(), 4096)


if __name__ == "__main__":
    unittest.main()
