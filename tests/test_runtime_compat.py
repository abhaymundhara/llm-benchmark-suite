import importlib.metadata as metadata
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from utils.runtime_compat import patch_importlib_metadata_version


class RuntimeCompatTests(unittest.TestCase):
    def test_recovers_string_version_when_metadata_version_returns_none(self) -> None:
        fake_distributions = [
            SimpleNamespace(
                metadata={"Name": "fsspec"},
                version="2025.9.0",
            )
        ]

        def broken_version(name: str):
            if name == "fsspec":
                return None
            return metadata.version(name)

        with patch.object(metadata, "version", broken_version), patch.object(
            metadata, "distributions", lambda: fake_distributions
        ):
            patch_importlib_metadata_version()
            self.assertEqual(metadata.version("fsspec"), "2025.9.0")


if __name__ == "__main__":
    unittest.main()
