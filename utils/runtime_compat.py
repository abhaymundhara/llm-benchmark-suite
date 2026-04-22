from __future__ import annotations

import importlib.metadata as metadata
from typing import Iterable


_PATCHED = False


def _iter_distribution_versions(package_name: str) -> Iterable[str]:
    target = package_name.lower().replace("-", "_")
    for dist in metadata.distributions():
        dist_name = (dist.metadata.get("Name") or "").lower().replace("-", "_")
        version = getattr(dist, "version", None)
        if dist_name == target and isinstance(version, str) and version.strip():
            yield version


def patch_importlib_metadata_version() -> None:
    """Recover from broken dist-info entries that make metadata.version() return None."""
    global _PATCHED
    if _PATCHED:
        return

    original_version = metadata.version

    def safe_version(package_name: str) -> str:
        version = original_version(package_name)
        if isinstance(version, str) and version.strip():
            return version

        for fallback_version in _iter_distribution_versions(package_name):
            return fallback_version

        return version

    metadata.version = safe_version
    _PATCHED = True


__all__ = ["patch_importlib_metadata_version"]
