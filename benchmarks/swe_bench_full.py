from __future__ import annotations

from typing import Optional

from .swe_bench_official import SWEbenchHarnessBenchmark


class SWEBenchFull(SWEbenchHarnessBenchmark):
    """Full SWE-bench benchmark that relies on the official harness with Docker."""

    dataset_name = "princeton-nlp/SWE-bench"
    split = "train"

    def __init__(
        self,
        *,
        limit: Optional[int] = None,
        evaluation_timeout: float = 1800.0,
        cache_dir: Optional[str] = None,
        harness_command: Optional[str] = None,
    ) -> None:
        super().__init__(
            limit=limit,
            evaluation_timeout=evaluation_timeout,
            cache_dir=cache_dir,
            harness_command=harness_command,
        )
