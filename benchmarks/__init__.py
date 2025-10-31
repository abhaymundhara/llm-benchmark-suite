"""Benchmark registry and convenience utilities."""

from __future__ import annotations

import logging
from typing import Callable, Dict, Iterable, Type, Union

from .base import Benchmark
from .bigcodebench import BigCodeBenchmark
from .humaneval import HumanEvalBenchmark
from .mbpp import MBPPBenchmark
from .swe_bench import SWEBenchDemo
from .swe_bench_full import SWEBenchFull
from .swe_bench_official import SWEBenchOfficial

logger = logging.getLogger(__name__)


class BenchmarkRegistry:
    """Registry for benchmark implementations."""

    def __init__(self) -> None:
        self._benchmarks: Dict[str, Union[Type[Benchmark], Callable]] = {}

    def register(self, key: str, benchmark_cls: Union[Type[Benchmark], Callable]) -> None:
        key_lower = key.lower()
        if key_lower in self._benchmarks:
            logger.warning("Overwriting benchmark registration for key '%s'.", key_lower)
        self._benchmarks[key_lower] = benchmark_cls

    def create(self, key: str, **kwargs) -> Benchmark:
        benchmark_cls_or_factory = self._benchmarks.get(key.lower())
        if not benchmark_cls_or_factory:
            raise KeyError(f"No benchmark registered with key '{key}'.")
        return benchmark_cls_or_factory(**kwargs)

    def keys(self) -> Iterable[str]:
        return self._benchmarks.keys()


# Factory functions for BigCodeBench variants
def _create_bigcodebench_instruct(**kwargs):
    """Create BigCodeBench with instruct split (natural language instructions)."""
    kwargs.setdefault('split', 'instruct')
    return BigCodeBenchmark(**kwargs)


def _create_bigcodebench_complete(**kwargs):
    """Create BigCodeBench with complete split (comprehensive docstrings)."""
    kwargs.setdefault('split', 'complete')
    return BigCodeBenchmark(**kwargs)


registry = BenchmarkRegistry()
# BigCodeBench variants
registry.register("bigcodebench", _create_bigcodebench_instruct)
registry.register("bigcodebench_instruct", _create_bigcodebench_instruct)
registry.register("bigcodebench_complete", _create_bigcodebench_complete)
# Other benchmarks
registry.register("human_eval", HumanEvalBenchmark)
registry.register("mbpp", MBPPBenchmark)
registry.register("swe_bench_demo", SWEBenchDemo)
registry.register("swe_bench_full", SWEBenchFull)
registry.register("swe_bench_official", SWEBenchOfficial)

__all__ = [
    "Benchmark",
    "BigCodeBenchmark",
    "registry",
    "HumanEvalBenchmark",
    "MBPPBenchmark",
    "SWEBenchDemo",
    "SWEBenchFull",
    "SWEBenchOfficial",
]
