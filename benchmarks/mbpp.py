from __future__ import annotations

import ast
import logging
from typing import Iterable, List, Optional

from datasets import Dataset, load_dataset

from .base import Benchmark, BenchmarkTask

logger = logging.getLogger(__name__)


class MBPPBenchmark(Benchmark):
    """Benchmark harness for the MBPP dataset."""

    name = "mbpp"
    description = "MBPP (Mostly Basic Python Problems) benchmark."

    def __init__(
        self,
        *,
        split: str = "test",
        limit: Optional[int] = None,
        evaluation_timeout: float = 30.0,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__(limit=limit, evaluation_timeout=evaluation_timeout)
        self.split = split
        self.cache_dir = cache_dir
        self._dataset: Optional[Dataset] = None

    def load_tasks(self) -> Iterable[BenchmarkTask]:
        dataset = self._load_dataset()
        tasks: List[BenchmarkTask] = []
        for idx, record in enumerate(dataset):
            prompt = record.get("text") or record.get("prompt") or ""
            code = record.get("code") or ""
            tests_list = record.get("test_list") or []
            if isinstance(tests_list, list):
                tests = "\n".join(tests_list)
            else:
                tests = str(tests_list)

            entry_point = self._extract_entry_point(code)
            task_id = record.get("task_id") or f"mbpp_{self.split}_{idx}"
            tasks.append(
                BenchmarkTask(
                    task_id=task_id,
                    tests=tests,
                    prompt=self._build_prompt(prompt, tests),
                    entry_point=entry_point,
                    metadata={
                        "canonical_solution": code,
                        "text_prompt": prompt,
                    },
                )
            )
        logger.info("Loaded %s MBPP tasks (split=%s)", len(tasks), self.split)
        return tasks

    def _load_dataset(self) -> Dataset:
        if self._dataset is None:
            logger.info("Loading MBPP dataset (split=%s) from Hugging Face.", self.split)
            self._dataset = load_dataset("mbpp", split=self.split, cache_dir=self.cache_dir)
        return self._dataset

    @staticmethod
    def _extract_entry_point(code: str) -> Optional[str]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                return node.name
        return None

    @staticmethod
    def _build_prompt(description: str, tests: str) -> str:
        header = "You are an expert Python programmer, and here is your task:"
        prompt = f"{header}{description.strip()}\n\nYour code should pass these tests:\n\n{tests.strip()}\n"
        return prompt
    