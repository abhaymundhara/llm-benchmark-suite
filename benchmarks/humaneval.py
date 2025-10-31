from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from datasets import Dataset, load_dataset

from .base import Benchmark, BenchmarkTask

logger = logging.getLogger(__name__)


class HumanEvalBenchmark(Benchmark):
    """Benchmark harness for the HumanEval dataset."""

    name = "human_eval"
    description = "Evaluation using the OpenAI HumanEval coding problems."

    def __init__(self, *, limit: Optional[int] = None, evaluation_timeout: float = 30.0, cache_dir: Optional[str] = None) -> None:
        super().__init__(limit=limit, evaluation_timeout=evaluation_timeout)
        self.cache_dir = cache_dir
        self._dataset: Optional[Dataset] = None

    def load_tasks(self) -> Iterable[BenchmarkTask]:
        dataset = self._load_dataset()
        tasks: List[BenchmarkTask] = []
        for record in dataset:
            tasks.append(
                BenchmarkTask(
                    task_id=record["task_id"],
                    prompt=record["prompt"],
                    tests=record["test"],
                    entry_point=record["entry_point"],
                    metadata={
                        "canonical_solution": record.get("canonical_solution"),
                        "problem_id": record.get("task_id"),
                        "docstring": record.get("prompt"),
                    },
                )
            )
        logger.info("Loaded %s HumanEval tasks", len(tasks))
        return tasks

    def _load_dataset(self) -> Dataset:
        if self._dataset is None:
            logger.info("Loading HumanEval dataset from Hugging Face.")
            self._dataset = load_dataset("openai_humaneval", split="test", cache_dir=self.cache_dir)
        return self._dataset
