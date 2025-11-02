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
            # Remove METADATA section from tests
            test_code = record['test'].strip()
            # Remove METADATA block if present
            if 'METADATA' in test_code:
                # Split by 'def check' and keep only the check function
                parts = test_code.split('def check')
                if len(parts) > 1:
                    test_code = 'def check' + parts[1]
            
            # Format prompt: You are an expert python programmer. Here is your task: [prompt]\n\nYour code should pass these tests:\n\n{tests}
            formatted_prompt = f"You are an expert python programmer. Here is your task: {record['prompt']}\n\nYour code should pass these tests:\n\n{test_code}\n \nOnly output the code"
            
            tasks.append(
                BenchmarkTask(
                    task_id=record["task_id"],
                    prompt=formatted_prompt,
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
