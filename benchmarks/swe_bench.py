from __future__ import annotations

import difflib
import logging
from typing import Iterable, List, Optional

from datasets import Dataset, load_dataset

from .base import Benchmark, BenchmarkTask

logger = logging.getLogger(__name__)


class SWEBenchDemo(Benchmark):
    """Lightweight SWE-bench evaluation using similarity scoring."""

    name = "swe_bench_demo"
    description = "Approximate SWE-bench evaluation using similarity metrics against reference patches."

    def __init__(
        self,
        *,
        split: str = "test",
        limit: Optional[int] = 5,
        evaluation_timeout: float = 60.0,
        cache_dir: Optional[str] = None,
        similarity_threshold: float = 0.8,
    ) -> None:
        super().__init__(limit=limit, evaluation_timeout=evaluation_timeout)
        self.split = split
        self.cache_dir = cache_dir
        self.similarity_threshold = similarity_threshold
        self._dataset: Optional[Dataset] = None

    def load_tasks(self) -> Iterable[BenchmarkTask]:
        dataset = self._load_dataset()
        tasks: List[BenchmarkTask] = []
        for record in dataset:
            prompt = self._build_prompt(record)
            patch = record.get("patch") or record.get("solution_patch") or ""
            task_id = record.get("instance_id") or record.get("task_id") or record.get("commit_id")
            tasks.append(
                BenchmarkTask(
                    task_id=task_id,
                    prompt=prompt,
                    tests=None,
                    metadata={
                        "repo": record.get("repo"),
                        "base_commit": record.get("base_commit"),
                        "problem_statement": record.get("problem_statement"),
                        "reference_patch": patch,
                    },
                )
            )
        logger.info("Loaded %s SWE-bench demo tasks", len(tasks))
        return tasks

    def evaluate_completion(
        self,
        task: BenchmarkTask,
        completion: str,
    ) -> tuple[bool, Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
        reference_patch = task.metadata.get("reference_patch") or ""
        similarity = self._patch_similarity(reference_patch, completion)
        passed = similarity >= self.similarity_threshold
        stdout = f"Patch similarity score: {similarity:.2f}"
        error = None if passed else "Patch similarity below threshold; manual verification recommended."
        return passed, stdout, None, error, completion, None

    def _load_dataset(self) -> Dataset:
        if self._dataset is None:
            logger.info("Loading SWE-bench Lite dataset for demo (split=%s).", self.split)
            self._dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split=self.split, cache_dir=self.cache_dir)
        return self._dataset

    @staticmethod
    def _build_prompt(record: dict) -> str:
        problem = record.get("problem_statement") or ""
        repo = record.get("repo") or ""
        base_commit = record.get("base_commit") or ""
        return (
            f"You are assisting with a bug fix in the repository '{repo}'. "
            f"The current base commit is {base_commit}.\n\n"
            f"Problem description:\n{problem}\n\n"
            "Produce a unified diff patch that resolves the issue."
        )

    @staticmethod
    def _patch_similarity(reference_patch: str, candidate_patch: str) -> float:
        if not reference_patch or not candidate_patch:
            return 0.0
        matcher = difflib.SequenceMatcher(None, reference_patch, candidate_patch)
        return matcher.ratio()
