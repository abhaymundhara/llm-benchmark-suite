from __future__ import annotations

import ast
import logging
from typing import Iterable, List, Optional

from datasets import Dataset, load_dataset

from .base import Benchmark, BenchmarkTask

logger = logging.getLogger(__name__)


class BigCodeBenchmark(Benchmark):
    """
    Benchmark harness for BigCodeBench.
    
    BigCodeBench evaluates code generation with diverse function calls and complex instructions.
    It features 1,140 tasks requiring multiple library calls across 7 domains.
    
    The benchmark has two variants:
    - 'complete': Code completion based on comprehensive docstrings
    - 'instruct': Code generation based on natural language instructions
    
    Subsets:
    - 'full': All 1,140 tasks
    - 'hard': Subset of more challenging tasks
    """

    name = "bigcodebench"
    description = "BigCodeBench: Benchmarking code generation with diverse function calls and complex instructions."

    def __init__(
        self,
        *,
        limit: Optional[int] = None,
        evaluation_timeout: float = 60.0,
        cache_dir: Optional[str] = None,
        split: str = "instruct",  # 'complete' or 'instruct'
        subset: str = "full",  # 'full' or 'hard'
    ) -> None:
        """
        Initialize BigCodeBench.
        
        Args:
            limit: Maximum number of tasks to load
            evaluation_timeout: Timeout for test execution in seconds
            cache_dir: Directory to cache the dataset
            split: 'complete' for docstring-based or 'instruct' for NL instructions
            subset: 'full' for all tasks or 'hard' for challenging subset
        """
        super().__init__(limit=limit, evaluation_timeout=evaluation_timeout)
        self.cache_dir = cache_dir
        self.split = split
        self.subset = subset
        self._dataset: Optional[Dataset] = None
        
        # Validate parameters
        if split not in ["complete", "instruct"]:
            raise ValueError(f"Invalid split '{split}'. Must be 'complete' or 'instruct'.")
        if subset not in ["full", "hard"]:
            raise ValueError(f"Invalid subset '{subset}'. Must be 'full' or 'hard'.")

    def load_tasks(self) -> Iterable[BenchmarkTask]:
        """Load BigCodeBench tasks from HuggingFace."""
        dataset = self._load_dataset()
        tasks: List[BenchmarkTask] = []
        
        for record in dataset:
            # Use the appropriate prompt based on split
            if self.split == "complete":
                prompt = record.get("complete_prompt", "")
            else:  # instruct
                prompt = record.get("instruct_prompt", "")
            
            # Skip if prompt is empty
            if not prompt:
                continue
            
            # Filter for hard subset if specified
            if self.subset == "hard":
                # Hard tasks are typically marked in the task_id or have specific characteristics
                # For now, we'll load all and let the dataset filtering handle it
                # The actual BigCodeBench dataset may have a 'hard' subset indicator
                pass
            
            # Parse libs field (stored as string representation of list)
            libs_raw = record.get("libs", "[]")
            try:
                libs = ast.literal_eval(libs_raw) if isinstance(libs_raw, str) else libs_raw
            except (ValueError, SyntaxError):
                libs = []
            
            tasks.append(
                BenchmarkTask(
                    task_id=record.get("task_id", ""),
                    prompt= "You are an expert Python programmer, and here is your task: "+ prompt + "\n Only return the code",
                    tests=record.get("test", ""),
                    entry_point=record.get("entry_point", "task_func"),
                    metadata={
                        "canonical_solution": record.get("canonical_solution"),
                        "complete_prompt": record.get("complete_prompt"),
                        "instruct_prompt": record.get("instruct_prompt"),
                        "code_prompt": record.get("code_prompt"),
                        "libs": libs,
                        "doc_struct": record.get("doc_struct", {}),
                    },
                )
            )
        
        logger.info(
            "Loaded %s BigCodeBench tasks (split=%s, subset=%s)",
            len(tasks),
            self.split,
            self.subset,
        )
        return tasks

    def _load_dataset(self) -> Dataset:
        """Load the BigCodeBench dataset from HuggingFace."""
        if self._dataset is None:
            logger.info(
                "Loading BigCodeBench dataset from Hugging Face (split=%s, subset=%s).",
                self.split,
                self.subset,
            )
            
            # BigCodeBench dataset from HuggingFace
            # The dataset uses version tags like 'v0.1.0_hf', 'v0.1.4', etc.
            try:
                # Load the latest version - adjust version as needed
                self._dataset = load_dataset(
                    "bigcode/bigcodebench",
                    split="v0.1.0_hf",  # Use the appropriate version/split
                    cache_dir=self.cache_dir,
                    trust_remote_code=False,  # BigCodeBench doesn't require remote code
                )
                logger.info("Loaded BigCodeBench dataset with %s examples", len(self._dataset))
            except Exception as e:
                logger.error("Failed to load BigCodeBench dataset: %s", e)
                raise
        
        return self._dataset
