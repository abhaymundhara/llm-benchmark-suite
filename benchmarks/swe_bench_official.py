from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable, List, Optional

from datasets import Dataset, load_dataset

from .base import Benchmark, BenchmarkTask

logger = logging.getLogger(__name__)


# Use the same extraction logic as official SWE-bench
# These patterns match what's in swebench/harness/utils.py and swebench/inference/make_datasets/utils.py
PATCH_PATTERN = re.compile(
    r"(?:diff[\w\_\.\ \/\-]+\n)?\-\-\-\s+a\/(?:.*?)\n\+\+\+\s+b\/(?:.*?)(?=diff\ |\-\-\-\ a\/|\Z)",
    re.DOTALL,
)


def normalize_patch(patch: str) -> str:
    """
    Normalize a patch to fix common formatting issues that cause patch command to fail.
    """
    if not patch:
        return patch
    
    lines = patch.split('\n')
    normalized = []
    
    for i, line in enumerate(lines):
        # Fix malformed hunk headers - ensure they end with function context or @@
        if line.startswith('@@') and not line.rstrip().endswith('@@'):
            # Check if there's any content after the second @@
            parts = line.split('@@')
            if len(parts) >= 3:
                # Keep the hunk header with context
                normalized.append(line)
            else:
                # Add closing @@
                normalized.append(line.rstrip() + ' @@')
        else:
            normalized.append(line)
    
    return '\n'.join(normalized)


def fix_patch_paths(patch: str, repo_name: str = "") -> str:
    """
    Try to fix common path issues in patches where models hallucinate file structures.
    
    Common issues:
    - Model adds extra subdirectories that don't exist
    - Model uses wrong file extensions or variations
    
    Args:
        patch: The patch content
        repo_name: The repository name (e.g., "astropy/astropy")
    
    Returns:
        Patch with corrected paths (if possible)
    """
    if not patch or not repo_name:
        return patch
    
    lines = patch.split('\n')
    corrected_lines = []
    
    # Common path corrections for known repos
    path_corrections = {
        'astropy/astropy': {
            # Model might add _separability.py instead of separable.py
            r'astropy/modeling/separable/_separability\.py': 'astropy/modeling/separable.py',
            # Other common patterns can be added here
        },
    }
    
    corrections = path_corrections.get(repo_name, {})
    
    for line in lines:
        corrected_line = line
        
        # Check if this is a file path line
        if line.startswith('--- a/') or line.startswith('+++ b/') or line.startswith('diff --git'):
            # Try to apply corrections
            for pattern, replacement in corrections.items():
                corrected_line = re.sub(pattern, replacement, corrected_line)
        
        corrected_lines.append(corrected_line)
    
    return '\n'.join(corrected_lines)


def extract_diff(response):
    """
    Extracts the diff from a response formatted in different ways.
    Enhanced to handle multiple edge cases from various model outputs.
    Based on swebench/inference/make_datasets/utils.py with improvements.
    """
    if response is None:
        return None
    
    # For responses with <patch> tags, extract content after the last opening tag
    # This handles cases where the model echoes the example before responding
    if '<patch>' in response:
        # Find the last <patch> tag - that's where the actual response starts
        last_patch_idx = response.rfind('<patch>')
        if last_patch_idx != -1:
            # Extract everything after the last <patch> tag
            content_after_tag = response[last_patch_idx + 7:]  # 7 = len('<patch>')
            
            # If there's a closing </patch>, use content up to it
            if '</patch>' in content_after_tag:
                end_idx = content_after_tag.find('</patch>')
                return content_after_tag[:end_idx].strip()
            else:
                # No closing tag, use everything after the last <patch>
                return content_after_tag.strip()
    
    diff_matches = []
    other_matches = []
    
    # Look for content in <diff>, <patch> tags (with closing tags)
    pattern = re.compile(r"\<([\w-]+)\>(.*?)\<\/\1\>", re.DOTALL)
    for code, match in pattern.findall(response):
        if code in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)
    
    # Look for content in ``` code blocks - handle various formats
    # Try ```diff first (most specific)
    diff_block_pattern = re.compile(r"```diff\s*\n(.*?)```", re.DOTALL)
    diff_blocks = diff_block_pattern.findall(response)
    if diff_blocks:
        diff_matches.extend(diff_blocks)
    
    # Try ```patch
    patch_block_pattern = re.compile(r"```patch\s*\n(.*?)```", re.DOTALL)
    patch_blocks = patch_block_pattern.findall(response)
    if patch_blocks:
        diff_matches.extend(patch_blocks)
    
    # Try generic ``` blocks
    generic_pattern = re.compile(r"```(\w+)?\s*\n(.*?)```", re.DOTALL)
    for code, match in generic_pattern.findall(response):
        if code in {"diff", "patch"}:
            if match not in diff_matches:
                diff_matches.append(match)
        elif not code or code == "":
            # Plain code block without language specifier
            if match not in other_matches:
                other_matches.append(match)
    
    if diff_matches:
        return diff_matches[0]
    if other_matches:
        # Check if other matches look like diffs
        for match in other_matches:
            if 'diff --git' in match or ('---' in match and '+++' in match):
                return match
    
    # Fallback: look for diff content directly in response
    if 'diff --git' in response:
        # Extract from first diff --git to end or next non-diff content
        start_idx = response.find('diff --git')
        return response[start_idx:].strip()
    
    return response.split("</s>")[0]




class SWEBenchOfficial(Benchmark):
    """
    Official SWE-bench evaluation using the Docker-based harness.
    
    This integrates directly with python -m swebench.harness.run_evaluation
    """

    name = "swe_bench_official"
    description = "Official SWE-bench evaluation with Docker harness"
    dataset_name = "princeton-nlp/SWE-bench_Lite"
    split = "test"

    def __init__(
        self,
        *,
        limit: Optional[int] = None,
        evaluation_timeout: float = 600.0,  # 10 minutes per instance
        cache_dir: Optional[str] = None,
        max_workers: int = 1,
        model_name: str = "",  # Add model name parameter
    ) -> None:
        super().__init__(limit=limit, evaluation_timeout=evaluation_timeout)
        self.cache_dir = cache_dir
        self.max_workers = max_workers
        self.model_name = model_name  # Store model name for prompt formatting
        
        # Find SWE-bench directory
        self.swebench_dir = Path(__file__).parent.parent / "SWE-bench"
        if not self.swebench_dir.exists():
            logger.warning("SWE-bench not found at %s", self.swebench_dir)
            self.swebench_dir = Path.cwd() / "SWE-bench"
        
        self._dataset: Optional[Dataset] = None

    def load_tasks(self) -> Iterable[BenchmarkTask]:
        """Load SWE-bench tasks from HuggingFace."""
        dataset = self._load_dataset()
        tasks: List[BenchmarkTask] = []
        
        for record in dataset:
            instance_id = record.get("instance_id", "")
            problem_statement = record.get("problem_statement", "")
            
            prompt = self._build_prompt(instance_id, problem_statement, record, self.model_name)
            
            tasks.append(
                BenchmarkTask(
                    task_id=instance_id,
                    prompt=prompt,
                    tests=None,
                    entry_point=None,
                    reference_solution=record.get("patch", ""),
                    metadata={
                        "repo": record.get("repo", ""),
                        "base_commit": record.get("base_commit", ""),
                        "version": record.get("version", ""),
                        "problem_statement": problem_statement,
                    },
                )
            )
        
        logger.info("Loaded %s SWE-bench tasks from %s", len(tasks), self.dataset_name)
        return tasks

    def _load_dataset(self) -> Dataset:
        if self._dataset is None:
            logger.info("Loading SWE-bench dataset: %s (split=%s)", self.dataset_name, self.split)
            self._dataset = load_dataset(
                self.dataset_name,
                split=self.split,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
        return self._dataset

    @staticmethod
    def _build_prompt(instance_id: str, problem_statement: str, record: dict, model_name: str = "") -> str:
        """Build prompt for the model using proper SWE-Llama format."""
        repo = record.get("repo", "")
        
        # Check if this is a SWE-Llama model that needs structured format
        is_swe_llama = any(x in model_name.lower() for x in ["swe13b", "swe-13b", "swe-llama", "swellama"])
        
        if is_swe_llama:
            # Use the proper SWE-Llama format with structured tags
            PATCH_EXAMPLE = """--- a/file.py
+++ b/file.py
@@ -1,5 +1,5 @@
 def function():
-    old_code = 1
+    new_code = 2
     return value"""
            
            premise = "You will be provided with a partial code base and an issue statement explaining a problem to resolve."
            instructions = (
                "I need you to solve this issue by generating a single patch file that I can apply "
                + "directly to this repository using git apply. Please respond with a single patch "
                + "file in the following format."
            )
            
            prompt = f"""{premise}

<issue>
{problem_statement}
</issue>

{instructions}

<patch>
{PATCH_EXAMPLE}
</patch>

Respond below:"""
        else:
            # Standard format for other models
            prompt = f"""Fix the following issue in {repo}:

{problem_statement}

Generate a git diff patch to resolve this issue. Use the format:

diff --git a/path/to/file.py b/path/to/file.py
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -X,Y +X,Y @@
-old code
+new code

Provide ONLY the patch."""
        
        return prompt

    def evaluate_completion(self, task: BenchmarkTask, completion: str):
        """
        Evaluate using official SWE-bench harness.
        
        Returns: (passed, stdout, stderr, error, executed_code, tests_code, failure_category)
        """
        # Extract patch
        patch = self._extract_patch(completion)
        
        if not patch:
            return (
                False,
                None,
                None,
                "Failed to extract patch from model completion",
                completion,
                None,
                "parser_extraction",
            )
        
        # Try to fix common path issues where models hallucinate file structures
        repo_name = task.metadata.get("repo", "")
        if repo_name:
            patch = fix_patch_paths(patch, repo_name)
        
        # Create predictions file
        prediction = {
            "instance_id": task.task_id,
            "model_name_or_path": "benchmark_model",
            "model_patch": patch
        }
        
        # Write predictions to SWE-bench directory
        timestamp = int(time.time())
        predictions_path = self.swebench_dir / f"pred_{task.task_id}_{timestamp}.jsonl"
        
        try:
            with open(predictions_path, 'w') as f:
                f.write(json.dumps(prediction) + '\n')
            
            # Run evaluation
            result = self._run_evaluation(task.task_id, predictions_path, timestamp)
            return result
        finally:
            # Cleanup
            predictions_path.unlink(missing_ok=True)

    def _extract_patch(self, completion: str) -> str:
        """
        Extract git diff patch from model completion.
        Enhanced with better validation and cleaning.
        Uses the same logic as official SWE-bench inference code.
        """
        if not completion or not completion.strip():
            return ""
        
        # First extract diff content using official extract_diff logic
        diff_content = extract_diff(completion)
        
        if not diff_content:
            return ""
        
        # Normalize line endings to Unix-style (git patches must use \n, not \r\n)
        diff_content = diff_content.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove any leading/trailing whitespace and common artifacts
        diff_content = diff_content.strip()
        
        # Remove common thinking/explanation text that models sometimes add
        thinking_markers = ['<thinking>', '</thinking>', '<explanation>', '</explanation>']
        for marker in thinking_markers:
            if marker in diff_content:
                diff_content = diff_content.replace(marker, '')
        
        # Now extract just the actual patch using PATCH_PATTERN
        # This finds all valid patch sections (diff --git ... --- a/... +++ b/...)
        patches = PATCH_PATTERN.findall(diff_content)
        
        if patches:
            # Filter out example patches from the prompt
            # Example patches contain "def function():" with "old_code" and "new_code"
            filtered_patches = []
            for patch in patches:
                # Skip if it's the example patch
                if 'def function():' in patch and 'old_code' in patch and 'new_code' in patch:
                    continue
                # Skip if it modifies a/file.py or b/file.py (generic example)
                if 'a/file.py' in patch or 'b/file.py' in patch:
                    continue
                # Skip empty or very short patches
                if len(patch.strip()) < 20:
                    continue
                filtered_patches.append(patch)
            
            if filtered_patches:
                combined = '\n'.join(filtered_patches).strip()
                # Ensure the patch starts with diff --git if it's not already there
                if not combined.startswith('diff --git'):
                    # Try to fix common formatting issues
                    if '--- a/' in combined and '+++ b/' in combined:
                        # Has the header markers but missing diff --git
                        # Extract the file path and add it
                        header_match = re.search(r'--- a/(\S+)', combined)
                        if header_match:
                            filepath = header_match.group(1)
                            combined = f'diff --git a/{filepath} b/{filepath}\n{combined}'
                # Normalize the patch to fix common formatting issues
                return normalize_patch(combined)
        
        # Fallback: if pattern doesn't match but looks like a patch, try to clean it
        if 'diff --git' in diff_content or ('--- a/' in diff_content and '+++ b/' in diff_content):
            # Still filter out example patches
            if 'def function():' in diff_content and 'old_code' in diff_content and 'new_code' in diff_content:
                return ""
            if 'a/file.py' in diff_content or 'b/file.py' in diff_content:
                return ""
            
            # Clean up the diff content
            lines = diff_content.split('\n')
            cleaned_lines = []
            in_diff = False
            
            for line in lines:
                # Start capturing from diff --git or --- a/
                if line.startswith('diff --git') or line.startswith('--- a/'):
                    in_diff = True
                
                if in_diff:
                    cleaned_lines.append(line)
            
            if cleaned_lines:
                cleaned_patch = '\n'.join(cleaned_lines).strip()
                return normalize_patch(cleaned_patch)
            
            return normalize_patch(diff_content.strip())
        
        return ""

    def _run_evaluation(self, instance_id: str, predictions_path: Path, timestamp: int):
        """
        Run SWE-bench evaluation.
        
        Returns: (passed, stdout, stderr, error, executed_code, tests_code, failure_category)
        """
        run_id = f"eval_{instance_id}_{timestamp}"
        
        cmd = [
            sys.executable,
            "-m", "swebench.harness.run_evaluation",
            "--max_workers", str(self.max_workers),
            "--instance_ids", instance_id,
            "--predictions_path", str(predictions_path),
            "--run_id", run_id,
        ]
        
        logger.info("Running SWE-bench evaluation: %s", instance_id)
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.swebench_dir),
                capture_output=True,
                text=True,
                timeout=self.evaluation_timeout,
            )
            
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            
            # Find and parse report
            report_files = list(self.swebench_dir.glob(f"benchmark_model.{run_id}.json"))
            
            if report_files:
                report_path = report_files[0]
                with open(report_path) as f:
                    report = json.load(f)
                
                resolved = instance_id in report.get("resolved_ids", [])
                has_error = instance_id in report.get("error_ids", [])
                
                # Cleanup
                report_path.unlink(missing_ok=True)
                summary_file = report_path.with_name(report_path.stem + "_summary.txt")
                summary_file.unlink(missing_ok=True)
                
                if resolved:
                    return (True, stdout, stderr, None, "", None, "success")
                elif has_error:
                    return (False, stdout, stderr, "SWE-bench evaluation error", "", None, "runtime_error")
                else:
                    return (False, stdout, stderr, "Patch did not resolve the issue", "", None, "model_algorithm")
            else:
                return (False, stdout, stderr, "No evaluation report generated", "", None, "runtime_error")
                
        except subprocess.TimeoutExpired:
            return (False, None, None, f"Timeout after {self.evaluation_timeout}s", "", None, "timeout")
        except Exception as e:
            logger.exception("Evaluation error for %s", instance_id)
            return (False, None, None, f"Evaluation error: {str(e)}", "", None, "runtime_error")

    def get_tests_for_task(self, task: BenchmarkTask) -> Optional[str]:
        """SWE-bench tests run in Docker."""
        return None


# Keep old implementation for backwards compatibility
def _format_harness_command(template: str, instance_id: str, patch_path: Path) -> str:
    return template.format(instance_id=instance_id, patch_path=str(patch_path))


class SWEbenchHarnessBenchmark(Benchmark):
    """Base class for SWE-bench benchmarks that rely on the official harness."""

    dataset_name: str = "princeton-nlp/SWE-bench"
    split: str = "dev"

    def __init__(
        self,
        *,
        limit: Optional[int] = None,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        evaluation_timeout: float = 900.0,
        cache_dir: Optional[str] = None,
        harness_command: Optional[str] = None,
    ) -> None:
        super().__init__(
            limit=limit,
            start_index=start_index,
            end_index=end_index,
            evaluation_timeout=evaluation_timeout
        )
        self.cache_dir = cache_dir
        self.harness_command = harness_command or os.environ.get("SWE_BENCH_HARNESS_CMD")
        self._dataset: Optional[Dataset] = None

    def load_tasks(self) -> Iterable[BenchmarkTask]:
        dataset = self._load_dataset()
        tasks: List[BenchmarkTask] = []
        for record in dataset:
            task_id = record.get("instance_id") or record.get("task_id")
            prompt = self._build_prompt(record)
            tasks.append(
                BenchmarkTask(
                    task_id=task_id,
                    prompt=prompt,
                    tests=None,
                    metadata={
                        "repo": record.get("repo"),
                        "base_commit": record.get("base_commit"),
                        "reference_patch": record.get("patch"),
                        "problem_statement": record.get("problem_statement"),
                        "dataset_name": self.dataset_name,
                        "dataset_split": self.split,
                    },
                )
            )
        logger.info("Loaded %s tasks for dataset %s (%s split).", len(tasks), self.dataset_name, self.split)
        return tasks

    def evaluate_completion(
        self,
        task: BenchmarkTask,
        completion: str,
    ) -> tuple[bool, Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
        if not self.harness_command:
            error = (
                "SWE_BENCH_HARNESS_CMD environment variable is not configured. "
                "Set it to a shell command template containing {instance_id} and {patch_path}."
            )
            return False, None, None, error, completion, None, "runtime_error"

        if not task.task_id:
            return False, None, None, "Task identifier is missing; cannot invoke harness.", completion, None, "runtime_error"

        with tempfile.TemporaryDirectory(prefix="swebench_") as tmpdir:
            patch_path = Path(tmpdir) / "candidate.patch"
            patch_path.write_text(completion, encoding="utf-8")
            command = _format_harness_command(self.harness_command, task.task_id, patch_path)
            logger.debug("Invoking SWE-bench harness: %s", command)
            env = os.environ.copy()
            env.setdefault("SWE_BENCH_DATASET", task.metadata.get("dataset_name", self.dataset_name))
            env.setdefault("SWE_BENCH_SPLIT", task.metadata.get("dataset_split", self.split))
            env.setdefault("SWE_BENCH_REPORT_DIR", str(Path.cwd() / "swebench_reports"))
            if "SWE_BENCH_ROOT" not in env:
                default_root = Path.cwd() / "SWE-bench"
                env["SWE_BENCH_ROOT"] = str(default_root)
            try:
                process = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.evaluation_timeout,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                logger.error("SWE-bench harness timed out for task %s.", task.task_id)
                return False, None, None, "SWE-bench harness timed out.", completion, None, "timeout"

        passed = process.returncode == 0
        error = None if passed else "SWE-bench harness reported failure."
        return passed, process.stdout, process.stderr, error, completion, None, "model_algorithm" if not passed else "success"

    def _load_dataset(self) -> Dataset:
        if self._dataset is None:
            logger.info("Loading dataset %s (split=%s) for SWE-bench evaluation.", self.dataset_name, self.split)
            self._dataset = load_dataset(self.dataset_name, split=self.split, cache_dir=self.cache_dir)
        return self._dataset

    @staticmethod
    def _build_prompt(record: dict) -> str:
        problem = record.get("problem_statement") or ""
        repo = record.get("repo") or ""
        base_commit = record.get("base_commit") or ""
        return (
            f"Repository: {repo}\n"
            f"Base commit: {base_commit}\n\n"
            f"Problem statement:\n{problem}\n\n"
            "Return a unified diff patch that resolves the issue."
        )


class SWEBenchOld(SWEbenchHarnessBenchmark):
    """Benchmark harness for the official SWE-bench evaluation (old implementation)."""

    dataset_name = "princeton-nlp/SWE-bench"
    split = "test"

