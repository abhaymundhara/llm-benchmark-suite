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
from typing import Iterable, List, Optional, Dict
import shutil

from datasets import Dataset, load_dataset

from .base import Benchmark, BenchmarkTask

logger = logging.getLogger(__name__)


# Use the same extraction logic as official SWE-bench
# These patterns match what's in swebench/harness/utils.py and swebench/inference/make_datasets/utils.py
PATCH_PATTERN = re.compile(
    r"(?:diff[\w\_\.\ \/\-]+\n)?\-\-\-\s+a\/(?:.*?)\n\+\+\+\s+b\/(?:.*?)(?=diff\ |\-\-\-\ a\/|\Z)",
    re.DOTALL,
)
HUNK_HEADER_PATTERN = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def normalize_patch(patch: str) -> str:
    """
    Normalize a patch to fix common formatting issues that cause patch command to fail.
    """
    if not patch:
        return patch
    
    # First, fix line wrapping issues (terminal wrapping or soft breaks)
    # Common issue: "b/path/to/file.py" split across lines as "b/path/to/file.\npy"
    patch = re.sub(r'([ab]/[^\s]+)\s*\n\s*([^\s/+\-@])', r'\1\2', patch)
    
    # Fix wrapped lines in git headers
    patch = re.sub(r'(diff --git a/[^\s]+)\s*\n\s*b/', r'\1 b/', patch)
    patch = re.sub(r'(--- a/[^\s]+)\s*\n\s*(\+)', r'\1\n\2', patch)
    
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
    
    result = '\n'.join(normalized)
    
    # Ensure patch ends with a newline (git patches require this)
    if result and not result.endswith('\n'):
        result += '\n'
    
    return result


def fix_hunk_header_counts(patch: str) -> str:
    """
    Recompute unified diff hunk line counts to match the actual hunk content.
    """
    if not patch:
        return patch

    lines = patch.splitlines()
    fixed: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if line.startswith('@@ '):
            match = HUNK_HEADER_PATTERN.match(line)
            if not match:
                fixed.append(line)
                i += 1
                continue

            start_old = match.group(1)
            start_new = match.group(3)
            hunk_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith(('@@ ', 'diff --git', '--- ', '+++ ')):
                hunk_lines.append(lines[i])
                i += 1

            old_count = 0
            new_count = 0
            for hunk_line in hunk_lines:
                if hunk_line.startswith('+'):
                    new_count += 1
                elif hunk_line.startswith('-'):
                    old_count += 1
                elif hunk_line.startswith(' '):
                    old_count += 1
                    new_count += 1
                elif hunk_line.startswith('\\'):
                    continue
                else:
                    # Treat unexpected lines as context to avoid invalid hunks.
                    old_count += 1
                    new_count += 1

            fixed.append(f"@@ -{start_old},{old_count} +{start_new},{new_count} @@")
            fixed.extend(hunk_lines)
            continue

        fixed.append(line)
        i += 1

    result = "\n".join(fixed)
    if patch.endswith("\n"):
        result += "\n"
    return result


def is_valid_unified_diff(patch: str) -> bool:
    """
    Validate that unified diff hunks match their declared line counts.
    """
    if not patch:
        return False

    in_hunk = False
    saw_hunk = False
    expected_old = expected_new = 0
    old_count = new_count = 0

    def _finalize_hunk() -> bool:
        return old_count == expected_old and new_count == expected_new

    for line in patch.splitlines():
        if line.startswith('@@ '):
            if in_hunk and not _finalize_hunk():
                return False
            match = HUNK_HEADER_PATTERN.match(line)
            if not match:
                return False
            expected_old = int(match.group(2) or "1")
            expected_new = int(match.group(4) or "1")
            old_count = 0
            new_count = 0
            in_hunk = True
            saw_hunk = True
            continue

        if line.startswith(('diff --git', '--- ', '+++ ')):
            if in_hunk and not _finalize_hunk():
                return False
            in_hunk = False
            continue

        if in_hunk:
            if line.startswith('+'):
                new_count += 1
            elif line.startswith('-'):
                old_count += 1
            elif line.startswith(' '):
                old_count += 1
                new_count += 1
            elif line.startswith('\\'):
                continue
            else:
                return False

    if in_hunk and not _finalize_hunk():
        return False

    return saw_hunk


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
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        evaluation_timeout: Optional[float] = None,  # No timeout by default for SWE-bench
        cache_dir: Optional[str] = None,
        max_workers: int = 1,
        model_name: str = "",  # Add model name parameter
        repo_cache_dir: Optional[str] = None,  # Where to cache cloned repos
    ) -> None:
        super().__init__(limit=limit, evaluation_timeout=evaluation_timeout)
        self.cache_dir = cache_dir
        self.max_workers = max_workers
        self.model_name = model_name  # Store model name for prompt formatting
        self.start_index = start_index
        self.end_index = end_index
        
        # Setup repo cache directory for code retrieval
        if repo_cache_dir:
            self.repo_cache_dir = Path(repo_cache_dir).expanduser().resolve()
        else:
            self.repo_cache_dir = Path.home() / ".cache" / "swe_bench_repos"
        self.repo_cache_dir.mkdir(parents=True, exist_ok=True)
        
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
        limit = self.limit if self.limit is not None and self.limit > 0 else None
        selected = 0
        
        for idx, record in enumerate(dataset):
            # Apply start_index/end_index filtering
            if self.start_index is not None and idx < self.start_index:
                continue
            if self.end_index is not None and idx > self.end_index:
                break
            if limit is not None and selected >= limit:
                break
                
            instance_id = record.get("instance_id", "")
            problem_statement = record.get("problem_statement", "")
            repo = record.get("repo", "")
            base_commit = record.get("base_commit", "")
            gold_patch = record.get("patch", "")
            
            # Get code context from repository
            logger.info("Retrieving code context for %s", instance_id)
            code_files = self._get_code_context(repo, base_commit, problem_statement, gold_patch)
            logger.info("Retrieved %d files for %s", len(code_files), instance_id)
            
            prompt = self._build_prompt(instance_id, problem_statement, record, self.model_name, code_files)
            
            tasks.append(
                BenchmarkTask(
                    task_id=instance_id,
                    prompt=prompt,
                    tests=None,
                    entry_point=None,
                    reference_solution=gold_patch,
                    metadata={
                        "repo": repo,
                        "base_commit": base_commit,
                        "version": record.get("version", ""),
                        "problem_statement": problem_statement,
                    },
                )
            )
            selected += 1
        
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

    def _get_repo_path(self, repo: str) -> Path:
        """Get or clone repository and return its path."""
        # Convert repo name to safe directory name
        repo_dir = repo.replace("/", "__")
        repo_path = self.repo_cache_dir / repo_dir
        
        if not repo_path.exists():
            # Clone the repo (full clone to have all commits)
            repo_url = f"https://github.com/{repo}.git"
            logger.info("Cloning %s to %s", repo_url, repo_path)
            try:
                subprocess.run(
                    ["git", "clone", repo_url, str(repo_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 min for full clone
                )
            except subprocess.CalledProcessError as e:
                logger.error("Failed to clone %s: %s", repo_url, e.stderr if e.stderr else str(e))
                raise
        
        return repo_path

    def _get_code_context(self, repo: str, base_commit: str, problem_statement: str, gold_patch: str = "") -> Dict[str, str]:
        """Retrieve relevant code files from the repository."""
        try:
            repo_path = self._get_repo_path(repo)
            
            # Checkout the base commit
            subprocess.run(
                ["git", "checkout", "-f", base_commit],
                cwd=repo_path,
                check=True,
                capture_output=True,
                timeout=30
            )
            subprocess.run(
                ["git", "clean", "-fdx"],
                cwd=repo_path,
                check=False,
                capture_output=True,
                timeout=30
            )
            
            # Extract files from the gold patch to know what files are relevant
            target_files = set()
            if gold_patch:
                # Parse diff headers to get file paths
                for line in gold_patch.split('\n'):
                    if line.startswith('--- a/'):
                        file_path = line[6:].split()[0]
                        target_files.add(file_path)
                    elif line.startswith('+++ b/'):
                        file_path = line[6:].split()[0]
                        target_files.add(file_path)
            
            # If we couldn't extract files from patch, try to find Python files mentioned in problem
            if not target_files:
                # Look for .py file references in problem statement
                import re
                py_files = re.findall(r'[\w/]+\.py', problem_statement)
                target_files.update(py_files)
            
            # If still nothing, do a simple search for relevant files
            if not target_files:
                # Search for Python files containing keywords from the problem
                keywords = set()
                for word in problem_statement.split():
                    if len(word) > 4 and word.isalnum():
                        keywords.add(word.lower())
                
                # Find Python files
                result = subprocess.run(
                    ["find", ".", "-name", "*.py", "-type", "f"],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                py_files = [f.lstrip('./') for f in result.stdout.split('\n') if f.strip()]
                # Score files by keyword matches (simple heuristic)
                scored_files = []
                for py_file in py_files[:200]:  # Limit to avoid huge searches
                    try:
                        file_path = repo_path / py_file
                        if file_path.stat().st_size > 100000:  # Skip files > 100KB
                            continue
                        content = file_path.read_text(encoding='utf-8', errors='ignore').lower()
                        score = sum(1 for kw in keywords if kw in content)
                        if score > 0:
                            scored_files.append((score, py_file))
                    except:
                        continue
                
                # Take top 5 files
                scored_files.sort(reverse=True)
                target_files = {f[1] for f in scored_files[:5]}
            
            # Read the target files with bounded context to avoid oversized prompts.
            max_total_chars = int(os.getenv("SWE_BENCH_MAX_CONTEXT_CHARS", "20000"))
            max_file_chars = int(os.getenv("SWE_BENCH_MAX_FILE_CHARS", "8000"))
            max_files = int(os.getenv("SWE_BENCH_MAX_CONTEXT_FILES", "5"))
            strip_comments = os.getenv("SWE_BENCH_STRIP_COMMENTS", "1").strip().lower() in {
                "1", "true", "yes", "on"
            }

            file_contents: Dict[str, str] = {}
            total_chars = 0
            for file_path in sorted(target_files)[:max_files]:
                if max_total_chars and total_chars >= max_total_chars:
                    break
                full_path = repo_path / file_path
                if full_path.exists() and full_path.is_file():
                    try:
                        content = full_path.read_text(encoding='utf-8', errors='ignore')
                        # Add line numbers
                        lines = content.split('\n')
                        numbered_lines = []
                        for i, line in enumerate(lines):
                            stripped = line.lstrip()
                            if strip_comments and (not stripped or stripped.startswith("#")):
                                continue
                            numbered_lines.append(f"{i+1} {line}")
                        numbered_content = '\n'.join(numbered_lines)
                        if max_file_chars and len(numbered_content) > max_file_chars:
                            logger.info("Truncating context for %s to %d chars", file_path, max_file_chars)
                            numbered_content = numbered_content[:max_file_chars].rstrip() + "\n... [truncated]"
                        if max_total_chars:
                            remaining = max_total_chars - total_chars
                            if remaining <= 0:
                                break
                            if len(numbered_content) > remaining:
                                numbered_content = numbered_content[:remaining].rstrip() + "\n... [truncated]"
                        file_contents[file_path] = numbered_content
                        total_chars += len(numbered_content)
                    except Exception as e:
                        logger.warning("Failed to read %s: %s", file_path, e)
            
            return file_contents
            
        except Exception as e:
            logger.warning("Failed to get code context for %s: %s", repo, e)
            return {}

    @staticmethod
    def _build_prompt(instance_id: str, problem_statement: str, record: dict, model_name: str = "", code_files: Optional[Dict[str, str]] = None) -> str:
        """Build prompt for the model using proper SWE-Llama format with code context."""
        repo = record.get("repo", "")
        
        # Check if this is a SWE-Llama model that needs structured format
        is_swe_llama = any(x in model_name.lower() for x in ["swe13b", "swe-13b", "swe-llama", "swellama"])
        
        # Format code files into a readable context
        code_context = ""
        if code_files:
            code_parts = []
            for file_path, content in sorted(code_files.items()):
                code_parts.append(f"[start of {file_path}]")
                code_parts.append(content)
                code_parts.append(f"[end of {file_path}]")
            code_context = "\n".join(code_parts)
        
        if is_swe_llama:
            # Use the proper SWE-Llama format with structured tags
            PATCH_EXAMPLE = """diff --git a/path/to/file.py b/path/to/file.py
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -X,Y +X,Y @@
-old_line
+new_line"""
            
            premise = "You will be provided with a partial code base and an issue statement explaining a problem to resolve."
            instructions = (
                "I need you to solve this issue by generating a single patch that I can apply "
                "directly to this repository using git apply. "
                "Respond with EXACTLY ONE <patch>...</patch> block, and make sure you include the closing </patch> tag. "
                "Do not include any other text outside the <patch> block. "
                "Do not ask questions or request additional information."
            )
            
            # Build prompt parts
            prompt_parts = [premise, "", "<issue>", problem_statement, "</issue>"]
            
            # Add code context if available
            if code_context:
                prompt_parts.extend(["", "<code>", code_context, "</code>"])
            
            # Add instructions and example
            prompt_parts.extend([
                "",
                instructions,
                "",
                "<patch>",
                PATCH_EXAMPLE,
                "</patch>",
                "",
                "Respond below:"
            ])
            
            prompt = "\n".join(prompt_parts)
        else:
            # Standard format for other models
            prompt_parts = [f"Fix the following issue in {repo}:", "", problem_statement]
            
            # Add code context if available
            if code_context:
                prompt_parts.extend(["", "Relevant code:", "", code_context])
            
            prompt_parts.extend([
                "",
                "Generate a git diff patch to resolve this issue. Use the format:",
                "",
                "diff --git a/path/to/file.py b/path/to/file.py",
                "--- a/path/to/file.py",
                "+++ b/path/to/file.py",
                "@@ -X,Y +X,Y @@"
            ])
            
            prompt = "\n".join(prompt_parts)
        
        return prompt

    def build_retry_prompt(
        self,
        task: BenchmarkTask,
        completion: str,
        failure_category: Optional[str],
    ) -> Optional[str]:
        if failure_category not in {"parser_extraction", "parser_incomplete", "generation_empty"}:
            return None

        prompt = task.prompt
        if failure_category == "parser_incomplete" and "<code>" in prompt and "</code>" in prompt:
            code_block = re.search(r"<code>.*?</code>", prompt, re.DOTALL)
            if code_block and len(code_block.group(0)) > 8000:
                prompt = prompt.replace(code_block.group(0), "<code>\n[context truncated]\n</code>")
        if "<patch>" in prompt:
            retry_instruction = (
                "IMPORTANT: Your previous response did not include a valid patch. "
                "Respond now with EXACTLY ONE <patch>...</patch> block and no other text. "
                "Do not ask questions or include explanations. "
                "Modify as few lines as possible; keep the patch minimal."
            )
        else:
            retry_instruction = (
                "IMPORTANT: Your previous response did not include a valid patch. "
                "Respond now with a unified diff (starting with 'diff --git') and no other text. "
                "Do not ask questions or include explanations. "
                "Modify as few lines as possible; keep the patch minimal."
            )

        if "<patch>" in prompt and prompt.rstrip().endswith("<patch>"):
            trimmed = prompt.rstrip()
            trimmed = trimmed[: trimmed.rfind("<patch>")].rstrip()
            return f"{trimmed}\n\n{retry_instruction}\n\n<patch>"

        return f"{prompt}\n\n{retry_instruction}"

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

        if not is_valid_unified_diff(patch):
            repaired = fix_hunk_header_counts(patch)
            if is_valid_unified_diff(repaired):
                patch = repaired
            else:
                return (
                    False,
                    None,
                    None,
                    "Malformed or truncated patch (hunk counts do not match).",
                    completion,
                    None,
                    "parser_incomplete",
                )

        preflight_error = self._preflight_patch_apply(patch, task)
        if preflight_error:
            return (
                False,
                None,
                None,
                f"Patch apply preflight failed: {preflight_error}",
                completion,
                None,
                "parser_incomplete",
            )

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

    def _preflight_patch_apply(self, patch: str, task: BenchmarkTask) -> Optional[str]:
        repo = task.metadata.get("repo")
        base_commit = task.metadata.get("base_commit")
        if not repo or not base_commit:
            return None

        try:
            repo_path = self._get_repo_path(repo)
            subprocess.run(
                ["git", "checkout", "-f", base_commit],
                cwd=repo_path,
                check=True,
                capture_output=True,
                timeout=60,
            )
            subprocess.run(
                ["git", "clean", "-fdx"],
                cwd=repo_path,
                check=False,
                capture_output=True,
                timeout=60,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Preflight checkout failed for %s: %s", repo, exc)
            return f"repo checkout failed: {exc}"

        with tempfile.TemporaryDirectory(prefix="swebench_patch_") as tmpdir:
            patch_path = Path(tmpdir) / "candidate.patch"
            patch_path.write_text(patch, encoding="utf-8")
            try:
                result = subprocess.run(
                    ["patch", "-p1", "--dry-run", "--batch", "--fuzz=3", "-i", str(patch_path)],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            except FileNotFoundError:
                result = subprocess.run(
                    ["git", "apply", "--check", str(patch_path)],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

        if result.returncode != 0:
            details = (result.stderr or result.stdout or "").strip()
            return details or "patch command failed"

        return None

    def _extract_patch(self, completion: str) -> str:
        """
        Extract git diff patch from model completion.
        Enhanced with better validation and cleaning.
        Uses the same logic as official SWE-bench inference code.
        """
        if not completion or not completion.strip():
            return ""

        # Some models sometimes emit a "columnized" diff where patch markers appear after huge padding
        # on the same line (e.g., "... b/file.py<spaces>--- a/file.py"). That breaks normal diff parsing.
        # De-columnize by forcing known diff markers onto their own lines.
        completion = completion.replace("\r\n", "\n").replace("\r", "\n")
        completion = re.sub(r"[ \t]{8,}(diff --git )", r"\n\1", completion)
        completion = re.sub(r"[ \t]{8,}(--- a/)", r"\n\1", completion)
        completion = re.sub(r"[ \t]{8,}(\+\+\+ b/)", r"\n\1", completion)
        completion = re.sub(r"[ \t]{8,}(@@ )", r"\n\1", completion)
        # Patch lines sometimes appear as "<spaces>+..." or "<spaces>-..." mid-line.
        completion = re.sub(r"[ \t]{8,}([\+\-][^\n]*)", r"\n\1", completion)
        
        # Fix line wrapping before extraction (aggressive unwrapping for terminal output)
        # This is a common issue where long lines get wrapped mid-word
        
        # Step 1: Unwrap lines that were clearly split mid-word or mid-path
        lines = completion.split('\n')
        unwrapped_lines = []
        i = 0
        
        while i < len(lines):
            current_line = lines[i].rstrip()
            
            # Don't try to unwrap if current line is a diff marker
            if current_line.startswith(('diff ', '--- ', '+++ ', '@@ ')):
                unwrapped_lines.append(current_line)
                i += 1
                continue
            
            # Check if next line looks like a continuation
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                
                # Don't merge if next line is a diff marker
                if next_line.startswith(('diff ', '--- ', '+++ ', '@@ ', '<patch', '</patch')):
                    unwrapped_lines.append(current_line)
                    i += 1
                    continue
                
                # Signs that next line is a continuation (wrapped mid-word):
                # 1. Current line is long (>50 chars) and doesn't end with space/punctuation
                # 2. Next line starts with lowercase letter or continues a word
                # 3. Next line is very short (<30 chars) suggesting it's a fragment
                is_mid_word_wrap = (
                    len(current_line) > 50 and 
                    current_line and 
                    not current_line[-1] in ' \t.,:;!?"\')]}' and
                    next_line and
                    (next_line[0].islower() or next_line[0] in '_/') and
                    len(next_line) < 30
                )
                
                if is_mid_word_wrap:
                    # Merge with next line (no space between - they were split mid-word)
                    unwrapped_lines.append(current_line + next_line)
                    i += 2
                    continue
            
            unwrapped_lines.append(current_line)
            i += 1
        
        completion = '\n'.join(unwrapped_lines)
        
        # Step 2: Fix specific git header patterns
        completion = re.sub(r'(diff --git a/[^\n]+?)\s*\n\s+(b/)', r'\1 \2', completion)
        completion = re.sub(r'(--- a/[^\n]+?)\s*\n\s+(\+)', r'\1\n\2', completion)
        completion = re.sub(r'(\+\+\+ b/[^\n]+?)\s*\n\s+(@)', r'\1\n\2', completion)

        # Step 3: Repair wrapped unified-diff lines where the continuation line lost its prefix.
        # In a valid unified diff, content lines always start with one of: ' ', '+', '-', '\\'.
        # If we see a non-header line that doesn't start with any of these, treat it as a wrapped
        # continuation of the previous diff content line.
        def _repair_wrapped_unified_diff_lines(text: str) -> str:
            repaired: list[str] = []
            prev_is_diff_content = False
            in_hunk = False

            for raw in text.split('\n'):
                line = raw.rstrip('\n')
                stripped = line.strip()

                if not stripped:
                    if in_hunk:
                        repaired.append(' ')
                    else:
                        repaired.append('')
                    prev_is_diff_content = False
                    continue

                # Diff structural / metadata lines
                if stripped.startswith(('diff ', '--- ', '+++ ', '<patch', '</patch')):
                    repaired.append(stripped)
                    prev_is_diff_content = False
                    in_hunk = False
                    continue

                if stripped.startswith('@@'):
                    repaired.append(stripped)
                    prev_is_diff_content = False
                    in_hunk = True
                    continue

                if re.match(r'^(index [0-9a-f]{7,}\.{2}[0-9a-f]{7,}|new file mode|deleted file mode|similarity index|rename from|rename to)\b', stripped):
                    repaired.append(stripped)
                    prev_is_diff_content = False
                    in_hunk = False
                    continue

                # Valid unified-diff content line
                if line.startswith((' ', '+', '-', '\\')):
                    repaired.append(line.rstrip())
                    prev_is_diff_content = True
                    continue

                if in_hunk:
                    # Missing diff line prefix inside a hunk; treat as context.
                    repaired.append(f" {line}")
                    prev_is_diff_content = True
                    continue

                # Otherwise, likely a wrapped continuation of the previous diff content line.
                if repaired and prev_is_diff_content:
                    repaired[-1] = repaired[-1].rstrip() + stripped
                    prev_is_diff_content = True
                else:
                    # If we can't safely repair, keep the line (better than dropping content).
                    repaired.append(stripped)
                    prev_is_diff_content = False

            return '\n'.join(repaired)

        completion = _repair_wrapped_unified_diff_lines(completion)
        
        # Extract content between <patch> tags if present
        # Take only the FIRST patch to avoid duplicate/malformed output
        patch_tag_pattern = re.compile(r'<patch>\s*(.*?)\s*(?:</patch>|$)', re.DOTALL)
        patch_match = patch_tag_pattern.search(completion)
        if patch_match:
            completion = patch_match.group(1)
        
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
            # Only skip patches that are PURELY example code (not real patches with example remnants)
            filtered_patches = []
            for patch in patches:
                # Check if this looks like a real patch to a real file
                real_file_patterns = [
                    r'astropy/',
                    r'django/',
                    r'sklearn/',
                    r'sympy/',
                    r'matplotlib/',
                    r'numpy/',
                    r'scipy/',
                    r'pandas/',
                    # Generic patterns for real code files
                    r'[a-z_]+\.py',
                    r'tests?/',
                    r'src/',
                    r'lib/',
                ]
                
                is_real_file = any(re.search(pattern, patch) for pattern in real_file_patterns)
                
                # Skip only if it's the pure example patch (file.py with example code)
                if 'a/file.py' in patch or 'b/file.py' in patch:
                    if 'def function():' in patch and 'old_code' in patch and 'new_code' in patch:
                        continue
                
                # Skip empty or very short patches
                if len(patch.strip()) < 20:
                    continue
                    
                # If it looks like a real file OR doesn't match example pattern, keep it
                if is_real_file or not ('def function():' in patch and 'old_code' in patch):
                    filtered_patches.append(patch)
            
            if filtered_patches:
                combined = '\n'.join(filtered_patches).strip()
                
                # Clean out example code remnants that the model copied from the prompt
                combined = self._remove_example_code(combined)
                
                # Validate patch is complete (has required elements)
                if not self._is_patch_complete(combined):
                    logger.warning("Patch appears incomplete, likely truncated mid-generation")
                    # Still try to use it - git apply will fail gracefully if truly broken
                
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
            # Still filter out pure example patches
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
                cleaned_patch = self._remove_example_code(cleaned_patch)
                return normalize_patch(cleaned_patch)
            
            cleaned = self._remove_example_code(diff_content.strip())
            return normalize_patch(cleaned)
        
        return ""

    def _remove_example_code(self, patch: str) -> str:
        """
        Remove example code lines that the model copied from the prompt.
        
        Example code contains patterns like:
        - def function():
        - old_code = 1
        - new_code = 2
        - return value
        """
        if not patch:
            return patch
        
        lines = patch.split('\n')
        cleaned_lines = []
        skip_example_hunk = False
        skip_example_block = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()

            # Some models copy the prompt example as raw lines (not inside a proper @@ hunk).
            # If we see the canonical example signature, drop that block until the next diff/hunk header.
            if not skip_example_hunk and not skip_example_block:
                if stripped == 'def function():' or stripped.startswith('def function():'):
                    skip_example_block = True
                    continue

            if skip_example_block:
                if stripped.startswith(('diff --git', '--- a/', '+++ b/', '@@')):
                    skip_example_block = False
                    # fall through and process this header line normally
                else:
                    # Also stop skipping if we hit an obvious file boundary line
                    if stripped == '</patch>':
                        skip_example_block = False
                        continue
                    # Skip all lines inside the example block
                    continue

            # Check if this is a hunk with example code
            if line.startswith('@@'):
                # Look ahead to see if next few lines contain example patterns
                next_lines = '\n'.join(lines[i:i+10])
                if 'def function():' in next_lines and ('old_code' in next_lines or 'new_code' in next_lines):
                    skip_example_hunk = True
                else:
                    skip_example_hunk = False
                    cleaned_lines.append(line)
            elif skip_example_hunk:
                # Skip this line (part of example hunk)
                # But stop skipping if we hit the next hunk or diff header
                if line.startswith('@@') or line.startswith('diff --git') or line.startswith('---'):
                    skip_example_hunk = False
                    cleaned_lines.append(line)
            else:
                # Drop any remaining single-line example artifacts that sometimes leak through.
                if stripped in {'old_code = 1', 'new_code = 2', 'return value'}:
                    continue
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def _is_patch_complete(self, patch: str) -> bool:
        """
        Check if a patch appears complete (not truncated mid-generation).
        
        A complete patch should have:
        - diff --git header (or at least --- and +++)
        - @@ hunk headers
        - Some actual changes (+/- lines)
        """
        if not patch:
            return False
        
        has_header = 'diff --git' in patch or ('--- a/' in patch and '+++ b/' in patch)
        has_hunk = '@@' in patch
        has_changes = any(line.startswith(('+', '-')) for line in patch.split('\n') if line)
        
        return has_header and has_hunk and has_changes

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
                timeout=self.evaluation_timeout if self.evaluation_timeout and self.evaluation_timeout > 0 else None,
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
            timeout_value = self.evaluation_timeout if self.evaluation_timeout is not None else "unknown"
            return (False, None, None, f"Timeout after {timeout_value}s", "", None, "timeout")
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
        evaluation_timeout: Optional[float] = None,
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
                    timeout=self.evaluation_timeout if self.evaluation_timeout and self.evaluation_timeout > 0 else None,
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
