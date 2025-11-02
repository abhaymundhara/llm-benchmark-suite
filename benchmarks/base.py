from __future__ import annotations

import abc
import ast
import json
import logging
import subprocess
import sys
import tempfile
import textwrap
import time
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence

from models.base import BaseModelAdapter, GenerationMetrics

logger = logging.getLogger(__name__)


class ProgressCallback(Protocol):
    """Interface for reporting progress while running benchmarks."""

    def __call__(self, result: "TaskResult") -> None:
        ...


@dataclass(frozen=True)
class BenchmarkTask:
    """Represents a single benchmark task prompt and associated metadata."""

    task_id: str
    prompt: str
    tests: Optional[str] = None
    entry_point: Optional[str] = None
    reference_solution: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Outcome of a single benchmark task execution."""

    task_id: str
    prompt: str
    completion: str
    passed: bool
    error: Optional[str]
    evaluation_stdout: Optional[str]
    evaluation_stderr: Optional[str]
    generation_metrics: GenerationMetrics
    metadata: Dict[str, Any] = field(default_factory=dict)
    evaluated_code: Optional[str] = None
    tests_code: Optional[str] = None
    failure_category: Optional[str] = None  # NEW: categorizes why it failed


@dataclass
class BenchmarkReport:
    """Aggregate benchmark results."""

    benchmark_name: str
    model_name: str
    started_at: float
    completed_at: float
    task_results: List[TaskResult]
    metrics: Dict[str, Any]

    def to_json(self) -> str:
        """Serialize report to JSON."""
        return json.dumps(
            {
                "benchmark_name": self.benchmark_name,
                "model_name": self.model_name,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "metrics": self.metrics,
                "task_results": [
                    {
                        "task_id": result.task_id,
                        "prompt": result.prompt,  # ADDED
                        "model_completion": result.completion,  # ADDED
                        "passed": result.passed,
                        "error": result.error,
                        "evaluation_stdout": result.evaluation_stdout,
                        "evaluation_stderr": result.evaluation_stderr,
                        "generation_metrics": result.generation_metrics.__dict__,
                        "metadata": result.metadata,
                        "evaluated_code": result.evaluated_code,
                        "tests_code": result.tests_code,
                        "failure_category": result.failure_category,  # NEW
                    }
                    for result in self.task_results
                ],
            },
            indent=2,
            sort_keys=True,
        )


class BenchmarkExecutionError(RuntimeError):
    """Raised when benchmark execution fails."""


class Benchmark(abc.ABC):
    """Abstract base class for benchmarks."""

    name: str = "benchmark"
    description: str = "Generic benchmark"

    def __init__(self, *, limit: Optional[int] = None, evaluation_timeout: float = 30.0) -> None:
        self.limit = limit
        self.evaluation_timeout = evaluation_timeout

    @abc.abstractmethod
    def load_tasks(self) -> Iterable[BenchmarkTask]:
        """Load tasks associated with this benchmark."""

    def run(
        self,
        adapter: BaseModelAdapter,
        *,
        temperature: float,
        max_tokens: int,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> BenchmarkReport:
        """Execute the benchmark for the selected model adapter."""
        started_at = time.time()
        task_results: List[TaskResult] = []
        if self.limit is not None:
            if self.limit <= 0:
                tasks = []
            else:
                tasks = []
                for task in self.load_tasks():
                    tasks.append(task)
                    if len(tasks) >= self.limit:
                        break
        else:
            tasks = list(self.load_tasks())

        for task in tasks:
            logger.debug("Evaluating task %s with model %s", task.task_id, adapter.model_name)
            try:
                generation_result = adapter.generate(
                    prompt=task.prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                completion_text = generation_result.output_text
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("Generation failed for task %s: %s", task.task_id, exc)
                task_result = TaskResult(
                    task_id=task.task_id,
                    prompt=task.prompt,
                    completion="",
                    passed=False,
                    error=str(exc),
                    evaluation_stdout=None,
                    evaluation_stderr=None,
                    generation_metrics=GenerationMetrics(),
                    metadata=task.metadata,
                    evaluated_code=None,
                    tests_code=self.get_tests_for_task(task),
                )
            else:
                (
                    passed,
                    stdout,
                    stderr,
                    evaluation_error,
                    executed_code,
                    tests_code,
                    failure_category,
                ) = self.evaluate_completion(task, completion_text)
                task_result = TaskResult(
                    task_id=task.task_id,
                    prompt=task.prompt,
                    completion=completion_text,
                    passed=passed,
                    error=evaluation_error,
                    evaluation_stdout=stdout,
                    evaluation_stderr=stderr,
                    generation_metrics=generation_result.metrics,
                    metadata=task.metadata,
                    evaluated_code=executed_code,
                    tests_code=tests_code,
                    failure_category=failure_category if not passed else None,
                )

            if not task_result.passed:
                logger.warning(
                    "Task %s failed. Error: %s",
                    task_result.task_id,
                    task_result.error or "Unknown failure.",
                )
                if task_result.evaluation_stdout:
                    logger.debug("Task %s stdout:\n%s", task_result.task_id, task_result.evaluation_stdout)
                if task_result.evaluation_stderr:
                    logger.debug("Task %s stderr:\n%s", task_result.task_id, task_result.evaluation_stderr)

            task_results.append(task_result)
            if progress_callback:
                progress_callback(task_result)

        completed_at = time.time()
        metrics = self._aggregate_metrics(task_results)
        return BenchmarkReport(
            benchmark_name=self.name,
            model_name=adapter.model_name,
            started_at=started_at,
            completed_at=completed_at,
            task_results=task_results,
            metrics=metrics,
        )

    def evaluate_completion(
        self,
        task: BenchmarkTask,
        completion: str,
    ) -> tuple[bool, Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Evaluate a completion using the benchmark's test harness.
        
        Returns:
            (passed, stdout, stderr, error, executed_code, tests_code, failure_category)
        """
        entry_point = task.entry_point or task.metadata.get("entry_point")
        executed_code = self._extract_code_for_execution(completion, entry_point, task.prompt)
        tests_code = self.get_tests_for_task(task)
        
        # Categorize the extraction/compilation
        failure_category = self._categorize_extraction(completion, executed_code, entry_point)
        
        if not task.tests:
            logger.debug("No tests provided for task %s; treating as pass.", task.task_id)
            return True, None, None, None, executed_code, tests_code, failure_category

        try:
            result = self._evaluate_python_completion(task, executed_code, tests_code)
            # Add failure category to the result
            passed = result[0]
            if not passed and failure_category == "success":
                # If extraction/compilation succeeded but tests failed, it's a model algorithm issue
                failure_category = "model_algorithm"
            return result + (failure_category,)
        except subprocess.TimeoutExpired:
            logger.warning("Evaluation timed out for task %s", task.task_id)
            if failure_category == "success":
                failure_category = "timeout"
            return False, None, None, "Evaluation timed out.", executed_code, tests_code, failure_category
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Evaluation failed for task %s: %s", task.task_id, exc)
            if failure_category == "success":
                failure_category = "runtime_error"
            return False, None, None, f"Evaluation error: {exc}", executed_code, tests_code, failure_category
    
    @staticmethod
    def _categorize_extraction(completion: str, extracted_code: str, entry_point: Optional[str]) -> str:
        """
        Categorize the extraction result to identify parser issues vs model issues.
        
        Returns one of:
        - "success": Code extracted and compiles successfully
        - "parser_extraction": Failed to extract any meaningful code
        - "parser_syntax": Extracted code has syntax errors
        - "parser_incomplete": Missing entry point function or incomplete code
        """
        # Check if extraction failed
        if not extracted_code or len(extracted_code.strip()) < 10:
            return "parser_extraction"
        
        # Check for syntax errors
        try:
            compile(extracted_code, '<string>', 'exec')
        except SyntaxError:
            return "parser_syntax"
        
        # Check if entry point function exists (if required)
        if entry_point:
            if not re.search(rf'\bdef\s+{re.escape(entry_point)}\s*\(', extracted_code):
                return "parser_incomplete"
        
        # Extraction and compilation successful
        return "success"

    def _evaluate_python_completion(
        self, task: BenchmarkTask, executed_code: str, tests_code: Optional[str]
    ) -> tuple[bool, Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Evaluate Python completion by executing tests in an isolated subprocess."""
        with tempfile.TemporaryDirectory(prefix="benchmark_") as tmpdir:
            tmp_path = Path(tmpdir)
            solution_path = tmp_path / "solution.py"
            evaluation_path = tmp_path / "evaluate.py"

            solution_code = executed_code
            if task.reference_solution and task.reference_solution not in solution_code:
                # Provide prompt context if the model returned only the body.
                solution_code = task.reference_solution + "\n" + solution_code

            solution_path.write_text(solution_code, encoding="utf-8")

            entry_point = task.entry_point or task.metadata.get("entry_point")
            test_body = task.tests or ""

            eval_script = textwrap.dedent(
                f"""
                import importlib.util
                import sys
                from pathlib import Path

                solution_module_path = Path(r"{solution_path}")
                spec = importlib.util.spec_from_file_location("solution", solution_module_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules["solution"] = module
                spec.loader.exec_module(module)  # type: ignore[union-attr]

                def run_tests():
                    # Import all names from solution module into test namespace
                    namespace = {{k: v for k, v in module.__dict__.items() if not k.startswith('_')}}
                    exec(compile({test_body!r}, "tests.py", "exec"), namespace, namespace)
                    entry_point = {entry_point!r}
                    if entry_point:
                        candidate = getattr(module, entry_point)
                        check_fn = namespace.get("check")
                        if callable(check_fn):
                            check_fn(candidate)
                    return namespace

                if __name__ == "__main__":
                    run_tests()
                """
            )
            evaluation_path.write_text(eval_script, encoding="utf-8")

            process = subprocess.run(
                [sys.executable, str(evaluation_path)],
                capture_output=True,
                text=True,
                timeout=self.evaluation_timeout,
            )
            passed = process.returncode == 0
            error = None if passed else "Tests failed."
            return passed, process.stdout or None, process.stderr or None, error, solution_code, tests_code

    @staticmethod
    def _aggregate_metrics(task_results: Sequence[TaskResult]) -> Dict[str, Any]:
        total = len(task_results)
        passed = sum(1 for result in task_results if result.passed)
        latencies = [
            result.generation_metrics.latency_ms
            for result in task_results
            if result.generation_metrics.latency_ms is not None
        ]
        total_latency = sum(latencies) if latencies else None
        avg_latency = (total_latency / len(latencies)) if latencies else None
        
        # Aggregate token usage
        input_tokens_list = [
            result.generation_metrics.input_tokens
            for result in task_results
            if result.generation_metrics.input_tokens is not None
        ]
        output_tokens_list = [
            result.generation_metrics.output_tokens
            for result in task_results
            if result.generation_metrics.output_tokens is not None
        ]
        
        total_input_tokens = sum(input_tokens_list) if input_tokens_list else None
        total_output_tokens = sum(output_tokens_list) if output_tokens_list else None
        total_tokens = None
        if total_input_tokens is not None and total_output_tokens is not None:
            total_tokens = total_input_tokens + total_output_tokens
        
        avg_input_tokens = (total_input_tokens / len(input_tokens_list)) if input_tokens_list else None
        avg_output_tokens = (total_output_tokens / len(output_tokens_list)) if output_tokens_list else None
        
        # Count failures by category
        failure_categories = {}
        for result in task_results:
            if not result.passed and result.failure_category:
                category = result.failure_category
                failure_categories[category] = failure_categories.get(category, 0) + 1
        
        metrics = {
            "total_tasks": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": (passed / total) if total else 0.0,
            "average_latency_ms": avg_latency,
            "total_latency_ms": total_latency,
            "failure_categories": failure_categories,
        }
        
        # Add token metrics if available
        if total_input_tokens is not None:
            metrics["total_input_tokens"] = total_input_tokens
        if total_output_tokens is not None:
            metrics["total_output_tokens"] = total_output_tokens
        if total_tokens is not None:
            metrics["total_tokens"] = total_tokens
        if avg_input_tokens is not None:
            metrics["average_input_tokens"] = avg_input_tokens
        if avg_output_tokens is not None:
            metrics["average_output_tokens"] = avg_output_tokens
        
        return metrics

    @staticmethod
    def _extract_code_for_execution(
        completion: str,
        entry_point: Optional[str],
        prompt: Optional[str] = None,
    ) -> str:
        """
        Enhanced code extraction that works with any model output format.
        Handles: GPT, Claude, Llama, Gemini, and other LLMs with different styles.
        """
        prompt_text = prompt or ""
        
        # Strategy 1: Extract from markdown code blocks (most common format)
        extracted = Benchmark._extract_from_code_blocks(completion, entry_point)
        if extracted:
            return Benchmark._finalize_code_for_execution(extracted, prompt_text)
        
        # Strategy 2: Extract from XML-style tags (Claude, some Llama formats)
        extracted = Benchmark._extract_from_xml_tags(completion, entry_point)
        if extracted:
            return Benchmark._finalize_code_for_execution(extracted, prompt_text)
        
        # Strategy 3: Extract from labeled sections (Llama often uses this)
        extracted = Benchmark._extract_from_labeled_sections(completion, entry_point)
        if extracted:
            return Benchmark._finalize_code_for_execution(extracted, prompt_text)
        
        # Strategy 4: Intelligent line-by-line extraction (fallback)
        extracted = Benchmark._extract_by_code_detection(completion, entry_point)
        if extracted:
            return Benchmark._finalize_code_for_execution(extracted, prompt_text)
        
        # Final fallback: Return cleaned completion
        return Benchmark._finalize_code_for_execution(completion.strip(), prompt_text)

    @staticmethod
    def _extract_from_code_blocks(completion: str, entry_point: Optional[str]) -> Optional[str]:
        """Extract code from markdown-style code blocks (```python ... ```)."""
        # Match various markdown fence patterns
        patterns = [
            r"```python\s*(.*?)```",  # ```python
            r"```py\s*(.*?)```",      # ```py
            r"```\s*(.*?)```",        # ``` (generic)
        ]
        
        all_blocks = []
        for pattern in patterns:
            fence_pattern = re.compile(pattern, re.IGNORECASE | re.DOTALL)
            matches = fence_pattern.findall(completion)
            for match in matches:
                # Properly dedent and strip the code
                dedented = textwrap.dedent(match)
                stripped = dedented.strip()
                if stripped:
                    all_blocks.append(stripped)
        
        # Handle incomplete code blocks (missing closing fence)
        if not all_blocks:
            # Check if there's an opening fence without a closing one
            incomplete_patterns = [
                r"```python\s*(.*)",  # ```python to end of string
                r"```py\s*(.*)",      # ```py to end of string
                r"```\s*(.*)",        # ``` to end of string (least specific)
            ]
            
            for pattern in incomplete_patterns:
                incomplete_pattern = re.compile(pattern, re.IGNORECASE | re.DOTALL)
                match = incomplete_pattern.search(completion)
                if match:
                    code = match.group(1)
                    # Clean up the incomplete code
                    dedented = textwrap.dedent(code)
                    stripped = dedented.strip()
                    
                    # Remove incomplete trailing lines (usually truncated comments or strings)
                    lines = stripped.split('\n')
                    if lines:
                        last_line = lines[-1].strip()
                        # Check if last line looks incomplete
                        incomplete_indicators = []
                        
                        # Check for inline or standalone comments
                        if '#' in last_line:
                            # Extract comment part (everything after #)
                            comment_part = last_line[last_line.index('#'):].rstrip()
                            # Check if comment doesn't end naturally
                            if not any(comment_part.endswith(end) for end in 
                                     ('.', '!', '?', ')', ']', '}', '"', "'", ':', ';', ',')):
                                incomplete_indicators.append(True)
                        
                        # Unclosed string
                        if last_line.count('"') % 2 == 1 or last_line.count("'") % 2 == 1:
                            incomplete_indicators.append(True)
                        
                        # Ends with continuation keywords
                        if last_line.rstrip().endswith(('and', 'or', 'if', 'for', 'while', 'with', ',', '\\')):
                            incomplete_indicators.append(True)
                        
                        # Very short line that's likely a fragment (variable name, partial statement)
                        # But exclude common complete short statements
                        if (len(last_line) < 20 and 
                            last_line and 
                            not last_line.startswith(('return', 'pass', 'break', 'continue', 'raise')) and
                            not last_line.endswith((')', ']', '}', ':', '.', ';')) and
                            not re.match(r'^\s*(return|pass|break|continue|raise|import|from)\b', last_line)):
                            # Check if it's just a bare identifier or incomplete expression
                            # Valid short complete statements: "return x", "pass", "x = 5"
                            # Invalid fragments: "current", "value", "x +", etc.
                            if (re.match(r'^\w+$', last_line) or  # Just a variable name
                                re.search(r'[\+\-\*/=<>!&|^]$', last_line)):  # Ends with operator
                                incomplete_indicators.append(True)
                        
                        if any(incomplete_indicators):
                            # Remove the incomplete last line
                            lines = lines[:-1]
                            stripped = '\n'.join(lines).strip()
                    
                    if stripped:
                        all_blocks.append(stripped)
                        break  # Only take the first incomplete block we find
        
        if not all_blocks:
            return None
        
        # Remove duplicates while preserving order
        seen = set()
        unique_blocks = []
        for block in all_blocks:
            if block and block not in seen:
                seen.add(block)
                unique_blocks.append(block)
        
        if not unique_blocks:
            return None
        
        # Score and rank blocks
        def score_block(block: str) -> tuple[int, int]:
            score = 0
            # High priority: contains the entry point function
            if entry_point and re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", block):
                score += 10
            # Medium priority: contains any function definition
            if re.search(r"\bdef\s+\w+\s*\(", block):
                score += 5
            # Contains class definitions
            if re.search(r"\bclass\s+\w+", block):
                score += 3
            # Contains imports
            if re.search(r"^(?:from|import)\s+", block, re.MULTILINE):
                score += 2
            # Has proper Python structure
            if re.search(r":\s*$", block, re.MULTILINE):  # Contains colons (def/class/if/etc.)
                score += 1
            return score, len(block)
        
        # Check for diff format (SWE-bench style)
        diff_blocks = [block for block in unique_blocks if re.search(r"^diff --git ", block, re.MULTILINE)]
        if diff_blocks:
            return max(diff_blocks, key=len)
        
        # Score all blocks and pick the best one
        scored_blocks = [(score_block(block), block) for block in unique_blocks]
        scored_blocks.sort(key=lambda item: item[0], reverse=True)
        
        return scored_blocks[0][1]

    @staticmethod
    def _extract_from_xml_tags(completion: str, entry_point: Optional[str]) -> Optional[str]:
        """Extract code from XML-style tags like <code>, <solution>, <answer>, etc."""
        # Common XML-style tags used by various models
        tag_patterns = [
            r"<code>(.*?)</code>",
            r"<solution>(.*?)</solution>",
            r"<answer>(.*?)</answer>",
            r"<python>(.*?)</python>",
            r"<implementation>(.*?)</implementation>",
            r"\[CODE\](.*?)\[/CODE\]",
            r"\[SOLUTION\](.*?)\[/SOLUTION\]",
        ]
        
        all_blocks = []
        for pattern in tag_patterns:
            tag_pattern = re.compile(pattern, re.IGNORECASE | re.DOTALL)
            matches = [textwrap.dedent(match).strip() for match in tag_pattern.findall(completion)]
            all_blocks.extend([m for m in matches if m])
        
        if not all_blocks:
            return None
        
        # Prefer blocks with entry point, otherwise return longest
        for block in all_blocks:
            if entry_point and re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", block):
                return block
        
        return max(all_blocks, key=len)

    @staticmethod
    def _extract_from_labeled_sections(completion: str, entry_point: Optional[str]) -> Optional[str]:
        """Extract code from labeled sections (common in Llama and instruction-tuned models)."""
        # Patterns like "Here's the solution:", "Code:", "Implementation:", etc.
        section_markers = [
            r"(?:here(?:'s| is)(?: the)?)?\s*(?:solution|implementation|code|answer)s?\s*:?\s*$",
            r"^def\s+",  # Direct function definition
            r"^class\s+",  # Direct class definition
        ]
        
        lines = completion.splitlines()
        code_sections = []
        current_section = []
        capturing = False
        
        for i, line in enumerate(lines):
            stripped = line.strip().lower()
            
            # Check if this line is a section marker
            is_marker = any(re.search(pattern, stripped, re.IGNORECASE) for pattern in section_markers)
            
            # Check if line looks like code
            is_code_line = (
                re.match(r"^(def |class |from |import |@|if |for |while |return |#)", line.lstrip()) or
                (entry_point and re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", line)) or
                re.match(r'^[a-zA-Z_]\w*\s*=', line.lstrip()) or  # Variable assignment
                re.match(r'^[a-zA-Z_]\w*\s*\(', line.lstrip())    # Function call
            )
            
            # Skip markdown fences
            if stripped.startswith("```"):
                continue
            
            # Start capturing if we hit a marker followed by code
            if is_marker:
                if current_section:
                    code_sections.append("\n".join(current_section))
                    current_section = []
                capturing = True
                # If the marker line itself is code, include it
                if is_code_line:
                    current_section.append(line)
                continue
            
            # Start capturing on first code-like line
            if not capturing and is_code_line:
                capturing = True
            
            # Add line to current section if capturing
            if capturing:
                # Stop if we hit explanatory text after code
                if current_section and stripped and not is_code_line and not line.startswith(" ") and not line.startswith("\t"):
                    # Check if it's really explanatory text (not just a comment or string)
                    if not stripped.startswith("#") and "def " not in stripped:
                        code_sections.append("\n".join(current_section))
                        current_section = []
                        capturing = False
                        continue
                
                current_section.append(line)
        
        # Add final section
        if current_section:
            code_sections.append("\n".join(current_section))
        
        if not code_sections:
            return None
        
        # If we have multiple sections, we need to combine them intelligently
        # This handles cases where imports and function definitions are separated
        
        # Separate imports from code sections
        import_sections = []
        code_with_entry_point = None
        other_sections = []
        
        for section in code_sections:
            # Check if section is primarily imports
            section_lines = [line for line in section.splitlines() if line.strip()]
            is_import_section = all(
                line.lstrip().startswith(('import ', 'from ')) or not line.strip()
                for line in section_lines
            )
            
            if is_import_section:
                import_sections.append(section)
            elif entry_point and re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", section):
                code_with_entry_point = section
            else:
                other_sections.append(section)
        
        # If we found the entry point section, combine it with imports
        if code_with_entry_point:
            all_imports = "\n".join(import_sections)
            if all_imports:
                return (all_imports + "\n\n" + code_with_entry_point).strip()
            return code_with_entry_point.strip()
        
        # Otherwise, return the longest section
        return max(code_sections, key=len).strip()

    @staticmethod
    def _extract_by_code_detection(completion: str, entry_point: Optional[str]) -> Optional[str]:
        """Fallback: Extract code by detecting Python syntax patterns."""
        tag_line_pattern = re.compile(r"\[(?:/?)[A-Za-z0-9_-]+\]")
        lines = completion.splitlines()
        code_lines: List[str] = []
        capture = False
        
        for line in lines:
            stripped = line.strip()
            
            # Skip markdown fences and XML-like tags
            if stripped.startswith("```") or tag_line_pattern.fullmatch(stripped):
                continue
            
            # Start capturing on Python keywords or entry point
            if not capture and (
                re.match(r"^(def |class |from |import |@|if __name__ ==)", stripped)
                or (entry_point and re.match(rf"\s*def\s+{re.escape(entry_point)}\s*\(", stripped))
            ):
                capture = True
            
            if capture:
                # Stop if we hit obvious explanatory text
                if stripped and not stripped.startswith("#"):
                    # Check if line looks like prose (no Python keywords/operators)
                    if re.match(r"^[A-Z][a-z]+.*[.!?]$", stripped) and "def " not in stripped and "=" not in stripped:
                        break
                code_lines.append(line)
        
        result = "\n".join(code_lines).strip()
        return result if result else None

    @staticmethod
    def _strip_language_tags(code: str) -> str:
        """
        Remove various language tags and metadata that models add to code.
        Handles: [PYTHON], <python>, <!-- comments -->, etc.
        """
        lines = code.splitlines()
        cleaned_lines = []
        
        # Patterns to remove
        tag_pattern = re.compile(r"^\[(?:/?)[A-Za-z0-9_-]+\]$")  # [PYTHON], [/PYTHON]
        xml_tag_pattern = re.compile(r"^</?(?:python|code|solution|answer)>$", re.IGNORECASE)
        html_comment_pattern = re.compile(r"^<!--.*?-->$")
        note_pattern = re.compile(r"^(?:Note|Explanation|Output|Example|Usage):", re.IGNORECASE)
        language_marker = re.compile(r"^(python|py|javascript|js|java|cpp|c\+\+)$", re.IGNORECASE)
        
        for line in lines:
            stripped = line.strip()
            
            # Skip various tag formats
            if (tag_pattern.match(stripped) or 
                xml_tag_pattern.match(stripped) or 
                html_comment_pattern.match(stripped) or
                language_marker.match(stripped)):  # Skip standalone language names
                continue
            
            # Skip explanatory notes (but keep comments in code)
            if note_pattern.match(stripped) and not line.lstrip().startswith("#"):
                continue
            
            cleaned_lines.append(line)
        
        result = "\n".join(cleaned_lines).strip()
        
        # Remove common prefixes that models add
        prefixes_to_remove = [
            "Here's the solution:",
            "Here is the solution:",
            "Here's the code:",
            "Here is the code:",
            "Solution:",
            "Code:",
            "Answer:",
        ]
        
        for prefix in prefixes_to_remove:
            if result.startswith(prefix):
                result = result[len(prefix):].lstrip()
        
        return result

    @staticmethod
    def _finalize_code_for_execution(code: str, prompt: str) -> str:
        """
        Final processing of extracted code before execution.
        Strips tags, merges imports, removes trailing prose, and handles incomplete functions.
        """
        stripped = Benchmark._strip_language_tags(code)
        cleaned = Benchmark._remove_trailing_prose(stripped)
        
        # IMPORTANT: Complete function BEFORE removing test code
        # Because _remove_test_code expects proper function definitions
        complete = Benchmark._complete_function_if_needed(cleaned, prompt)
        
        # NOW remove test/demo code (after function is complete)
        complete = Benchmark._remove_test_code(complete, prompt)
        
        merged = Benchmark._merge_imports_with_prompt(complete, prompt)
        return Benchmark._ensure_typing_imports(merged)

    @staticmethod
    def _complete_function_if_needed(code: str, prompt: str) -> str:
        """
        If the code is just a function body without the def line,
        merge it with the function signature from the prompt.
        """
        if not prompt:
            return code
        
        # Check if code has a function definition
        if re.search(r'^\s*def\s+\w+', code, re.MULTILINE):
            # Code already has function definition
            return code
        
        # Check if code looks like a function body (has return, or indented code)
        looks_like_body = (
            'return ' in code or
            re.search(r'^\s{4,}', code, re.MULTILINE) or  # Has indented lines
            re.search(r'^\s+(if|for|while|with|try)\s+', code, re.MULTILINE)  # Starts with control flow
        )
        
        if not looks_like_body:
            return code
        
        # Extract function signature from prompt
        func_match = re.search(
            r'^(def\s+\w+\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:)\s*(?:""".*?"""|\'\'\'.*?\'\'\')?',
            prompt,
            re.MULTILINE | re.DOTALL
        )
        
        if not func_match:
            return code
        
        func_signature = func_match.group(1)
        
        # Extract just the function signature line(s), not the docstring
        # Find the def line and include continuation lines if it's multi-line
        signature_lines = []
        in_signature = False
        paren_count = 0
        
        for line in prompt.split('\n'):
            stripped = line.strip()
            
            # Start when we find 'def '
            if stripped.startswith('def '):
                in_signature = True
                signature_lines.append(line)
                # Count parentheses to handle multi-line signatures
                paren_count += line.count('(') - line.count(')')
                # If signature ends on this line (found ':'), stop
                if ':' in line and paren_count == 0:
                    break
                continue
            
            # Continue collecting lines if we're in a multi-line signature
            if in_signature:
                signature_lines.append(line)
                paren_count += line.count('(') - line.count(')')
                # Stop when we find the closing ')' and ':'
                if ':' in line and paren_count == 0:
                    break
        
        if not signature_lines:
            return code
        
        # Reconstruct the function with its body
        func_def = '\n'.join(signature_lines)
        
        # Ensure code is properly indented (should have 4 spaces)
        code_lines = code.split('\n')
        indented_lines = []
        
        for line in code_lines:
            if not line.strip():
                indented_lines.append(line)
            elif line.startswith('    '):
                # Already indented correctly
                indented_lines.append(line)
            else:
                # Add indentation
                indented_lines.append('    ' + line)
        
        indented_body = '\n'.join(indented_lines)
        
        # Combine signature and body
        complete_function = func_def + '\n' + indented_body
        
        return complete_function

    @staticmethod
    def _remove_trailing_prose(code: str) -> str:
        """
        Remove explanatory text that appears after the code.
        This is common in Llama and other instruction-tuned models.
        """
        lines = code.splitlines()
        
        # Comments that typically mark the end of implementation
        end_markers = [
            '# test', '# example', '# usage', '# demo', '# sample',
            '# output', '# explanation', '# note', '# testing'
        ]
        
        # Find the last line that looks like actual implementation code
        last_code_idx = -1
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check if this is an end marker comment
            is_end_marker = any(line.lower().startswith(marker) for marker in end_markers)
            if is_end_marker:
                # Don't include this or anything after it
                continue
            
            # This looks like code if it:
            # - Is a regular comment (but not an end marker)
            # - Contains Python keywords
            # - Has assignment or operators
            # - Is indented (continuation of code block)
            # - Ends with typical code characters
            is_code = (
                (line.startswith('#') and not is_end_marker) or
                any(kw in line for kw in ['def ', 'class ', 'return', 'if ', 'elif ', 'else:', 'for ', 'while ', 'import ', 'from ']) or
                any(op in line for op in ['=', '(', ')', '[', ']', '{', '}']) or
                (i > 0 and line and lines[i].startswith(('    ', '\t'))) or
                line.endswith((':', ',', ')', ']', '}')) or
                re.match(r'^[a-z_][a-z0-9_]*\s*=', line, re.IGNORECASE)  # variable assignment
            )
            
            # This looks like prose if it:
            # - Starts with a capital letter and contains common prose words
            # - Ends with punctuation typical of sentences
            # - Contains question marks
            prose_indicators = [
                'this', 'the', 'you', 'can', 'will', 'should', 'would', 'note', 'example',
                'explanation', 'usage', 'hope', 'let', 'here', 'output', 'result', 'test'
            ]
            is_prose = (
                (line[0].isupper() and any(word in line.lower() for word in prose_indicators)) or
                re.search(r'[.!?]\s*$', line) or
                '?' in line
            )
            
            # If it looks like code and not prose, mark it
            if is_code and not is_prose:
                last_code_idx = i
                break
        
        # If we found actual code, keep everything up to that point
        if last_code_idx >= 0:
            return '\n'.join(lines[:last_code_idx + 1]).strip()
        
        # Otherwise return as-is
        return code

    @staticmethod
    def _remove_test_code(code: str, prompt: str) -> str:
        """
        Remove test/demo code that models often add after the implementation.
        This includes:
        - print() statements calling the function
        - Example usage at module level
        - Test cases outside of function definitions
        """
        if not code:
            return code
        
        lines = code.splitlines()
        
        # Find where the main implementation ends
        # We want to keep: imports, function/class definitions
        # We want to remove: top-level function calls, print statements, etc.
        
        result_lines = []
        in_function_or_class = False
        function_indent_level = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Track if we're inside a function or class
            if stripped.startswith('def ') or stripped.startswith('class '):
                in_function_or_class = True
                # Calculate indent level
                function_indent_level = len(line) - len(line.lstrip())
                result_lines.append(line)
                continue
            
            # If we're in a function/class, check if we've exited
            if in_function_or_class:
                current_indent = len(line) - len(line.lstrip())
                # If we're back to the same or less indentation and line has content
                if stripped and current_indent <= function_indent_level:
                    in_function_or_class = False
                else:
                    # Still inside function/class
                    result_lines.append(line)
                    continue
            
            # Outside function/class: only keep imports and blank lines
            if (stripped.startswith('import ') or 
                stripped.startswith('from ') or 
                not stripped):  # blank line
                result_lines.append(line)
            else:
                # This is module-level code (test/demo code) - skip it
                # Examples: print(...), result = func(...), func(...), etc.
                pass
        
        return '\n'.join(result_lines).strip()

    @staticmethod
    def _merge_imports_with_prompt(code: str, prompt: str) -> str:
        if not prompt:
            return code

        # Extract imports from prompt
        prompt_imports = Benchmark._extract_import_lines(prompt)
        
        # Extract helper functions from prompt (functions that are not the main entry point)
        prompt_helper_functions = Benchmark._extract_helper_functions(prompt, code)
        
        code_imports = set(Benchmark._extract_import_lines(code))
        missing_imports = [imp for imp in prompt_imports if imp not in code_imports]
        
        # Combine missing imports and helper functions
        prefix_code = missing_imports + prompt_helper_functions
        
        if not prefix_code:
            return code

        if not code.strip():
            return "\n".join(prefix_code)

        lines = code.splitlines()
        insert_idx = 0
        while insert_idx < len(lines) and not lines[insert_idx].strip():
            insert_idx += 1

        if insert_idx < len(lines) and lines[insert_idx].strip().startswith(('"""', "'''")):
            insert_idx = Benchmark._find_docstring_end(lines, insert_idx) + 1

        insertion_block = prefix_code[:]
        if insert_idx < len(lines) and lines[insert_idx].strip():
            insertion_block.append("")

        new_lines = lines[:insert_idx] + insertion_block + lines[insert_idx:]
        return "\n".join(new_lines).strip()
    
    @staticmethod
    def _extract_helper_functions(prompt: str, code: str) -> List[str]:
        """Extract helper functions from prompt that are not already in the code."""
        helper_functions = []
        
        # Only extract functions that have proper Python syntax
        try:
            tree = ast.parse(prompt)
        except SyntaxError:
            # If prompt isn't valid Python, can't extract functions
            return []
        
        # Find all function names already in the code
        code_func_names = set()
        for match in re.finditer(r'def\s+(\w+)\s*\(', code):
            code_func_names.add(match.group(1))
        
        # Extract function definitions from the parsed AST
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                if func_name not in code_func_names:
                    # Get the source code for this function
                    # Find the function in the original prompt text
                    func_pattern = re.compile(
                        rf'^\s*def\s+{re.escape(func_name)}\s*\([^)]*\).*?(?=^\s*(?:def\s+|class\s+|\Z))',
                        re.MULTILINE | re.DOTALL
                    )
                    match = func_pattern.search(prompt)
                    if match:
                        helper_functions.append(match.group(0).strip())
        
        return helper_functions

    @staticmethod
    def _extract_import_lines(source: str) -> List[str]:
        imports: List[str] = []
        seen = set()
        for line in source.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("import ") or stripped.startswith("from "):
                if stripped not in seen:
                    imports.append(stripped)
                    seen.add(stripped)
        return imports

    @staticmethod
    def _find_docstring_end(lines: List[str], start_idx: int) -> int:
        opening_line = lines[start_idx].strip()
        quote_types = ('"""', "'''")
        quote = next((q for q in quote_types if opening_line.startswith(q)), None)
        if not quote:
            return start_idx
        if opening_line.endswith(quote) and len(opening_line) > len(quote):
            return start_idx
        idx = start_idx + 1
        while idx < len(lines):
            if quote in lines[idx]:
                return idx
            idx += 1
        return start_idx

    @staticmethod
    def _ensure_typing_imports(code: str) -> str:
        typing_import_map = {
            "Any": "from typing import Any",
            "Callable": "from typing import Callable",
            "Counter": "from typing import Counter",
            "DefaultDict": "from typing import DefaultDict",
            "Deque": "from typing import Deque",
            "Dict": "from typing import Dict",
            "Iterable": "from typing import Iterable",
            "List": "from typing import List",
            "Mapping": "from typing import Mapping",
            "MutableMapping": "from typing import MutableMapping",
            "Optional": "from typing import Optional",
            "Sequence": "from typing import Sequence",
            "Set": "from typing import Set",
            "Tuple": "from typing import Tuple",
            "Union": "from typing import Union",
        }

        try:
            tree = ast.parse(code)
        except SyntaxError:
            tree = None

        used_names = set()

        if tree is not None:
            class TypingNameCollector(ast.NodeVisitor):
                def visit_Name(self, node: ast.Name) -> None:
                    if node.id in typing_import_map:
                        used_names.add(node.id)
                    self.generic_visit(node)

            TypingNameCollector().visit(tree)
        else:
            for name in typing_import_map:
                if re.search(rf"(?<!\.)\b{name}\b", code):
                    used_names.add(name)

        existing_import_lines = Benchmark._extract_import_lines(code)
        imported_typing_names = set()

        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "typing":
                    for alias in node.names:
                        if alias.asname is None:
                            imported_typing_names.add(alias.name)

        missing_imports: List[str] = []
        for name in typing_import_map:
            if name not in used_names:
                continue
            if name in imported_typing_names:
                continue
            import_line = typing_import_map[name]
            if import_line not in existing_import_lines and import_line not in missing_imports:
                missing_imports.append(import_line)

        if not missing_imports:
            return code

        lines = code.splitlines()
        insert_idx = 0
        while insert_idx < len(lines) and not lines[insert_idx].strip():
            insert_idx += 1

        if insert_idx < len(lines) and lines[insert_idx].strip().startswith(('"""', "'''")):
            insert_idx = Benchmark._find_docstring_end(lines, insert_idx) + 1

        insertion_block = missing_imports[:]
        if insert_idx < len(lines) and lines[insert_idx].strip():
            insertion_block.append("")

        new_lines = lines[:insert_idx] + insertion_block + lines[insert_idx:]
        return "\n".join(new_lines).strip()

    @staticmethod
    def get_tests_for_task(task: BenchmarkTask) -> Optional[str]:
        """Return the raw test code associated with a task."""
        return task.tests
