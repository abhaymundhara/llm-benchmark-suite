from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from benchmarks import registry as benchmark_registry
from benchmarks.base import Benchmark, BenchmarkReport, ProgressCallback
from models import registry as model_registry
from models.base import BaseModelAdapter

logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """Configure application-wide logging with optional file output."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    handlers = [logging.StreamHandler()]
    file_path = log_file or os.getenv("BENCHMARK_LOG_FILE")
    if file_path:
        path = Path(file_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=handlers,
    )


@dataclass
class GenerationConfig:
    """Model generation parameters."""

    temperature: float = 0.2
    max_tokens: int = 512


@dataclass
class RunConfig:
    """Configuration for executing a benchmark run."""

    model_provider: str
    model_name: str
    benchmark_key: str
    limit: Optional[int] = None
    generation: GenerationConfig = field(default_factory=GenerationConfig)


class BenchmarkRunner:
    """High-level orchestrator for executing benchmarks across model adapters."""

    def __init__(self, *, reports_dir: Optional[str] = None) -> None:
        self.reports_dir = Path(reports_dir or os.getenv("BENCHMARK_REPORTS_DIR", "reports")).expanduser()
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.last_report_paths: Dict[str, Path] = {}

    def list_model_providers(self) -> Iterable[str]:
        return model_registry.keys()

    def list_benchmarks(self) -> Iterable[str]:
        return benchmark_registry.keys()

    def available_models(self) -> Dict[str, List[str]]:
        return {provider: [info.name for info in infos] for provider, infos in model_registry.available_models().items()}

    def run(self, config: RunConfig, *, progress_callback: Optional[ProgressCallback] = None) -> BenchmarkReport:
        logger.info(
            "Starting benchmark '%s' with model '%s/%s'.",
            config.benchmark_key,
            config.model_provider,
            config.model_name,
        )
        adapter = self._create_adapter(config.model_provider, config.model_name)
        benchmark = self._create_benchmark(config.benchmark_key, limit=config.limit)

        report = benchmark.run(
            adapter,
            temperature=config.generation.temperature,
            max_tokens=config.generation.max_tokens,
            progress_callback=progress_callback,
        )
        pass_rate = report.metrics.get("pass_rate", 0.0)
        total_latency_ms = report.metrics.get("total_latency_ms")
        if total_latency_ms is not None:
            logger.info(
                "Completed benchmark '%s' for model '%s/%s'. Pass rate: %.2f. Total latency: %.0f ms.",
                report.benchmark_name,
                report.model_name,
                config.model_provider,
                pass_rate,
                total_latency_ms,
            )
        else:
            logger.info(
                "Completed benchmark '%s' for model '%s/%s'. Pass rate: %.2f.",
                report.benchmark_name,
                report.model_name,
                config.model_provider,
                pass_rate,
            )

        self._persist_report(report)
        return report

    def _persist_report(self, report: BenchmarkReport) -> None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        benchmark_part = self._sanitize_component(report.benchmark_name)
        model_part = self._sanitize_component(report.model_name)
        base_name = f"{benchmark_part}_{model_part}_{timestamp}"
        json_path = self.reports_dir / f"{base_name}.json"
        json_path.write_text(report.to_json(), encoding="utf-8")

        failed = [result for result in report.task_results if not result.passed]
        total_latency_ms = report.metrics.get("total_latency_ms")
        failure_categories = report.metrics.get("failure_categories", {})
        
        # Calculate parser vs model issues
        parser_issues = sum(failure_categories.get(cat, 0) for cat in [
            "parser_extraction", "parser_syntax", "parser_incomplete"
        ])
        model_issues = sum(failure_categories.get(cat, 0) for cat in [
            "model_algorithm", "timeout", "runtime_error"
        ])
        
        summary_lines = [
            f"Benchmark: {report.benchmark_name}",
            f"Model: {report.model_name}",
            f"Pass rate: {report.metrics.get('pass_rate', 0.0):.2f}",
            f"Total tasks: {report.metrics.get('total_tasks')}",
            f"Passed: {report.metrics.get('passed')}",
            f"Failed: {report.metrics.get('failed')}",
            (
                f"Total latency (ms): {total_latency_ms:.0f}"
                if total_latency_ms is not None
                else "Total latency (ms): N/A"
            ),
            "",
        ]
        
        # Add token usage metrics if available
        total_input = report.metrics.get('total_input_tokens')
        total_output = report.metrics.get('total_output_tokens')
        total_all = report.metrics.get('total_tokens')
        avg_input = report.metrics.get('average_input_tokens')
        avg_output = report.metrics.get('average_output_tokens')
        
        if any([total_input, total_output, total_all]):
            summary_lines.append("=== TOKEN USAGE ===")
            if total_input is not None:
                summary_lines.append(f"Total Input Tokens: {total_input:,}")
            if total_output is not None:
                summary_lines.append(f"Total Output Tokens: {total_output:,}")
            if total_all is not None:
                summary_lines.append(f"Total Tokens: {total_all:,}")
            if avg_input is not None:
                summary_lines.append(f"Average Input Tokens: {avg_input:.1f}")
            if avg_output is not None:
                summary_lines.append(f"Average Output Tokens: {avg_output:.1f}")
            summary_lines.append("")
        
        summary_lines.extend([
            "=== FAILURE ANALYSIS ===",
            f"Parser Issues (NEED TO FIX): {parser_issues}",
            f"  - Extraction failures: {failure_categories.get('parser_extraction', 0)}",
            f"  - Syntax errors: {failure_categories.get('parser_syntax', 0)}",
            f"  - Incomplete code: {failure_categories.get('parser_incomplete', 0)}",
            "",
            f"Model Issues (Algorithm/Logic): {model_issues}",
            f"  - Wrong algorithm: {failure_categories.get('model_algorithm', 0)}",
            f"  - Timeouts: {failure_categories.get('timeout', 0)}",
            f"  - Runtime errors: {failure_categories.get('runtime_error', 0)}",
            "",
        ])
        
        if parser_issues > 0:
            summary_lines.append("⚠️  PARSER ISSUES FOUND - Focus on improving code extraction!")
            summary_lines.append("")
            parser_failures = [r for r in failed if r.failure_category in [
                "parser_extraction", "parser_syntax", "parser_incomplete"
            ]]
            summary_lines.append("Parser Issue Tasks:")
            for result in parser_failures:
                summary_lines.append(f"- {result.task_id} [{result.failure_category}]: {result.error or 'Failed'}")
            summary_lines.append("")
        
        if failed:
            summary_lines.append("All Failed Tasks:")
            for result in failed:
                category_label = f" [{result.failure_category}]" if result.failure_category else ""
                summary_lines.append(f"- {result.task_id}{category_label}: {result.error or 'Tests failed.'}")
        else:
            summary_lines.append("All tasks passed.")

        summary_path = self.reports_dir / f"{base_name}_summary.txt"
        summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
        self.last_report_paths = {"json": json_path, "summary": summary_path}
        
        # Print the failure analysis to console
        print("\n" + "=" * 80)
        print("BENCHMARK COMPLETE - FAILURE ANALYSIS")
        print("=" * 80)
        print(f"\n✅ Passed: {report.metrics.get('passed')}/{report.metrics.get('total_tasks')}")
        print(f"❌ Failed: {report.metrics.get('failed')}/{report.metrics.get('total_tasks')}")
        
        if failed:
            print(f"\n{'='*80}")
            print(f"🔧 PARSER ISSUES: {parser_issues}")
            print(f"{'='*80}")
            if parser_issues > 0:
                parser_failures = [r for r in failed if r.failure_category in [
                    "parser_extraction", "parser_syntax", "parser_incomplete"
                ]]
                for result in parser_failures:
                    print(f"  ❌ {result.task_id} [{result.failure_category}]")
            else:
                print("  ✅ No parser issues!")
            
            print(f"\n{'='*80}")
            print(f"🤖 MODEL ISSUES: {model_issues}")
            print(f"{'='*80}")
            if model_issues > 0:
                model_failures = [r for r in failed if r.failure_category in [
                    "model_algorithm", "timeout", "runtime_error"
                ]]
                for result in model_failures:
                    print(f"  ❌ {result.task_id} [{result.failure_category}]")
            else:
                print("  ✅ No model issues!")
        else:
            print("\n🎉 ALL TESTS PASSED!")
        
        print(f"\n{'='*80}")
        print(f"📄 Full report: {json_path.name}")
        print(f"📄 Summary: {summary_path.name}")
        print("=" * 80 + "\n")
        
        logger.info("Report saved to %s; summary saved to %s", json_path, summary_path)

    @staticmethod
    def _sanitize_component(value: str) -> str:
        if not value:
            return "unknown"
        sanitized = re.sub(r"[\\/:*?\"<>|\s]+", "_", value)
        sanitized = sanitized.strip("._")
        return sanitized or "unknown"

    @staticmethod
    def _create_adapter(provider_key: str, model_name: str) -> BaseModelAdapter:
        try:
            return model_registry.create(provider_key, model_name=model_name)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Failed to initialize model adapter '%s/%s'.", provider_key, model_name)
            raise

    @staticmethod
    def _create_benchmark(benchmark_key: str, *, limit: Optional[int]) -> Benchmark:
        try:
            return benchmark_registry.create(benchmark_key, limit=limit)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Failed to initialize benchmark '%s'.", benchmark_key)
            raise
