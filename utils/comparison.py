"""Utilities for comparing benchmark results across multiple runs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class RunSummary:
    """Summary information about a single benchmark run."""
    
    file_path: Path
    benchmark_name: str
    model_name: str
    timestamp: str
    total_tasks: int
    passed: int
    failed: int
    pass_rate: float
    average_latency_ms: Optional[float]
    task_ids: Set[str]
    full_data: Dict[str, Any]
    
    def display_name(self) -> str:
        """Generate a user-friendly display name for this run."""
        return f"{self.model_name} ({self.timestamp}) - {self.total_tasks} tasks, {self.pass_rate:.1%} pass"


@dataclass
class ComparisonResult:
    """Result of comparing multiple benchmark runs."""
    
    runs: List[RunSummary]
    overlapping_task_ids: Set[str]
    comparison_metrics: List[Dict[str, Any]]
    per_task_results: Dict[str, Dict[str, bool]]  # task_id -> {run_name: passed}


def load_reports_from_directory(reports_dir: Path) -> List[RunSummary]:
    """Load all benchmark reports from a directory."""
    summaries = []
    
    for json_file in reports_dir.glob("*.json"):
        if "parser_analysis" in json_file.name:
            continue
            
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            parts = json_file.stem.split('_')
            timestamp = parts[-1] if parts else "unknown"
            
            task_results = data.get('task_results', [])
            task_ids = {result['task_id'] for result in task_results}
            
            metrics = data.get('metrics', {})
            
            summary = RunSummary(
                file_path=json_file,
                benchmark_name=data.get('benchmark_name', 'unknown'),
                model_name=data.get('model_name', 'unknown'),
                timestamp=timestamp,
                total_tasks=metrics.get('total_tasks', 0),
                passed=metrics.get('passed', 0),
                failed=metrics.get('failed', 0),
                pass_rate=metrics.get('pass_rate', 0.0),
                average_latency_ms=metrics.get('average_latency_ms'),
                task_ids=task_ids,
                full_data=data
            )
            summaries.append(summary)
            
        except Exception as exc:
            logger.warning("Failed to load report %s: %s", json_file.name, exc)
            continue
    
    summaries.sort(key=lambda s: s.timestamp, reverse=True)
    return summaries


def filter_by_benchmark(summaries: List[RunSummary], benchmark_name: str) -> List[RunSummary]:
    """Filter summaries to only include a specific benchmark."""
    return [s for s in summaries if s.benchmark_name == benchmark_name]


def compare_runs(selected_runs: List[RunSummary]) -> ComparisonResult:
    """Compare multiple benchmark runs by finding overlapping tasks."""
    if len(selected_runs) < 2:
        raise ValueError("Need at least 2 runs to compare")
    
    overlapping_task_ids = set.intersection(*[run.task_ids for run in selected_runs])
    
    if not overlapping_task_ids:
        logger.warning("No overlapping tasks found between selected runs")
    
    comparison_metrics = []
    per_task_results = {task_id: {} for task_id in overlapping_task_ids}
    
    for run in selected_runs:
        overlapping_results = [
            result for result in run.full_data.get('task_results', [])
            if result['task_id'] in overlapping_task_ids
        ]
        
        if overlapping_results:
            passed_count = sum(1 for r in overlapping_results if r['passed'])
            failed_count = len(overlapping_results) - passed_count
            pass_rate = passed_count / len(overlapping_results) if overlapping_results else 0.0
            
            latencies = [
                r['generation_metrics'].get('latency_ms')
                for r in overlapping_results
                if r['generation_metrics'].get('latency_ms') is not None
            ]
            avg_latency = sum(latencies) / len(latencies) if latencies else None
            
            input_tokens = [
                r['generation_metrics'].get('input_tokens')
                for r in overlapping_results
                if r['generation_metrics'].get('input_tokens') is not None
            ]
            output_tokens = [
                r['generation_metrics'].get('output_tokens')
                for r in overlapping_results
                if r['generation_metrics'].get('output_tokens') is not None
            ]
            
            total_input = sum(input_tokens) if input_tokens else None
            total_output = sum(output_tokens) if output_tokens else None
            avg_input = sum(input_tokens) / len(input_tokens) if input_tokens else None
            avg_output = sum(output_tokens) / len(output_tokens) if output_tokens else None
            
            tokens_per_success = None
            if passed_count > 0 and total_output is not None:
                tokens_per_success = total_output / passed_count
            
            metrics = {
                'run_name': run.display_name(),
                'model_name': run.model_name,
                'timestamp': run.timestamp,
                'total_tasks_run': run.total_tasks,
                'tasks_compared': len(overlapping_results),
                'passed': passed_count,
                'failed': failed_count,
                'pass_rate': pass_rate,
                'average_latency_ms': avg_latency,
                'total_input_tokens': total_input,
                'total_output_tokens': total_output,
                'avg_input_tokens': avg_input,
                'avg_output_tokens': avg_output,
                'tokens_per_success': tokens_per_success,
            }
            comparison_metrics.append(metrics)
            
            for result in overlapping_results:
                per_task_results[result['task_id']][run.model_name] = result['passed']
        else:
            comparison_metrics.append({
                'run_name': run.display_name(),
                'model_name': run.model_name,
                'timestamp': run.timestamp,
                'total_tasks_run': run.total_tasks,
                'tasks_compared': 0,
                'passed': 0,
                'failed': 0,
                'pass_rate': 0.0,
                'average_latency_ms': None,
                'total_input_tokens': None,
                'total_output_tokens': None,
                'avg_input_tokens': None,
                'avg_output_tokens': None,
                'tokens_per_success': None,
            })
    
    return ComparisonResult(
        runs=selected_runs,
        overlapping_task_ids=overlapping_task_ids,
        comparison_metrics=comparison_metrics,
        per_task_results=per_task_results
    )


def get_available_benchmarks(summaries: List[RunSummary]) -> List[str]:
    """Get unique benchmark names from summaries."""
    benchmarks = sorted(set(s.benchmark_name for s in summaries))
    return benchmarks
