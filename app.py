from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from benchmarks import registry as benchmark_registry
from benchmarks.base import Benchmark, BenchmarkReport, TaskResult
from runner import BenchmarkRunner, GenerationConfig, RunConfig, configure_logging


def _initialize_logging() -> Optional[str]:
    log_path = os.getenv("BENCHMARK_LOG_FILE")
    configure_logging(logging.INFO, log_path)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    return log_path


def _load_benchmark_stub(benchmark_key: str, limit: Optional[int]) -> Benchmark:
    return benchmark_registry.create(benchmark_key, limit=limit)


def _task_results_to_dataframe(results: List[TaskResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        metrics = result.generation_metrics
        rows.append(
            {
                "Task ID": result.task_id,
                "Passed": "✅" if result.passed else "❌",
                "Error": result.error or "",
                "Latency (ms)": (
                    f"{metrics.latency_ms:.0f}"
                    if metrics.latency_ms is not None
                    else ""
                ),
                "Input Tokens": (
                    f"{metrics.input_tokens:,}"
                    if metrics.input_tokens is not None
                    else ""
                ),
                "Output Tokens": (
                    f"{metrics.output_tokens:,}"
                    if metrics.output_tokens is not None
                    else ""
                ),
                "Total Tokens": (
                    f"{metrics.input_tokens + metrics.output_tokens:,}"
                    if metrics.input_tokens is not None and metrics.output_tokens is not None
                    else ""
                ),
            }
        )
    return pd.DataFrame(rows)


def _render_task_details(result: TaskResult, *, collapsible: bool = False) -> None:
    """Render detailed information for a single task."""
    title = f"{'✅' if result.passed else '❌'} {result.task_id}"
    container = st.expander(title, expanded=False) if collapsible else st.container()
    with container:
        st.markdown("**Prompt**")
        st.code(result.prompt, language="markdown")

        st.markdown("**Model Completion**")
        st.code(result.completion, language="python")

        test_output = ""
        if result.evaluation_stdout:
            test_output += result.evaluation_stdout
        if result.evaluation_stderr:
            if test_output:
                test_output += "\n"
            test_output += result.evaluation_stderr
        st.markdown("**Test Output**")
        if test_output.strip():
            st.code(test_output, language="text")
        else:
            if result.passed:
                message = "Tests completed successfully with no stdout/stderr output."
            else:
                message = "No stdout/stderr captured for this run."
            st.code(message, language="text")

        if result.tests_code:
            st.markdown("**Tests Executed**")
            st.code(result.tests_code, language="python")

        metrics = result.generation_metrics
        latency = metrics.latency_ms
        latency_str = f"{latency:.0f} ms" if latency is not None else "N/A"
        
        # Build metrics display
        metrics_parts = [
            f"- Passed: {'Yes' if result.passed else 'No'}",
            f"- Latency: {latency_str}"
        ]
        
        if metrics.input_tokens is not None:
            metrics_parts.append(f"- Input Tokens: {metrics.input_tokens:,}")
        if metrics.output_tokens is not None:
            metrics_parts.append(f"- Output Tokens: {metrics.output_tokens:,}")
        if metrics.input_tokens is not None and metrics.output_tokens is not None:
            total = metrics.input_tokens + metrics.output_tokens
            metrics_parts.append(f"- Total Tokens: {total:,}")
        if metrics.cost_usd is not None:
            metrics_parts.append(f"- Cost: ${metrics.cost_usd:.6f}")
        
        st.markdown("\n".join(metrics_parts))

        if result.error:
            st.error(result.error)
        if not test_output:
            if result.evaluation_stdout:
                st.text_area(
                    "Evaluation Stdout",
                    result.evaluation_stdout,
                    height=200,
                    key=f"stdout_{result.task_id}",
                )
            if result.evaluation_stderr:
                st.text_area(
                    "Evaluation Stderr",
                    result.evaluation_stderr,
                    height=200,
                    key=f"stderr_{result.task_id}",
                )


def _render_report(report: BenchmarkReport) -> None:
    st.subheader("Aggregate Metrics")
    metrics_df = pd.DataFrame([report.metrics])
    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("Task Results")
    results_df = _task_results_to_dataframe(report.task_results)
    st.dataframe(results_df, use_container_width=True)

    failed_results = [result for result in report.task_results if not result.passed]
    if failed_results:
        st.error(f"{len(failed_results)} task(s) failed. See details below or check the log file for full traces.")
        with st.expander("Failed Task Summary", expanded=False):
            for result in failed_results:
                st.write(f"- `{result.task_id}` — {result.error or 'No error message provided.'}")

    if report.task_results:
        st.subheader("Task Details")
        st.session_state.setdefault("detail_mode", "Single Task")
        detail_mode = st.radio(
            "Detail Mode",
            ["Single Task", "All Tasks"],
            horizontal=True,
            key="detail_mode",
        )
        if detail_mode == "Single Task":
            task_ids = [result.task_id for result in report.task_results]
            selected_task_id = st.selectbox("Inspect Task", task_ids, key="task_selector")
            selected_result = next(result for result in report.task_results if result.task_id == selected_task_id)
            _render_task_details(selected_result)
        else:
            for result in report.task_results:
                _render_task_details(result, collapsible=True)

    st.download_button(
        label="Download JSON Report",
        data=report.to_json(),
        file_name=f"{report.benchmark_name}_{report.model_name}_report.json",
        mime="application/json",
    )

    st.info("Detailed stdout/stderr for each task is available in the console logs.")


def run_benchmark_with_progress(runner: BenchmarkRunner, config: RunConfig) -> BenchmarkReport:
    benchmark_preview = _load_benchmark_stub(config.benchmark_key, config.limit)
    total_tasks = len(list(benchmark_preview.load_tasks()))
    if config.limit is not None:
        total_tasks = min(total_tasks, config.limit)
    st.write(f"Benchmark will run on **{total_tasks}** tasks.")

    progress_bar = st.progress(0, text="Starting benchmark...")
    task_log_container = st.container()

    completed_results: List[TaskResult] = []

    def progress_callback(task_result: TaskResult) -> None:
        completed_results.append(task_result)
        progress = len(completed_results) / total_tasks if total_tasks else 0.0
        status_text = f"{len(completed_results)}/{total_tasks} tasks completed"
        progress_bar.progress(progress, text=status_text)
        with task_log_container:
            metrics = task_result.generation_metrics
            latency_value = (
                f"{metrics.latency_ms:.0f} ms"
                if metrics.latency_ms is not None
                else "— ms"
            )
            
            # Build token info string
            token_info = ""
            if metrics.input_tokens is not None and metrics.output_tokens is not None:
                total = metrics.input_tokens + metrics.output_tokens
                token_info = f" | Tokens: {metrics.input_tokens:,}+{metrics.output_tokens:,}={total:,}"
            elif metrics.output_tokens is not None:
                token_info = f" | Out: {metrics.output_tokens:,}"
            
            st.write(f"{'✅' if task_result.passed else '❌'} `{task_result.task_id}` - {latency_value}{token_info}")

    with st.spinner("Running benchmark..."):
        report = runner.run(config, progress_callback=progress_callback)

    progress_bar.progress(1.0, text="Benchmark completed.")
    total_latency_ms = report.metrics.get("total_latency_ms")
    if total_latency_ms is not None:
        st.success(f"Total generation latency: {total_latency_ms:.0f} ms")

    return report


def main() -> None:
    st.set_page_config(page_title="LLM Benchmark Dashboard", layout="wide")
    log_path = _initialize_logging()

    runner = BenchmarkRunner()

    st.title("LLM Benchmark Dashboard")
    st.caption("Evaluate large language models across multiple coding benchmarks.")

    available_models: Dict[str, List[str]] = runner.available_models()
    if not available_models:
        st.warning(
            "No models detected. Ensure your environment variables are configured and model backends are reachable."
        )
        available_models = {provider: [] for provider in runner.list_model_providers()}

    model_provider = st.sidebar.selectbox("Model Provider", list(available_models.keys()))
    model_choices = available_models.get(model_provider, [])
    if model_choices:
        model_name = st.sidebar.selectbox("Model", model_choices)
    else:
        model_name = st.sidebar.text_input("Model", "")

    benchmark_key = st.sidebar.selectbox("Benchmark", list(runner.list_benchmarks()))
    limit = st.sidebar.number_input("Task Limit (0 for all)", min_value=0, value=10, step=1)
    limit_value = None if limit == 0 else limit

    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.2, step=0.1)
    max_tokens = st.sidebar.number_input("Max Tokens", min_value=64, max_value=8192, value=2048, step=64)

    st.sidebar.markdown("---")
    st.sidebar.write("Configure .env with provider API keys before running.")
    if log_path:
        st.sidebar.caption(f"Detailed logs writing to `{log_path}`.")

    report: Optional[BenchmarkReport] = st.session_state.get("last_report")

    if st.button("Run Benchmark", type="primary"):
        if not model_name:
            st.error("No model selected for the chosen provider.")
            return
        config = RunConfig(
            model_provider=model_provider,
            model_name=model_name,
            benchmark_key=benchmark_key,
            limit=limit_value,
            generation=GenerationConfig(temperature=temperature, max_tokens=int(max_tokens)),
        )
        report = run_benchmark_with_progress(runner, config)
        st.session_state["last_report"] = report

    if report:
        _render_report(report)
        if runner.last_report_paths:
            summary_path = runner.last_report_paths.get("summary")
            json_path = runner.last_report_paths.get("json")
            msg = "Report artifacts saved."
            if summary_path or json_path:
                details = []
                if summary_path:
                    details.append(f"Summary: `{summary_path}`")
                if json_path:
                    details.append(f"JSON: `{json_path}`")
                msg = msg + " " + " | ".join(details)
            st.success(msg)
    else:
        st.info("Configure the benchmark and click **Run Benchmark** to start.")


if __name__ == "__main__":
    main()
