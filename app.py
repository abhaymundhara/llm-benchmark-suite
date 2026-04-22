from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from benchmarks import registry as benchmark_registry
from benchmarks.base import Benchmark, BenchmarkReport, TaskResult
from runner import BenchmarkRunner, GenerationConfig, RunConfig, configure_logging
from utils.comparison import (
    ComparisonResult,
    RunSummary,
    compare_runs,
    filter_by_benchmark,
    get_available_benchmarks,
    load_reports_from_directory,
)


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
    # Calculate total tasks based on config without preloading all tasks.
    if config.start_index is not None and config.end_index is not None:
        total_tasks = config.end_index - config.start_index + 1
    elif config.limit is not None:
        total_tasks = max(0, config.limit)
    else:
        all_tasks_preview = list(_load_benchmark_stub(config.benchmark_key, limit=None).load_tasks())
        total_tasks = len(all_tasks_preview)
    
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


@st.cache_data(ttl=300)
def _load_all_reports(reports_dir_str: str) -> List[RunSummary]:
    """Load all reports with caching to avoid re-parsing on every interaction."""
    return load_reports_from_directory(Path(reports_dir_str))


def _render_comparison(comparison: ComparisonResult) -> None:
    """Render comparison visualization for multiple benchmark runs."""
    
    # Display overlap info
    num_overlapping = len(comparison.overlapping_task_ids)
    if num_overlapping == 0:
        st.error("❌ No overlapping tasks found between the selected runs. Cannot compare.")
        return
    
    # Info banner
    min_tasks = min(m['total_tasks_run'] for m in comparison.comparison_metrics)
    max_tasks = max(m['total_tasks_run'] for m in comparison.comparison_metrics)
    
    if min_tasks != max_tasks:
        st.info(f"ℹ️ Comparing **{num_overlapping}** common tasks (runs had {min_tasks}-{max_tasks} total tasks)")
    else:
        st.success(f"✅ Comparing all **{num_overlapping}** tasks")
    
    # Metrics table
    st.subheader("Comparison Metrics")
    
    df = pd.DataFrame(comparison.comparison_metrics)
    
    # Format the dataframe for display
    display_df = df[[
        'model_name', 'timestamp', 'total_tasks_run', 'tasks_compared', 
        'passed', 'failed', 'pass_rate', 'average_latency_ms'
    ]].copy()
    
    display_df['pass_rate'] = display_df['pass_rate'].apply(lambda x: f"{x:.1%}")
    display_df['average_latency_ms'] = display_df['average_latency_ms'].apply(
        lambda x: f"{x:.0f}" if pd.notna(x) else "N/A"
    )
    
    display_df.columns = [
        'Model', 'Timestamp', 'Total Tasks Run', 'Tasks Compared',
        'Passed', 'Failed', 'Pass Rate', 'Avg Latency (ms)'
    ]
    
    st.dataframe(display_df, use_container_width=True)
    
    # Pass rate bar chart
    st.subheader("Pass Rate Comparison")
    chart_data = pd.DataFrame({
        'Model': [m['model_name'] for m in comparison.comparison_metrics],
        'Pass Rate': [m['pass_rate'] for m in comparison.comparison_metrics]
    })
    
    fig_pass = px.bar(
        chart_data, 
        x='Model', 
        y='Pass Rate', 
        color='Model',
        text_auto='.1%',
        title='Pass Rate by Model'
    )
    fig_pass.update_layout(showlegend=False, yaxis_tickformat=".0%")
    st.plotly_chart(fig_pass, use_container_width=True)
    
    # Latency Comparison
    st.subheader("Latency Comparison")
    latency_data = pd.DataFrame({
        'Model': [m['model_name'] for m in comparison.comparison_metrics],
        'Avg Latency': [m['average_latency_ms'] for m in comparison.comparison_metrics]
    })
    
    fig_latency = px.bar(
        latency_data,
        x='Model',
        y='Avg Latency',
        color='Model',
        text_auto='.0f',
        title='Average Latency (ms)',
        labels={'Avg Latency': 'Latency (ms)'}
    )
    fig_latency.update_layout(showlegend=False)
    st.plotly_chart(fig_latency, use_container_width=True)

    # Token metrics if available
    if any(m.get('total_output_tokens') for m in comparison.comparison_metrics):
        st.subheader("Token Usage Comparison")
        
        token_df = pd.DataFrame(comparison.comparison_metrics)[[
            'model_name', 'total_input_tokens', 'total_output_tokens', 
            'avg_input_tokens', 'avg_output_tokens', 'tokens_per_success'
        ]].copy()
        
        token_df.columns = [
            'Model', 'Total Input', 'Total Output', 
            'Avg Input', 'Avg Output', 'Tokens/Success'
        ]
        
        # Format with commas
        for col in ['Total Input', 'Total Output']:
            token_df[col] = token_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
        
        for col in ['Avg Input', 'Avg Output', 'Tokens/Success']:
            token_df[col] = token_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
        
        st.dataframe(token_df, use_container_width=True)
        
        # Token efficiency chart
        efficiency_data = pd.DataFrame({
            'Model': [m['model_name'] for m in comparison.comparison_metrics],
            'Tokens per Success': [
                m.get('tokens_per_success', 0) if m.get('tokens_per_success') else 0 
                for m in comparison.comparison_metrics
            ]
        })
        if efficiency_data['Tokens per Success'].sum() > 0:
            fig_eff = px.bar(
                efficiency_data,
                x='Model',
                y='Tokens per Success',
                color='Model',
                text_auto='.0f',
                title='Tokens per Success'
            )
            fig_eff.update_layout(showlegend=False)
            st.plotly_chart(fig_eff, use_container_width=True)
    
    # Per-task breakdown
    st.subheader("Per-Task Results")
    
    task_comparison_data = []
    for task_id in sorted(comparison.per_task_results.keys()):
        row = {'Task ID': task_id}
        for run in comparison.runs:
            result = comparison.per_task_results[task_id].get(run.model_name)
            row[run.model_name] = "✅" if result else ("❌" if result is False else "—")
        task_comparison_data.append(row)
    
    task_df = pd.DataFrame(task_comparison_data)
    st.dataframe(task_df, use_container_width=True, height=400)
    
    # Download comparison CSV
    csv = task_df.to_csv(index=False)
    st.download_button(
        label="Download Task Comparison CSV",
        data=csv,
        file_name="benchmark_comparison.csv",
        mime="text/csv"
    )


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

    model_provider = st.sidebar.selectbox(
        "Model Provider",
        list(available_models.keys()),
        key="model_provider",
    )
    model_choices = available_models.get(model_provider, [])
    if model_choices:
        model_name = st.sidebar.selectbox("Model", model_choices, key="model_name")
    else:
        model_name = st.sidebar.text_input("Model", "", key="model_name")

    benchmark_key = st.sidebar.selectbox("Benchmark", list(runner.list_benchmarks()))
    
    # Task range selection
    st.sidebar.markdown("### Task Selection")
    task_mode = st.sidebar.radio(
        "Mode",
        ["Limit", "Range"],
        help="Limit: Run first N tasks | Range: Run tasks from start to end index"
    )
    
    if task_mode == "Limit":
        limit = st.sidebar.number_input("Task Limit (0 for all)", min_value=0, value=10, step=1)
        limit_value = None if limit == 0 else limit
        start_index = None
        end_index = None
    else:
        # Get benchmark to know total tasks
        try:
            benchmark_preview = _load_benchmark_stub(benchmark_key, limit=None)
            total_available = len(list(benchmark_preview.load_tasks()))
        except Exception:
            total_available = 1000  # Fallback if we can't load
        
        st.sidebar.caption(f"Total tasks available: ~{total_available}")
        start_index = st.sidebar.number_input(
            "Start Index (inclusive, 0-based)",
            min_value=0,
            max_value=max(0, total_available - 1),
            value=0,
            step=1,
            help="First task to run (0 = first task)"
        )
        end_index = st.sidebar.number_input(
            "End Index (inclusive, 0-based)",
            min_value=start_index,
            max_value=max(start_index, total_available - 1),
            value=min(start_index + 9, total_available - 1),
            step=1,
            help="Last task to run (inclusive)"
        )
        limit_value = None
        st.sidebar.info(f"Will run tasks {start_index} to {end_index} ({end_index - start_index + 1} tasks)")

    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.2, step=0.1)
    
    # Get model-specific max_tokens for Ollama models
    suggested_max_tokens = 2048  # Default
    raw_model_max_tokens = None
    if model_provider == "ollama" and model_name:
        from models.ollama_adapter import OllamaAdapter
        try:
            adapter = OllamaAdapter(model_name)
            raw_model_max_tokens = adapter._get_model_max_tokens()
            suggested_max_tokens = adapter.get_suggested_generation_tokens()
        except Exception:
            suggested_max_tokens = 2048

    selected_model_key = (model_provider, model_name)
    current_max_tokens = st.session_state.get("max_tokens")
    should_reset_tokens = st.session_state.get("last_model_key") != selected_model_key
    if (
        not should_reset_tokens
        and model_provider == "ollama"
        and raw_model_max_tokens is not None
        and current_max_tokens == raw_model_max_tokens
        and raw_model_max_tokens > suggested_max_tokens
    ):
        should_reset_tokens = True

    if should_reset_tokens:
        st.session_state["last_model_key"] = selected_model_key
        st.session_state["max_tokens"] = suggested_max_tokens
    
    max_tokens = st.sidebar.number_input(
        "Max Tokens", 
        min_value=64, 
        max_value=2000000,  # Support models with up to 2M context (e.g., Gemini, Nemotron)
        value=st.session_state.get("max_tokens", suggested_max_tokens),
        step=64,
        key="max_tokens",
        help=f"Model-specific recommended: {suggested_max_tokens}" if model_provider == "ollama" else None
    )
    
    # Show info if using recommended value for Ollama
    if model_provider == "ollama" and max_tokens == suggested_max_tokens and suggested_max_tokens > 2048:
        st.sidebar.success(f"✓ Using optimized {suggested_max_tokens} tokens for {model_name}")
    elif model_provider == "ollama" and max_tokens > 8192:
        st.sidebar.warning("Very large max token values will make local benchmark runs much slower.")

    st.sidebar.markdown("---")
    
    # Comparison Section
    st.sidebar.subheader("📊 Compare Results")
    
    if st.sidebar.button("Compare Benchmark Runs", use_container_width=True):
        st.session_state["show_comparison"] = not st.session_state.get("show_comparison", False)
    
    st.sidebar.markdown("---")
    st.sidebar.write("Configure .env with provider API keys before running.")
    if log_path:
        st.sidebar.caption(f"Detailed logs writing to `{log_path}`.")

    report: Optional[BenchmarkReport] = st.session_state.get("last_report")
    
    # Comparison Interface
    if st.session_state.get("show_comparison", False):
        st.header("📊 Benchmark Comparison")
        
        try:
            # Load all available reports
            reports_dir = Path("reports")
            if not reports_dir.exists():
                st.warning("No reports directory found. Run some benchmarks first!")
            else:
                all_summaries = _load_all_reports(str(reports_dir))
                
                if not all_summaries:
                    st.warning("No benchmark reports found. Run some benchmarks first!")
                else:
                    # Get available benchmarks
                    available_benchmarks = get_available_benchmarks(all_summaries)
                    
                    if not available_benchmarks:
                        st.warning("No valid benchmarks found in reports.")
                    else:
                        # Benchmark selector
                        selected_benchmark = st.selectbox(
                            "Select Benchmark to Compare",
                            available_benchmarks,
                            help="Only runs from the same benchmark can be compared"
                        )
                        
                        # Filter by selected benchmark
                        filtered_summaries = filter_by_benchmark(all_summaries, selected_benchmark)
                        
                        # Limit to most recent 30 runs for better UI performance
                        display_summaries = filtered_summaries[:30]
                        
                        if len(filtered_summaries) > 30:
                            st.info(f"Showing 30 most recent runs (total: {len(filtered_summaries)})")
                        
                        # Multiselect for runs
                        run_options = {summary.display_name(): summary for summary in display_summaries}
                        
                        selected_run_names = st.multiselect(
                            "Select Runs to Compare (2-5 recommended)",
                            list(run_options.keys()),
                            help="Select 2 or more runs to compare. Comparison uses only overlapping tasks."
                        )
                        
                        if len(selected_run_names) < 2:
                            st.info("Select at least 2 runs to enable comparison.")
                        elif len(selected_run_names) > 5:
                            st.warning("Comparing more than 5 runs may be hard to visualize. Consider selecting fewer runs.")
                        
                        if len(selected_run_names) >= 2:
                            if st.button("Generate Comparison", type="primary"):
                                selected_runs = [run_options[name] for name in selected_run_names]
                                
                                with st.spinner("Analyzing overlapping tasks..."):
                                    comparison = compare_runs(selected_runs)
                                    _render_comparison(comparison)
        
        except Exception as exc:
            st.error(f"Error loading comparison: {exc}")
            import traceback
            st.code(traceback.format_exc())
        
        st.markdown("---")

    if st.button("Run Benchmark", type="primary"):
        if not model_name:
            st.error("No model selected for the chosen provider.")
            return
        config = RunConfig(
            model_provider=model_provider,
            model_name=model_name,
            benchmark_key=benchmark_key,
            limit=limit_value,
            start_index=start_index,
            end_index=end_index,
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
