# llm-benchmark-suite

A benchmark suite for evaluating large language models on coding tasks. The project supports HumanEval, MBPP, BigCodeBench, and SWE-bench, and provides both a command-line runner and a Streamlit interface for interactive analysis.

Repository: https://github.com/abhaymundhara/llm-benchmark-suite

## Overview

The suite is designed to compare cloud and local model providers under a shared evaluation pipeline. It records pass rate, latency, token usage, and failure categories, and exports results in both JSON and human-readable summary formats.

## Features

- Multiple benchmarks: HumanEval, MBPP, BigCodeBench, and SWE-bench variants
- CLI workflow for reproducible batch execution
- Streamlit UI for run configuration, progress monitoring, and result inspection
- Run comparison tools with overlap-aware task matching
- Support for OpenAI, Anthropic, Google Gemini, and Ollama-based local models
- Structured reporting with task-level diagnostics and summary metrics

## Getting Started

### Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

### Run the Streamlit app

```bash
streamlit run app.py
```

### Run a CLI benchmark

```bash
python3 runner.py --model ollama:qwen2.5-coder:7b --benchmark bigcodebench --limit 5
```

### Environment configuration

For cloud providers, set the required API keys before running benchmarks:

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

For SWE-bench runs, use `setup_docker.sh` to prepare the required Docker images.

## Project Structure

- `app.py` — Streamlit application
- `runner.py` — CLI benchmark runner and orchestration layer
- `benchmarks/` — benchmark implementations and registry
- `models/` — model adapters for supported providers
- `requirements.txt` — Python dependencies
- `run.sh` — convenience launcher for the Streamlit app
- `setup_docker.sh` — SWE-bench environment setup
- `Documentation/` — project notes, guides, and dataset references

## Benchmarks

### HumanEval

- 164 Python function-synthesis tasks
- Lightweight evaluation suitable for quick iteration

### MBPP (Mostly Basic Python Problems)

- 500 Python programming tasks
- Useful for basic code-generation evaluation

### BigCodeBench

- 1,140 tasks focused on practical coding workflows
- Includes `instruct` and `complete` variants

### SWE-bench

- Repository-level software-engineering tasks based on real GitHub issues
- Requires Docker-based evaluation

## Supported Models

### Cloud providers

- OpenAI models such as GPT-4 and GPT-3.5
- Anthropic Claude models
- Google Gemini models

### Local models via Ollama

- Qwen2.5-Coder
- DeepSeek-Coder
- Code Llama
- Other models available through Ollama

When Ollama is used with the default `max_tokens=512`, the adapter applies model-specific token limits automatically to reduce truncation and improve output quality.

## Metrics

- Pass rate
- Latency per task
- Input, output, and total token usage
- Estimated API cost for cloud models
- Failure categorization and diagnostics

## Output

Benchmark runs are written to the `reports/` directory:

- `benchmark_<name>_<timestamp>.json` — full structured results
- `benchmark_<name>_<timestamp>_summary.txt` — concise summary report

## Acknowledgments

- HumanEval — OpenAI
- MBPP — Google
- SWE-bench — Princeton NLP
- BigCodeBench — BigCode Project
