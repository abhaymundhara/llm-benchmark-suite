# llm-benchmark-suite

A benchmark suite for evaluating large language models on coding tasks. The project supports HumanEval, MBPP, BigCodeBench, and SWE-bench, and provides both a command-line runner and a Streamlit interface for interactive analysis.

Repository: https://github.com/abhaymundhara/llm-benchmark-suite

## Artifact Documentation

- [REQUIREMENTS](REQUIREMENTS) — hardware, software, storage, and environment prerequisites
- [INSTALL](INSTALL) — installation, smoke-test, and basic usage validation
- [REPLICATION_GUIDE](REPLICATION_GUIDE) — instructions for repeating and reproducing benchmark results

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

## Common Installation and Runtime Issues

The checks below are intended to prevent the most frequent installation and execution failures.

### 1) Python or dependency mismatch

- Symptom: package install failures or import errors at startup.
- Check: `python --version` and confirm a supported interpreter (3.10 or 3.11).
- Resolution: create a fresh virtual environment and reinstall from `requirements.txt`.

### 2) Streamlit command not found

- Symptom: `streamlit` is missing when running `run.sh`.
- Check: ensure the virtual environment is active.
- Resolution: reinstall dependencies and run with `python -m streamlit run app.py` if needed.

### 3) Docker daemon unavailable (SWE-bench)

- Symptom: setup script reports daemon communication failure.
- Check: run `bash setup_docker.sh` and verify daemon health.
- Resolution: start Docker Desktop/Engine and rerun setup.

### 4) Cloud API authentication failures

- Symptom: provider returns authorization or credential errors.
- Check: verify `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GOOGLE_API_KEY` values.
- Resolution: export valid keys in current shell session or `.env`.

### 5) Local model not available (Ollama)

- Symptom: generation fails because model tag is missing.
- Check: verify Ollama service is running and model is pulled.
- Resolution: pull the model first (for example, `ollama pull qwen2.5-coder:7b`).

### 6) Insufficient disk for SWE-bench/container workflows

- Symptom: Docker pulls/builds fail or runs terminate unexpectedly.
- Check: available disk space.
- Resolution: free disk space and rerun setup; SWE-bench workflows require substantially more storage than lightweight benchmarks.

If a problem persists, open a GitHub issue with: OS version, Python version, benchmark/model configuration, command used, and full error output.

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

- OpenAI models such as GPT-5.4 and GPT-5.3-Codex
- Anthropic Claude models
- Google Gemini models

### Local models via Ollama

- Qwen2.5-Coder:7b
- GPT-OSS:20B
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
