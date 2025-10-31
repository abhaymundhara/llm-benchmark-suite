# LLM Benchmark Suite

The LLM Benchmark Suite is a modular evaluation platform for large language models (LLMs) focused on software engineering benchmarks. It combines a Streamlit dashboard, configurable model adapters, and benchmark-specific harnesses to help teams measure performance across HumanEval, MBPP, and SWE-bench scenarios.

## Highlights
- Unified Streamlit dashboard with live progress and exportable reports
- Pluggable adapters for OpenAI, Anthropic Claude, Google Gemini, and local Ollama models
- Modular benchmark implementations with shared orchestration and metrics collection
- Extensible SWE-bench integration supporting both quick similarity scoring and the official Docker harness
- Typed, documented Python code with structured logging and robust error handling

## Directory Overview
```
project/
├── app.py                 # Streamlit UI
├── runner.py              # Benchmark orchestrator
├── models/                # Model adapter implementations
├── benchmarks/            # Benchmark logic and harnesses
├── Documentation/         # Guides and reference material
└── run.sh                 # Helper script for launching the UI
```

## Supported Model Providers
- **Ollama** – local models discovered via the REST `/api/tags` endpoint
- **OpenAI** – GPT-4 family (4o, 4.1, 3.5) via the `openai` SDK
- **Anthropic** – Claude 3 (Opus, Sonnet, Haiku)
- **Google** – Gemini 1.5 Pro and Flash via `google-generativeai`

Configure provider credentials in `.env`; see `Documentation/INSTALLATION.md`.

## Benchmarks
- **HumanEval** – canonical function synthesis with unit-test verification
- **MBPP** – natural language prompts with Python assertions
- **SWE-bench Demo** – similarity-based scoring using the SWE-bench Lite dataset
- **SWE-bench Full/Official** – hooks into the official harness via Docker

Benchmark behavior, dataset requirements, and evaluation details are covered in `Documentation/DATASETS.md` and `Documentation/FULL_SWE_BENCH.md`.

## Next Steps
1. Complete installation (dependencies and environment variables)
2. Run `./run.sh` to launch the Streamlit dashboard
3. Review `Documentation/CONTROLS_GUIDE.md` for an overview of the UI controls
4. Configure SWE-bench harness settings if you plan to evaluate full-scale tasks

For troubleshooting and advanced usage examples, consult the remaining documents in the `Documentation/` directory.
