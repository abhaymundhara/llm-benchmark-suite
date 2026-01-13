# llm-benchmark-suite

A comprehensive benchmark suite for evaluating Large Language Models on coding tasks (HumanEval, MBPP, SWE-bench, BigCodeBench). Includes an interactive Streamlit UI, support for cloud and local models, and detailed reporting for pass rates, latency, and token usage.

Repository: https://github.com/abhaymundhara/llm-benchmark-suite

Quick links

- Documentation: `Documentation/`
- Issues / PRs: use the GitHub repo above

## 🎯 Features

- **Multiple Benchmarks**: HumanEval, MBPP, SWE-bench, BigCodeBench
- **Streamlit UI**: Interactive web interface for running benchmarks and viewing results
- **Benchmark Comparison**: Compare multiple runs side-by-side with smart task overlap detection
- **Model Support**: OpenAI, Anthropic Claude, Google Gemini, Ollama (local models)
- **Detailed Metrics**: Pass rates, latency, token usage, and cost tracking
- **Comprehensive Reports**: JSON and text summaries with task-level details

## 🚀 Quickstart

1. Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

2. Run the Streamlit UI (recommended for interactive use)

```bash
streamlit run app.py
```

3. Run a quick CLI benchmark (example)

```bash
python3 runner.py --model ollama:qwen2.5-coder:7b --benchmark bigcodebench --limit 5
```

Notes

- For cloud models set API keys in environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
- Use `setup_docker.sh` to prepare SWE-bench Docker images when evaluating SWE-bench tasks.

## 📁 Project layout (important files)

- `app.py` — Streamlit UI
- `runner.py` — CLI benchmark runner
- `requirements.txt` — Dependencies
- `setup_docker.sh` / `run.sh` — helper scripts
- `benchmarks/` — benchmark implementations (HumanEval, MBPP, BigCodeBench, SWE-bench variants)
- `models/` — model adapters for OpenAI, Claude, Gemini, Ollama
- `Documentation/` — user guides and dataset notes

## 🎮 Available Benchmarks

### HumanEval

- 164 Python programming problems
- Function implementation tasks
- Fast evaluation (~5-10 minutes for full set)

### MBPP (Mostly Basic Python Problems)

- 500 entry-level Python problems
- Beginner to intermediate difficulty
- Good for testing basic coding abilities

### BigCodeBench

- 1,140 practical coding tasks
- Multi-library function calls
- Two modes: `instruct` (NL) and `complete` (docstrings)

### SWE-bench

- Real-world GitHub issues
- Repository-level code changes
- Requires Docker for evaluation

## 🤖 Supported models

### Cloud APIs

- **OpenAI**: GPT-4, GPT-3.5, etc.
- **Anthropic**: Claude 3 (Opus, Sonnet, Haiku)
- **Google**: Gemini Pro, Gemini Flash

### Local (via Ollama)

- Qwen2.5-Coder
- DeepSeek Coder
- Code Llama
- And any other Ollama model

**Automatic Token Limit Configuration**: When using Ollama models with the default `max_tokens=512`, the adapter automatically adjusts the limit based on the model's architecture:

- Qwen2.5-Coder: 4096 tokens
- DeepSeek-Coder: 4096 tokens
- DeepSeek-R1: 8192 tokens
- CodeLlama: 4096 tokens
- Llama 3.x: 8192 tokens
- Mistral: 8192 tokens
- Mixtral: 32768 tokens

This ensures optimal output quality without manual configuration.

## 📊 Metrics Tracked

- **Pass Rate**: Percentage of tasks solved correctly
- **Latency**: Average time per task
- **Token Usage**: Input, output, and total tokens
- **Cost**: Estimated API costs (for cloud models)
- **Failure Analysis**: Categorized error types

## 🔧 Configuration & setup

### Environment Variables

For cloud models, set API keys:

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

### Ollama (local) example

```bash
# Install Ollama (if needed)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a coder model
ollama pull qwen2.5-coder:7b
```


## 📝 Output

Results are saved to `reports/` directory:

- `benchmark_<name>_<timestamp>.json` - Full results
- `benchmark_<name>_<timestamp>_summary.txt` - Human-readable summary


## 🙏 Acknowledgments

- **HumanEval**: OpenAI
- **MBPP**: Google
- **SWE-bench**: Princeton NLP
- **BigCodeBench**: BigCode Project
