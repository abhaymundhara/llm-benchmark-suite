# LLM Benchmark Suite

A comprehensive benchmarking system for evaluating Large Language Models on coding tasks, featuring a Streamlit UI and support for multiple benchmark datasets.

## 🎯 Features

- **Multiple Benchmarks**: HumanEval, MBPP, SWE-bench, BigCodeBench
- **Streamlit UI**: Interactive web interface for running benchmarks and viewing results
- **Model Support**: OpenAI, Anthropic Claude, Google Gemini, Ollama (local models)
- **Detailed Metrics**: Pass rates, latency, token usage, and cost tracking
- **Comprehensive Reports**: JSON and text summaries with task-level details

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Streamlit UI

```bash
streamlit run app.py
```

### 3. Or Use Command Line

```bash
python runner.py --model ollama:qwen2.5-coder:7b --benchmark human_eval --limit 10
```

## 📁 Project Structure

```
├── app.py                      # Streamlit UI
├── runner.py                   # CLI benchmark runner
├── requirements.txt            # Python dependencies
├── run.sh                      # Quick run script
├── setup_docker.sh            # Docker setup for SWE-bench
│
├── benchmarks/                 # Benchmark implementations
│   ├── __init__.py            # Registry
│   ├── base.py                # Base classes
│   ├── humaneval.py           # HumanEval benchmark
│   ├── mbpp.py                # MBPP benchmark
│   ├── bigcodebench.py        # BigCodeBench benchmark
│   ├── swe_bench.py           # SWE-bench demo
│   ├── swe_bench_full.py      # SWE-bench full dataset
│   └── swe_bench_official.py  # SWE-bench official evaluation
│
├── models/                     # Model adapters
│   ├── __init__.py            # Model registry
│   ├── base.py                # Base adapter
│   ├── openai_adapter.py      # OpenAI models
│   ├── claude_adapter.py      # Anthropic Claude
│   ├── gemini_adapter.py      # Google Gemini
│   └── ollama_adapter.py      # Ollama (local)
│
├── scripts/                    # Helper scripts
│   └── run_single.py          # Run single task
│
└── Documentation/              # User guides
    ├── QUICKSTART.md
    ├── INSTALLATION.md
    ├── DATASETS.md
    ├── BIGCODEBENCH.md
    └── ...
```

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

## 🤖 Supported Models

### Cloud APIs

- **OpenAI**: GPT-4, GPT-3.5, etc.
- **Anthropic**: Claude 3 (Opus, Sonnet, Haiku)
- **Google**: Gemini Pro, Gemini Flash

### Local (via Ollama)

- Qwen2.5-Coder
- DeepSeek Coder
- Code Llama
- And any other Ollama model

## 📊 Metrics Tracked

- **Pass Rate**: Percentage of tasks solved correctly
- **Latency**: Average time per task
- **Token Usage**: Input, output, and total tokens
- **Cost**: Estimated API costs (for cloud models)
- **Failure Analysis**: Categorized error types

## 🔧 Configuration

### Environment Variables

For cloud models, set API keys:

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

### Ollama Setup

For local models:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull qwen2.5-coder:7b
```

## 📖 Documentation

See the `Documentation/` folder for detailed guides:

- [Installation Guide](Documentation/INSTALLATION.md)
- [Quick Start Guide](Documentation/QUICKSTART.md)
- [Dataset Information](Documentation/DATASETS.md)
- [BigCodeBench Guide](Documentation/BIGCODEBENCH.md)
- [SWE-bench Setup](Documentation/SWE_BENCH_SETUP.md)

## 🧪 Running Tests

The test files are excluded from the repository. For development:

```bash
# Example: Test HumanEval
python -c "from benchmarks import registry; b = registry.create('human_eval', limit=3); print(list(b.load_tasks()))"
```

## 📝 Output

Results are saved to `reports/` directory:

- `benchmark_<name>_<timestamp>.json` - Full results
- `benchmark_<name>_<timestamp>_summary.txt` - Human-readable summary

## 🤝 Contributing

This is a benchmarking tool. To add new benchmarks:

1. Create a new file in `benchmarks/`
2. Extend the `Benchmark` base class
3. Register in `benchmarks/__init__.py`

## 📄 License

See LICENSE file for details.

## 🙏 Acknowledgments

- **HumanEval**: OpenAI
- **MBPP**: Google
- **SWE-bench**: Princeton NLP
- **BigCodeBench**: BigCode Project
