# Inference Cost Optimizer

An intelligent LLM inference server that classifies requests by complexity and routes them to the optimal model — minimizing cost-per-token without sacrificing quality.

**Key idea**: Not every request needs a 7B parameter model. A simple factual query and a multi-step math proof need different amounts of reasoning. This system analyzes each request and routes it to the smallest model that can handle it well.

```
User Request
     |
     v
Classifier (analyzes complexity)
     |
     v
Router (decides which model)
     |
     v
Model Pool (TinyLlama / Phi-2 / Qwen2)
     |
     v
Dashboard (tracks cost savings)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python -m api.server

# Open interactive docs
# Visit http://localhost:8000/docs

# Run the benchmark suite
python benchmark/benchmark.py

# Test the classifier
python classifier/test_classifier.py --interactive
```

## Architecture

### Classifier (`classifier/classifier.py`)
Rule-based complexity analyzer. Checks:
- **Domain**: medical, legal, hard science -> larger model
- **Reasoning depth**: multi-step deduction -> larger model
- **Question type**: analysis, reasoning -> larger model
- **Technical content**: code, math -> larger model

Rules fire in priority order — the first matching rule wins. Easy to debug, easy to tune.

### Router (`router/router.py`)
Receives classification result, decides actual model to use. Handles:
- User-specified model override (respect their choice)
- SLA checking (does the model meet latency budget?)
- Routing statistics for observability

### Model Pool (`core/model_pool.py`)
Manages multiple HuggingFace models:
- Lazy loading (loads on first use)
- Async inference via ThreadPoolExecutor
- True GPU batch inference (multiple prompts in one forward pass)
- Per-model metrics

### KV Cache (`core/kv_cache.py`)
Simplified PagedAttention-style block allocator. Not production-grade, but demonstrates the concept: fixed-size blocks + page table = less memory fragmentation.

### Dashboard (`dashboard/dashboard.py`)
Observability layer:
- Per-model latency, throughput, error rate
- Routing distribution
- Cost savings vs naive baseline
- Complexity distribution

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `POST /generate` | POST | Generate text (auto-routes if model not specified) |
| `POST /classify` | POST | Preview routing decision without inference |
| `POST /batch` | POST | Batch multiple prompts (auto-routes each) |
| `GET /metrics` | GET | Full metrics dashboard |
| `GET /routing` | GET | Routing statistics |
| `GET /status` | GET | System health |

## Models

| Model | Size | Memory | Use Case |
|---|---|---|---|
| TinyLlama-1.1B | 1.1B params | ~2.2GB FP16 | Simple factual, greetings |
| Phi-2-2.7B | 2.7B params | ~5.4GB FP16 | Code, moderate reasoning |
| Qwen2-0.5B | 0.5B params | ~1GB FP16 | Fallback, long context |

## What This Is NOT

This is not a production inference system. It's a demonstration of:
- Request classification and routing
- Batch inference
- Memory management concepts
- System observability

For production inference, use [vLLM](https://github.com/vllm-project/vllm), [TGI](https://github.com/huggingface/text-generation-inference), or [llama.cpp](https://github.com/ggerganov/llama.cpp).

## Project Structure

```
inference-cost-optimizer/
├── api/
│   └── server.py          # FastAPI server, all endpoints
├── classifier/
│   ├── classifier.py       # Rule-based complexity classifier
│   └── test_classifier.py # Test suite + interactive mode
├── core/
│   ├── types.py           # Shared types (ModelSize enum)
│   ├── model_pool.py      # Model loading + inference
│   ├── kv_cache.py        # Block-based KV cache manager
│   └── day1_trace.py      # Generation tracing script
├── router/
│   └── router.py          # Routing logic + SLA enforcement
├── dashboard/
│   └── dashboard.py        # Metrics collection + observability
├── benchmark/
│   └── benchmark.py        # Benchmark suite
└── design/
    └── DESIGN.md          # Architecture decisions
```

## Key Design Decisions

See `design/DESIGN.md` for detailed reasoning on:
- Why batch inference improves throughput
- How the classifier avoids false positives
- Tradeoffs between latency and cost
- What a production KV cache allocator needs

## Benchmarks

Run `python benchmark/benchmark.py` to generate results. Expected outcomes:

- **Throughput**: 2-3x improvement vs naive single-model serving
- **Cost**: 40-60% reduction on simple queries
- **Latency**: 30-50% improvement for simple requests

(Benchmarks depend heavily on GPU hardware. Numbers above are approximate.)

## Motivation

Built as a learning project exploring LLM inference optimization. Inspired by vLLM's PagedAttention, HuggingFace's routing systems, and Perplexity's model selection strategies.

See `design/DESIGN.md` for full motivation and architecture.
