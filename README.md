# Inference Cost Optimizer

An intelligent LLM inference server that routes requests to the optimal model based on request complexity — minimizing cost-per-correct-answer.

> **Why this project?** vLLM optimizes for throughput. I'm trying to optimize for cost-per-correct-answer. These are fundamentally different objectives. [Read the design document](design/DESIGN.md)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Trace through LLM generation (Day 1 exercise)
python core/day1_trace.py

# 3. Start the server
uvicorn api.server:app --reload --port 8000

# 4. Visit http://localhost:8000/docs for the interactive API
```

## Project Structure

```
LLM-Inference/
├── api/
│   └── server.py          # FastAPI server, request/response models
├── core/
│   ├── day1_trace.py     # Trace through autoregressive generation
│   └── model_pool.py     # Model loading, inference, metrics
├── design/
│   └── DESIGN.md         # Architecture decisions and trade-offs
├── classifier/            # Request complexity classification
├── router/               # Routing logic and scheduling
├── dashboard/            # Observability and metrics
├── requirements.txt
└── README.md
```

## Architecture

```
User Request
     │
     ▼
Request Classifier (heuristic → simple / moderate / complex)
     │
     ▼
Router + Dynamic Batcher
     │
     ▼
┌────────┬────────┬────────┐
│ Tiny   │ Phi-2  │ Qwen2  │  ← Model Pool
│ Llama  │        │        │
│ 1.1B   │ 2.7B   │ 0.5B   │
└────────┴────────┴────────┘
     │
     ▼
Observability Dashboard
```
## Running Experiments

```python
from core.model_pool import get_model_pool
import asyncio

async def benchmark():
    pool = get_model_pool()

    # Test each model
    for size in [ModelSize.TINY, ModelSize.MEDIUM, ModelSize.LARGE]:
        result = await pool.generate(
            size,
            "Explain why the sky is blue in one sentence.",
            max_tokens=50
        )
        print(f"{size.value}: {result['latency_ms']:.0f}ms, "
              f"{result['completion_tokens']} tokens, "
              f"{result['completion_tokens']/result['latency_ms']*1000:.1f} tok/s")

asyncio.run(benchmark())
```

## Benchmark Results

*yet to be filled*

## What I'm Learning

> **The core insight**: LLM inference is memory-bound, not compute-bound. The GPU waits for weights to arrive from memory more than it computes. This is why batching helps — filling GPU compute while waiting for memory.

See [DESIGN.md](design/DESIGN.md) for detailed reasoning behind every decision.

