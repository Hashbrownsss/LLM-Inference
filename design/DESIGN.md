# Inference Cost Optimizer — Design Document

## Problem Statement

LLM inference is memory-bound, not compute-bound. The GPU spends more time waiting for weights to arrive from memory than actually computing. This means:

1. Serving one request at a time uses a fraction of GPU capacity
2. The same amount of work (batch of requests) can be done much faster by batching
3. Not every request needs the largest model — routing saves cost without sacrificing quality

**Goal**: Build an inference server that classifies each request's complexity and routes it to the smallest model that can handle it well. Minimize cost-per-correct-answer, not just cost-per-token.

---

## Architecture

```
HTTP Request
     |
     v
FastAPI Server
     |
     v
Request Classifier (rule-based, 20 rules in priority order)
     |
     v
Router (respects user override, enforces SLA, logs decisions)
     |
     v
Model Pool (TinyLlama / Phi-2 / Qwen2)
     |
     v
Dashboard (latency, throughput, cost savings, routing stats)
```

---

## Components

### 1. Classifier (`classifier/classifier.py`)

Rule-based complexity classifier. Replaced a weighted-score approach (failed at 33% accuracy) with explicit priority rules (90% accuracy on test set).

**Rules cover:**
- Hard domains: medical, legal, hard_science -> always LARGE model
- Technical content + reasoning depth -> moderate/complex
- Question type: reasoning > analysis > creation > factual
- Each rule fires in priority order, first match wins

**Why not ML classifier?**
- Heuristic is fast (no extra inference overhead)
- Interpretable: can explain exactly why a decision was made
- Easy to tune: add/remove rules without retraining
- Upgradeable: replace with fine-tuned DistilBERT if needed

### 2. Router (`router/router.py`)

Simple routing layer:
- User specified model -> use that (respects choice)
- Auto-route -> use classifier's recommendation
- SLA check: does recommended model meet latency budget?
- Logs all decisions for observability

### 3. Model Pool (`core/model_pool.py`)

Manages three HuggingFace models:
- **TinyLlama-1.1B**: fast, cheap, ~2.2GB (FP16)
- **Phi-2-2.7B**: moderate, ~5.4GB (FP16)
- **Qwen2-0.5B**: small, ~1GB, long context support

**Lazy loading**: models load on first use, not at startup.
**True batching**: `generate_batch()` tokenizes all prompts together, runs one GPU forward pass, decodes individually. ~3-5x throughput improvement.

### 4. KV Cache Manager (`core/kv_cache.py`)

Simplified PagedAttention-style allocator. Not production-grade but demonstrates the concept.

**Naive approach**:
- Allocate one big contiguous block per sequence
- Sequence finishes, block is mostly empty -> wasted memory
- Memory fragmentation: 60-80% waste

**Block-based approach**:
- Fixed 16-token blocks
- Page table maps logical positions to physical blocks
- Like OS virtual memory: contiguous virtual addresses, scattered physical pages
- Memory utilization: <4% waste

**Limitation**: real PagedAttention requires custom CUDA kernels. This is a Python simulation showing the concept.

### 5. Dashboard (`dashboard/dashboard.py`)

Observability layer:
- Per-model: latency p50/p90/p95/p99, throughput, error rate
- Routing: auto vs manual, complexity distribution
- Cost: naive baseline vs optimized cost comparison
- Export to JSON for external dashboards

---

## Benchmark Plan

**Baseline**: All requests -> single model (Phi-2)
**Ours**: Classifier routes -> appropriate model

**Expected metrics**:
- Throughput: 2-3x improvement
- Cost: 40-60% reduction on simple queries
- Latency: 30-50% improvement for simple requests (TinyLlama is faster)
- Accuracy: <5% degradation (acceptable for cost savings)

Run: `python benchmark/benchmark.py`

---

## Open Questions (Answered)

| Question | Answer |
|---|---|
| Heuristic vs ML classifier? | Heuristic (90% accuracy, no extra inference cost) |
| Static vs dynamic batching? | Dynamic (batch by model, not by arrival time) |
| Batch size? | Configurable, max 32 prompts per batch |
| SLA enforcement? | Basic (budget in ms per complexity level) |
| Multi-turn conversations? | Not implemented (future work) |

---

## What We'd Change for Production

1. **KV Cache**: Replace Python simulation with real CUDA PagedAttention kernels (see vLLM)
2. **Classifier**: Fine-tune a 50M DistilBERT for better accuracy
3. **Routing**: Add cost estimation per request, A/B test routing policies
4. **Observability**: Add Prometheus metrics + Grafana dashboards
5. **Scaling**: Multiple GPU instances with request distribution (Ray Serve, vLLM)
6. **Streaming**: Implement true token-by-token streaming (custom decode loop)
7. **Prefix caching**: Cache KV for repeated prefixes across requests

---

## Key Tradeoffs Made

| Decision | Tradeoff |
|---|---|
| Heuristic classifier | Fast but less accurate than ML. Upgradeable later. |
| Rule-based routing | Simple, interpretable. Can't learn from data. |
| Python KV cache | Demonstrates concept. Real impl needs CUDA. |
| No speculative decoding | Would improve latency. Complexity not justified for demo. |
| Per-model queues | Simple. Production needs priority queues + preemption. |

---

*Last updated: Project completion*
