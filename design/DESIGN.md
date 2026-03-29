# Inference Cost Optimizer — Design Document

> **Problem Statement**: LLM inference is expensive and memory-bound. Serving all requests with the same model wastes resources on simple queries and starves complex ones. We build an intelligent routing layer that classifies requests by complexity and routes them to the optimal model, minimizing cost-per-correct-answer.

---

## Problem Analysis

### Why LLM Inference Is Memory-Bound

A language model generates text one token at a time. Each new token must:
1. Load all model weights from GPU memory (~14GB for 7B FP16)
2. Load the entire KV cache of all previous tokens
3. Perform a small amount of computation
4. Repeat for every token

**The bottleneck is memory bandwidth, not compute.** Modern GPUs have compute that far exceeds memory transfer speed. The GPU sits idle waiting for weights to arrive.

### KV Cache Growth

During autoregressive generation, attention requires Key and Value tensors for **every previous token**:

```
Context length  512:  KV cache ≈  64 MB/layer
Context length 2048:  KV cache ≈ 256 MB/layer
Context length 4096:  KV cache ≈ 512 MB/layer
```

For a 32-layer model, that's gigabytes of KV cache. This memory must be managed efficiently — naive allocation leads to fragmentation and OOM.

### vLLM's Insight: PagedAttention

Instead of allocating one contiguous block per sequence, PagedAttention allocates fixed-size blocks (typically 16 tokens each). A page table maps logical token positions to physical blocks. This is identical to how OS virtual memory works.

**Key benefit**: Reduces memory fragmentation from ~60-80% to <4%, enabling 2-4x more sequences in GPU memory.

### Our Angle: Cost-Driven Routing

vLLM optimizes for **throughput**. We optimize for **cost-per-correct-answer**. These are fundamentally different objectives:

| Optimization Target | Metric | Strategy |
|---|---|---|
| vLLM | tokens/sec/GB | Maximize batch size, minimize memory waste |
| Ours | cost/token × accuracy | Route each request to minimum-cost sufficient model |

**Hypothesis**: Not every request needs a 7B parameter model. Simple factual queries, classification tasks, and short responses can be handled by small quantized models with equivalent quality.

---

## Architecture

```
User Request
     │
     ▼
┌──────────────────────────────────────────────┐
│           Inference Cost Optimizer             │
│                                               │
│  1. Request Ingestion (FastAPI)               │
│           │                                   │
│           ▼                                   │
│  2. Request Classifier (heuristic or ML)      │
│      "simple" → small model pool              │
│      "complex" → large model pool             │
│           │                                   │
│           ▼                                   │
│  3. Router + Scheduler                        │
│      Async queues, batching, backpressure       │
│           │                                   │
│           ▼                                   │
│  4. Model Pool                                │
│      TinyLlama-1.1B / Phi-2 / Qwen-0.5B      │
│                                               │
│  5. Observability Layer                       │
│      Cost tracking, latency breakdown          │
└──────────────────────────────────────────────┘
```

### Component Responsibilities

**Request Classifier**
- Input: raw prompt text
- Output: complexity label (simple / moderate / complex)
- Methods: heuristic (length, keywords) or ML classifier
- Why: separates concerns; we can improve classification independently

**Router**
- Input: complexity label + request queue
- Output: routes to appropriate model pool
- Handles: SLA enforcement, queue prioritization
- Why: decoupled from classifier; enables A/B testing different routing policies

**Model Pool**
- Multiple models at different sizes and quantization levels
- Each model runs in its own async process
- Exposes consistent interface regardless of underlying model

**Observability Layer**
- Metrics: tokens/sec, memory usage, cache hit rate, cost/token
- Latency breakdown: prefill time vs decode time
- Per-model accuracy tracking (ground truth evaluation)

---

## Design Decisions + Trade-offs

### Decision 1: Heuristic vs ML Classifier

**Option A — Heuristic**: Use prompt length, presence of code/math keywords, estimated reasoning steps.
- Pros: Fast, no extra model, interpretable
- Cons: Limited accuracy, can't capture nuanced complexity

**Option B — Small ML Classifier**: Fine-tune a 50M parameter model to classify request complexity.
- Pros: More accurate, can learn from data
- Cons: Extra inference cost for classification, adds latency

**Chosen**: Start with Option A (heuristic), validate on real traffic, upgrade to Option B if heuristic misses >20% of cases.

### Decision 2: Model Pool Composition

| Model | Parameters | Quantization | Memory | Use Case |
|---|---|---|---|---|
| TinyLlama-1.1B | 1.1B | Q4_K_M | ~600MB | Simple factual, short responses |
| Phi-2 | 2.7B | Q4_K_M | ~1.5GB | Moderate reasoning, code |
| Qwen2-0.5B | 0.5B | Q8 | ~600MB | Fallback, batched simple tasks |

### Decision 3: Batching Strategy

Naive batching: Wait for N requests, process together.
- Problem: Simple requests wait behind complex ones → high latency

**Chosen: Dynamic batching with max_wait_time**
- Batch requests up to max_wait_time (e.g., 100ms)
- After timeout, process whatever is in the batch
- Small/simple requests don't wait long; large batches maximize throughput

### Decision 4: Evaluation Metric

We track **cost-weighted accuracy**:
```
Score = (accuracy × request_complexity_weight) / cost_per_token
```

A request classified as "simple" but answered incorrectly costs more than a correct answer — the classifier made the wrong routing decision. We weight by complexity to prioritize correct complex requests over correct simple ones.

---

## Benchmark Plan

### Baseline: Naive Serving
All requests → single model (Phi-2 2.7B). Measure:
- Throughput: tokens/sec
- Memory usage: peak GPU memory
- Cost/token: computed from GPU-hours and tokens processed
- Accuracy: on a benchmark dataset

### Our System
Same benchmark, but with routing. Measure:
- Same metrics per model in the pool
- Routing accuracy: % of requests correctly classified
- Cost savings: (naive_cost - optimized_cost) / naive_cost
- Accuracy delta: did routing hurt accuracy?

### Expected Results

```
Hypothesis: 60-70% of requests are "simple" (can be handled by TinyLlama)
→ Cost savings: 40-60% reduction in compute cost
→ Accuracy impact: <5% accuracy loss (acceptable threshold)
```

---

## Implementation Plan

### Week 1 (Current Sprint)
- [ ] Day 1: Mental model, project setup, trace TinyLlama
- [ ] Day 2: FastAPI server, request ingestion
- [ ] Day 3: Model pool, basic serving
- [ ] Day 4: Request classifier (heuristic)
- [ ] Day 5: Router, dynamic batching
- [ ] Day 6: KV cache basics, memory tracking
- [ ] Day 7: First benchmark run

### Week 2
- [ ] Day 8-9: Observability dashboard
- [ ] Day 10-11: Refine classifier
- [ ] Day 12-13: Documentation + blog post
- [ ] Day 14: Final benchmarks + polish

---

## Open Questions

1. What is the accuracy threshold below which routing hurts more than it helps?
2. Can we predict cost savings from request characteristics before running inference?
3. Should we do speculative decoding on complex requests routed to small models?
4. How do we handle multi-turn conversations with context carry-over?

---

*Last updated: Day 1*
