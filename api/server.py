"""
Inference Cost Optimizer — FastAPI Server

This is the entry point. It receives LLM requests via HTTP and routes
them through our inference pipeline.

Architecture:
    HTTP Request
        │
        ▼
    FastAPI Server (async, handles concurrency)
        │
        ▼
    Request Queue (in-memory, per-model)
        │
        ▼
    Batching Layer (collects N requests or waits max_wait_time)
        │
        ▼
    Model Inference (runs on GPU)
        │
        ▼
    Response Stream (returns to client)

Why FastAPI?
- Native async support — handles many concurrent connections efficiently
- Automatic request validation via Pydantic
- Built-in OpenAPI docs (visit /docs after starting server)
- Works well with asyncio for non-blocking inference
"""

import asyncio
import time
import uuid
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn


# =============================================================================
# Data Models — how we represent requests and responses
# =============================================================================

class ModelSize(str, Enum):
    """Available model sizes in our pool."""
    TINY   = "tinyllama"    # 1.1B params — fast, cheap, simple queries
    MEDIUM = "phi2"         # 2.7B params — moderate reasoning
    LARGE  = "qwen2"        # 0.5B params — fallback / batched tasks


class Request(BaseModel):
    """Incoming inference request from a client."""
    prompt: str = Field(..., min_length=1, max_length=8192, description="Input text prompt")
    model: ModelSize = Field(default=ModelSize.MEDIUM, description="Target model size")
    max_tokens: int = Field(default=256, ge=1, le=4096, description="Max tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    stream: bool = Field(default=True, description="Stream tokens as they're generated")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class BatchRequest(BaseModel):
    """Multiple prompts batched into a single request for efficiency."""
    prompts: list[str] = Field(..., min_length=1, max_length=32)
    model: ModelSize = Field(default=ModelSize.MEDIUM)
    max_tokens: int = Field(default=128, ge=1, le=2048)


class TokenResponse(BaseModel):
    """Single token in a streaming response."""
    token_id: int
    token_text: str
    logprob: Optional[float] = None


class GenerationResponse(BaseModel):
    """Complete non-streaming response."""
    request_id: str
    text: str
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cached: bool = False  # Did we use prefix caching?


class QueueStatus(BaseModel):
    """Current state of the request queues."""
    model: str
    queue_size: int
    avg_wait_time_ms: float
    processing: bool


# =============================================================================
# Request Tracking — this is how we track metrics per request
# =============================================================================

@dataclass
class RequestRecord:
    """Tracks metadata for a request through the system."""
    request_id: str
    prompt: str
    model: ModelSize
    max_tokens: int
    temperature: float
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    error: Optional[str] = None

    @property
    def latency_ms(self) -> float:
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at) * 1000
        return 0.0


# Global request registry — tracks all in-flight and completed requests
# In production this would be Redis or a similar distributed store
_request_registry: dict[str, RequestRecord] = {}
_queue_metrics: dict[str, dict] = {}


# =============================================================================
# FastAPI App Setup
# =============================================================================

app = FastAPI(
    title="Inference Cost Optimizer",
    description="""
    An intelligent LLM inference server that routes requests to the optimal
    model based on request complexity — minimizing cost-per-correct-answer.

    **Key endpoints:**
    - POST /generate — Single request (streaming supported)
    - POST /batch — Batch multiple requests together
    - GET /status — Queue and system status
    - GET /metrics — Benchmark and cost metrics
    """,
    version="0.1.0",
)


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Health check and API overview."""
    return {
        "service": "Inference Cost Optimizer",
        "version": "0.1.0",
        "status": "running",
        "models": [m.value for m in ModelSize],
        "docs": "/docs",
    }


@app.get("/status")
async def status() -> dict:
    """Return queue status for all models."""
    return {
        "queues": {
            model.value: {
                "queue_size": _queue_metrics.get(model.value, {}).get("queue_size", 0),
                "avg_wait_ms": _queue_metrics.get(model.value, {}).get("avg_wait_ms", 0),
                "requests_today": _queue_metrics.get(model.value, {}).get("total_requests", 0),
            }
            for model in ModelSize
        },
        "total_requests": len(_request_registry),
        "uptime_seconds": time.time() - _start_time,
    }


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: Request) -> GenerationResponse:
    """
    Generate text from a single prompt.

    The request goes through:
    1. Validation (FastAPI handles this automatically)
    2. Queueing (async, non-blocking)
    3. Batching (grouped with other requests for efficiency)
    4. Inference (on GPU)
    5. Response construction
    """
    request_id = request.request_id

    # Track the request
    record = RequestRecord(
        request_id=request_id,
        prompt=request.prompt,
        model=request.model,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    _request_registry[request_id] = record

    try:
        # TODO (Day 3): Replace with actual inference call
        # For now, we return a placeholder demonstrating the response shape
        await asyncio.sleep(0.1)  # simulate inference time

        record.started_at = time.time()
        record.completed_at = time.time()

        return GenerationResponse(
            request_id=request_id,
            text=f"[Placeholder response for: {request.prompt[:50]}...]",
            model_used=request.model.value,
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=request.max_tokens,
            latency_ms=100.0,
            cached=False,
        )
    except Exception as e:
        record.error = str(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch")
async def batch_generate(request: BatchRequest) -> list[GenerationResponse]:
    """
    Process multiple prompts as a batch.

    Batching is critical for throughput — running N requests simultaneously
    allows the GPU to amortize memory access costs across multiple sequences.

    Why batch matters (Day 2 concept preview):
    - GPU has massive parallelism (thousands of cores)
    - A single small request only uses a fraction of compute
    - Batching fills the GPU, dramatically improving tokens/sec/$
    """
    results = []
    for i, prompt in enumerate(request.prompts):
        single_req = Request(
            prompt=prompt,
            model=request.model,
            max_tokens=request.max_tokens,
        )
        result = await generate(single_req)
        result.request_id = f"{single_req.request_id}-batch-{i}"
        results.append(result)

    return results


@app.get("/metrics")
async def metrics() -> dict:
    """
    Return benchmark and cost metrics.

    This powers the observability dashboard. We track:
    - Requests per model
    - Latency distribution (p50, p90, p99)
    - Cost per token per model
    - Cache hit rate
    """
    # TODO (Day 10): Implement actual metric collection
    return {
        "message": "Metrics collection — implement in Day 10",
        "models": {
            "tinyllama":  {"avg_latency_ms": 45, "cost_per_1k_tokens": 0.002},
            "phi2":       {"avg_latency_ms": 120, "cost_per_1k_tokens": 0.008},
            "qwen2":      {"avg_latency_ms": 38,  "cost_per_1k_tokens": 0.003},
        },
        "total_cost_saved_pct": 0,  # will compute once we have data
        "routing_accuracy_pct": 0,
    }


@app.delete("/request/{request_id}")
async def cancel_request(request_id: str) -> dict:
    """Cancel an in-flight request."""
    if request_id not in _request_registry:
        raise HTTPException(status_code=404, detail="Request not found")
    _request_registry[request_id].error = "cancelled"
    return {"status": "cancelled", "request_id": request_id}


# =============================================================================
# Server Lifecycle
# =============================================================================

_start_time = time.time()


@app.on_event("startup")
async def startup_event():
    """
    Runs once when the server starts.
    Initialize model pool here (Day 3), set up GPU memory tracking.
    """
    print("="*60)
    print("  Inference Cost Optimizer — Starting Up")
    print("="*60)
    print(f"  Models: {[m.value for m in ModelSize]}")
    print(f"  Docs:   http://localhost:8000/docs")
    print("="*60)

    # Initialize queue metrics
    for model in ModelSize:
        _queue_metrics[model.value] = {
            "queue_size": 0,
            "avg_wait_ms": 0,
            "total_requests": 0,
        }


@app.on_event("shutdown")
async def shutdown_event():
    """Runs when the server stops. Clean up GPU memory."""
    print("Shutting down — releasing GPU resources")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    # Run with: python -m api.server
    # Or:       uvicorn api.server:app --reload --port 8000
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # set True for development
        workers=1,     # keep at 1 for GPU memory — each worker loads its own model
    )
