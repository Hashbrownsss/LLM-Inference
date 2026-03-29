"""
FastAPI inference server — wires HTTP requests to the model pool.

Request flow:
    HTTP Request → FastAPI validates → model pool → batcher → GPU → response

Key design:
- Model pool is initialized once at startup (shared across all requests)
- Inference runs in ThreadPoolExecutor (doesn't block FastAPI's event loop)
- Batching collects multiple requests for GPU-parallel inference
- Streaming yields tokens as they're generated
"""

import asyncio
import time
import uuid
from typing import Optional
from dataclasses import dataclass, field
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from core.types import ModelSize


# =============================================================================
# Request / Response Models
# =============================================================================

class Request(BaseModel):
    """Incoming inference request."""
    prompt: str = Field(..., min_length=1, max_length=8192)
    model: ModelSize = Field(default=ModelSize.MEDIUM)
    max_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = Field(default=False, description="Stream tokens as they're generated")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class BatchRequest(BaseModel):
    """Multiple prompts processed as a batch — more efficient than single requests."""
    prompts: list[str] = Field(..., min_length=1, max_length=32)
    model: ModelSize = Field(default=ModelSize.MEDIUM)
    max_tokens: int = Field(default=128, ge=1, le=1024)


class GenerationResponse(BaseModel):
    """Non-streaming response."""
    request_id: str
    text: str
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    batched: bool = False


@dataclass
class RequestRecord:
    """Tracks a request through the system for metrics."""
    request_id: str
    prompt: str
    model: ModelSize
    max_tokens: int
    temperature: float
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None


_request_registry: dict[str, RequestRecord] = {}


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Inference Cost Optimizer",
    description="Routes LLM requests to the optimal model to minimize cost-per-token.",
    version="0.2.0",
)


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Health check."""
    return {
        "service": "Inference Cost Optimizer",
        "version": "0.2.0",
        "status": "running",
        "models": [m.value for m in ModelSize],
        "docs": "/docs",
    }


@app.get("/status")
async def status():
    """Queue and system health."""
    pool = get_model_pool()
    return {
        "queues": {m.value: pool.get_queue_size(m) for m in ModelSize},
        "metrics": pool.get_metrics(),
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: Request) -> GenerationResponse:
    """
    Generate text from a single prompt.

    If stream=True, returns a streaming response (tokens arrive incrementally).
    Otherwise returns the complete text.
    """
    record = RequestRecord(
        request_id=request.request_id,
        prompt=request.prompt,
        model=request.model,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    _request_registry[request.request_id] = record

    if request.stream:
        # Streaming path — return SSE stream
        return StreamingResponse(
            _stream_tokens(request),
            media_type="text/event-stream",
        )

    try:
        pool = get_model_pool()
        record.started_at = time.time()
        result = await pool.generate(
            request.model,
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        record.completed_at = time.time()

        return GenerationResponse(
            request_id=request.request_id,
            text=result["text"],
            model_used=request.model.value,
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            latency_ms=result["latency_ms"],
            batched=False,
        )

    except Exception as e:
        record.error = str(e)
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_tokens(request: Request):
    """
    Generator that yields tokens as they're produced.

    Uses Server-Sent Events (SSE) format:
        data: {"token": "Hello", "done": false}
        data: {"token": " world", "done": false}
        data: {"token": "", "done": true}

    Why SSE and not WebSocket?
    - Simpler to implement
    - Works over HTTP (no upgrade needed)
    - Good enough for token streaming
    - HuggingFace's Inference API uses SSE
    """
    import json
    try:
        pool = get_model_pool()
        async for token_text in pool.generate_stream(
            request.model,
            request.prompt,
            request.max_tokens,
            request.temperature,
        ):
            yield f"data: {json.dumps({'token': token_text, 'done': False})}\n\n"
        yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"

    except Exception as e:
        error_json = json.dumps({"error": str(e)})
        yield f"data: {error_json}\n\n"


@app.post("/batch", response_model=list[GenerationResponse])
async def batch_generate(request: BatchRequest) -> list[GenerationResponse]:
    """
    Process multiple prompts in a single batched inference call.

    Batching is significantly more efficient than individual requests:
    - GPU processes all sequences in parallel in one forward pass
    - Memory bandwidth cost is amortized across all sequences
    - Typical throughput improvement: 3-5x vs sequential processing

    Trade-off: all sequences in a batch share the same max_tokens and temperature.
    """
    if not request.prompts:
        return []

    try:
        pool = get_model_pool()
        start = time.time()
        results = await pool.generate_batch(
            request.model,
            request.prompts,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        total_latency = (time.time() - start) * 1000

        return [
            GenerationResponse(
                request_id=f"batch-{request.prompts.index(prompt)}",
                text=r["text"],
                model_used=request.model.value,
                prompt_tokens=r["prompt_tokens"],
                completion_tokens=r["completion_tokens"],
                latency_ms=total_latency,  # batch latency is shared
                batched=True,
            )
            for prompt, r in zip(request.prompts, results)
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Return performance metrics for all models."""
    pool = get_model_pool()
    return pool.get_metrics()


@app.delete("/request/{request_id}")
async def cancel_request(request_id: str):
    """Cancel a request (marks it, doesn't stop GPU computation mid-run)."""
    if request_id not in _request_registry:
        raise HTTPException(status_code=404, detail="Request not found")
    _request_registry[request_id].error = "cancelled"
    return {"status": "cancelled", "request_id": request_id}


# =============================================================================
# Lifecycle
# =============================================================================

_start_time = time.time()


@app.on_event("startup")
async def startup():
    """Initialize the model pool on server start."""
    from core.model_pool import get_model_pool
    print("="*60)
    print("  Inference Cost Optimizer — Starting")
    print("="*60)
    # Don't load models here — lazy load on first request
    print("  Models will load on first use (lazy loading)")
    print("  Docs: http://localhost:8000/docs")
    print("="*60)


@app.on_event("shutdown")
async def shutdown():
    print("Shutting down — releasing GPU resources")


# =============================================================================
# Entry Point
# =============================================================================

def get_model_pool():
    """Lazy initialization — creates model pool on first use."""
    global _model_pool
    if _model_pool is None:
        from core.model_pool import ModelPool
        _model_pool = ModelPool()
    return _model_pool


_model_pool = None


if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)
