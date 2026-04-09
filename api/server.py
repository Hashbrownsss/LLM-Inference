"""
FastAPI inference server — complete end-to-end pipeline.

Request flow:
    HTTP Request
        -> Classifier (determines complexity)
        -> Router (decides model, enforces SLA)
        -> Model Pool (batched GPU inference)
        -> Dashboard (metrics collection)
        -> Response

This is the fully wired version of the system.
"""

import asyncio
import time
import uuid
from typing import Optional
from dataclasses import dataclass, field
from fastapi import FastAPI, HTTPException
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
    model: ModelSize | None = Field(
        default=None,
        description="Model size. None = auto-select based on prompt complexity"
    )
    max_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = Field(default=False)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class BatchRequest(BaseModel):
    """Batch of prompts processed together."""
    prompts: list[str] = Field(..., min_length=1, max_length=32)
    model: ModelSize | None = Field(default=None)
    max_tokens: int = Field(default=64, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class GenerationResponse(BaseModel):
    """Response to a generation request."""
    request_id: str
    text: str
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    routed_automatically: bool = False
    routing_reason: str = ""


class RoutingInfo(BaseModel):
    """Information about how a request was routed."""
    requested_model: str | None
    actual_model: str
    was_auto_routed: bool
    reason: str
    complexity: str | None = None
    confidence: float | None = None


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Inference Cost Optimizer",
    description=(
        "An intelligent LLM inference server that classifies incoming requests "
        "by complexity and routes them to the optimal model — minimizing "
        "cost-per-token while maintaining quality."
    ),
    version="1.0.0",
)


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/")
async def root():
    return {
        "service": "Inference Cost Optimizer",
        "version": "1.0.0",
        "status": "running",
        "models": [m.value for m in ModelSize],
        "docs": "/docs",
    }


@app.get("/status")
async def status():
    pool = _get_pool()
    router = _get_router()
    dash = _get_dashboard()
    return {
        "uptime_seconds": round(time.time() - _start_time, 1),
        "metrics": pool.get_metrics(),
        "routing_stats": router.get_stats(),
        "dashboard": {
            "total_requests": sum(d.requests for d in dash._models.values()),
            "total_tokens": sum(d.total_tokens for d in dash._models.values()),
        },
    }


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: Request) -> GenerationResponse:
    """Generate text. If model=None, auto-routes based on prompt complexity."""
    router = _get_router()
    pool = _get_pool()
    dash = _get_dashboard()

    # Route the request
    decision = router.route(request.prompt, request.model)
    actual_model = decision.actual_model

    # Record routing decision
    dash.record_routing(
        was_auto=decision.was_auto_routed,
        complexity=(
            decision.classification.complexity_level.value
            if decision.classification else "unknown"
        ),
        model_used=actual_model,
    )

    try:
        start = time.time()
        result = await pool.generate(
            actual_model,
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        latency_ms = (time.time() - start) * 1000

        # Record metrics
        dash.record_request(
            model=actual_model,
            latency_ms=latency_ms,
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
        )

        return GenerationResponse(
            request_id=request.request_id,
            text=result["text"],
            model_used=actual_model.value,
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            latency_ms=round(latency_ms, 1),
            routed_automatically=decision.was_auto_routed,
            routing_reason=decision.reason,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify")
async def classify(request: Request):
    """
    Preview how a request would be routed — without running inference.
    Useful for debugging the classifier.
    """
    router = _get_router()
    decision = router.route(request.prompt, request.model)
    return {
        "prompt": request.prompt,
        "requested_model": request.model.value if request.model else None,
        "actual_model": decision.actual_model.value,
        "was_auto_routed": decision.was_auto_routed,
        "reason": decision.reason,
        "complexity": (
            decision.classification.complexity_level.value
            if decision.classification else None
        ),
        "confidence": (
            round(decision.classification.confidence, 2)
            if decision.classification else None
        ),
        "estimated_tokens": (
            decision.classification.estimated_response_tokens
            if decision.classification else 64
        ),
    }


@app.post("/batch", response_model=list[GenerationResponse])
async def batch_generate(request: BatchRequest) -> list[GenerationResponse]:
    """
    Process multiple prompts as a batch.

    If model=None, each prompt is individually routed through the classifier.
    This is slower but more cost-efficient than batching all with the same model.
    """
    if not request.prompts:
        return []

    pool = _get_pool()
    router = _get_router()
    dash = _get_dashboard()

    # Auto-route each prompt if no explicit model specified
    if request.model is None:
        routing_decisions = [
            router.route(p, None) for p in request.prompts
        ]
        # Group prompts by model for efficient batching
        model_groups: dict[ModelSize, list[tuple[int, str]]] = {}
        for i, (prompt, decision) in enumerate(zip(request.prompts, routing_decisions)):
            if decision.actual_model not in model_groups:
                model_groups[decision.actual_model] = []
            model_groups[decision.actual_model].append((i, prompt))

        results = [None] * len(request.prompts)
        for model, group in model_groups.items():
            prompts_in_group = [p for _, p in group]
            try:
                batch_results = await pool.generate_batch(
                    model,
                    prompts_in_group,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
                for (idx, _), result in zip(group, batch_results):
                    results[idx] = GenerationResponse(
                        request_id=f"batch-{idx}",
                        text=result["text"],
                        model_used=model.value,
                        prompt_tokens=result["prompt_tokens"],
                        completion_tokens=result["completion_tokens"],
                        latency_ms=result.get("latency_ms", 0),
                        routed_automatically=True,
                        routing_reason="auto-routed batch",
                    )
                    dash.record_request(
                        model=model,
                        latency_ms=result.get("latency_ms", 0),
                        prompt_tokens=result["prompt_tokens"],
                        completion_tokens=result["completion_tokens"],
                    )
            except Exception as e:
                for idx, _ in group:
                    results[idx] = GenerationResponse(
                        request_id=f"batch-{idx}",
                        text=f"Error: {e}",
                        model_used=model.value,
                        prompt_tokens=0,
                        completion_tokens=0,
                        latency_ms=0,
                    )

        return [r for r in results if r is not None]
    else:
        # Explicit model — batch all together
        try:
            batch_results = await pool.generate_batch(
                request.model,
                request.prompts,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            return [
                GenerationResponse(
                    request_id=f"batch-{i}",
                    text=r["text"],
                    model_used=request.model.value,
                    prompt_tokens=r["prompt_tokens"],
                    completion_tokens=r["completion_tokens"],
                    latency_ms=r.get("latency_ms", 0),
                    routed_automatically=False,
                    routing_reason=f"explicit model: {request.model.value}",
                )
                for i, r in enumerate(batch_results)
            ]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Return full metrics dashboard."""
    dash = _get_dashboard()
    return dash.get_summary()


@app.get("/routing")
async def routing_stats():
    """Return routing statistics and recent decisions."""
    router = _get_router()
    return {
        "stats": router.get_stats(),
        "recent_decisions": router.get_recent_decisions(10),
    }


@app.get("/dashboard")
async def dashboard():
    """Return full dashboard with human-readable printout."""
    dash = _get_dashboard()
    return dash.get_summary()


@app.delete("/request/{request_id}")
async def cancel_request(request_id: str):
    return {"status": "acknowledged", "request_id": request_id}


# =============================================================================
# Lifecycle
# =============================================================================

_start_time = time.time()


@app.on_event("startup")
async def startup():
    print("=" * 60)
    print("  Inference Cost Optimizer v1.0.0")
    print("=" * 60)
    print("  Pipeline: Classifier -> Router -> Model Pool")
    print("  Models: TinyLlama-1.1B, Phi-2-2.7B, Qwen2-0.5B")
    print("  Docs:   http://localhost:8000/docs")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown():
    print("Shutting down...")


# =============================================================================
# Lazy Initialization
# =============================================================================

_pool = None
_router = None
_dash = None


def _get_pool():
    global _pool
    if _pool is None:
        from core.model_pool import ModelPool
        _pool = ModelPool()
    return _pool


def _get_router():
    global _router
    if _router is None:
        from router.router import Router
        _router = Router()
    return _router


def _get_dashboard():
    global _dash
    if _dash is None:
        from dashboard.dashboard import Dashboard
        _dash = Dashboard()
    return _dash


if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)
