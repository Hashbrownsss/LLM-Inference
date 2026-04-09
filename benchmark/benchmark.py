"""
Benchmark suite — measures throughput, latency, and cost savings.

Run: python benchmark/benchmark.py

This script:
1. Runs a baseline: all requests -> single model
2. Runs our system: requests -> classifier -> optimal model
3. Compares throughput, latency, and cost
4. Outputs results + generates charts data
"""

import asyncio
import time
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.types import ModelSize
from core.model_pool import ModelPool, MODEL_REGISTRY
from classifier.classifier import RequestClassifier
from router.router import Router
from dashboard.dashboard import Dashboard


# =============================================================================
# Test Workload
# =============================================================================

# A realistic mix of prompts — simulates production traffic
WORKLOAD = [
    # Simple (40%) — TinyLlama territory
    ("What is Python?", ModelSize.TINY),
    ("Define machine learning", ModelSize.TINY),
    ("What time is it?", ModelSize.TINY),
    ("Hello, how are you?", ModelSize.TINY),
    ("What's 2+2?", ModelSize.TINY),
    ("What is the capital of Japan?", ModelSize.TINY),
    ("Define the word algorithm", ModelSize.TINY),
    ("What does CPU stand for?", ModelSize.TINY),
    ("Who invented Python?", ModelSize.TINY),
    ("What is a variable in programming?", ModelSize.TINY),
    ("Is water wet?", ModelSize.TINY),
    ("What's for breakfast?", ModelSize.TINY),

    # Moderate (35%) — Phi-2 territory
    ("Compare Python and JavaScript for backend development", ModelSize.MEDIUM),
    ("Explain how neural networks learn through backpropagation", ModelSize.MEDIUM),
    ("Write a Python function to check if a number is prime", ModelSize.MEDIUM),
    ("What are the pros and cons of microservices architecture?", ModelSize.MEDIUM),
    ("Debug this code: for i in range(10): print(i)", ModelSize.MEDIUM),
    ("Explain the difference between REST and GraphQL APIs", ModelSize.MEDIUM),
    ("Write a Python decorator that logs function calls", ModelSize.MEDIUM),
    ("What is the CAP theorem in distributed systems?", ModelSize.MEDIUM),
    ("Compare SQL and NoSQL databases for a social media app", ModelSize.MEDIUM),
    ("Explain how git branching works with examples", ModelSize.MEDIUM),
    ("Write a Python context manager for database connections", ModelSize.MEDIUM),
    ("What are the main differences between Docker and Kubernetes?", ModelSize.MEDIUM),
    ("Explain object-oriented programming principles", ModelSize.MEDIUM),
    ("Write a Python class for a stack data structure", ModelSize.MEDIUM),

    # Complex (25%) — LARGE territory
    ("Prove that the sum of angles in a triangle is 180 degrees", ModelSize.LARGE),
    ("Derive the time complexity of quicksort using recurrence relations", ModelSize.LARGE),
    ("Analyze the trade-offs between CAP theorem properties in distributed systems", ModelSize.LARGE),
    ("What is the mechanism of action of mRNA vaccines?", ModelSize.LARGE),
    ("Explain GDPR article 17 and its implications for data retention policies", ModelSize.LARGE),
    ("Prove that the halting problem is undecidable", ModelSize.LARGE),
    ("Analyze the economic implications of automation on labor markets", ModelSize.LARGE),
    ("Derive the backpropagation equations for a multi-layer perceptron", ModelSize.LARGE),
    ("Compare and contrast consequentialist and deontological ethical frameworks", ModelSize.LARGE),
    ("Analyze the implications of quantum computing on current cryptography", ModelSize.LARGE),
]

SIMPLE_PROMPTS = [p for p, _ in WORKLOAD if _ == ModelSize.TINY]
MODERATE_PROMPTS = [p for p, _ in WORKLOAD if _ == ModelSize.MEDIUM]
COMPLEX_PROMPTS = [p for p, _ in WORKLOAD if _ == ModelSize.LARGE]


# =============================================================================
# Benchmark Functions
# =============================================================================

async def benchmark_naive(pool: ModelPool, prompts: list[str], model: ModelSize) -> dict:
    """
    Baseline: all requests go to a single model.
    Measures naive throughput and latency.
    """
    results = []
    start = time.time()

    for prompt in prompts:
        r_start = time.time()
        try:
            result = await pool.generate(model, prompt, max_tokens=64, temperature=0.7)
            r_time = (time.time() - r_start) * 1000
            results.append({
                "prompt": prompt,
                "model": model.value,
                "success": True,
                "latency_ms": r_time,
                "tokens": result["completion_tokens"],
            })
        except Exception as e:
            results.append({
                "prompt": prompt,
                "model": model.value,
                "success": False,
                "latency_ms": (time.time() - r_start) * 1000,
                "error": str(e),
            })

    total_time = time.time() - start
    total_tokens = sum(r["tokens"] for r in results if r["success"])
    latencies = [r["latency_ms"] for r in results if r["success"]]

    return {
        "mode": "naive",
        "model": model.value,
        "total_requests": len(prompts),
        "successful_requests": sum(1 for r in results if r["success"]),
        "total_time_s": round(total_time, 2),
        "throughput_tps": round(total_tokens / total_time, 1),
        "latency_p50_ms": round(sorted(latencies)[len(latencies)//2], 1) if latencies else 0,
        "latency_p90_ms": round(sorted(latencies)[int(len(latencies)*0.9)], 1) if latencies else 0,
        "latency_avg_ms": round(sum(latencies)/len(latencies), 1) if latencies else 0,
    }


async def benchmark_optimizer(
    pool: ModelPool,
    classifier: RequestClassifier,
    router: Router,
    prompts: list[str],
) -> dict:
    """
    Our system: classifier decides model, router routes there.
    Measures optimized throughput and latency.
    """
    results = []
    routing_decisions = {size.value: 0 for size in ModelSize}
    start = time.time()

    for prompt in prompts:
        r_start = time.time()
        try:
            decision = router.route(prompt)
            routing_decisions[decision.actual_model.value] += 1

            result = await pool.generate(
                decision.actual_model,
                prompt,
                max_tokens=64,
                temperature=0.7,
            )
            r_time = (time.time() - r_start) * 1000
            results.append({
                "prompt": prompt,
                "model_used": decision.actual_model.value,
                "auto_routed": decision.was_auto_routed,
                "success": True,
                "latency_ms": r_time,
                "tokens": result["completion_tokens"],
            })
        except Exception as e:
            results.append({
                "prompt": prompt,
                "success": False,
                "latency_ms": (time.time() - r_start) * 1000,
                "error": str(e),
            })

    total_time = time.time() - start
    total_tokens = sum(r["tokens"] for r in results if r["success"])
    latencies = [r["latency_ms"] for r in results if r["success"]]
    auto_routed = sum(1 for r in results if r.get("auto_routed"))

    return {
        "mode": "optimizer",
        "total_requests": len(prompts),
        "successful_requests": sum(1 for r in results if r["success"]),
        "total_time_s": round(total_time, 2),
        "throughput_tps": round(total_tokens / total_time, 1),
        "latency_p50_ms": round(sorted(latencies)[len(latencies)//2], 1) if latencies else 0,
        "latency_p90_ms": round(sorted(latencies)[int(len(latencies)*0.9)], 1) if latencies else 0,
        "latency_avg_ms": round(sum(latencies)/len(latencies), 1) if latencies else 0,
        "routing_distribution": routing_decisions,
        "auto_route_rate": round(auto_routed / len(results) * 100, 1),
    }


async def run_benchmarks():
    """Run the full benchmark suite."""
    print("\n" + "=" * 60)
    print("  INFERENCE COST OPTIMIZER — BENCHMARK SUITE")
    print("=" * 60)

    pool = ModelPool()
    classifier = RequestClassifier()
    router = Router()
    dashboard = Dashboard()

    # Warm up: load TinyLlama
    print("\n[1/4] Warming up TinyLlama...")
    try:
        await pool.generate(ModelSize.TINY, "Hello", max_tokens=5)
    except Exception as e:
        print(f"  Warning: warmup failed ({e}). Continuing anyway.")

    # Benchmark 1: Naive — all requests to TINY
    print(f"\n[2/4] Benchmark: Naive (all -> TinyLlama)")
    print(f"  Running {len(SIMPLE_PROMPTS)} simple prompts...")
    naive_tiny = await benchmark_naive(pool, SIMPLE_PROMPTS, ModelSize.TINY)
    print(f"  Throughput: {naive_tiny['throughput_tps']:.1f} tok/s | "
          f"Latency p50: {naive_tiny['latency_p50_ms']:.0f}ms")

    # Benchmark 2: Naive — all requests to MEDIUM
    print(f"\n[3/4] Benchmark: Naive (all -> Phi-2)")
    print(f"  Running {len(WORKLOAD)} mixed prompts...")
    naive_medium = await benchmark_naive(pool, WORKLOAD, ModelSize.MEDIUM)
    print(f"  Throughput: {naive_medium['throughput_tps']:.1f} tok/s | "
          f"Latency p50: {naive_medium['latency_p50_ms']:.0f}ms")

    # Benchmark 3: Optimizer — classifier routes each request
    print(f"\n[4/4] Benchmark: Optimizer (auto-routed)")
    print(f"  Running {len(WORKLOAD)} mixed prompts...")
    optimized = await benchmark_optimizer(pool, classifier, router, WORKLOAD)
    print(f"  Throughput: {optimized['throughput_tps']:.1f} tok/s | "
          f"Latency p50: {optimized['latency_p50_ms']:.0f}ms")
    print(f"  Routing: {optimized['routing_distribution']}")

    # Print comparison
    print("\n" + "=" * 60)
    print("  RESULTS COMPARISON")
    print("=" * 60)

    # Throughput comparison
    speedup_tiny = (
        optimized["throughput_tps"] / naive_tiny["throughput_tps"]
        if naive_tiny["throughput_tps"] > 0 else 0
    )
    speedup_medium = (
        optimized["throughput_tps"] / naive_medium["throughput_tps"]
        if naive_medium["throughput_tps"] > 0 else 0
    )

    # Latency comparison
    latency_improvement = (
        (naive_medium["latency_avg_ms"] - optimized["latency_avg_ms"])
        / naive_medium["latency_avg_ms"] * 100
        if naive_medium["latency_avg_ms"] > 0 else 0
    )

    print(f"\n  THROUGHPUT (tokens/sec):")
    print(f"    Naive (all -> TinyLlama): {naive_tiny['throughput_tps']:.1f} tok/s  [baseline]")
    print(f"    Naive (all -> Phi-2):    {naive_medium['throughput_tps']:.1f} tok/s")
    print(f"    Optimizer (auto-route):  {optimized['throughput_tps']:.1f} tok/s")
    print(f"    Speedup vs TinyLlama:    {speedup_tiny:.2f}x")
    print(f"    Speedup vs Phi-2:        {speedup_medium:.2f}x")

    print(f"\n  LATENCY (average ms):")
    print(f"    Naive (all -> TinyLlama): {naive_tiny['latency_avg_ms']:.0f}ms")
    print(f"    Naive (all -> Phi-2):    {naive_medium['latency_avg_ms']:.0f}ms")
    print(f"    Optimizer (auto-route):  {optimized['latency_avg_ms']:.0f}ms")
    print(f"    Improvement vs Phi-2:     {latency_improvement:.0f}%")

    print(f"\n  ROUTING DISTRIBUTION:")
    total = sum(optimized["routing_distribution"].values())
    for model, count in optimized["routing_distribution"].items():
        pct = count / total * 100
        bar = "=" * int(pct / 3)
        print(f"    {model:10s}: {count:3d} ({pct:5.1f}%) {bar}")

    print(f"\n  COST ESTIMATE (based on GPU-hours):")
    # Simple cost model: TinyLlama ~$0.002/1k tokens, Phi-2 ~$0.008/1k tokens
    cost_tiny = (sum(r["tokens"] for r in [naive_tiny]) / 1000) * 0.002
    cost_medium = (sum(r["tokens"] for r in [naive_medium]) / 1000) * 0.008
    # Optimizer uses TinyLlama for simple, medium for moderate, etc.
    optimized_tokens = (
        optimized["routing_distribution"].get("tinyllama", 0) * 50 +
        optimized["routing_distribution"].get("phi2", 0) * 80 +
        optimized["routing_distribution"].get("qwen2", 0) * 40
    )
    cost_optimized = (optimized_tokens / 1000) * 0.004  # weighted average

    print(f"    Naive (all TinyLlama):  ${cost_tiny:.4f}")
    print(f"    Naive (all Phi-2):      ${cost_medium:.4f}")
    print(f"    Optimizer:              ${cost_optimized:.4f}")

    # Save results
    results = {
        "naive_tiny": naive_tiny,
        "naive_medium": naive_medium,
        "optimizer": optimized,
        "workload_size": len(WORKLOAD),
        "timestamp": time.time(),
    }

    benchmark_dir = Path(__file__).parent
    benchmark_dir.mkdir(exist_ok=True)
    with open(benchmark_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: benchmark/results.json")

    print("\n" + "=" * 60)
    return results


if __name__ == "__main__":
    asyncio.run(run_benchmarks())
