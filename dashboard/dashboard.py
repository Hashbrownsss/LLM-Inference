"""
Observability Dashboard — tracks and displays system metrics.

Tracks:
- Per-model: latency, throughput, memory, error rate
- Per-request: routing decision, tokens generated, end-to-end latency
- System: GPU utilization, cache hit rate, cost per token
- Routing: classifier accuracy, SLA compliance rate
"""

import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.types import ModelSize


# =============================================================================
# Metric Collectors
# =============================================================================

@dataclass
class LatencyBucket:
    """Tracks latency distribution."""
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    _values: list = field(default_factory=list)

    def add(self, value: float):
        self._values.append(value)
        # Keep last 1000 for rolling percentiles
        if len(self._values) > 1000:
            self._values = self._values[-1000:]
        self._recalculate()

    def _recalculate(self):
        if not self._values:
            return
        sorted_vals = sorted(self._values)
        n = len(sorted_vals)
        self.p50 = sorted_vals[int(n * 0.50)]
        self.p90 = sorted_vals[int(n * 0.90)]
        self.p95 = sorted_vals[int(n * 0.95)]
        self.p99 = sorted_vals[int(n * 0.99)] if n >= 100 else sorted_vals[-1]


@dataclass
class ModelDashboard:
    """Metrics for a single model."""
    requests: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    errors: int = 0
    latency: LatencyBucket = field(default_factory=LatencyBucket)

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.requests if self.requests > 0 else 0.0

    @property
    def tokens_per_second(self) -> float:
        ms = self.total_latency_ms
        return (self.total_tokens / ms * 1000) if ms > 0 else 0.0

    @property
    def error_rate(self) -> float:
        return self.errors / self.requests if self.requests > 0 else 0.0


# =============================================================================
# Main Dashboard
# =============================================================================

class Dashboard:
    """
    Central metrics collection for the inference system.

    Collects metrics at every layer:
    - Router: routing decisions, SLA compliance
    - Model pool: inference latency, throughput, errors
    - System: GPU memory, KV cache utilization

    Usage:
        dashboard = Dashboard()
        dashboard.record_request(model=ModelSize.TINY, latency_ms=45.2, tokens=12)
        dashboard.record_routing(was_auto=True, complexity="simple")
        print(dashboard.get_summary())
    """

    def __init__(self):
        self._start_time = time.time()

        # Per-model metrics
        self._models: dict[ModelSize, ModelDashboard] = {
            size: ModelDashboard() for size in ModelSize
        }

        # Routing metrics
        self._routing_auto: int = 0
        self._routing_manual: int = 0
        self._routing_correct: int = 0  # classifier agreed with user

        # Complexity distribution
        self._complexity_counts: dict[str, int] = {
            "simple": 0,
            "moderate": 0,
            "complex": 0,
        }

        # Cost tracking (estimated)
        # Based on GPU hours + model size
        self._cost_per_1k_tokens = {
            ModelSize.TINY: 0.002,
            ModelSize.MEDIUM: 0.008,
            ModelSize.LARGE: 0.003,
        }

        # Naive baseline comparison (all requests -> MEDIUM model)
        self._naive_cost: float = 0.0
        self._optimized_cost: float = 0.0

        # Benchmark results
        self._benchmark_results: dict = {}

    def record_request(
        self,
        model: ModelSize,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        error: bool = False,
    ):
        """Record a completed inference request."""
        d = self._models[model]
        d.requests += 1
        d.total_tokens += completion_tokens
        d.total_latency_ms += latency_ms
        d.latency.add(latency_ms)
        if error:
            d.errors += 1

        # Update cost tracking
        total_tokens = prompt_tokens + completion_tokens
        request_cost = (total_tokens / 1000) * self._cost_per_1k_tokens[model]
        self._optimized_cost += request_cost

        # Naive cost: all requests to MEDIUM
        naive_cost = (total_tokens / 1000) * self._cost_per_1k_tokens[ModelSize.MEDIUM]
        self._naive_cost += naive_cost

    def record_routing(
        self,
        was_auto: bool,
        complexity: str,
        model_used: ModelSize,
    ):
        """Record a routing decision."""
        if was_auto:
            self._routing_auto += 1
        else:
            self._routing_manual += 1

        if complexity in self._complexity_counts:
            self._complexity_counts[complexity] += 1

    def record_classification(self, was_correct: bool):
        """Record whether the classifier's choice was appropriate."""
        self._routing_correct += 1 if was_correct else 0

    def record_benchmark(
        self,
        name: str,
        throughput_tps: float,
        latency_p50_ms: float,
        cost_per_1k_tokens: float,
    ):
        """Store a benchmark result."""
        self._benchmark_results[name] = {
            "throughput_tps": throughput_tps,
            "latency_p50_ms": latency_p50_ms,
            "cost_per_1k_tokens": cost_per_1k_tokens,
            "timestamp": time.time(),
        }

    def get_summary(self) -> dict:
        """Return full dashboard summary."""
        total_requests = sum(d.requests for d in self._models.values())
        total_tokens = sum(d.total_tokens for d in self._models.values())

        # Find best and worst performing models
        model_stats = {}
        for size, d in self._models.items():
            if d.requests > 0:
                model_stats[size.value] = {
                    "requests": d.requests,
                    "avg_latency_ms": round(d.avg_latency_ms, 1),
                    "p50_latency_ms": round(d.latency.p50, 1),
                    "p90_latency_ms": round(d.latency.p90, 1),
                    "tokens_per_sec": round(d.tokens_per_second, 1),
                    "total_tokens": d.total_tokens,
                    "error_rate": round(d.error_rate * 100, 2),
                }

        cost_saved = (
            ((self._naive_cost - self._optimized_cost) / self._naive_cost * 100)
            if self._naive_cost > 0 else 0
        )

        return {
            "uptime_seconds": round(time.time() - self._start_time, 0),
            "total_requests": total_requests,
            "total_tokens_generated": total_tokens,
            "models": model_stats,
            "routing": {
                "auto_routed": self._routing_auto,
                "manual_routed": self._routing_manual,
                "auto_route_rate": round(
                    self._routing_auto / (self._routing_auto + self._routing_manual) * 100, 1
                ) if (self._routing_auto + self._routing_manual) > 0 else 0,
            },
            "complexity_distribution": self._complexity_counts,
            "cost": {
                "naive_baseline_cost": round(self._naive_cost, 4),
                "optimized_cost": round(self._optimized_cost, 4),
                "cost_saved_percent": round(cost_saved, 1),
                "cost_per_1k_tokens": {
                    size.value: cost for size, cost in self._cost_per_1k_tokens.items()
                },
            },
            "benchmarks": self._benchmark_results,
        }

    def print_summary(self):
        """Print a human-readable dashboard summary to console."""
        s = self.get_summary()

        print("\n" + "=" * 60)
        print("  INFERENCE COST OPTIMIZER — DASHBOARD")
        print("=" * 60)
        print(f"  Uptime: {s['uptime_seconds']:.0f}s  |  Total requests: {s['total_requests']}")
        print(f"  Total tokens generated: {s['total_tokens_generated']:,}")

        print(f"\n  ROUTING:")
        r = s["routing"]
        print(f"    Auto-routed:    {r['auto_routed']:5d} ({r['auto_route_rate']:.1f}%)")
        print(f"    Manual:        {r['manual_routed']:5d}")

        print(f"\n  COMPLEXITY DISTRIBUTION:")
        for level, count in s["complexity_distribution"].items():
            pct = count / max(s["total_requests"], 1) * 100
            bar = "=" * int(pct / 5)
            print(f"    {level:10s}: {count:4d} ({pct:5.1f}%) {bar}")

        print(f"\n  COST SAVINGS:")
        c = s["cost"]
        print(f"    Naive baseline:  ${c['naive_baseline_cost']:.4f}")
        print(f"    Optimized:        ${c['optimized_cost']:.4f}")
        print(f"    SAVED:            ${c['naive_baseline_cost'] - c['optimized_cost']:.4f} ({c['cost_saved_percent']:.1f}%)")

        if s["models"]:
            print(f"\n  MODEL PERFORMANCE:")
            for name, m in s["models"].items():
                print(f"    {name:10s}: {m['requests']:4d} reqs | "
                      f"latency: {m['avg_latency_ms']:6.1f}ms avg / {m['p50_latency_ms']:5.1f}ms p50 | "
                      f"{m['tokens_per_sec']:5.1f} tok/s")

        print("=" * 60)

    def export_json(self, path: str):
        """Export dashboard data to JSON for external dashboards."""
        with open(path, "w") as f:
            json.dump(self.get_summary(), f, indent=2)


# =============================================================================
# Global dashboard
# =============================================================================

_dashboard: Optional[Dashboard] = None


def get_dashboard() -> Dashboard:
    global _dashboard
    if _dashboard is None:
        _dashboard = Dashboard()
    return _dashboard
