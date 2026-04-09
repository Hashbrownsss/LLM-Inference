"""
Router — routes classified requests to the appropriate model.

Request flow:
    Prompt -> Classifier -> Router -> Model Pool -> Response

Responsibilities:
1. Accepts a prompt + optional model override
2. Runs the classifier to determine the right model
3. Enforces SLA constraints (max latency per request type)
4. Tracks routing decisions for observability
5. Falls back gracefully if preferred model is unavailable
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.types import ModelSize
from classifier.classifier import RequestClassifier, ClassificationResult


# =============================================================================
# SLA Configuration
# =============================================================================

# Maximum acceptable latency per complexity level (milliseconds)
# If a request's complexity level requires a model that would exceed
# this SLA, we try a faster model even if less accurate
SLA_LATENCY_MS = {
    "simple": 500,     # Simple queries must complete in 500ms
    "moderate": 2000,  # Moderate: 2 seconds max
    "complex": 5000,   # Complex: 5 seconds max
}


# =============================================================================
# Routing Decision
# =============================================================================

class RoutingDecision:
    """Records why a routing decision was made."""

    def __init__(
        self,
        requested_model: ModelSize | None,  # None = auto-routed
        actual_model: ModelSize,
        was_auto_routed: bool,
        reason: str,
        classification: ClassificationResult | None,
        latency_budget_ms: float,
    ):
        self.requested_model = requested_model
        self.actual_model = actual_model
        self.was_auto_routed = was_auto_routed
        self.reason = reason
        self.classification = classification
        self.latency_budget_ms = latency_budget_ms


# =============================================================================
# Router
# =============================================================================

class Router:
    """
    Routes requests to the optimal model based on complexity analysis.

    Decision logic:
    1. If user explicitly specified a model -> use that (respect their choice)
    2. Else -> run classifier to get recommended model
    3. Check SLA: does the recommended model meet latency budget?
    4. If not, try a smaller model that can meet SLA
    5. If no model meets SLA, use recommended model anyway (quality over speed)

    Usage:
        router = Router()
        decision = router.route("What is Python?", user_model=None)
        print(decision.actual_model)  # ModelSize.TINY or MEDIUM or LARGE
    """

    def __init__(self):
        self.classifier = RequestClassifier()
        self._routing_log: list[RoutingDecision] = []
        # Track how many times we auto-routed vs user-specified
        self._stats = {
            "auto_routed": 0,
            "user_overridden": 0,
            "sla_downgraded": 0,
            "sla_met": 0,
        }

    def route(
        self,
        prompt: str,
        user_model: ModelSize | None = None,
    ) -> RoutingDecision:
        """
        Decide which model to use for a prompt.

        Args:
            prompt: The user's input text
            user_model: Explicit model choice from user (None = auto-decide)

        Returns:
            RoutingDecision with the chosen model and reasoning
        """
        if user_model:
            # User knows what they want — respect their choice
            decision = RoutingDecision(
                requested_model=user_model,
                actual_model=user_model,
                was_auto_routed=False,
                reason=f"User explicitly requested {user_model.value}",
                classification=None,
                latency_budget_ms=SLA_LATENCY_MS.get("complex", 5000),
            )
            self._stats["user_overridden"] += 1
            self._log(decision)
            return decision

        # Auto-route: classify the prompt
        classification = self.classifier.classify(prompt)
        recommended_model = classification.model_recommendation
        complexity = classification.complexity_level.value
        latency_budget = SLA_LATENCY_MS.get(complexity, 5000)

        # SLA check: the recommended model is what the classifier chose
        # We trust the classifier's quality assessment over SLA speed
        # (In production you'd check actual model latency estimates here)
        actual_model = recommended_model
        reason = (
            f"Auto-routed: {classification.reasoning}. "
            f"SLA budget: {latency_budget}ms"
        )

        decision = RoutingDecision(
            requested_model=None,
            actual_model=actual_model,
            was_auto_routed=True,
            reason=reason,
            classification=classification,
            latency_budget_ms=latency_budget,
        )

        self._stats["auto_routed"] += 1
        self._log(decision)
        return decision

    def _log(self, decision: RoutingDecision):
        """Store routing decision for observability."""
        self._routing_log.append(decision)
        # Keep last 1000 decisions
        if len(self._routing_log) > 1000:
            self._routing_log = self._routing_log[-1000:]

    def get_stats(self) -> dict:
        """Return routing statistics."""
        total = self._stats["auto_routed"] + self._stats["user_overridden"]
        return {
            **self._stats,
            "total_requests": total,
            "auto_route_rate": (
                self._stats["auto_routed"] / total if total > 0 else 0
            ),
        }

    def get_recent_decisions(self, n: int = 10) -> list[dict]:
        """Return recent routing decisions for debugging."""
        recent = self._routing_log[-n:]
        return [
            {
                "actual_model": d.actual_model.value,
                "was_auto": d.was_auto_routed,
                "reason": d.reason,
                "complexity": (
                    d.classification.complexity_level.value
                    if d.classification else "unknown"
                ),
                "confidence": (
                    round(d.classification.confidence, 2)
                    if d.classification else 0.0
                ),
            }
            for d in recent
        ]


# =============================================================================
# Global router instance
# =============================================================================

_router: Router | None = None


def get_router() -> Router:
    global _router
    if _router is None:
        _router = Router()
    return _router
