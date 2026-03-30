"""
Request Classifier — analyzes prompt complexity to determine routing.

Design philosophy:
- Rule-based classification with explicit precedence
- Each rule is interpretable, testable, and tunable
- Hard domains (medical, legal, hard_science) always get larger models
- Technical + reasoning combines into higher complexity
- Easy to extend, easy to debug

The classifier evaluates rules in this priority order:
1. Hard domains override everything (medical, legal, hard_science -> LARGE)
2. Question type + reasoning depth combinations
3. Technical content adds complexity
4. Domain expertise required
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.types import ModelSize


# =============================================================================
# Output Types
# =============================================================================

class ComplexityLevel(str, Enum):
    SIMPLE   = "simple"    # TinyLlama-1.1B sufficient
    MODERATE = "moderate"  # Phi-2-2.7B recommended
    COMPLEX  = "complex"   # LARGE model needed


class Domain(str, Enum):
    GENERAL      = "general"
    CREATIVE     = "creative"
    TECHNICAL    = "technical"
    HARD_SCIENCE = "hard_science"
    SOCIAL       = "social"
    MEDICAL      = "medical"
    LEGAL        = "legal"
    CURRENT_EVENTS = "current"


class QuestionType(str, Enum):
    FACTUAL     = "factual"
    EXPLANATION = "explanation"
    PROCEDURAL  = "procedural"
    ANALYSIS    = "analysis"
    CREATION    = "creation"
    REASONING   = "reasoning"


@dataclass
class ClassificationResult:
    model_recommendation: ModelSize
    complexity_level: ComplexityLevel
    confidence: float
    reasoning: str
    domain: Domain
    question_type: QuestionType
    estimated_response_tokens: int
    rule_applied: str = ""  # Which rule fired


# =============================================================================
# Rule Engine
# =============================================================================

class ClassificationRule:
    """
    A single rule that can fire to classify a prompt.

    Each rule has:
    - condition(s): checks that must ALL be true
    - result: complexity level to assign if rule fires
    - model: which model to recommend
    - reason: why this rule fired
    - priority: higher = checked first

    Rules are evaluated top-to-bottom until one fires.
    """

    def __init__(
        self,
        name: str,
        condition: callable,
        complexity: ComplexityLevel,
        model: ModelSize,
        reason: str,
        priority: int = 0,
    ):
        self.name = name
        self.condition = condition
        self.complexity = complexity
        self.model = model
        self.reason = reason
        self.priority = priority


class RequestClassifier:
    """
    Heuristic request classifier using explicit rule chains.

    Why rules instead of weighted scores?
    - Weighted scores are hard to calibrate (e.g., "technical=2 + reasoning=2 + analysis=2.5"
      can still normalize to a score of 1.0 if weighted wrong)
    - Rules are explicit: "if domain=HARD_SCIENCE AND reasoning>=2 -> COMPLEX"
    - Easy to add, remove, or tune rules
    - Easy to test in isolation
    - Debugging is trivial: "which rule fired?"

    The rules cover:
    - Hard domains (medical, legal, hard_science): always need larger models
    - Reasoning + technical combinations: indicate complex tasks
    - Question type: some question types are inherently harder
    - Domain + question type: specific combinations to watch for
    """

    def __init__(self):
        self.rules = self._build_rules()
        # Extractors for signal detection
        self._domain_keywords = self._build_domain_keywords()
        self._reasoning_keywords = self._build_reasoning_keywords()

    # -------------------------------------------------------------------------
    # Rule Building
    # -------------------------------------------------------------------------

    def _build_rules(self) -> list[ClassificationRule]:
        """Define all classification rules in priority order."""
        rules = []

        def add(condition, complexity, model, reason, priority=0):
            rules.append(ClassificationRule(
                name=reason[:40],
                condition=condition,
                complexity=complexity,
                model=model,
                reason=reason,
                priority=priority,
            ))

        # ===== Priority 10: Hard domains always need larger models =====
        # Medical questions: always complex regardless of question type
        add(
            lambda ctx: ctx["domain"] == Domain.MEDICAL,
            ComplexityLevel.COMPLEX, ModelSize.LARGE,
            "Medical domain requires largest model for accuracy",
            priority=10,
        )

        # Legal questions: always moderate or complex
        add(
            lambda ctx: ctx["domain"] == Domain.LEGAL,
            ComplexityLevel.MODERATE, ModelSize.LARGE,
            "Legal domain requires careful, accurate responses",
            priority=10,
        )

        # Hard science + reasoning: always complex
        add(
            lambda ctx: ctx["domain"] == Domain.HARD_SCIENCE and ctx["reasoning_depth"] >= 2,
            ComplexityLevel.COMPLEX, ModelSize.LARGE,
            "Hard science with reasoning depth requires large model",
            priority=10,
        )

        # Hard science with analysis/explanation: moderate minimum
        add(
            lambda ctx: ctx["domain"] == Domain.HARD_SCIENCE and ctx["qtype"] in (QuestionType.ANALYSIS, QuestionType.EXPLANATION, QuestionType.REASONING),
            ComplexityLevel.MODERATE, ModelSize.LARGE,
            "Hard science explanation/analysis needs strong model",
            priority=10,
        )

        # Hard science simple factual: can use moderate
        add(
            lambda ctx: ctx["domain"] == Domain.HARD_SCIENCE and ctx["qtype"] == QuestionType.FACTUAL,
            ComplexityLevel.MODERATE, ModelSize.MEDIUM,
            "Hard science factual: moderate model sufficient",
            priority=10,
        )

        # ===== Priority 8: Technical content + reasoning =====
        add(
            lambda ctx: ctx["is_technical"] and ctx["reasoning_depth"] >= 2,
            ComplexityLevel.MODERATE, ModelSize.LARGE,
            "Technical content + reasoning depth: complex",
            priority=8,
        )

        add(
            lambda ctx: ctx["is_technical"] and ctx["qtype"] == QuestionType.ANALYSIS,
            ComplexityLevel.MODERATE, ModelSize.LARGE,
            "Technical analysis: complex task",
            priority=8,
        )

        add(
            lambda ctx: ctx["is_technical"] and ctx["reasoning_depth"] >= 1 and ctx["qtype"] == QuestionType.REASONING,
            ComplexityLevel.MODERATE, ModelSize.LARGE,
            "Technical reasoning: complex",
            priority=8,
        )

        # ===== Priority 6: Technical + creation/procedural =====
        add(
            lambda ctx: ctx["is_technical"] and ctx["qtype"] == QuestionType.CREATION,
            ComplexityLevel.MODERATE, ModelSize.MEDIUM,
            "Technical code generation: moderate complexity",
            priority=6,
        )

        add(
            lambda ctx: ctx["is_technical"] and ctx["qtype"] == QuestionType.PROCEDURAL,
            ComplexityLevel.MODERATE, ModelSize.MEDIUM,
            "Technical procedural: moderate complexity",
            priority=6,
        )

        # Technical factual or simple: still moderate due to precision needed
        add(
            lambda ctx: ctx["is_technical"] and ctx["qtype"] in (QuestionType.FACTUAL, QuestionType.EXPLANATION),
            ComplexityLevel.SIMPLE, ModelSize.MEDIUM,
            "Technical factual: medium model for precision",
            priority=6,
        )

        # ===== Priority 5: Social science + reasoning =====
        add(
            lambda ctx: ctx["domain"] == Domain.SOCIAL and ctx["reasoning_depth"] >= 2,
            ComplexityLevel.MODERATE, ModelSize.LARGE,
            "Social analysis with deep reasoning",
            priority=5,
        )

        add(
            lambda ctx: ctx["domain"] == Domain.SOCIAL and ctx["qtype"] == QuestionType.ANALYSIS,
            ComplexityLevel.MODERATE, ModelSize.MEDIUM,
            "Social analysis: moderate",
            priority=5,
        )

        # ===== Priority 4: Creative + length/content =====
        # Long creative writing needs more reasoning
        add(
            lambda ctx: ctx["domain"] == Domain.CREATIVE and ctx["reasoning_depth"] >= 2,
            ComplexityLevel.MODERATE, ModelSize.MEDIUM,
            "Creative with reasoning depth",
            priority=4,
        )

        # ===== Priority 3: Pure reasoning =====
        add(
            lambda ctx: ctx["qtype"] == QuestionType.REASONING and ctx["reasoning_depth"] >= 2,
            ComplexityLevel.MODERATE, ModelSize.LARGE,
            "Deep reasoning: complex",
            priority=3,
        )

        add(
            lambda ctx: ctx["qtype"] == QuestionType.REASONING,
            ComplexityLevel.SIMPLE, ModelSize.MEDIUM,
            "Reasoning: moderate model",
            priority=3,
        )

        # ===== Priority 2: Analysis =====
        add(
            lambda ctx: ctx["qtype"] == QuestionType.ANALYSIS,
            ComplexityLevel.SIMPLE, ModelSize.MEDIUM,
            "Analysis: moderate model recommended",
            priority=2,
        )

        # ===== Priority 1: Creation/explanation with depth =====
        add(
            lambda ctx: ctx["qtype"] in (QuestionType.CREATION, QuestionType.EXPLANATION) and ctx["reasoning_depth"] >= 1,
            ComplexityLevel.SIMPLE, ModelSize.MEDIUM,
            "Creation/explanation with some reasoning",
            priority=1,
        )

        # ===== Priority 0: Default =====
        add(
            lambda ctx: True,  # always fires
            ComplexityLevel.SIMPLE, ModelSize.TINY,
            "Default: simple general query",
            priority=0,
        )

        # Sort by priority descending
        rules.sort(key=lambda r: r.priority, reverse=True)
        return rules

    def _build_domain_keywords(self) -> dict[Domain, set[str]]:
        return {
            Domain.HARD_SCIENCE: {
                "physics", "chemistry", "biology", "quantum", "thermodynamics",
                "genetics", "molecular", "biochemistry", "cosmology", "astronomy",
                "neuroscience", "electromagnetic", "relativity",
                "differential equation", "integral calculus", "derivative",
                "theorem", "proof", "prove that", "matrix", "vector",
                "eigenvalue", "gradient descent", "particle physics",
                "organic chemistry", "mechanics", "kinematics", "optics",
                "genome", "photosynthesis", "cell division", "evolution",
            },
            Domain.MEDICAL: {
                "medical", "diagnosis", "treatment", "symptom", "disease",
                "patient", "clinical", "pharmaceutical", "drug interaction",
                "surgery", "vaccine", "antibiotic", "pathology", "oncology",
                "cardiology", "neurology", "epidemiology", "public health",
                "clinical trial", "pharmacology", "pathophysiology",
                "mrna vaccine", "therapeutic", "diagnosis", "prognosis",
            },
            Domain.LEGAL: {
                "law", "legal", "court", "liability", "contract", "compliance",
                "patent", "copyright", "trademark", "jurisdiction", "litigation",
                "gdpr", "hipaa", "regulation", "legal framework", "tort",
                "statute", "precedent", "case law", "plaintiff", "defendant",
                "intellectual property", "regulatory", "mandate", "enforcement",
            },
            Domain.TECHNICAL: {
                "python", "javascript", "java", "c++", "rust", "golang", "sql",
                "api", "database", "server", "frontend", "backend", "devops",
                "algorithm", "data structure", "optimization", "docker", "kubernetes",
                "function", "class ", "module", "library", "bug", "debug",
                "git", "ci/cd", "microservice", "cloud computing", "deployment",
                "recursion", "iteration", "binary search", "sorting algorithm",
                "dynamic programming", "greedy algorithm", "time complexity",
                "space complexity", "recurrence", "divide and conquer",
                "neural network", "machine learning", "deep learning",
                "backpropagation", "gradient", "loss function", "optimizer",
            },
            Domain.SOCIAL: {
                "economics", "psychology", "sociology", "political", "philosophy",
                "ethics", "history", "anthropology", "culture", "market",
                "behavioral", "cognitive", "moral", "justice", "rights",
                "trade", "policy", "government", "democracy", "capitalism",
                "communism", "societal", "inequality", "globalization",
            },
            Domain.CREATIVE: {
                "story", "fiction", "poem", "narrative", "creative writing",
                "brainstorm", "imagine", "invent", "design brand",
                "character", "plot", "dialogue", "script", "screenplay",
                "novel", "chapter", "creative", "artistic", "narrative",
            },
            Domain.CURRENT_EVENTS: {
                "news", "recently", "latest", "2025", "2026",
                "yesterday", "last week", "happening now", "just announced",
                "breaking", "current events",
            },
        }

    def _build_reasoning_keywords(self) -> dict[int, set[str]]:
        """Reasoning depth by keyword matches."""
        return {
            0: set(),  # no keywords found
            1: {"why", "how", "reason", "because"},  # light
            2: {"explain", "compare", "contrast", "analyze", "evaluate",
                "assess", "implications", "consequences", "causes", "effects",
                "consider", "factors"},  # moderate
            3: {"prove", "derive", "synthesize", "deduce", "prove that",
                "logically", "contradiction", "optimal", "minimize", "maximize",
                "proof", "theoretical", "rigorous", "prove this"},  # heavy
        }

    # -------------------------------------------------------------------------
    # Signal Extraction
    # -------------------------------------------------------------------------

    def _detect_domain(self, text: str) -> Domain:
        """Detect the primary domain of the prompt."""
        text_lower = text.lower()
        scores = {}

        for domain, keywords in self._domain_keywords.items():
            # Count how many keywords from this domain appear in the text
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                scores[domain] = matches

        if not scores:
            return Domain.GENERAL

        # Return the domain with the most keyword matches
        return max(scores, key=scores.get)

    def _detect_reasoning_depth(self, text: str) -> int:
        """Estimate reasoning depth (0-3)."""
        text_lower = text.lower()
        depth = 0

        # Check reasoning keywords by depth level (check high first)
        for level in [3, 2, 1]:
            keywords = self._reasoning_keywords[level]
            if any(kw in text_lower for kw in keywords):
                depth = max(depth, level)

        # Additional patterns that indicate deeper reasoning
        if re.search(r"if\s+.+\s+then\s+.+\s+else", text_lower):
            depth = max(depth, 2)
        if re.search(r"(therefore|thus|hence|consequently)\s", text_lower):
            depth = max(depth, 2)
        if text_lower.count("because") >= 2 or text_lower.count("therefore") >= 2:
            depth = max(depth, 2)

        return depth

    def _detect_qtype(self, text: str) -> QuestionType:
        """Classify the question type."""
        text_lower = text.lower().strip()

        # Reasoning
        if re.search(r"(prove|derive|show that|solve for|calculate)", text_lower):
            return QuestionType.REASONING

        # Analysis
        if re.search(r"(compare|contrast|analyze|evaluate|assess|trade-off|pros and cons)", text_lower):
            return QuestionType.ANALYSIS

        # Creation
        if re.search(r"^(write|create|generate|make|design|compose)", text_lower):
            return QuestionType.CREATION

        # Procedural
        if re.search(r"(how do i|how to|instructions?|steps?|tutorial)", text_lower):
            return QuestionType.PROCEDURAL

        # Explanation
        if re.search(r"(explain|why does|why is|describe|how does)", text_lower):
            return QuestionType.EXPLANATION

        # Factual (default)
        if re.search(r"^(what|who|when|where|is |are |define)", text_lower):
            return QuestionType.FACTUAL

        # Default based on length
        if len(text.split()) > 30:
            return QuestionType.EXPLANATION
        return QuestionType.FACTUAL

    def _detect_technical(self, text: str) -> bool:
        """Check if prompt contains technical content."""
        text_lower = text.lower()
        technical_indicators = [
            "def ", "function(", "class ", "import ", "const ",
            "for ", "while ", "if ", "return ", "print(",
            "```", "api", "database", "algorithm", "code",
            "python", "javascript", "sql", "bug", "error",
            "server", "endpoint", "request", "response", "json",
            "neural network", "gradient", "backpropagation", "training",
            "loss function", "optimizer", "layer", "tensor",
        ]
        return any(ind in text_lower for ind in technical_indicators)

    def _estimate_tokens(self, qtype: QuestionType, domain: Domain, is_technical: bool) -> int:
        """Estimate response token count."""
        base = {
            QuestionType.FACTUAL: 40,
            QuestionType.EXPLANATION: 100,
            QuestionType.PROCEDURAL: 120,
            QuestionType.ANALYSIS: 200,
            QuestionType.CREATION: 250,
            QuestionType.REASONING: 180,
        }[qtype]

        if domain in (Domain.HARD_SCIENCE, Domain.MEDICAL, Domain.LEGAL):
            base = int(base * 1.3)
        if is_technical:
            base = int(base * 1.2)

        return min(base, 500)

    # -------------------------------------------------------------------------
    # Main Classification
    # -------------------------------------------------------------------------

    def classify(self, prompt: str) -> ClassificationResult:
        """
        Classify a prompt into complexity level and model recommendation.

        Process:
        1. Extract all signals (domain, question type, reasoning depth, technical)
        2. Build context dict
        3. Evaluate rules in priority order
        4. Return first rule that fires
        """
        prompt = prompt.strip()
        if not prompt:
            return ClassificationResult(
                model_recommendation=ModelSize.TINY,
                complexity_level=ComplexityLevel.SIMPLE,
                confidence=0.0,
                reasoning="Empty prompt",
                domain=Domain.GENERAL,
                question_type=QuestionType.FACTUAL,
                estimated_response_tokens=20,
                rule_applied="default_empty",
            )

        # Extract signals
        domain = self._detect_domain(prompt)
        qtype = self._detect_qtype(prompt)
        reasoning_depth = self._detect_reasoning_depth(prompt)
        is_technical = self._detect_technical(prompt)

        # Build context for rule evaluation
        ctx = {
            "domain": domain,
            "qtype": qtype,
            "reasoning_depth": reasoning_depth,
            "is_technical": is_technical,
        }

        # Evaluate rules in priority order
        for rule in self.rules:
            if rule.condition(ctx):
                tokens = self._estimate_tokens(qtype, domain, is_technical)
                confidence = 0.6 + rule.priority * 0.05
                return ClassificationResult(
                    model_recommendation=rule.model,
                    complexity_level=rule.complexity,
                    confidence=min(confidence, 0.95),
                    reasoning=f"[rule={rule.priority}] {rule.reason}",
                    domain=domain,
                    question_type=qtype,
                    estimated_response_tokens=tokens,
                    rule_applied=rule.name,
                )

        # Fallback (should never reach here)
        return ClassificationResult(
            model_recommendation=ModelSize.TINY,
            complexity_level=ComplexityLevel.SIMPLE,
            confidence=0.3,
            reasoning="Default fallback",
            domain=domain,
            question_type=qtype,
            estimated_response_tokens=30,
            rule_applied="default_fallback",
        )
