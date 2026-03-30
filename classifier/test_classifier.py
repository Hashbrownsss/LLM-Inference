"""
Test script for the Request Classifier.

Run: python classifier/test_classifier.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from classifier.classifier import RequestClassifier, ComplexityLevel, ModelSize


def test_classifier():
    classifier = RequestClassifier()

    # (prompt, minimum expected_model, reason)
    test_cases = [
        # Simple — TinyLlama should handle these
        ("What is the capital of France?", ModelSize.TINY, "simple factual"),
        ("Hello, how are you?", ModelSize.TINY, "simple greeting"),
        ("What time is it?", ModelSize.TINY, "simple factual"),
        ("Define the word 'algorithm'", ModelSize.TINY, "simple definition"),
        ("What's 2+2?", ModelSize.TINY, "simple math"),

        # Moderate — Phi-2 or LARGE recommended
        ("Compare Python and JavaScript for backend development", ModelSize.LARGE, "technical analysis"),
        ("Explain how a neural network learns through backpropagation", ModelSize.LARGE, "technical explanation with reasoning"),
        ("Write a Python function to check if a number is prime", ModelSize.MEDIUM, "technical code creation"),
        ("What are the pros and cons of microservices?", ModelSize.LARGE, "technical analysis"),
        ("Debug this code: for i in range(10): print(i)", ModelSize.LARGE, "technical debugging"),

        # Complex — LARGE needed
        ("Prove that the sum of angles in a triangle is 180 degrees", ModelSize.LARGE, "hard science + reasoning"),
        ("Derive the time complexity of quicksort using recurrence relations", ModelSize.LARGE, "hard science reasoning"),
        ("Analyze the trade-offs between CAP theorem properties in distributed systems", ModelSize.LARGE, "technical analysis"),
        ("Why did the Roman Empire fall? Consider economic, military, and political factors.", ModelSize.LARGE, "social analysis multi-factor"),

        # Medical/Legal — LARGE due to domain
        ("What is the mechanism of action of mRNA vaccines?", ModelSize.LARGE, "medical domain"),
        ("Explain GDPR article 17 and its implications for data retention policies", ModelSize.LARGE, "legal domain"),

        # Creative — moderate
        ("Write a short story about dragons", ModelSize.MEDIUM, "creative writing moderate length"),
        ("Write a 10000 word story about dragons", ModelSize.MEDIUM, "creative but manageable"),

        # Tricky
        ("Hello", ModelSize.TINY, "minimal prompt"),
        ("Prove this theorem", ModelSize.LARGE, "pure reasoning command"),
    ]

    print("=" * 80)
    print("  Request Classifier — Test Results")
    print("=" * 80)

    results = []
    for prompt, expected_min, description in test_cases:
        result = classifier.classify(prompt)
        # Check: did we recommend at least the minimum model?
        model_order = [ModelSize.TINY, ModelSize.MEDIUM, ModelSize.LARGE]
        got_rank = model_order.index(result.model_recommendation)
        expect_rank = model_order.index(expected_min)
        passed = got_rank <= expect_rank

        results.append({
            "prompt": prompt[:55],
            "got": result.model_recommendation.value,
            "expected": expected_min.value,
            "passed": passed,
            "rule": result.rule_applied[:30] if result.rule_applied else "none",
            "reasoning": result.reasoning[:60],
            "domain": result.domain.value,
            "qtype": result.question_type.value,
            "depth": getattr(result, 'reasoning_depth', '?'),
        })

    # Print results in a table
    print(f"\n{'#':<3} {'Prompt':<50} {'Got':<10} {'Expected':<10} {'Rule'}")
    print("-" * 120)
    for i, r in enumerate(results):
        status = "[OK]" if r["passed"] else "[FAIL]"
        print(f"{i+1:<3} {r['prompt']:<50} {r['got']:<10} {r['expected']:<10} {r['rule']}")

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"\n{'=' * 80}")
    print(f"  Result: {passed}/{total} = {passed/total*100:.0f}% passed")
    print(f"{'=' * 80}")

    # Show detailed failures
    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for r in failures:
            print(f"  FAIL: '{r['prompt']}'")
            print(f"    Got: {r['got']}, Expected: {r['expected']} or larger")
            print(f"    Rule: {r['rule']}")
            print(f"    Reasoning: {r['reasoning']}")
            print(f"    Domain: {r['domain']}, QType: {r['qtype']}")


def interactive_test():
    classifier = RequestClassifier()
    print("\nInteractive mode — type a prompt and press Enter.")
    print("Type 'quit' to exit.\n")
    while True:
        try:
            prompt = input("Prompt: ").strip()
            if prompt.lower() == "quit":
                break
            if not prompt:
                continue
            result = classifier.classify(prompt)
            print(f"  Model:         {result.model_recommendation.value}")
            print(f"  Complexity:   {result.complexity_level.value}")
            print(f"  Domain:       {result.domain.value}")
            print(f"  QType:        {result.question_type.value}")
            print(f"  Rule fired:   {result.rule_applied}")
            print(f"  Confidence:   {result.confidence:.0%}")
            print(f"  Est. tokens:  {result.estimated_response_tokens}")
            print(f"  Reasoning:    {result.reasoning}")
            print()
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_test()
    else:
        test_classifier()
