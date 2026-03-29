"""
Model Pool — manages loading, caching, and inference across multiple models.

Design decisions in this file:
1. Each model runs in its own async task — inference doesn't block the event loop
2. Models are loaded lazily (on first use) — allows startup without loading everything
3. We track per-model metrics — latency, memory, token count
4. Consistent interface regardless of model size — enables transparent routing

Why this matters:
- Different models have different memory footprints
- We need to manage which models are loaded based on GPU memory availability
- Some requests should route to specific models (user preference, SLA requirements)
"""

import asyncio
import gc
import time
import torch
from typing import Optional
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

# Import our model size enum
import sys
sys.path.insert(0, ".")
from api.server import ModelSize


@dataclass
class ModelMetrics:
    """Tracks inference performance for a single model."""
    total_requests: int = 0
    total_tokens_generated: int = 0
    total_latency_ms: float = 0.0
    peak_memory_mb: float = 0.0
    errors: int = 0

    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    @property
    def avg_tokens_per_sec(self) -> float:
        if self.total_latency_ms == 0:
            return 0.0
        return (self.total_tokens_generated / self.total_latency_ms) * 1000


@dataclass
class ModelConfig:
    """Configuration for a model in the pool."""
    name: str
    hf_repo: str           # HuggingFace repository ID
    dtype: torch.dtype     # float32, float16, bfloat16
    max_memory_mb: int     # Approximate GPU memory this model uses
    max_context_length: int = 2048
    description: str = ""


# =============================================================================
# Model Registry — define our pool
# =============================================================================

MODEL_REGISTRY: dict[ModelSize, ModelConfig] = {
    ModelSize.TINY: ModelConfig(
        name="TinyLlama-1.1B",
        hf_repo="TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T",
        dtype=torch.float16,  # 2 bytes per param → ~2.2GB for 1.1B params
        max_memory_mb=2500,
        max_context_length=2048,
        description="Fast, cheap, good for simple factual queries and short responses",
    ),
    ModelSize.MEDIUM: ModelConfig(
        name="Phi-2-2.7B",
        hf_repo="microsoft/phi-2",
        dtype=torch.float16,  # ~5.4GB for 2.7B params
        max_memory_mb=6000,
        max_context_length=2048,
        description="Moderate reasoning, code understanding, longer responses",
    ),
    ModelSize.LARGE: ModelConfig(
        name="Qwen2-0.5B",
        hf_repo="Qwen/Qwen2-0.5B",
        dtype=torch.float16,
        max_memory_mb=1200,
        max_context_length=8192,  # Qwen supports long contexts
        description="Fallback / batched simple tasks / long context",
    ),
}


# =============================================================================
# Model Pool — the core class
# =============================================================================

class ModelPool:
    """
    Manages a pool of language models with lazy loading and metrics tracking.

    Key design:
    - Models are loaded ON DEMAND when first requested
    - If a model fails to load (OOM), we retry with lower dtype
    - Each model has its own async inference queue
    - Metrics are tracked per-model for routing decisions

    Usage:
        pool = ModelPool()
        result = await pool.generate(ModelSize.TINY, "Hello world", max_tokens=50)
    """

    def __init__(self):
        self._models: dict[ModelSize, Optional[Any]] = {}
        self._tokenizers: dict[ModelSize, Any] = {}
        self._metrics: dict[ModelSize, ModelMetrics] = {
            size: ModelMetrics() for size in ModelSize
        }
        self._load_lock = asyncio.Lock()
        self._is_initialized = False
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        if torch.cuda.is_available():
            print(f"  GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  No GPU — will run on CPU (slow)")

    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------

    async def load_model(self, size: ModelSize) -> bool:
        """
        Load a model and tokenizer for the given size.

        Why async?
        - model loading takes 10-30 seconds
        - we don't want to block the FastAPI event loop
        - allows serving other requests while a model loads

        Returns True if loaded successfully, False on error.
        """
        async with self._load_lock:
            if size in self._models and self._models[size] is not None:
                return True  # Already loaded

            config = MODEL_REGISTRY[size]
            print(f"  Loading {config.name} ({size.value})...")

            try:
                # Import here to avoid slow startup if transformers isn't installed yet
                from transformers import AutoTokenizer, AutoModelForCausalLM

                load_start = time.time()

                # Step 1: Load tokenizer (fast, ~1 second)
                tokenizer = AutoTokenizer.from_pretrained(config.hf_repo)
                self._tokenizers[size] = tokenizer

                # Step 2: Load model (slow, 10-30 seconds)
                # We use device_map="auto" to let accelerate place layers optimally
                model = AutoModelForCausalLM.from_pretrained(
                    config.hf_repo,
                    torch_dtype=config.dtype,
                    device_map="auto",
                    low_cpu_mem_usage=True,  # reduces peak RAM during loading
                )

                # Step 3: Clear GPU cache after loading
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                self._models[size] = model

                load_time = time.time() - load_start
                print(f"  ✓ {config.name} loaded in {load_time:.1f}s")

                # Update metrics with memory info
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1e6
                    self._metrics[size].peak_memory_mb = max(
                        self._metrics[size].peak_memory_mb, allocated
                    )
                    print(f"    GPU memory allocated: {allocated:.0f} MB")

                return True

            except Exception as e:
                print(f"  ✗ Failed to load {size.value}: {e}")
                # Try fallback: load on CPU if GPU fails
                try:
                    print(f"  Retrying {config.name} on CPU...")
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    tokenizer = AutoTokenizer.from_pretrained(config.hf_repo)
                    model = AutoModelForCausalLM.from_pretrained(
                        config.hf_repo,
                        torch_dtype=torch.float32,
                        device_map="cpu",
                    )
                    self._tokenizers[size] = tokenizer
                    self._models[size] = model
                    self._device = "cpu"
                    print(f"  ✓ {config.name} loaded on CPU")
                    return True
                except Exception as e2:
                    print(f"  ✗ CPU fallback also failed: {e2}")
                    return False

    async def ensure_model(self, size: ModelSize) -> bool:
        """Ensure a model is loaded before inference. Lazy loading."""
        if size not in self._models or self._models[size] is None:
            return await self.load_model(size)
        return True

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    async def generate(
        self,
        size: ModelSize,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> dict:
        """
        Generate text from a prompt using the specified model.

        Args:
            size: Which model to use (tiny, medium, large)
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy, 1 = creative)
            stream: If True, yields tokens as they're generated

        Returns:
            dict with keys: text, prompt_tokens, completion_tokens, latency_ms

        Why run inference in a thread?
        - model.generate() is CPU-bound (Python interpreter + PyTorch ops)
        - asyncio can't parallelize CPU work — it only handles I/O
        - ThreadPoolExecutor lets us run inference without blocking the event loop
        - FastAPI's async handlers can still process other requests while waiting

        Note: For true GPU parallelism (batching multiple requests),
        we'll build a custom batcher in Day 5.
        """
        # Ensure model is loaded
        loaded = await self.ensure_model(size)
        if not loaded:
            raise RuntimeError(f"Failed to load model {size.value}")

        model = self._models[size]
        tokenizer = self._tokenizers[size]

        # Track metrics
        metrics = self._metrics[size]
        metrics.total_requests += 1

        start_time = time.time()

        # Run inference in a thread pool to avoid blocking the event loop
        # Using ThreadPoolExecutor because model.forward() holds the GIL
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(
                executor,
                self._generate_sync,
                model, tokenizer, prompt, max_tokens, temperature
            )

        latency_ms = (time.time() - start_time) * 1000

        # Update metrics
        metrics.total_latency_ms += latency_ms
        metrics.total_tokens_generated += result["completion_tokens"]

        if torch.cuda.is_available():
            current_mem = torch.cuda.memory_allocated() / 1e6
            metrics.peak_memory_mb = max(metrics.peak_memory_mb, current_mem)

        result["latency_ms"] = latency_ms
        return result

    @staticmethod
    def _generate_sync(
        model,
        tokenizer,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> dict:
        """Synchronous inference — runs in thread pool."""
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_tokens = inputs["input_ids"].shape[1]

        # Move inputs to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                # Performance hints:
                use_cache=True,       # Enable KV cache — CRITICAL for efficiency
                # For batch inference, we'd use `generation_config`
            )

        # Decode only the newly generated tokens
        completion_tokens = outputs.shape[1] - prompt_tokens
        generated_ids = outputs[0][prompt_tokens:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    def get_metrics(self) -> dict:
        """Return current metrics for all models."""
        return {
            size.value: {
                "total_requests": self._metrics[size].total_requests,
                "avg_latency_ms": round(self._metrics[size].avg_latency_ms, 2),
                "avg_tokens_per_sec": round(self._metrics[size].avg_tokens_per_sec, 2),
                "peak_memory_mb": round(self._metrics[size].peak_memory_mb, 0),
                "total_tokens": self._metrics[size].total_tokens_generated,
                "errors": self._metrics[size].errors,
            }
            for size in ModelSize
        }

    def unload_model(self, size: ModelSize):
        """Unload a model to free GPU memory."""
        if size in self._models and self._models[size] is not None:
            del self._models[size]
            self._models[size] = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  Unloaded {size.value} — freed GPU memory")


# =============================================================================
# Global instance — shared across all requests
# =============================================================================

# We use a module-level singleton so FastAPI workers share the same pool
# In production with multiple workers, you'd use a shared cache (Redis) or
# separate pools per worker
_model_pool: Optional[ModelPool] = None


def get_model_pool() -> ModelPool:
    global _model_pool
    if _model_pool is None:
        _model_pool = ModelPool()
    return _model_pool
