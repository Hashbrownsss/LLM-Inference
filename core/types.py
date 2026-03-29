"""
Shared types across the project.

This file exists to avoid circular imports:
- core/model_pool.py needs ModelSize
- api/server.py needs ModelSize and Request/Response models
- router/ and classifier/ also need these types

Centralizing here means no circular dependency.
"""

from enum import Enum


class ModelSize(str, Enum):
    """Available model sizes in our pool."""
    TINY   = "tinyllama"    # 1.1B params — fast, cheap, simple queries
    MEDIUM = "phi2"         # 2.7B params — moderate reasoning
    LARGE  = "qwen2"        # 0.5B params — fallback / batched tasks
