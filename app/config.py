"""Settings loaded from environment variables. Everything has a safe local
default so the service runs (and CI passes) with zero external dependencies;
set BACKEND=openai and the OPENAI_* vars to point it at a real vLLM/OpenAI-
compatible server.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    backend: str = os.environ.get("BACKEND", "mock")  # "mock" | "openai"
    openai_base_url: str = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    small_model: str = os.environ.get("SMALL_MODEL", "small-tier")
    large_model: str = os.environ.get("LARGE_MODEL", "large-tier")
    cache_ttl_seconds: float = float(os.environ.get("CACHE_TTL_SECONDS", "300"))
    cache_max_size: int = int(os.environ.get("CACHE_MAX_SIZE", "2048"))
    batch_window_ms: float = float(os.environ.get("BATCH_WINDOW_MS", "20"))
    max_batch_size: int = int(os.environ.get("MAX_BATCH_SIZE", "8"))
    complexity_threshold: float = float(os.environ.get("COMPLEXITY_THRESHOLD", "0.5"))
    # Optional API-key auth: when set, write endpoints require a matching
    # X-API-Key header. Empty (the default) leaves the service open.
    api_key: str = os.environ.get("API_KEY", "")
    # Request-size guards to keep a single caller from exhausting memory.
    max_prompt_chars: int = int(os.environ.get("MAX_PROMPT_CHARS", "100000"))
    max_batch_requests: int = int(os.environ.get("MAX_BATCH_REQUESTS", "128"))


settings = Settings()
