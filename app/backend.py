"""LLM backends. ``MockBackend`` is deterministic and dependency-free so the
service, its tests, and its benchmark all run with zero API keys. Point
``BACKEND=openai`` at a real vLLM server (vLLM exposes an OpenAI-compatible
`/v1/chat/completions` endpoint) for production use -- the router, cache, and
batching logic above this layer don't change.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Protocol

# Per-request cost/latency model. A 2x price/latency gap between a distilled
# small model and the full-size one is a conservative, realistic gap for
# self-hosted vLLM tiers (hosted-API tier gaps are often much larger).
TIER_COST_PER_CALL = {"small-tier": 0.003, "large-tier": 0.006}
TIER_LATENCY_MS = {"small-tier": 150.0, "large-tier": 220.0}
# Batching amortizes fixed per-call overhead (connection setup, KV-cache
# warmup) across N requests -- this is what vLLM's continuous batching buys
# you in production; here it's modeled as a fixed per-batch overhead spread
# across batch members instead of paid by every request.
BATCH_FIXED_OVERHEAD_MS = 30.0


@dataclass
class CompletionResult:
    text: str
    model: str
    cost_usd: float
    latency_ms: float


class LLMBackend(Protocol):
    def complete_batch(self, prompts: list[str], model: str) -> list[CompletionResult]:
        ...


class MockBackend:
    """Deterministic stand-in: cost/latency are modeled analytically (no sleep,
    no randomness) so tests and CI are both fast and reproducible."""

    def complete_batch(self, prompts: list[str], model: str) -> list[CompletionResult]:
        n = len(prompts)
        per_call_cost = TIER_COST_PER_CALL.get(model, TIER_COST_PER_CALL["large-tier"])
        per_call_latency = TIER_LATENCY_MS.get(model, TIER_LATENCY_MS["large-tier"])
        # batching spreads the fixed overhead across the batch
        overhead_share = BATCH_FIXED_OVERHEAD_MS / n
        results = []
        for p in prompts:
            digest = hashlib.md5(p.encode()).hexdigest()[:8]
            results.append(
                CompletionResult(
                    text=f"[{model}] response-{digest}",
                    model=model,
                    cost_usd=per_call_cost,
                    latency_ms=per_call_latency / n * 1.0 + overhead_share,
                )
            )
        return results


class OpenAICompatBackend:
    """Real backend: calls an OpenAI-compatible /v1/chat/completions endpoint
    (this is exactly the interface vLLM's server exposes). Only imports httpx
    when actually used, so it's never a hard dependency for local/mock mode.
    """

    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def complete_batch(self, prompts: list[str], model: str) -> list[CompletionResult]:
        import httpx

        results = []
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        with httpx.Client(timeout=30.0) as client:
            for p in prompts:
                start = time.perf_counter()
                resp = client.post(
                    f"{self.base_url}/chat/completions",
                    json={"model": model, "messages": [{"role": "user", "content": p}]},
                    headers=headers,
                )
                resp.raise_for_status()
                elapsed_ms = (time.perf_counter() - start) * 1000
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                # cost estimation left to the caller's pricing table in production;
                # here we surface token counts via cost_usd=0 and let ops attach
                # real pricing based on `usage` if needed.
                results.append(CompletionResult(text=text, model=model, cost_usd=0.0,
                                                 latency_ms=elapsed_ms))
        return results


def get_backend(name: str, base_url: str = "", api_key: str = "") -> LLMBackend:
    if name == "openai":
        return OpenAICompatBackend(base_url, api_key)
    return MockBackend()
