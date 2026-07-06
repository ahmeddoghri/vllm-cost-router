"""The gateway: ties together routing, caching, and batching in front of an
LLM backend. This is the whole point of the service -- none of cost routing,
caching, or batching individually is novel, but stacking them in front of
every request is what actually moves the cost/latency needle in production.
"""
from __future__ import annotations

from dataclasses import dataclass

from .backend import CompletionResult, LLMBackend
from .cache import TTLCache, cache_key
from .metrics import Metrics
from .router import choose_tier


@dataclass
class Request:
    prompt: str
    max_tokens: int = 256


class Gateway:
    def __init__(self, backend: LLMBackend, cache: TTLCache, metrics: Metrics,
                 threshold: float, small_model: str, large_model: str) -> None:
        self.backend = backend
        self.cache = cache
        self.metrics = metrics
        self.threshold = threshold
        self.small_model = small_model
        self.large_model = large_model

    def complete_one(self, req: Request) -> CompletionResult:
        """Single-request path (typical interactive traffic): cache, then a
        single-item backend call at whatever tier the router picks."""
        tier, _ = choose_tier(req.prompt, req.max_tokens, self.threshold,
                              self.small_model, self.large_model)
        key = cache_key(req.prompt, tier, req.max_tokens)
        cached = self.cache.get(key)
        if cached is not None:
            self.metrics.record(0.0, 1.0, tier, cache_hit=True)  # cache hits are ~free & instant
            return cached  # type: ignore[return-value]

        result = self.backend.complete_batch([req.prompt], tier)[0]
        self.cache.set(key, result)
        self.metrics.record(result.cost_usd, result.latency_ms, tier, cache_hit=False)
        return result

    def complete_batch(self, reqs: list[Request]) -> list[CompletionResult]:
        """Batch path: group cache-missed requests by tier and dispatch each
        tier as one backend call, so fixed per-batch overhead is amortized
        across every request in it -- the actual mechanism behind vLLM's
        continuous batching cost/latency win.
        """
        results: list[CompletionResult | None] = [None] * len(reqs)
        groups: dict[str, list[int]] = {}

        for i, req in enumerate(reqs):
            tier, _ = choose_tier(req.prompt, req.max_tokens, self.threshold,
                                  self.small_model, self.large_model)
            key = cache_key(req.prompt, tier, req.max_tokens)
            cached = self.cache.get(key)
            if cached is not None:
                results[i] = cached  # type: ignore[assignment]
                self.metrics.record(0.0, 1.0, tier, cache_hit=True)
            else:
                groups.setdefault(tier, []).append(i)

        for tier, idxs in groups.items():
            prompts = [reqs[i].prompt for i in idxs]
            batch_results = self.backend.complete_batch(prompts, tier)
            for i, result in zip(idxs, batch_results):
                results[i] = result
                key = cache_key(reqs[i].prompt, tier, reqs[i].max_tokens)
                self.cache.set(key, result)
                self.metrics.record(result.cost_usd, result.latency_ms, tier, cache_hit=False)

        return results  # type: ignore[return-value]
