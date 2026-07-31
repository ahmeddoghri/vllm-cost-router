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
from .router import choose_tier as choose_tier_v1
from .router_v2 import choose_tier as choose_tier_v2


def get_router(version: str):
    """Pick a routing function by name.

    Defaults to v2 for anything unrecognized: the failure mode of the keyword
    router is silent (hard requests quietly answered by the small model), so
    it should never be reached by accident.
    """
    return choose_tier_v1 if version == "v1" else choose_tier_v2


#: Each scorer produces values on its own scale, so a threshold is only
#: meaningful paired with the router it was chosen for. Passing v1's 0.5 to the
#: structural scorer silently sends nearly everything to the cheap tier, which
#: is exactly the failure this package exists to prevent.
DEFAULT_THRESHOLDS = {"v1": 0.50, "v2": 0.10}


def default_threshold(version: str) -> float:
    return DEFAULT_THRESHOLDS.get(version, DEFAULT_THRESHOLDS["v2"])


@dataclass
class Request:
    prompt: str
    max_tokens: int = 256


class Gateway:
    def __init__(self, backend: LLMBackend, cache: TTLCache, metrics: Metrics,
                 threshold: float | None = None, small_model: str = "small-tier",
                 large_model: str = "large-tier", router_version: str = "v2") -> None:
        self.backend = backend
        self.cache = cache
        self.metrics = metrics
        # None means "whatever is right for this router", which is the only
        # safe default given the two scorers use different scales.
        self.threshold = default_threshold(router_version) if threshold is None else threshold
        self.small_model = small_model
        self.large_model = large_model
        self.router_version = router_version
        self.choose_tier = get_router(router_version)

    def complete_one(self, req: Request) -> CompletionResult:
        """Single-request path (typical interactive traffic): cache, then a
        single-item backend call at whatever tier the router picks."""
        tier, _ = self.choose_tier(req.prompt, req.max_tokens, self.threshold,
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
            tier, _ = self.choose_tier(req.prompt, req.max_tokens, self.threshold,
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
