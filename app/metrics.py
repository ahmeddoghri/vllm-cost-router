"""Rolling request metrics: cost, latency percentiles, cache hit rate."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Metrics:
    total_requests: int = 0
    cache_hits: int = 0
    total_cost_usd: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)
    tier_counts: dict[str, int] = field(default_factory=dict)

    def record(self, cost_usd: float, latency_ms: float, model: str, cache_hit: bool) -> None:
        self.total_requests += 1
        if cache_hit:
            self.cache_hits += 1
        self.total_cost_usd += cost_usd
        self.latencies_ms.append(latency_ms)
        self.tier_counts[model] = self.tier_counts.get(model, 0) + 1

    def percentile(self, p: float) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        idx = min(len(s) - 1, max(0, math.ceil(p / 100 * len(s)) - 1))
        return s[idx]

    def summary(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "cache_hit_rate": round(self.cache_hits / self.total_requests, 4) if self.total_requests else 0.0,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_cost_usd": round(self.total_cost_usd / self.total_requests, 6) if self.total_requests else 0.0,
            "p50_latency_ms": round(self.percentile(50), 2),
            "p95_latency_ms": round(self.percentile(95), 2),
            "tier_counts": dict(self.tier_counts),
        }
