"""Does the router+cache+batch gateway actually beat naive always-large-model
serving on cost and p95 latency?

We simulate a realistic support-bot-style workload: a quarter of requests are
genuinely complex (code/analysis, correctly needing the large model), a
quarter are literal repeated FAQs (cacheable), and half are simple but
user-specific templated requests (an order number or account name makes each
one unique, so they're cheap-tier appropriate but not cacheable) -- the
realistic shape of production support traffic, not an artificially repetitive
toy set.

    python -m app.eval
"""
from __future__ import annotations

import argparse
import random

from .backend import MockBackend
from .cache import TTLCache
from .gateway import Gateway, Request
from .metrics import Metrics

FAQS = [
    "What are your business hours?",
    "How do I reset my password?",
    "Where is my order?",
    "What is your return policy?",
    "How do I contact support?",
    "Do you ship internationally?",
    "How long does shipping take?",
    "Can I change my shipping address?",
    "How do I cancel my subscription?",
    "What payment methods do you accept?",
    "Is there a student discount?",
    "How do I update my billing info?",
    "What is your refund policy?",
    "Do you offer a free trial?",
    "How do I delete my account?",
    "Can I use multiple coupons?",
    "What is the warranty period?",
    "How do I track my package?",
    "Do you have a mobile app?",
    "How do I upgrade my plan?",
    "What happens if I miss a payment?",
    "Can I get an invoice for my purchase?",
    "How do I add a team member?",
    "Is my data encrypted?",
    "What is your uptime guarantee?",
]
COMPLEX_PROMPTS = [
    "Explain the trade-offs between eventual consistency and strong consistency for our order pipeline.",
    "Analyze this function and suggest a more efficient algorithm:\n```def f(x):\n  for i in range(x):\n    for j in range(x):\n      pass\n```",
    "Design a database schema for a multi-tenant SaaS billing system and explain your reasoning.",
    "Compare REST and gRPC for our internal service mesh and recommend one with justification.",
]
# realistic support traffic is mostly *templated but user-specific* -- an order
# number or account name makes each message unique even though the underlying
# question is simple, so it's cheap-tier appropriate but not cacheable
UNIQUE_TEMPLATES = [
    "Where is my order #{n}?",
    "Can you look up account {n} for me?",
    "I was charged twice on invoice #{n}, please check.",
    "My tracking number {n} hasn't updated in 3 days.",
]


def generate_workload(n: int = 500, complex_fraction: float = 0.25,
                      faq_fraction: float = 0.25, seed: int = 0) -> list[str]:
    """A support-bot-style mix: a minority of genuinely complex asks (large
    tier), a minority of literal repeated FAQs (cacheable), and the rest
    unique-but-simple templated requests (small tier, not cacheable) -- the
    realistic shape of production support traffic.
    """
    rng = random.Random(seed)
    workload = []
    for i in range(n):
        r = rng.random()
        if r < complex_fraction:
            workload.append(rng.choice(COMPLEX_PROMPTS))
        elif r < complex_fraction + faq_fraction:
            workload.append(rng.choice(FAQS))
        else:
            template = rng.choice(UNIQUE_TEMPLATES)
            workload.append(template.format(n=100000 + i))
    return workload


def run_naive(prompts: list[str]) -> Metrics:
    """Baseline: every request hits the large model individually, no cache."""
    from .backend import TIER_COST_PER_CALL, TIER_LATENCY_MS
    metrics = Metrics()
    for p in prompts:
        metrics.record(TIER_COST_PER_CALL["large-tier"], TIER_LATENCY_MS["large-tier"],
                       "large-tier", cache_hit=False)
    return metrics


def run_router(prompts: list[str], threshold: float = 0.5, batch_size: int = 8) -> Metrics:
    """Router policy: complexity-based tiering + caching + micro-batching."""
    metrics = Metrics()
    gateway = Gateway(MockBackend(), TTLCache(max_size=4096, ttl_seconds=600), metrics,
                      threshold, "small-tier", "large-tier")
    for i in range(0, len(prompts), batch_size):
        batch = [Request(p) for p in prompts[i:i + batch_size]]
        gateway.complete_batch(batch)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    prompts = generate_workload(args.n)
    naive = run_naive(prompts)
    router = run_router(prompts, batch_size=args.batch_size)

    cost_reduction = 1 - (router.total_cost_usd / naive.total_cost_usd)
    p95_reduction = 1 - (router.percentile(95) / naive.percentile(95))

    print(f"workload: {args.n} requests (25% genuinely complex, 25% repeated FAQs, "
          f"50% unique-but-simple templated requests)\n")
    print(f"{'policy':<12}{'total cost':>14}{'p95 latency':>16}{'cache hit rate':>18}")
    print(f"{'naive':<12}{'$' + f'{naive.total_cost_usd:.4f}':>14}{naive.percentile(95):>15.1f}ms{'--':>18}")
    print(f"{'router':<12}{'$' + f'{router.total_cost_usd:.4f}':>14}{router.percentile(95):>15.1f}ms"
          f"{router.summary()['cache_hit_rate']:>17.0%}")
    print(f"\ncost reduction: {cost_reduction:.0%}   p95 latency reduction: {p95_reduction:.0%}")


if __name__ == "__main__":
    main()
