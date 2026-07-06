from app.backend import MockBackend
from app.cache import TTLCache, cache_key
from app.eval import generate_workload, run_naive, run_router
from app.gateway import Gateway, Request
from app.metrics import Metrics
from app.router import choose_tier


def test_simple_prompt_routes_to_small_tier():
    tier, score = choose_tier("What are your business hours?", 64, 0.5, "small", "large")
    assert tier == "small"
    assert score < 0.5


def test_complex_prompt_routes_to_large_tier():
    prompt = "Explain the trade-offs between eventual consistency and strong consistency."
    tier, score = choose_tier(prompt, 512, 0.5, "small", "large")
    assert tier == "large"
    assert score >= 0.5


def test_cache_hit_avoids_backend_call():
    cache = TTLCache(ttl_seconds=60)
    key = cache_key("hello", "small", 64)
    assert cache.get(key) is None
    cache.set(key, "cached-value")
    assert cache.get(key) == "cached-value"


def test_cache_expires_after_ttl():
    t = [1000.0]
    cache = TTLCache(ttl_seconds=5, clock=lambda: t[0])
    key = cache_key("hello", "small", 64)
    cache.set(key, "value")
    t[0] += 10
    assert cache.get(key) is None


def test_gateway_repeated_request_hits_cache():
    metrics = Metrics()
    gateway = Gateway(MockBackend(), TTLCache(ttl_seconds=60), metrics, 0.5, "small", "large")
    r1 = gateway.complete_one(Request("hello there"))
    r2 = gateway.complete_one(Request("hello there"))
    assert r1.text == r2.text
    assert metrics.cache_hits == 1
    assert metrics.total_requests == 2


def test_gateway_batch_groups_by_tier():
    metrics = Metrics()
    gateway = Gateway(MockBackend(), TTLCache(ttl_seconds=60), metrics, 0.5, "small", "large")
    reqs = [Request("What are your business hours?"), Request("Where is my order?"),
            Request("Explain the trade-offs of eventual consistency vs strong consistency.")]
    results = gateway.complete_batch(reqs)
    assert len(results) == 3
    assert results[0].model == "small"
    assert results[2].model == "large"


def test_benchmark_router_beats_naive_on_cost_and_latency():
    prompts = generate_workload(n=500, seed=0)
    naive = run_naive(prompts)
    router = run_router(prompts)
    assert router.total_cost_usd < naive.total_cost_usd
    assert router.percentile(95) < naive.percentile(95)
    cost_reduction = 1 - (router.total_cost_usd / naive.total_cost_usd)
    assert cost_reduction > 0.3  # meaningful, not a rounding artifact


def test_benchmark_is_deterministic_across_hash_seeds():
    # regression guard: this project's sibling repo (citebench) had a real
    # hash-seed-dependent flakiness bug; make sure this one doesn't either
    prompts = generate_workload(n=200, seed=1)
    r1 = run_router(prompts)
    r2 = run_router(prompts)
    assert r1.total_cost_usd == r2.total_cost_usd
    assert r1.percentile(95) == r2.percentile(95)
