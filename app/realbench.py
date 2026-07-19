"""The same router, pointed at real models.

The main benchmark (``python -m app.eval``) uses an analytical mock so CI
stays fast and free. This one loads two real HuggingFace chat models as the
small and large tiers, pushes a graded workload through the exact same
Gateway (router + cache + batching), and measures what routing actually
buys you when the latency is wall-clock and the quality gap is real.

    python -m app.realbench

Quality scoring is deliberately narrow: only prompts with an exactly
checkable answer are graded (factual containment, JSON parses, exactly
three bullets). The hard prompts stay in the workload because they drive
routing, cost, and latency, but their open-ended answers are not graded.
An early version graded them by keyword coverage and it mostly measured
truncation, not quality, so it got cut. No judge model, no API, every
check is in this file.

Generations are cached to realbench_cache.json (gitignored) so re-runs are
fast; delete the file to re-measure from scratch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re

from .backend import CompletionResult
from .cache import TTLCache
from .gateway import Gateway, Request
from .metrics import Metrics
from .realbackend import LocalHFBackend

SMALL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LARGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CACHE_PATH = pathlib.Path(__file__).resolve().parent.parent / "realbench_cache.json"

# ---------------------------------------------------------------- workload
# Each item: (prompt, kind, payload). Factual prompts avoid the router's
# reasoning-marker words on purpose; hard prompts use them on purpose. That
# is not gaming the benchmark, it is what the router keys on by design.

FACTUAL = [
    ("What is the capital of France? Answer in one word.", "paris"),
    ("What is the capital of Japan? Answer in one word.", "tokyo"),
    ("What is 12 times 11? Answer with just the number.", "132"),
    ("What is 45 plus 55? Answer with just the number.", "100"),
    ("How many days are in a leap year? Answer with just the number.", "366"),
    ("What does CPU stand for? Answer briefly.", "central processing unit"),
    ("What does HTTP stand for? Answer briefly.", "hypertext transfer protocol"),
    ("What planet is closest to the sun? Answer in one word.", "mercury"),
    ("What is the chemical symbol for gold? Answer briefly.", "au"),
    ("How many minutes are in three hours? Answer with just the number.", "180"),
    ("What language is spoken in Brazil? Answer in one word.", "portuguese"),
    ("What is the largest ocean on Earth? Answer briefly.", "pacific"),
    ("What is 20 percent of 250? Answer with just the number.", "50"),
    ("How many sides does a hexagon have? Answer with just the number.", "6"),
    ("What is the boiling point of water in Celsius at sea level? Just the number.", "100"),
    ("What does RAM stand for? Answer briefly.", "random access memory"),
]

FORMAT = [
    ("Return a JSON object with exactly two keys, name set to Bob and age set to 30. "
     "Output only the JSON.", "json_name_age"),
    ("List exactly three primary colors as a comma-separated list, nothing else.",
     "comma_list_3"),
    ("Answer with exactly one word, yes or no: is the Earth round?", "yes_no"),
    ("Return a JSON array of the numbers 1, 2, and 3. Output only the JSON.",
     "json_array_123"),
    ("Write the word hello in uppercase, and output nothing else.", "hello_upper"),
    ("List exactly three fruits, one per line, each line starting with a dash.",
     "dash_lines_3"),
    ("Answer with exactly one word, yes or no: is 7 an even number?", "no_word"),
    ("Return a JSON object with a single key city set to Toronto. Output only the JSON.",
     "json_city"),
]

# Open-ended prompts. They exercise the router's "this needs the big model"
# path and dominate token cost, but they are not quality-graded: keyword
# checks on essay answers measure truncation and phrasing, not correctness.
HARD = [
    "Explain the trade-offs between eventual consistency and strong consistency "
    "in a distributed database.",
    "Compare REST and gRPC for internal microservices and recommend one with justification.",
    "Explain why quicksort is usually faster than bubble sort, mentioning time complexity.",
    "Design a database schema for a simple library lending system and explain your choices.",
    "Explain why TCP uses a three-way handshake instead of a two-way handshake.",
    "Analyze the pros and cons of microservices versus a monolith for a five-person startup.",
    "Explain how a hash table achieves O(1) average lookup and what breaks that guarantee.",
    "Compare SQL and NoSQL databases and explain when each is the right choice.",
    "Explain why caching can return stale data and describe two invalidation strategies.",
    "Design a rate limiter for an API and explain the algorithm you would use.",
    "Explain the difference between processes and threads, including memory sharing.",
    "Analyze this code and explain its time complexity:\n```\ndef f(x):\n  for i in range(x):\n    for j in range(x):\n      pass\n```",
]


def check_factual(response: str, gold: str) -> bool:
    return gold.lower() in response.lower()


def check_format(response: str, spec: str) -> bool:
    text = response.strip()
    fenced = re.sub(r"^```[a-z]*\n?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        if spec == "json_name_age":
            obj = json.loads(fenced)
            return obj.get("name") == "Bob" and obj.get("age") == 30
        if spec == "json_array_123":
            return json.loads(fenced) == [1, 2, 3]
        if spec == "json_city":
            return json.loads(fenced).get("city") == "Toronto"
    except (json.JSONDecodeError, AttributeError):
        return False
    if spec == "comma_list_3":
        return len([p for p in text.split(",") if p.strip()]) == 3
    if spec == "yes_no":
        return text.lower().rstrip(".") == "yes"
    if spec == "no_word":
        return text.lower().rstrip(".") == "no"
    if spec == "hello_upper":
        return "HELLO" in text and "hello" not in text.replace("HELLO", "")
    if spec == "dash_lines_3":
        lines = [ln for ln in text.splitlines() if ln.strip()]
        return len(lines) == 3 and all(ln.lstrip().startswith("-") for ln in lines)
    return False


def build_workload() -> list[tuple[str, str, object]]:
    items: list[tuple[str, str, object]] = []
    items += [(p, "factual", gold) for p, gold in FACTUAL]
    items += [(p, "format", spec) for p, spec in FORMAT]
    items += [(p, "hard", None) for p in HARD]
    return items


def grade(kind: str, payload: object, response: str) -> bool | None:
    """True/False for exactly checkable prompts, None for ungraded ones."""
    if kind == "factual":
        return check_factual(response, payload)  # type: ignore[arg-type]
    if kind == "format":
        return check_format(response, payload)  # type: ignore[arg-type]
    return None


# ------------------------------------------------------- generation caching
class MemoBackend:
    """Wraps LocalHFBackend with a JSON memo of past generations (including
    measured latency), so the three policies share work across runs. Real
    measurements happen once per (tier, prompt); re-runs are free."""

    def __init__(self, inner: LocalHFBackend, path: pathlib.Path) -> None:
        self.inner = inner
        self.path = path
        self.memo: dict[str, dict] = {}
        if path.exists():
            self.memo = json.loads(path.read_text())

    def _key(self, prompt: str, model: str) -> str:
        return hashlib.sha256(f"{model}::{prompt}".encode()).hexdigest()

    def complete_batch(self, prompts: list[str], model: str) -> list[CompletionResult]:
        results = []
        dirty = False
        for p in prompts:
            k = self._key(p, model)
            if k not in self.memo:
                res = self.inner.complete_batch([p], model)[0]
                self.memo[k] = {
                    "text": res.text, "model": res.model,
                    "cost_usd": res.cost_usd, "latency_ms": res.latency_ms,
                }
                dirty = True
            m = self.memo[k]
            results.append(CompletionResult(m["text"], m["model"], m["cost_usd"],
                                            m["latency_ms"]))
        if dirty:
            self.path.write_text(json.dumps(self.memo, indent=1))
        return results


# ----------------------------------------------------------------- policies
def run_fixed(backend: MemoBackend, items, tier: str) -> tuple[Metrics, int, int]:
    """Returns (metrics over all prompts, graded correct, graded total)."""
    metrics = Metrics()
    correct = graded = 0
    for prompt, kind, payload in items:
        res = backend.complete_batch([prompt], tier)[0]
        metrics.record(res.cost_usd, res.latency_ms, tier, cache_hit=False)
        verdict = grade(kind, payload, res.text)
        if verdict is not None:
            graded += 1
            correct += verdict
    return metrics, correct, graded


def run_routed(backend: MemoBackend, items,
               threshold: float = 0.5) -> tuple[Metrics, int, int, dict]:
    metrics = Metrics()
    gateway = Gateway(backend, TTLCache(max_size=256, ttl_seconds=600), metrics,
                      threshold, "small-tier", "large-tier")
    correct = graded = 0
    for prompt, kind, payload in items:
        res = gateway.complete_one(Request(prompt))
        verdict = grade(kind, payload, res.text)
        if verdict is not None:
            graded += 1
            correct += verdict
    return metrics, correct, graded, dict(metrics.tier_counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    items = build_workload()
    n = len(items)
    inner = LocalHFBackend(tier_models={"small-tier": SMALL_MODEL,
                                        "large-tier": LARGE_MODEL})
    backend = MemoBackend(inner, CACHE_PATH)

    print(f"real-model benchmark: {n} prompts "
          f"({len(FACTUAL)} factual + {len(FORMAT)} format, exactly graded; "
          f"{len(HARD)} hard, ungraded)")
    print(f"small tier: {SMALL_MODEL}")
    print(f"large tier: {LARGE_MODEL}\n")

    small_m, small_ok, n_graded = run_fixed(backend, items, "small-tier")
    large_m, large_ok, _ = run_fixed(backend, items, "large-tier")
    routed_m, routed_ok, _, tiers = run_routed(backend, items, args.threshold)

    def row(name: str, m: Metrics, ok: int, extra: str = "") -> None:
        mean_lat = sum(m.latencies_ms) / len(m.latencies_ms)
        print(f"{name:<14}{ok}/{n_graded} = {ok / n_graded:>4.0%}"
              f"{mean_lat:>12.0f}ms{m.percentile(95):>12.0f}ms"
              f"{'$' + f'{m.total_cost_usd:.4f}':>12}  {extra}")

    print(f"{'policy':<14}{'quality*':>11}{'mean lat':>13}{'p95 lat':>13}{'est cost':>12}")
    row("always_small", small_m, small_ok)
    row("always_large", large_m, large_ok)
    row("router", routed_m, routed_ok,
        f"({tiers.get('small-tier', 0)} small / {tiers.get('large-tier', 0)} large)")

    cost_saving = 1 - routed_m.total_cost_usd / large_m.total_cost_usd
    quality_delta = (routed_ok - large_ok) / n_graded
    print(f"\nrouter vs always_large: {cost_saving:.0%} lower cost, "
          f"{quality_delta:+.0%} on graded quality")
    print(f"* quality is over the {n_graded} exactly checkable prompts; "
          "the hard prompts drive cost and latency but are not graded")
    print("cost estimated with gpt-4o-mini vs gpt-4o per-token pricing; "
          "latency is measured wall-clock on this machine")


if __name__ == "__main__":
    main()
