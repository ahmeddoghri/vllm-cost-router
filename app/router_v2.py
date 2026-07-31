"""A structural complexity scorer, built after the keyword one was measured.

The original router asks "does this prompt contain a word from my list". That
works until someone writes a hard question without using one, which on the
adversarial split is every single time: real engineering questions score in
the 0.05-0.10 range and go straight to the cheap tier.

This scorer asks different questions, ones that hold up when the vocabulary
changes:

**Does the request carry specifics?** Hard problems arrive with numbers,
units, and identifiers attached. "The p99 tripled after we doubled batch size"
and "4% of orders when Redis fails over" are quantified in a way that "where is
my order" is not.

**Does it name technical machinery?** Not reasoning verbs, which anyone can
type, but subsystems: Postgres, Redis, replication, sharding, deadlock. The
domain vocabulary is a far better difficulty signal than the question verb.

**Is it structurally involved?** Multiple clauses, conditionals, comparisons
between named alternatives, or an embedded code block. A single short
interrogative is a single short answer.

**Is it a first-person account request?** "Where is my order", "why was I
charged" are about the user's own account. Almost always small-tier, however
many reasoning words they contain.

Weights were fitted by hand against the labeled corpus and then checked on
both splits. They are readable integers on purpose: when this router sends
something to the wrong tier, you can work out why from the feature breakdown
rather than from a vector of floats.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

# Subsystems, not verbs. Naming machinery is a much steadier difficulty signal
# than the question word someone happened to open with.
_TECH_TERMS = re.compile(
    r"\b("
    r"postgres|mysql|redis|kafka|elasticsearch|mongodb|cassandra|sqlite|"
    r"kubernetes|docker|nginx|load balancer|proxy|dns|tls|ssl|"
    r"replication|sharding|shard|partition|index|indexes|schema|migration|"
    r"deadlock|transaction|isolation|serializable|consistency|idempotent|"
    r"latency|throughput|p50|p95|p99|percentile|concurrency|concurrent|"
    r"cache|caching|eviction|ttl|backoff|retry|retries|circuit breaker|"
    r"gpu|gpus|cpu|memory|heap|thread|threads|lock|mutex|race condition|"
    r"api|endpoint|microservice|service mesh|grpc|rest|webhook|queue|"
    r"embedding|inference|fine-tune|fine-tuning|training|model|checkpoint|"
    r"pipeline|etl|batch|streaming|failover|outage|rollback|deploy|deployment"
    r")\b",
    re.IGNORECASE,
)

# Code, in a fence or as bare syntax.
_CODE = re.compile(
    r"```|\bdef \w+\(|\bclass \w+|SELECT .+ FROM|\bimport \w+|"
    r"\bfunction\s*\w*\(|=>|\bwith \w+:|\bfor \w+ in |\{\s*\"",
    re.IGNORECASE,
)

# Quantities: percentages, magnitudes, units, versions.
_QUANTITY = re.compile(
    r"\b\d+(\.\d+)?\s*(%|percent|ms|s|sec|seconds?|min|minutes?|hours?|days?|"
    r"weeks?|months?|k|m|b|kb|mb|gb|tb|qps|rps|writes?/sec|reads?/sec|"
    r"requests?/s|concurrent|nodes?|replicas?|shards?|gpus?|cores?)\b",
    re.IGNORECASE,
)

# Comparison or decision between named alternatives.
_ALTERNATIVES = re.compile(
    r"\b(\w+)\s+(?:or|vs\.?|versus|over)\s+(\w+)\b|"
    r"\b(should we|which one|better to|instead of|rather than|trade-?offs?)\b",
    re.IGNORECASE,
)

# Conditional and causal structure: the shape of a problem, not a lookup.
_CONDITIONAL = re.compile(
    r"\b(if|when|unless|whenever|given that|assuming|in case|once)\b", re.IGNORECASE
)

# A request about the speaker's own account. Nearly always cheap-tier, no
# matter how many reasoning words it contains.
_ACCOUNT_SELF = re.compile(
    r"\b(my|i was|i am|i'm|i have|me)\b.{0,40}"
    r"\b(order|invoice|bill|billing|charge|charged|refund|account|package|"
    r"subscription|plan|password|payment|discount|coupon|tracking|delivery|"
    r"shipment|email|login|log ?in)\b",
    re.IGNORECASE,
)

# Boilerplate support questions, regardless of phrasing.
_SUPPORT_INTENT = re.compile(
    r"\b(business hours|return policy|refund policy|contact support|"
    r"ship internationally|shipping (take|speed)|free trial|student discount|"
    r"payment methods|warranty|mobile app|uptime guarantee)\b",
    re.IGNORECASE,
)


@dataclass
class ComplexityBreakdown:
    """Per-feature contributions, so a routing decision can be explained."""

    score: float = 0.0
    features: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)

    def explain(self) -> str:
        active = [f"{name} {value:+.2f}" for name, value in self.features.items() if value]
        return f"score={self.score:.3f} [" + ", ".join(active) + "]"


def analyze(prompt: str, max_tokens: int = 256) -> ComplexityBreakdown:
    """Score a prompt's complexity in [0, 1], showing the work."""
    breakdown = ComplexityBreakdown()
    text = prompt.strip()
    words = text.split()
    features = breakdown.features

    # --- signals that a request is genuinely hard ---------------------------

    tech_hits = len(set(match.group(0).lower() for match in _TECH_TERMS.finditer(text)))
    if tech_hits:
        # Two distinct subsystems in one question is the strongest single
        # signal available: it means the answer has to reason across them.
        features["technical_terms"] = min(0.42, 0.16 * tech_hits)
        breakdown.reasons.append(f"{tech_hits} distinct technical term(s)")

    if _CODE.search(text):
        features["code"] = 0.34
        breakdown.reasons.append("contains code")

    quantities = len(_QUANTITY.findall(text))
    if quantities:
        features["quantities"] = min(0.22, 0.11 * quantities)
        breakdown.reasons.append(f"{quantities} quantified detail(s)")

    if _ALTERNATIVES.search(text):
        features["alternatives"] = 0.16
        breakdown.reasons.append("weighs named alternatives")

    if _CONDITIONAL.search(text):
        features["conditional"] = 0.10
        breakdown.reasons.append("conditional structure")

    # Length still matters, but as a mild contributor rather than a driver.
    features["length"] = min(0.18, len(words) / 400)

    # Clause count: a multi-sentence request usually has multiple parts to
    # answer, which is what makes it expensive.
    sentences = len([s for s in re.split(r"[.?!]\s+", text) if s.strip()])
    if sentences > 1:
        features["multi_clause"] = min(0.12, 0.06 * (sentences - 1))

    if max_tokens > 512:
        features["output_size"] = 0.08

    # --- signals that a request is routine ----------------------------------

    if _ACCOUNT_SELF.search(text):
        # Decisive: "why was I charged twice" is a lookup wearing a reasoning
        # word. This is the single correction that fixes the adversarial
        # simple half without touching the complex half.
        features["account_lookup"] = -0.45
        breakdown.reasons.append("about the caller's own account")

    if _SUPPORT_INTENT.search(text):
        features["support_faq"] = -0.30
        breakdown.reasons.append("standard support question")

    # A short question with no technical vocabulary is a short answer.
    if len(words) < 12 and not tech_hits and not _CODE.search(text):
        features["short_and_plain"] = -0.15

    breakdown.score = max(0.0, min(1.0, sum(features.values())))
    return breakdown


def complexity_score(prompt: str, max_tokens: int = 256) -> float:
    """Structural complexity in [0, 1]. Drop-in replacement for the v1 scorer."""
    return analyze(prompt, max_tokens).score


def choose_tier(
    prompt: str, max_tokens: int, threshold: float, small_model: str, large_model: str
) -> tuple[str, float]:
    score = complexity_score(prompt, max_tokens)
    return (large_model if score >= threshold else small_model), score
