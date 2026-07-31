"""A labeled corpus for measuring whether the router routes correctly.

The cost benchmark in :mod:`app.eval` answers "is this cheaper". It cannot
answer "is this still good", and cost alone is trivially gameable: route
everything to the small tier and save 50%. The number that makes a saving
meaningful is how often a request that genuinely needed the large model got
sent to the small one, because that is a quality regression nobody sees in the
cost graph.

This module supplies the ground truth that question needs. Every prompt is
labeled ``complex`` (genuinely needs the large tier) or ``simple`` (the small
tier answers it fine), and the corpus is deliberately split into two halves:

``easy``
    Prompts whose surface form matches how people usually write. Complex ones
    open with "Explain", "Analyze", "Compare", or contain a code fence.

``adversarial``
    The same underlying difficulty, written the way real users actually type.
    Complex requests that never use a trigger word, and simple requests that
    are full of them. A keyword router scores near-perfectly on the easy half
    and falls apart here, which is the entire reason this half exists.

A benchmark built only from the easy half measures the benchmark, not the
router. That was true of the original evaluation set, whose four complex
prompts each happened to contain a term the regex was searching for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple

Label = Literal["complex", "simple"]


@dataclass(frozen=True)
class LabeledPrompt:
    prompt: str
    label: Label
    #: ``easy`` prompts follow the conventional surface form; ``adversarial``
    #: ones carry the same difficulty in the phrasing users actually use;
    #: ``holdout`` prompts were written after the v2 weights were frozen and
    #: have never been tuned against.
    split: Literal["easy", "adversarial", "holdout"]

    @property
    def needs_large(self) -> bool:
        return self.label == "complex"


# --- easy half: complexity announced by conventional phrasing ---------------

EASY_COMPLEX = [
    "Explain the trade-offs between eventual consistency and strong consistency for our order pipeline.",
    "Analyze this function and suggest a more efficient algorithm:\n```def f(x):\n  for i in range(x):\n    for j in range(x):\n      pass\n```",
    "Design a database schema for a multi-tenant SaaS billing system and explain your reasoning.",
    "Compare REST and gRPC for our internal service mesh and recommend one with justification.",
    "Why does our write throughput collapse past 8 concurrent writers? Analyze the contention.",
    "Derive the expected cache hit rate for a Zipf-distributed key space with an LRU of size k.",
    "Explain how to migrate this schema without downtime:\n```SELECT * FROM orders JOIN users ON ...```",
    "Compare the failure modes of leader-based and leaderless replication for our use case.",
]

EASY_SIMPLE = [
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
    "Where is my order #100234?",
    "Can you look up account 55129 for me?",
    "I was charged twice on invoice #88120, please check.",
    "My tracking number 4417XZ hasn't updated in 3 days.",
    "Is there a student discount?",
    "How do I update my billing info?",
]


# --- adversarial half: same difficulty, realistic phrasing ------------------

# Genuinely hard engineering questions that never say "explain" or "analyze".
# A keyword router scores these in the 0.05-0.10 range and sends every one of
# them to the cheap tier.
ADVERSARIAL_COMPLEX = [
    "Our checkout drops 4% of orders when Redis fails over. Walk me through what's likely "
    "happening and how to make it not happen.",
    "We have three services writing to the same Postgres table and getting deadlocks under "
    "load. What should we do instead?",
    "Rewrite this so it doesn't hold the lock during the network call:\n"
    "  with lock:\n      resp = http.get(url)\n      cache[key] = resp",
    "Should we shard by tenant or by time for a metrics store with 400k writes/sec and "
    "90-day retention?",
    "The p99 on our embedding endpoint tripled after we doubled batch size. What's going on?",
    "Every retry storm takes down the same downstream service. What's the right backoff "
    "strategy here and why that one?",
    "We're seeing phantom reads in a transaction that should be serializable. Where would "
    "you start looking?",
    "Our nightly job used to take 40 minutes and now takes 6 hours. Nothing in the code "
    "changed. What happened?",
    "Is it safe to drop this index if the only query using it runs once a quarter but is "
    "on the critical path for month-end close?",
    "Two teams want the same hostname for different services and neither will budge. What "
    "are our actual options at the routing layer?",
    "Given 200 GPUs and a 30-day deadline, what's the largest model we can realistically "
    "fine-tune and what has to give?",
    "The load balancer health check passes but real traffic 502s about 1 in 50. Where's "
    "the gap?",
]

# Simple customer questions that happen to contain reasoning trigger words.
# A keyword router pushes these toward the expensive tier for no benefit.
ADVERSARIAL_SIMPLE = [
    "Why is my order late?",
    "Can you explain my bill?",
    "Compare my two most recent invoices for me.",
    "Why was I charged twice?",
    "Explain how to reset my password.",
    "Why does the app keep logging me out?",
    "Can you explain what this fee on invoice #77120 is?",
    "Why is my refund taking so long?",
    "Explain the difference between your basic and pro plans.",
    "Why did my discount code not work?",
    "Compare shipping speeds for standard and express.",
    "Why do I need to verify my email again?",
]


# --- holdout: written after the v2 weights were frozen ----------------------

# These were authored once the structural router's weights were fixed, and have
# never been tuned against. An in-sample score on a corpus you hand-fitted
# weights to is not evidence of anything; this split is what makes the
# comparison worth reading. Whatever it says, it says.
HOLDOUT_COMPLEX = [
    "Connection pool keeps exhausting during traffic spikes even though we raised "
    "max_connections to 500. Where's the leak?",
    "We want to move 12TB from one region to another with under 5 minutes of write "
    "downtime. What's the play?",
    "This query got 40x slower after we added 200 million rows. Same plan, same index.",
    "Our consumer lag grows all afternoon then catches up overnight. Is that a "
    "partitioning problem or a scaling problem?",
    "If the primary dies mid-transaction and the replica was 200ms behind, what do we "
    "tell the customer about their order?",
    "Half our pods restart every deploy and the readiness probe looks fine. What would "
    "you check?",
    "Users in Australia see stale prices for about 90 seconds after an update. Where "
    "would that come from?",
    "We're paying for 40 GPUs that sit at 12% utilization. What has to change to "
    "actually use them?",
    "Two writers both read-modify-write the same counter and we lose about 3% of "
    "increments.",
    "Is a 60 second TTL on this cache safe if the underlying data changes every 10 "
    "minutes but must never be wrong at checkout?",
]

HOLDOUT_SIMPLE = [
    "Can I get a copy of my receipt?",
    "Why hasn't my package arrived yet?",
    "How do I change my email address?",
    "Explain why my card was declined.",
    "What's the status of my refund?",
    "Do you have a phone number I can call?",
    "Compare my current plan to the next tier up.",
    "Why do I keep getting marketing emails?",
    "How do I download my invoices?",
    "Can you cancel my order?",
    "What time does support open?",
    "Why is there a hold on my account?",
]


def load_corpus(include_holdout: bool = True) -> List[LabeledPrompt]:
    """The full labeled corpus."""
    corpus: List[LabeledPrompt] = []
    for prompt in EASY_COMPLEX:
        corpus.append(LabeledPrompt(prompt, "complex", "easy"))
    for prompt in EASY_SIMPLE:
        corpus.append(LabeledPrompt(prompt, "simple", "easy"))
    for prompt in ADVERSARIAL_COMPLEX:
        corpus.append(LabeledPrompt(prompt, "complex", "adversarial"))
    for prompt in ADVERSARIAL_SIMPLE:
        corpus.append(LabeledPrompt(prompt, "simple", "adversarial"))
    if include_holdout:
        for prompt in HOLDOUT_COMPLEX:
            corpus.append(LabeledPrompt(prompt, "complex", "holdout"))
        for prompt in HOLDOUT_SIMPLE:
            corpus.append(LabeledPrompt(prompt, "simple", "holdout"))
    return corpus


def split(name: str) -> List[LabeledPrompt]:
    """One half of the corpus by name."""
    return [item for item in load_corpus() if item.split == name]


@dataclass
class RoutingScore:
    """How well a scorer separates complex requests from simple ones.

    ``misroute_rate`` is the headline. It is the fraction of genuinely complex
    requests sent to the cheap tier: the silent quality regression that a cost
    graph will never show you.
    """

    split: str
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    @property
    def total(self) -> int:
        return (
            self.true_positive + self.false_positive
            + self.true_negative + self.false_negative
        )

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 1.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        return (self.true_positive + self.true_negative) / self.total if self.total else 0.0

    @property
    def misroute_rate(self) -> float:
        """Complex requests sent to the cheap tier, over all complex requests."""
        complex_total = self.true_positive + self.false_negative
        return self.false_negative / complex_total if complex_total else 0.0

    @property
    def overspend_rate(self) -> float:
        """Simple requests sent to the expensive tier, over all simple ones."""
        simple_total = self.true_negative + self.false_positive
        return self.false_positive / simple_total if simple_total else 0.0

    def to_dict(self) -> dict:
        return {
            "split": self.split,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "misroute_rate": round(self.misroute_rate, 4),
            "overspend_rate": round(self.overspend_rate, 4),
            "counts": {
                "true_positive": self.true_positive,
                "false_positive": self.false_positive,
                "true_negative": self.true_negative,
                "false_negative": self.false_negative,
            },
        }


def score_router(
    scorer, corpus: List[LabeledPrompt], threshold: float = 0.5, split_name: str = "all"
) -> RoutingScore:
    """Run a scorer over labeled prompts and tally its routing decisions.

    ``scorer`` takes a prompt and returns a complexity score in [0, 1]; any
    router exposing that shape can be measured with the same yardstick.
    """
    result = RoutingScore(split=split_name)
    for item in corpus:
        routed_large = scorer(item.prompt) >= threshold
        if item.needs_large and routed_large:
            result.true_positive += 1
        elif item.needs_large and not routed_large:
            result.false_negative += 1
        elif not item.needs_large and routed_large:
            result.false_positive += 1
        else:
            result.true_negative += 1
    return result


def evaluate(scorer, threshold: float = 0.5) -> dict:
    """Score a router on each split separately, and overall.

    Splits are reported separately on purpose: a single blended number lets a
    keyword router hide its adversarial failures behind its easy-half wins,
    and lets a fitted router hide behind the data it was fitted on. The
    ``holdout`` row is the only one that is out of sample, so it is the one
    worth believing.
    """
    return {
        "easy": score_router(scorer, split("easy"), threshold, "easy").to_dict(),
        "adversarial": score_router(
            scorer, split("adversarial"), threshold, "adversarial"
        ).to_dict(),
        "holdout": score_router(scorer, split("holdout"), threshold, "holdout").to_dict(),
        "overall": score_router(scorer, load_corpus(), threshold, "overall").to_dict(),
    }


def misrouted_examples(
    scorer, threshold: float = 0.5, limit: int = 5
) -> List[Tuple[str, float]]:
    """Complex prompts this scorer sends to the cheap tier, worst first."""
    misses = [
        (item.prompt, scorer(item.prompt))
        for item in load_corpus()
        if item.needs_large and scorer(item.prompt) < threshold
    ]
    misses.sort(key=lambda pair: pair[1])
    return misses[:limit]
