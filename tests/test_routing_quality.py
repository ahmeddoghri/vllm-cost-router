"""Tests for routing quality: does the router send hard requests to the big model?"""

from __future__ import annotations

import pytest

from app import routebench as rb
from app.eval import generate_workload, run_naive, run_router
from app.gateway import Gateway, Request, get_router
from app.router import complexity_score as v1_score
from app.router_v2 import analyze, complexity_score as v2_score

V1_THRESHOLD = 0.50
V2_THRESHOLD = 0.10


def v1(prompt: str) -> float:
    return v1_score(prompt, 256)


def v2(prompt: str) -> float:
    return v2_score(prompt, 256)


# --- the corpus -------------------------------------------------------------

def test_corpus_has_all_splits():
    splits = {item.split for item in rb.load_corpus()}
    assert splits == {"easy", "adversarial", "holdout"}


def test_corpus_is_balanced_enough_to_be_meaningful():
    corpus = rb.load_corpus()
    complex_count = sum(1 for item in corpus if item.needs_large)
    assert 0.25 < complex_count / len(corpus) < 0.75


def test_holdout_can_be_excluded():
    assert {item.split for item in rb.load_corpus(include_holdout=False)} == {
        "easy", "adversarial"
    }


def test_no_prompt_appears_in_two_splits():
    """A leaked prompt would make the holdout meaningless."""
    corpus = rb.load_corpus()
    by_prompt: dict[str, set[str]] = {}
    for item in corpus:
        by_prompt.setdefault(item.prompt, set()).add(item.split)
    assert not [p for p, splits in by_prompt.items() if len(splits) > 1]


# --- the finding this whole module exists to pin down ------------------------

def test_keyword_router_is_perfect_on_conventional_phrasing():
    """v1 looks flawless, which is why nobody noticed it was broken."""
    score = rb.score_router(v1, rb.split("easy"), V1_THRESHOLD)
    assert score.accuracy == 1.0
    assert score.misroute_rate == 0.0


def test_keyword_router_collapses_on_realistic_phrasing():
    """The same router, same difficulty, phrased the way users type."""
    score = rb.score_router(v1, rb.split("adversarial"), V1_THRESHOLD)
    assert score.misroute_rate > 0.85


def test_keyword_router_misroutes_everything_out_of_sample():
    score = rb.score_router(v1, rb.split("holdout"), V1_THRESHOLD)
    assert score.misroute_rate == 1.0


def test_structural_router_holds_up_on_realistic_phrasing():
    score = rb.score_router(v2, rb.split("adversarial"), V2_THRESHOLD)
    assert score.misroute_rate == 0.0


def test_structural_router_holds_up_out_of_sample():
    """The only row that is out of sample, so the only one worth believing."""
    score = rb.score_router(v2, rb.split("holdout"), V2_THRESHOLD)
    assert score.accuracy > 0.9
    assert score.misroute_rate <= 0.1


def test_structural_router_beats_keyword_router_overall():
    corpus = rb.load_corpus()
    assert (
        rb.score_router(v2, corpus, V2_THRESHOLD).accuracy
        > rb.score_router(v1, corpus, V1_THRESHOLD).accuracy
    )


def test_neither_router_overspends():
    """Routing simple requests to the big model is the other failure mode."""
    corpus = rb.load_corpus()
    assert rb.score_router(v1, corpus, V1_THRESHOLD).overspend_rate == 0.0
    assert rb.score_router(v2, corpus, V2_THRESHOLD).overspend_rate == 0.0


# --- metric arithmetic ------------------------------------------------------

def test_misroute_rate_is_complex_sent_cheap():
    score = rb.RoutingScore("t", true_positive=3, false_negative=1)
    assert score.misroute_rate == 0.25


def test_overspend_rate_is_simple_sent_expensive():
    score = rb.RoutingScore("t", true_negative=3, false_positive=1)
    assert score.overspend_rate == 0.25


def test_always_small_misroutes_everything():
    score = rb.score_router(lambda _: 0.0, rb.load_corpus(), 0.5)
    assert score.misroute_rate == 1.0
    assert score.overspend_rate == 0.0


def test_always_large_overspends_on_everything():
    score = rb.score_router(lambda _: 1.0, rb.load_corpus(), 0.5)
    assert score.misroute_rate == 0.0
    assert score.overspend_rate == 1.0


def test_empty_corpus_does_not_divide_by_zero():
    score = rb.score_router(v2, [], V2_THRESHOLD)
    assert score.misroute_rate == 0.0
    assert score.accuracy == 0.0


# --- the structural scorer --------------------------------------------------

def test_scores_stay_in_range():
    for item in rb.load_corpus():
        assert 0.0 <= v2(item.prompt) <= 1.0


def test_complex_and_simple_are_separable():
    """No threshold can rescue a scorer whose classes overlap."""
    corpus = rb.load_corpus()
    hardest_simple = max(v2(i.prompt) for i in corpus if not i.needs_large)
    easiest_complex = min(v2(i.prompt) for i in corpus if i.needs_large)
    assert easiest_complex > hardest_simple


def test_account_questions_stay_cheap_despite_reasoning_words():
    """The correction that fixes 'why was I charged twice' without breaking real work."""
    for prompt in ("Why was I charged twice?", "Can you explain my bill?",
                   "Compare my two most recent invoices."):
        assert v2(prompt) < V2_THRESHOLD


def test_technical_questions_route_large_without_trigger_words():
    prompt = ("Our checkout drops 4% of orders when Redis fails over. "
              "Walk me through what's likely happening.")
    assert v2(prompt) >= V2_THRESHOLD
    assert v1(prompt) < V1_THRESHOLD  # the regression being fixed


def test_code_still_routes_large():
    assert v2("Rewrite this:\n```def f(x):\n  return x\n```") >= V2_THRESHOLD


def test_decisions_are_explainable():
    """A routing decision you cannot explain is one you cannot debug."""
    breakdown = analyze("Should we shard by tenant or by time for 400k writes/sec?")
    assert breakdown.reasons
    assert "technical_terms" in breakdown.features
    assert "score=" in breakdown.explain()


def test_larger_max_tokens_raises_the_score():
    prompt = "Our Postgres replication lag spikes every hour."
    assert v2_score(prompt, 1024) >= v2_score(prompt, 128)


def test_scoring_is_deterministic():
    prompt = "Why does our write throughput collapse past 8 concurrent writers?"
    assert v2(prompt) == v2(prompt)


def test_empty_prompt_scores_zero():
    assert v2("") == 0.0


# --- gateway wiring ---------------------------------------------------------

def test_gateway_defaults_to_the_structural_router():
    """v1's failure is silent, so it must never be reached by accident."""
    from app.router_v2 import choose_tier as v2_choose

    assert get_router("v2") is v2_choose
    assert get_router("nonsense") is v2_choose


def test_gateway_can_still_select_v1():
    from app.router import choose_tier as v1_choose

    assert get_router("v1") is v1_choose


def test_gateway_routes_a_hard_request_to_the_large_tier():
    from app.backend import MockBackend
    from app.cache import TTLCache
    from app.metrics import Metrics

    metrics = Metrics()
    gateway = Gateway(MockBackend(), TTLCache(), metrics, V2_THRESHOLD,
                      "small-tier", "large-tier", "v2")
    gateway.complete_one(
        Request("Our Postgres deadlocks under load with three concurrent writers.")
    )
    assert metrics.tier_counts.get("large-tier") == 1


def test_gateway_routes_a_simple_request_to_the_small_tier():
    from app.backend import MockBackend
    from app.cache import TTLCache
    from app.metrics import Metrics

    metrics = Metrics()
    gateway = Gateway(MockBackend(), TTLCache(), metrics, V2_THRESHOLD,
                      "small-tier", "large-tier", "v2")
    gateway.complete_one(Request("Where is my order?"))
    assert metrics.tier_counts.get("small-tier") == 1


# --- the cost benchmark's blind spot ----------------------------------------

def test_cost_benchmark_cannot_tell_the_routers_apart_on_easy_phrasing():
    """The reason the original benchmark never caught this."""
    prompts = generate_workload(200, seed=0)
    v1_cost = run_router(prompts, threshold=V1_THRESHOLD, router_version="v1").total_cost_usd
    v2_cost = run_router(prompts, threshold=V2_THRESHOLD, router_version="v2").total_cost_usd
    assert v1_cost == pytest.approx(v2_cost)


def test_broken_router_looks_cheaper_on_realistic_traffic():
    """v1 wins on cost precisely because it is answering hard questions cheaply."""
    prompts = generate_workload(200, seed=0, realistic=True)
    v1_cost = run_router(prompts, threshold=V1_THRESHOLD, router_version="v1").total_cost_usd
    v2_cost = run_router(prompts, threshold=V2_THRESHOLD, router_version="v2").total_cost_usd
    assert v1_cost < v2_cost


def test_both_routers_still_beat_naive_serving():
    prompts = generate_workload(200, seed=0, realistic=True)
    naive = run_naive(prompts).total_cost_usd
    for version, threshold in (("v1", V1_THRESHOLD), ("v2", V2_THRESHOLD)):
        assert run_router(prompts, threshold=threshold,
                          router_version=version).total_cost_usd < naive * 0.5


def test_realistic_workload_uses_realistic_phrasing():
    prompts = set(generate_workload(300, seed=1, realistic=True))
    assert prompts & set(rb.ADVERSARIAL_COMPLEX)


# --- the report -------------------------------------------------------------

def test_report_covers_every_router_and_split():
    from app.routereport import build_report

    report = build_report()
    assert set(report["routers"]) == {"keyword (v1)", "structural (v2)"}
    for data in report["routers"].values():
        assert set(data["splits"]) == {"easy", "adversarial", "holdout", "overall"}


def test_report_is_reproducible():
    from app.routereport import build_report

    assert build_report() == build_report()


def test_report_formats_readably():
    from app.routereport import build_report, format_report

    text = format_report(build_report())
    assert "misroute" in text
    assert "holdout" in text


def test_misrouted_examples_lists_only_complex_prompts():
    misses = rb.misrouted_examples(v1, V1_THRESHOLD, limit=5)
    complex_prompts = {i.prompt for i in rb.load_corpus() if i.needs_large}
    assert misses
    assert all(prompt in complex_prompts for prompt, _ in misses)
