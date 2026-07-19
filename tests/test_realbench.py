"""Tests for the real-model benchmark plumbing.

Everything here runs without torch/transformers installed: grading logic,
workload construction, routing behavior on the graded workload, cost math,
and the generation memo. The only thing that needs the heavy extras is the
actual model forward pass, and that is exercised by `python -m app.realbench`
locally, not in CI.
"""
import json

from app.backend import CompletionResult
from app.realbackend import PRICING_PER_MTOK, LocalHFBackend
from app.realbench import (
    CACHE_PATH,
    FACTUAL,
    FORMAT,
    HARD,
    MemoBackend,
    build_workload,
    check_factual,
    check_format,
    grade,
)
from app.router import choose_tier


def test_workload_is_complete_and_typed():
    items = build_workload()
    assert len(items) == len(FACTUAL) + len(FORMAT) + len(HARD)
    kinds = {k for _, k, _ in items}
    assert kinds == {"factual", "format", "hard"}


def test_router_splits_workload_as_designed():
    # easy prompts avoid reasoning markers, hard prompts use them; the router
    # should send every factual/format item small and every hard item large
    for prompt, kind, _ in build_workload():
        tier, _score = choose_tier(prompt, 256, 0.5, "small-tier", "large-tier")
        expected = "large-tier" if kind == "hard" else "small-tier"
        assert tier == expected, f"{kind} prompt routed to {tier}: {prompt[:50]}"


def test_factual_grading_is_containment():
    assert check_factual("The capital is Paris.", "paris")
    assert not check_factual("The capital is Lyon.", "paris")


def test_format_grading_json_and_lists():
    assert check_format('{"name": "Bob", "age": 30}', "json_name_age")
    assert check_format('```json\n{"name": "Bob", "age": 30}\n```', "json_name_age")
    assert not check_format('{"name": "Alice", "age": 30}', "json_name_age")
    assert check_format("red, blue, yellow", "comma_list_3")
    assert not check_format("red, blue", "comma_list_3")
    assert check_format("- apple\n- pear\n- plum", "dash_lines_3")
    assert not check_format("apple\npear\nplum", "dash_lines_3")
    assert check_format("Yes.", "yes_no")
    assert not check_format("Definitely yes", "yes_no")


def test_grade_dispatches_by_kind():
    assert grade("factual", "paris", "Paris") is True
    assert grade("format", "yes_no", "yes") is True
    # open-ended prompts are deliberately ungraded: keyword checks on essay
    # answers measure truncation, not quality
    assert grade("hard", None, "a long correct essay") is None


def test_cost_model_uses_published_price_pair():
    backend = LocalHFBackend(tier_models={})
    small = backend.cost_usd("small-tier", 1000, 1000)
    large = backend.cost_usd("large-tier", 1000, 1000)
    assert small == (1000 * 0.15 + 1000 * 0.60) / 1_000_000
    assert large > small
    assert PRICING_PER_MTOK["large"]["output"] == 10.00


def test_memo_backend_replays_without_touching_inner(tmp_path):
    class Recorder:
        calls = 0

        def complete_batch(self, prompts, model):
            Recorder.calls += 1
            return [CompletionResult(f"resp:{p}", model, 0.001, 42.0) for p in prompts]

    path = tmp_path / "memo.json"
    memo = MemoBackend(Recorder(), path)
    first = memo.complete_batch(["hello"], "small-tier")
    again = memo.complete_batch(["hello"], "small-tier")
    assert Recorder.calls == 1
    assert first[0].text == again[0].text == "resp:hello"
    assert again[0].latency_ms == 42.0
    # survives a reload from disk
    memo2 = MemoBackend(Recorder(), path)
    replay = memo2.complete_batch(["hello"], "small-tier")
    assert Recorder.calls == 1
    assert replay[0].text == "resp:hello"


def test_cache_file_is_gitignored():
    gitignore = (CACHE_PATH.parent / ".gitignore").read_text()
    assert "realbench_cache.json" in gitignore


def test_memo_persists_valid_json(tmp_path):
    class Inner:
        def complete_batch(self, prompts, model):
            return [CompletionResult("x", model, 0.0, 1.0) for _ in prompts]

    path = tmp_path / "memo.json"
    MemoBackend(Inner(), path).complete_batch(["a", "b"], "large-tier")
    data = json.loads(path.read_text())
    assert len(data) == 2
