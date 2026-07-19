"""A real local-model backend for the same gateway.

vLLM itself won't run on a Mac (no CUDA), so this backend loads two real
HuggingFace chat models locally and serves them behind the exact same
``LLMBackend`` protocol the mock uses. Real forward passes, real wall-clock
latency, real quality gaps between tiers. The router, cache, and batching
code above this layer are untouched, which is the whole point: swap the
backend, keep the economics.

Requires the extras in requirements-real.txt (torch + transformers). Never
imported by the service or the tests unless explicitly requested, so the
core stays dependency-free.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .backend import CompletionResult

# Cost model for the estimate column in the real benchmark: a real published
# small/large price pair (OpenAI gpt-4o-mini vs gpt-4o, USD per 1M tokens,
# input/output, as listed on openai.com/api/pricing at time of writing).
# The local Qwen models stand in for the tiers; the pricing pair makes the
# cost column concrete instead of inventing per-call numbers.
PRICING_PER_MTOK = {
    "small": {"input": 0.15, "output": 0.60},
    "large": {"input": 2.50, "output": 10.00},
}


@dataclass
class LocalHFBackend:
    """Runs real HuggingFace chat models as the small/large tiers.

    ``tier_models`` maps the tier names the router emits (e.g. "small-tier")
    to HF model ids. Models are loaded lazily on first use and kept resident.
    Prompts inside a batch are processed sequentially: local HF generation on
    a laptop has no continuous batching, and pretending otherwise would fake
    the latency numbers.
    """

    tier_models: dict[str, str]
    tier_pricing: dict[str, str] = field(
        default_factory=lambda: {"small-tier": "small", "large-tier": "large"}
    )
    max_new_tokens: int = 160
    _loaded: dict = field(default_factory=dict, repr=False)

    def _load(self, model_id: str):
        if model_id in self._loaded:
            return self._loaded[model_id]
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16)
        model = model.to(device).eval()
        self._loaded[model_id] = (tok, model, device)
        return self._loaded[model_id]

    def _generate(self, prompt: str, model_id: str) -> tuple[str, float, int, int]:
        """Returns (text, latency_ms, input_tokens, output_tokens)."""
        import torch

        tok, model, device = self._load(model_id)
        messages = [{"role": "user", "content": prompt}]
        text_prompt = tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = tok(text_prompt, return_tensors="pt").to(device)
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        latency_ms = (time.perf_counter() - start) * 1000
        n_in = inputs["input_ids"].shape[1]
        new_tokens = out[0][n_in:]
        text = tok.decode(new_tokens, skip_special_tokens=True)
        return text, latency_ms, n_in, len(new_tokens)

    def cost_usd(self, tier: str, n_in: int, n_out: int) -> float:
        prices = PRICING_PER_MTOK[self.tier_pricing.get(tier, "large")]
        return (n_in * prices["input"] + n_out * prices["output"]) / 1_000_000

    def complete_batch(self, prompts: list[str], model: str) -> list[CompletionResult]:
        model_id = self.tier_models.get(model)
        if model_id is None:
            raise KeyError(f"no local model configured for tier {model!r}")
        results = []
        for p in prompts:
            text, latency_ms, n_in, n_out = self._generate(p, model_id)
            results.append(
                CompletionResult(
                    text=text,
                    model=model,
                    cost_usd=self.cost_usd(model, n_in, n_out),
                    latency_ms=latency_ms,
                )
            )
        return results
