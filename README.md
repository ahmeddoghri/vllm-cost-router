# vllm-cost-router

![CI](https://github.com/ahmeddoghri/vllm-cost-router/actions/workflows/ci.yml/badge.svg)
![tests](https://img.shields.io/badge/tests-28%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-black)

> **73% lower cost and 73% lower p95 latency** on a 500-request mixed
> workload, same model quality, zero API keys to reproduce it:
> `python -m app.eval`.

Sending "what are your business hours?" to your biggest, most expensive
model is like renting a moving truck to pick up one bag of groceries.
Technically it works. Your cloud bill will have questions. vllm-cost-router
sits in front of your LLM serving layer and decides, per request, which
model tier actually needs to handle this, whether the answer is already
cached, and whether this can be batched with what's already in flight.
FastAPI service, not a notebook you run once and forget about.

I built this after spending a chunk of last year on exactly this problem:
getting inference cost and p95 latency down on a SageMaker/vLLM endpoint
serving tens of thousands of requests a day. The mechanism there was
model distillation plus batched serving. The routing/caching/batching
stack here is the same idea distilled into something you can actually
run and poke at. Not the same code (that was proprietary), same shape of
solution, rebuilt from scratch as something shareable.

## Why this exists

Most teams solve "inference is expensive" by either always calling the
big model and eating the cost, or always calling the small model and
eating the quality hit on the requests that actually needed the big one.
Neither is right. The fix is boring and it works: **route on complexity,
cache what repeats, batch what's concurrent.** None of those three ideas
is novel by itself. Stacking all three in front of every request is what
actually moves the number.

## The result, on a synthetic but disclosed workload

```bash
python -m app.eval
```
```
workload: 500 requests (25% genuinely complex, 25% repeated FAQs,
50% unique-but-simple templated requests)

policy          total cost     p95 latency    cache hit rate
naive              $3.0000          220.0ms                --
router             $0.8190           60.0ms              46%

cost reduction: 73%   p95 latency reduction: 73%
```

I want to be upfront about how this number is built, because a
suspiciously round "94% savings!" is a red flag and I'd rather you trust
the methodology than take the headline at face value. The workload mix
and the 2x cost/latency gap between tiers are both defined in
`app/eval.py` and `app/backend.py`, in plain sight. Real production
numbers depend entirely on your actual traffic mix and your actual tier
pricing. Plug those into the same harness and you'll get your own
number, not mine.

## Same router, real models

The mock benchmark above proves the mechanism. This one proves it against
actual model weights. vLLM itself will not run on a Mac (no CUDA), so the
harness loads two real HuggingFace chat models locally, Qwen2.5-0.5B-Instruct
as the small tier and Qwen2.5-1.5B-Instruct as the large one, and pushes a
graded workload through the exact same Gateway. Real forward passes, real
wall-clock latency, and a real quality gap. Nothing above the backend layer
changes, which is the entire argument for having a backend layer.

```bash
pip install -r requirements-real.txt
python -m app.realbench
```
```
policy           quality*     mean lat      p95 lat    est cost
always_small  19/24 =  79%        1797ms        4502ms     $0.0016
always_large  24/24 = 100%        2421ms        6721ms     $0.0248
router        19/24 =  79%        2549ms        6721ms     $0.0210  (24 small / 12 large)

router vs always_large: 15% lower cost
```

Two honest findings, both less flattering than the mock and both more
useful:

1. **The savings shrink when cost is per-token.** The mock's 73% assumed a
   per-call price gap. With real per-token pricing (the gpt-4o-mini vs
   gpt-4o published rates), the long analytical answers that correctly stay
   on the large tier dominate total spend, so routing the short stuff down
   only trims 15%. Routing helps most when your traffic skews short and
   simple; measure your mix before promising your CFO a number.

2. **The small tier is not free.** Qwen2.5-0.5B got 79% on prompts with
   exactly checkable answers. Its misses are not subtle: it answered 90 for
   45 plus 55, and Spanish for the language of Brazil. The 1.5B went 24 for
   24. If your tiers are this far apart in capability, the router's cost
   win comes straight out of your quality budget. Pick a small tier you
   would actually let talk to a user.

Quality is graded only where grading is honest: factual containment and
strict format checks (24 prompts). The 12 open-ended prompts drive routing,
cost, and latency but are not scored, because keyword-matching an essay
mostly measures truncation. An earlier version of this harness did exactly
that and got cut. Generations are cached to `realbench_cache.json`
(gitignored), so re-runs are instant; delete it to re-measure.

## Install & run

```bash
git clone https://github.com/ahmeddoghri/vllm-cost-router
cd vllm-cost-router
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

Or with Docker:

```bash
docker build -t vllm-cost-router .
docker run -p 8000:8000 vllm-cost-router
```

Or docker-compose:

```bash
docker compose up
```

Try it:

```bash
curl -X POST localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What are your business hours?"}'

curl localhost:8000/metrics
```

## How it decides

```
request
  ├─ cache lookup (prompt + model + max_tokens) ── hit? return immediately
  ├─ complexity_score(prompt)  -- length, code markers, reasoning language
  │     ├─ below threshold  -> small tier
  │     └─ above threshold  -> large tier
  └─ (batch endpoint only) group cache-misses by tier, dispatch each
     tier as one backend call so fixed per-batch overhead amortizes
     across every request in it -- the actual mechanism behind vLLM's
     continuous batching win
```

## Point it at a real backend

Defaults to a deterministic in-process mock so CI and local dev need
zero API keys. Point it at a real vLLM server or anything
OpenAI-compatible:

```bash
export BACKEND=openai
export OPENAI_BASE_URL=http://your-vllm-host:8000/v1
export OPENAI_API_KEY=your-key
export SMALL_MODEL=your-distilled-model-name
export LARGE_MODEL=your-full-model-name
uvicorn app.main:app
```

## API

| Endpoint | What it does |
|---|---|
| `POST /v1/completions` | Single request: cache, route, complete |
| `POST /v1/batch/completions` | Batch of requests, grouped by tier and dispatched together |
| `GET /metrics` | Running cost, p50/p95 latency, cache hit rate, tier distribution |
| `GET /healthz` | Liveness probe |
| `GET /readyz` | Readiness probe |

## Production configuration

All settings have safe local defaults; override via environment variables.

| Variable | Default | Purpose |
|---|---|---|
| `API_KEY` | *(empty)* | When set, write endpoints require a matching `X-API-Key` header. Empty leaves the service open. |
| `MAX_PROMPT_CHARS` | `100000` | Rejects (422) prompts larger than this to bound memory. |
| `MAX_BATCH_REQUESTS` | `128` | Caps how many requests one batch call may carry. |

Every response carries an `X-Request-ID` header (echoed from the request
if provided, otherwise generated), requests are logged with method, path,
status, and latency, and unhandled errors return a structured `500`
without leaking stack traces.

## Tests

```bash
pip install -r requirements-dev.txt && pytest -q      # 28 passing
```

## More in this series

Nine small, dependency-light, benchmarked tools for LLM/ML infrastructure. Each one reproduces its headline number locally with no API keys:

[agentmem](https://github.com/ahmeddoghri/agentmem) · [rubricagent](https://github.com/ahmeddoghri/rubricagent) · [clarifyrag](https://github.com/ahmeddoghri/clarifyrag) · [churnfm](https://github.com/ahmeddoghri/churnfm) · [citebench](https://github.com/ahmeddoghri/citebench) · [guardrail-gate](https://github.com/ahmeddoghri/guardrail-gate) · [tablextract](https://github.com/ahmeddoghri/tablextract) · [taggate](https://github.com/ahmeddoghri/taggate)

## License

MIT © Ahmed Doghri
