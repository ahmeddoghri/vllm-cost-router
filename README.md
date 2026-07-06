# vllm-cost-router

![CI](https://github.com/ahmeddoghri/vllm-cost-router/actions/workflows/ci.yml/badge.svg)
![tests](https://img.shields.io/badge/tests-19%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-black)

> **73% lower cost and 73% lower p95 latency** on a 500-request mixed workload —
> same model quality, zero API keys to reproduce it: `python -m app.eval`.

A cost-and-latency-aware gateway that sits in front of your LLM serving layer
and decides, per request: which model tier actually needs to handle this, is
the answer already cached, and can this be batched with what's already in
flight. FastAPI service, not a notebook.

I built this after spending a chunk of last year on exactly this problem —
getting inference cost and p95 latency down on a SageMaker/vLLM endpoint
serving tens of thousands of requests a day. The mechanism there was model
distillation plus batched serving; the routing/caching/batching stack here is
the same idea distilled into something you can actually run and poke at.
It's not the same code (that was proprietary), but it's the same shape of
solution, rebuilt from scratch as something shareable.

## Why this exists

Most teams solve "inference is expensive" by either (a) always calling the
big model and eating the cost, or (b) always calling the small model and
eating the quality hit on the requests that actually needed the big one.
Neither is right. The fix is boring and it works: **route on complexity,
cache what repeats, batch what's concurrent.** None of those three ideas is
novel on its own — stacking all three in front of every request is what
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

I want to be upfront about how this number is constructed, because a
suspiciously round "94% savings!" is a red flag and I'd rather you trust the
methodology than take the headline at face value: the workload mix and the
2x cost/latency gap between tiers are both defined in `app/eval.py` and
`app/backend.py`, in plain sight. Real production numbers depend entirely on
your actual traffic mix and your actual tier pricing — plug those into the
same harness and you'll get your own number, not mine.

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

Defaults to a deterministic in-process mock so CI and local dev need zero
API keys. Point it at a real vLLM server (or anything OpenAI-compatible):

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

Every response carries an `X-Request-ID` header (echoed from the request if
provided, otherwise generated) and requests are logged with method, path,
status, and latency. Unhandled errors return a structured `500` without
leaking stack traces.

## Tests

```bash
pip install -r requirements-dev.txt && pytest -q      # 19 passing
```

## More in this series

Nine small, dependency-light, benchmarked tools for LLM/ML infrastructure — each reproduces its headline number locally with no API keys:

[agentmem](https://github.com/ahmeddoghri/agentmem) · [rubricagent](https://github.com/ahmeddoghri/rubricagent) · [clarifyrag](https://github.com/ahmeddoghri/clarifyrag) · [churnfm](https://github.com/ahmeddoghri/churnfm) · [citebench](https://github.com/ahmeddoghri/citebench) · [guardrail-gate](https://github.com/ahmeddoghri/guardrail-gate) · [tablextract](https://github.com/ahmeddoghri/tablextract) · [taggate](https://github.com/ahmeddoghri/taggate)

## License

MIT © Ahmed Doghri
