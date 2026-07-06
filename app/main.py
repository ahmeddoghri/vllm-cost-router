"""FastAPI gateway: cost-aware routing, caching, and batching in front of an
LLM backend. Run locally with `uvicorn app.main:app --reload`.
"""
from __future__ import annotations

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from . import security
from .backend import get_backend
from .cache import TTLCache
from .config import settings
from .gateway import Gateway
from .gateway import Request as GatewayRequest
from .metrics import Metrics
from .security import require_api_key

app = FastAPI(title="vllm-cost-router", version="0.1.0")
security.install(app)

_backend = get_backend(settings.backend, settings.openai_base_url, settings.openai_api_key)
_cache = TTLCache(max_size=settings.cache_max_size, ttl_seconds=settings.cache_ttl_seconds)
_metrics = Metrics()
_gateway = Gateway(_backend, _cache, _metrics, settings.complexity_threshold,
                    settings.small_model, settings.large_model)


class CompletionRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=settings.max_prompt_chars)
    max_tokens: int = Field(default=256, ge=1, le=8192)


class CompletionResponse(BaseModel):
    text: str
    model: str
    cost_usd: float
    latency_ms: float


class BatchCompletionRequest(BaseModel):
    requests: list[CompletionRequest] = Field(
        min_length=1, max_length=settings.max_batch_requests)


@app.get("/healthz")
def healthz() -> dict:
    """Liveness probe: the process is up."""
    return {"status": "ok", "backend": settings.backend}


@app.get("/readyz")
def readyz() -> dict:
    """Readiness probe: dependencies are wired and the service can serve."""
    return {"status": "ready", "backend": settings.backend}


@app.post("/v1/completions", response_model=CompletionResponse,
          dependencies=[Depends(require_api_key)])
def complete(req: CompletionRequest) -> CompletionResponse:
    result = _gateway.complete_one(GatewayRequest(req.prompt, req.max_tokens))
    return CompletionResponse(text=result.text, model=result.model,
                              cost_usd=result.cost_usd, latency_ms=result.latency_ms)


@app.post("/v1/batch/completions", response_model=list[CompletionResponse],
          dependencies=[Depends(require_api_key)])
def complete_batch(req: BatchCompletionRequest) -> list[CompletionResponse]:
    gw_reqs = [GatewayRequest(r.prompt, r.max_tokens) for r in req.requests]
    results = _gateway.complete_batch(gw_reqs)
    return [
        CompletionResponse(text=r.text, model=r.model, cost_usd=r.cost_usd, latency_ms=r.latency_ms)
        for r in results
    ]


@app.get("/metrics")
def metrics() -> dict:
    return _metrics.summary()
