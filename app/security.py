"""Production hardening helpers: optional API-key auth, request-id logging,
and a catch-all exception handler that never leaks stack traces to callers.

Auth is opt-in. Set API_KEY in the environment to require an `X-API-Key`
header on write endpoints; with no API_KEY set (the default, and what CI
uses) the service stays open so existing behaviour is unchanged.
"""
from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from .config import settings

logger = logging.getLogger("vllm_cost_router")


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency enforcing API-key auth when one is configured."""
    if not settings.api_key:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing API key",
        )


def install(app: FastAPI) -> None:
    """Attach request-id logging middleware and a catch-all exception handler."""

    @app.middleware("http")
    async def _log_requests(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception("unhandled error rid=%s %s %s (%.1fms)",
                             request_id, request.method, request.url.path, elapsed_ms)
            return JSONResponse(status_code=500,
                                content={"detail": "internal server error"},
                                headers={"x-request-id": request_id})
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("rid=%s %s %s -> %s (%.1fms)", request_id,
                    request.method, request.url.path, response.status_code, elapsed_ms)
        response.headers["x-request-id"] = request_id
        return response
