"""A small TTL + LRU in-memory cache keyed on (prompt, model, max_tokens).

Repeated or near-duplicate requests (the same FAQ asked by different users,
retried requests, etc.) are extremely common in production traffic and cost
nothing to skip once seen. Swap this for Redis behind the same interface if
you need cross-process sharing.
"""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Optional


def cache_key(prompt: str, model: str, max_tokens: int) -> str:
    raw = f"{model}:{max_tokens}:{prompt}"
    return hashlib.sha256(raw.encode()).hexdigest()


class TTLCache:
    def __init__(self, max_size: int = 2048, ttl_seconds: float = 300.0, clock=time.time) -> None:
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._clock = clock
        self._store: "OrderedDict[str, tuple[float, object]]" = OrderedDict()

    def get(self, key: str) -> Optional[object]:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if self._clock() >= expires_at:
            del self._store[key]
            return None
        self._store.move_to_end(key)  # LRU touch
        return value

    def set(self, key: str, value: object) -> None:
        self._store[key] = (self._clock() + self.ttl, value)
        self._store.move_to_end(key)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def __len__(self) -> int:
        return len(self._store)
