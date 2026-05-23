"""In-memory TTL response cache.

Caches analysis responses keyed on a SHA-256 hash of
``(task_type, financial_text)`` to avoid redundant inference for
identical requests.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CacheEntry:
    """An immutable snapshot of a cached inference result."""

    response: str
    model: str
    confidence: float
    routing_decision: str
    tokens_used: int
    estimated_cost_usd: float
    created_at: float


class ResponseCache:
    """Thread-safe LRU cache with per-entry TTL expiration.

    Args:
        max_size: Maximum number of entries to keep.
        ttl_seconds: Time-to-live for each entry in seconds.
    """

    def __init__(self, max_size: int = 128, ttl_seconds: int = 300) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

    # ── public API ──────────────────────────────────────────────

    @staticmethod
    def make_key(task_type: str, financial_text: str) -> str:
        """Compute a deterministic cache key from request content."""
        raw = f"{task_type.strip().lower()}::{financial_text.strip()}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> CacheEntry | None:
        """Return the cached entry for *key*, or ``None`` on miss / expiry."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if time.time() - entry.created_at > self._ttl:
                # Expired — evict silently.
                del self._store[key]
                logger.debug("Cache expired | key=%s", key[:12])
                return None
            # Move to end so LRU eviction works correctly.
            self._store.move_to_end(key)
            logger.info("Cache HIT | key=%s", key[:12])
            return entry

    def put(self, key: str, entry: CacheEntry) -> None:
        """Store an entry, evicting the oldest if at capacity."""
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = entry
            if len(self._store) > self._max_size:
                evicted_key, _ = self._store.popitem(last=False)
                logger.debug("Cache evicted | key=%s", evicted_key[:12])
        logger.info("Cache STORE | key=%s  ttl=%ds", key[:12], self._ttl)

    def clear(self) -> None:
        """Remove all entries (useful for testing)."""
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        """Current number of entries in the cache."""
        return len(self._store)


# Module-level singleton — configured via Settings at startup.
response_cache = ResponseCache()
