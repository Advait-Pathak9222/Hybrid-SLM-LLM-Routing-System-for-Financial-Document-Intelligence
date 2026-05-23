"""Tests for the response cache."""

import time

import pytest

from app.services.cache import CacheEntry, ResponseCache


def _make_entry(response: str = "test response", **kwargs) -> CacheEntry:
    """Helper to create a CacheEntry with sensible defaults."""
    defaults = {
        "response": response,
        "model": "phi3:mini",
        "confidence": 0.85,
        "routing_decision": "slm",
        "tokens_used": 100,
        "estimated_cost_usd": 0.0,
        "created_at": time.time(),
    }
    defaults.update(kwargs)
    return CacheEntry(**defaults)


class TestResponseCache:
    """Unit tests for ResponseCache."""

    def setup_method(self):
        self.cache = ResponseCache(max_size=3, ttl_seconds=10)

    def test_make_key_deterministic(self):
        """Same input always produces the same key."""
        key1 = ResponseCache.make_key("summarization", "Hello world")
        key2 = ResponseCache.make_key("summarization", "Hello world")
        assert key1 == key2

    def test_make_key_differs_by_task(self):
        """Different task types produce different keys."""
        key1 = ResponseCache.make_key("summarization", "Hello world")
        key2 = ResponseCache.make_key("extraction", "Hello world")
        assert key1 != key2

    def test_make_key_case_insensitive_task(self):
        """Task type comparison is case-insensitive."""
        key1 = ResponseCache.make_key("Summarization", "Hello world")
        key2 = ResponseCache.make_key("summarization", "Hello world")
        assert key1 == key2

    def test_miss_returns_none(self):
        """Cache miss returns None."""
        assert self.cache.get("nonexistent") is None

    def test_put_and_get(self):
        """Stored entry can be retrieved."""
        entry = _make_entry("cached result")
        key = ResponseCache.make_key("summarization", "test text")
        self.cache.put(key, entry)

        result = self.cache.get(key)
        assert result is not None
        assert result.response == "cached result"
        assert result.model == "phi3:mini"

    def test_ttl_expiry(self):
        """Entries expire after TTL."""
        cache = ResponseCache(max_size=10, ttl_seconds=0)  # instant expiry
        entry = _make_entry(created_at=time.time() - 1)
        key = "test_key"
        cache.put(key, entry)

        # Entry should be expired immediately.
        assert cache.get(key) is None

    def test_max_size_eviction(self):
        """Oldest entry is evicted when cache is full."""
        for i in range(4):  # max_size is 3
            key = f"key_{i}"
            self.cache.put(key, _make_entry(f"response_{i}"))

        # First entry should have been evicted.
        assert self.cache.get("key_0") is None
        # Latest entries should still be present.
        assert self.cache.get("key_3") is not None
        assert self.cache.size == 3

    def test_clear(self):
        """Clear removes all entries."""
        self.cache.put("k1", _make_entry())
        self.cache.put("k2", _make_entry())
        self.cache.clear()
        assert self.cache.size == 0
        assert self.cache.get("k1") is None
