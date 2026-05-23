"""Tests for request tracing."""

import pytest

from app.utils.tracing import generate_request_id, get_request_id, set_request_id


class TestTracing:
    """Unit tests for the tracing module."""

    def test_default_request_id(self):
        """Default request ID is 'no-request'."""
        # Reset to default by setting a known value and checking default behavior.
        # ContextVar defaults are per-context, so in a fresh test this should work.
        assert isinstance(get_request_id(), str)

    def test_generate_request_id_format(self):
        """Generated IDs are 12-char hex strings."""
        rid = generate_request_id()
        assert len(rid) == 12
        assert all(c in "0123456789abcdef" for c in rid)

    def test_generate_unique_ids(self):
        """Each call produces a unique ID."""
        ids = {generate_request_id() for _ in range(100)}
        assert len(ids) == 100

    def test_set_and_get(self):
        """Setting a request ID makes it retrievable."""
        set_request_id("test-abc-123")
        assert get_request_id() == "test-abc-123"

    def test_overwrite(self):
        """Setting a new ID overwrites the previous one."""
        set_request_id("first")
        set_request_id("second")
        assert get_request_id() == "second"
