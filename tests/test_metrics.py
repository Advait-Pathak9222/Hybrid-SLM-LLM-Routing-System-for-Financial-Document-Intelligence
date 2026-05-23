"""Tests for the pipeline metrics module."""

from app.utils.metrics import PipelineMetrics


class TestPipelineMetrics:
    """Verify metrics tracking."""

    def test_record_slm_request(self) -> None:
        """Recording an SLM request should increment counters."""
        m = PipelineMetrics()
        m.record("slm", 100.0, 0.85)
        assert m.total_requests == 1
        assert m.slm_requests == 1
        assert m.llm_requests == 0

    def test_record_llm_request(self) -> None:
        """Recording an LLM request should increment counters."""
        m = PipelineMetrics()
        m.record("llm", 200.0, 1.0)
        assert m.total_requests == 1
        assert m.llm_requests == 1

    def test_record_fallback(self) -> None:
        """Fallback requests should increment both SLM and fallback counters."""
        m = PipelineMetrics()
        m.record("slm→llm_fallback", 300.0, 0.4)
        assert m.fallback_count == 1
        assert m.slm_requests == 1

    def test_avg_latency(self) -> None:
        """Average latency should be computed correctly."""
        m = PipelineMetrics()
        m.record("slm", 100.0, 0.8)
        m.record("llm", 200.0, 1.0)
        assert m.avg_latency_ms == 150.0

    def test_fallback_rate(self) -> None:
        """Fallback rate should be fallbacks / SLM total."""
        m = PipelineMetrics()
        m.record("slm", 100.0, 0.9)
        m.record("slm→llm_fallback", 300.0, 0.4)
        assert m.fallback_rate == 0.5

    def test_summary(self) -> None:
        """Summary should return all metrics as a dict."""
        m = PipelineMetrics()
        m.record("slm", 100.0, 0.85)
        s = m.summary()
        assert "total_requests" in s
        assert "fallback_rate" in s
        assert s["total_requests"] == 1

    def test_reset(self) -> None:
        """Reset should zero all counters."""
        m = PipelineMetrics()
        m.record("slm", 100.0, 0.85)
        m.reset()
        assert m.total_requests == 0
