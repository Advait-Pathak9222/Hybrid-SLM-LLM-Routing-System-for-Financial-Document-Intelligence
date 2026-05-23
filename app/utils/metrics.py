import threading
import time
from dataclasses import dataclass, field
from app.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class PipelineMetrics:
    """Thread-safe in-memory metrics for pipeline observability.
    
    Tracks request counts, routing decisions, fallbacks, and latency.
    Designed for lightweight monitoring — not a replacement for
    production metrics systems like Prometheus.
    """
    total_requests: int = 0
    slm_requests: int = 0
    llm_requests: int = 0
    fallback_count: int = 0
    total_latency_ms: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, routing_decision: str, latency_ms: float, confidence: float) -> None:
        """Record metrics for a completed request."""
        with self._lock:
            self.total_requests += 1
            self.total_latency_ms += latency_ms
            if routing_decision == "slm":
                self.slm_requests += 1
            elif routing_decision == "llm":
                self.llm_requests += 1
            elif "fallback" in routing_decision:
                self.slm_requests += 1  # started as SLM
                self.fallback_count += 1
        
        logger.info(
            "Metrics  | routing=%s  confidence=%.2f  latency=%.0fms  "
            "total=%d  fallbacks=%d",
            routing_decision, confidence, latency_ms,
            self.total_requests, self.fallback_count,
        )

    @property
    def avg_latency_ms(self) -> float:
        """Average latency across all requests."""
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    @property
    def fallback_rate(self) -> float:
        """Percentage of SLM requests that fell back to LLM."""
        slm_total = self.slm_requests
        if slm_total == 0:
            return 0.0
        return self.fallback_count / slm_total

    def summary(self) -> dict:
        """Return a snapshot of all metrics."""
        return {
            "total_requests": self.total_requests,
            "slm_requests": self.slm_requests,
            "llm_requests": self.llm_requests,
            "fallback_count": self.fallback_count,
            "fallback_rate": round(self.fallback_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }

    def reset(self) -> None:
        """Reset all counters (useful for testing)."""
        with self._lock:
            self.total_requests = 0
            self.slm_requests = 0
            self.llm_requests = 0
            self.fallback_count = 0
            self.total_latency_ms = 0.0


# Module-level singleton
pipeline_metrics = PipelineMetrics()
