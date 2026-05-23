"""In-memory pipeline metrics with cache and cost tracking."""

import threading
from dataclasses import dataclass, field

from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineMetrics:
    """Thread-safe in-memory metrics for pipeline observability.

    Tracks request counts, routing decisions, fallbacks, latency,
    cache performance, and cost estimation.  Designed for lightweight
    monitoring — not a replacement for production metrics systems
    like Prometheus.
    """

    total_requests: int = 0
    slm_requests: int = 0
    llm_requests: int = 0
    fallback_count: int = 0
    total_latency_ms: float = 0.0

    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0

    # Cost metrics
    total_tokens_used: int = 0
    total_estimated_cost_usd: float = 0.0
    total_cost_saved_vs_gpt4o: float = 0.0

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self,
        routing_decision: str,
        latency_ms: float,
        confidence: float,
        *,
        cache_hit: bool = False,
        tokens_used: int = 0,
        estimated_cost_usd: float = 0.0,
        cost_saved_vs_gpt4o: float = 0.0,
    ) -> None:
        """Record metrics for a completed request."""
        with self._lock:
            self.total_requests += 1
            self.total_latency_ms += latency_ms

            # Routing
            if routing_decision == "slm":
                self.slm_requests += 1
            elif routing_decision == "llm":
                self.llm_requests += 1
            elif "fallback" in routing_decision:
                self.slm_requests += 1  # started as SLM
                self.fallback_count += 1

            # Cache
            if cache_hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1

            # Cost
            self.total_tokens_used += tokens_used
            self.total_estimated_cost_usd += estimated_cost_usd
            self.total_cost_saved_vs_gpt4o += cost_saved_vs_gpt4o

        logger.info(
            "Metrics  | routing=%s  confidence=%.2f  latency=%.0fms  "
            "total=%d  fallbacks=%d  cache=%s  tokens=%d",
            routing_decision,
            confidence,
            latency_ms,
            self.total_requests,
            self.fallback_count,
            "HIT" if cache_hit else "MISS",
            tokens_used,
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

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate as a fraction."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    @property
    def avg_tokens_per_request(self) -> float:
        """Average tokens per request."""
        if self.total_requests == 0:
            return 0.0
        return self.total_tokens_used / self.total_requests

    def summary(self) -> dict:
        """Return a snapshot of all metrics."""
        return {
            "total_requests": self.total_requests,
            "slm_requests": self.slm_requests,
            "llm_requests": self.llm_requests,
            "fallback_count": self.fallback_count,
            "fallback_rate": round(self.fallback_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            # Cache
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            # Cost
            "total_tokens_used": self.total_tokens_used,
            "avg_tokens_per_request": round(self.avg_tokens_per_request, 1),
            "total_estimated_cost_usd": round(self.total_estimated_cost_usd, 6),
            "total_cost_saved_vs_gpt4o": round(self.total_cost_saved_vs_gpt4o, 6),
        }

    def reset(self) -> None:
        """Reset all counters (useful for testing)."""
        with self._lock:
            self.total_requests = 0
            self.slm_requests = 0
            self.llm_requests = 0
            self.fallback_count = 0
            self.total_latency_ms = 0.0
            self.cache_hits = 0
            self.cache_misses = 0
            self.total_tokens_used = 0
            self.total_estimated_cost_usd = 0.0
            self.total_cost_saved_vs_gpt4o = 0.0


# Module-level singleton
pipeline_metrics = PipelineMetrics()
