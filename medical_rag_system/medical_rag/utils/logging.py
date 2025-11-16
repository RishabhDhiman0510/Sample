"""Structured logging with Prometheus metrics."""
import logging
import structlog
from prometheus_client import Counter, Histogram, CollectorRegistry, generate_latest

PROM_REGISTRY = CollectorRegistry()
REQUEST_COUNT = Counter('medical_rag_requests_total', 'Total requests', ['endpoint', 'method'], registry=PROM_REGISTRY)
ERROR_COUNT = Counter('medical_rag_errors_total', 'Total errors', ['error_type'], registry=PROM_REGISTRY)
RETRIEVAL_LATENCY = Histogram('medical_rag_retrieval_latency_seconds', 'Retrieval latency', buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0], registry=PROM_REGISTRY)
GENERATION_LATENCY = Histogram('medical_rag_generation_latency_seconds', 'Generation latency', buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0], registry=PROM_REGISTRY)
FEEDBACK_COUNT = Counter('medical_rag_feedback_total', 'Total feedback', registry=PROM_REGISTRY)

def setup_logging(log_level: str = "INFO") -> None:
    structlog.configure(processors=[structlog.contextvars.merge_contextvars, structlog.processors.add_log_level, structlog.processors.StackInfoRenderer(), structlog.dev.set_exc_info, structlog.processors.TimeStamper(fmt="iso", utc=True), structlog.processors.JSONRenderer()], wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, log_level.upper())), context_class=dict, logger_factory=structlog.PrintLoggerFactory(), cache_logger_on_first_use=True)

def get_logger(name=None):
    return structlog.get_logger(name)

def get_metrics() -> bytes:
    return generate_latest(PROM_REGISTRY)
