"""KVCacheBench: failure-focused diagnostics for KV cache compression."""

from .retention import slot_retention_metrics

__version__ = "0.2.0"

__all__ = ["__version__", "slot_retention_metrics"]
