"""Feature extraction wrapper.

Thin layer over cricket_features.extract_features() and
cricket_classifier.features_from_extract(). Handles path resolution
and converts raw dicts to Pydantic Snapshot models.
"""

import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mindprint.config import CacheScopeConfig
from mindprint.snapshot import AggregateFeatures, LayerFeatures, Snapshot

# Lazy imports — resolved at init time
_extract_features = None
_features_from_extract = None
_cache_to_cpu = None
_normalize_cache = None


def _resolve_paths(config: CacheScopeConfig) -> Tuple[str, str]:
    """Find cricket src and kv-experiments code directories."""
    cricket = config.cricket_src_path
    kv_exp = config.kv_experiments_code_path

    if not cricket:
        # Try common locations relative to this file
        for candidate in [
            Path.home() / "jiminai-cricket" / "src",
            Path("/home/agent/jiminai-cricket/src"),
        ]:
            if (candidate / "cricket_features.py").exists():
                cricket = str(candidate)
                break

    if not kv_exp:
        for candidate in [
            Path.home() / "KV-Experiments" / "code",
            Path("/home/agent/KV-Experiments/code"),
        ]:
            if (candidate / "gpu_utils.py").exists():
                kv_exp = str(candidate)
                break

    return cricket or "", kv_exp or ""


def _ensure_imports(config: CacheScopeConfig) -> None:
    """Import cricket_features and gpu_utils on first use."""
    global _extract_features, _features_from_extract, _cache_to_cpu, _normalize_cache

    if _extract_features is not None:
        return

    cricket_path, kv_path = _resolve_paths(config)

    if cricket_path and cricket_path not in sys.path:
        sys.path.insert(0, cricket_path)
    if kv_path and kv_path not in sys.path:
        sys.path.insert(0, kv_path)

    from cricket_features import extract_features
    from cricket_classifier import features_from_extract
    from gpu_utils import cache_to_cpu, normalize_cache

    _extract_features = extract_features
    _features_from_extract = features_from_extract
    _cache_to_cpu = cache_to_cpu
    _normalize_cache = normalize_cache


class CacheScopeExtractor:
    """Extracts KV-cache geometry features and produces Snapshots."""

    def __init__(self, config: CacheScopeConfig):
        self.config = config
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._imports_ready = False

    def extract(
        self,
        cache,
        n_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Snapshot:
        """Extract features from a KV-cache and return a Snapshot.

        Args:
            cache: past_key_values (already on CPU, normalized).
            n_tokens: Token count for norm_per_token computation.
            metadata: Optional prompt/model info to attach.
        """
        if not self._imports_ready:
            _ensure_imports(self.config)
            self._imports_ready = True

        t0 = time.perf_counter()

        # Count layers without consuming the cache
        # cache is typically a tuple, but normalize first if needed
        if hasattr(cache, '__len__'):
            n_layers = len(cache)
        else:
            cache = tuple(cache)
            n_layers = len(cache)
        layer_subset = self.config.resolve_layer_subset(n_layers)

        result = _extract_features(
            cache,
            variance_threshold=self.config.variance_threshold,
            layer_subset=layer_subset,
        )

        agg_4d = _features_from_extract(result, n_tokens=n_tokens)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        with self._seq_lock:
            self._seq += 1
            seq = self._seq

        per_layer = [LayerFeatures(**lf) for lf in result["per_layer"]]
        aggregate = AggregateFeatures(
            norm=float(agg_4d[0]),
            norm_per_token=float(agg_4d[1]),
            key_rank=float(agg_4d[2]),
            key_entropy=float(agg_4d[3]),
        )

        return Snapshot(
            snapshot_id=str(uuid.uuid4()),
            timestamp=time.time(),
            sequence_number=seq,
            per_layer=per_layer,
            aggregate=aggregate,
            n_layers=result["n_layers"],
            n_extracted=result["n_extracted"],
            n_tokens=n_tokens,
            extraction_time_ms=elapsed_ms,
            metadata=metadata or {},
        )

    def cache_to_cpu(self, cache):
        """Move cache tensors to CPU, handling all cache formats.

        Handles DynamicCache, HybridCache, tuple-of-tuples, and
        entries with None values (common in newer transformers).
        """
        if not self._imports_ready:
            _ensure_imports(self.config)
            self._imports_ready = True

        # Normalize first (handles DynamicCache → list of tuples)
        normalized = _normalize_cache(cache)

        result = []
        for layer in normalized:
            if isinstance(layer, tuple):
                cpu_layer = tuple(
                    t.detach().cpu() if t is not None else None
                    for t in layer
                )
                # Skip layers where both K and V are None
                if cpu_layer[0] is not None or (len(cpu_layer) > 1 and cpu_layer[1] is not None):
                    result.append(cpu_layer)
            else:
                result.append(layer)

        return tuple(result)

    def normalize_cache(self, cache):
        """Normalize cache format (delegates to gpu_utils)."""
        if not self._imports_ready:
            _ensure_imports(self.config)
            self._imports_ready = True
        return _normalize_cache(cache)
