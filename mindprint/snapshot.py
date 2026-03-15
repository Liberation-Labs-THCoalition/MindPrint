"""Snapshot data models — standalone copy from CacheScope.

These models define the geometric observation format that MindPrint
operates on. They can be populated from CacheScope's extractor or
from any other source that computes the same 12 per-layer features.
"""

import time
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LayerFeatures(BaseModel):
    """12 geometric features for a single transformer layer's KV-cache."""

    layer_idx: int

    # Magnitude features (6)
    key_norm: float
    value_norm: float
    key_mean: float
    value_mean: float
    key_std: float
    value_std: float

    # SVD features (6)
    key_eff_rank: float
    value_eff_rank: float
    key_spectral_entropy: float
    value_spectral_entropy: float
    key_rank_ratio: float
    value_rank_ratio: float


class AggregateFeatures(BaseModel):
    """4 aggregate features matching Cricket's classifier input."""

    norm: float
    norm_per_token: float
    key_rank: float
    key_entropy: float


class Snapshot(BaseModel):
    """A single KV-cache geometry observation."""

    snapshot_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    sequence_number: int = 0

    per_layer: List[LayerFeatures]
    aggregate: AggregateFeatures

    n_layers: int
    n_extracted: int
    n_tokens: Optional[int] = None

    extraction_time_ms: float = 0.0

    metadata: Dict[str, Any] = Field(default_factory=dict)


FEATURE_NAMES = [
    "key_norm", "value_norm", "key_mean", "value_mean", "key_std", "value_std",
    "key_eff_rank", "value_eff_rank", "key_spectral_entropy",
    "value_spectral_entropy", "key_rank_ratio", "value_rank_ratio",
]

AGGREGATE_NAMES = ["norm", "norm_per_token", "key_rank", "key_entropy"]
