"""MindPrint — compact geometric fingerprint of KV-cache inference.

A MindPrint captures the geometric signature of a model's KV-cache
during inference. Same model + same input + honest computation produces
a deterministic MindPrint under greedy decoding.

The fingerprint includes:
  - 4 aggregate features (norm, norm_per_token, key_rank, key_entropy)
  - Per-layer effective rank profile (the "shape" of computation)
  - Per-layer spectral entropy profile
  - A content hash binding the print to specific input/output
"""

import hashlib
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from mindprint.snapshot import Snapshot


class MindPrint(BaseModel):
    """Compact geometric fingerprint of model inference."""

    # Identity
    version: int = 1
    model_id: str
    timestamp: float = Field(default_factory=time.time)

    # Content binding — hash of (prompt + output + model_id)
    # Prevents replaying a MindPrint against a different input
    content_hash: str

    # Aggregate geometry (4 features — matches Cricket classifier input)
    norm: float
    norm_per_token: float
    key_rank: float
    key_entropy: float

    # Per-layer profiles — the geometric "shape" of the computation
    # These are the discriminative signals: same model + different input
    # produces different profiles; different model + same input also differs
    layer_rank_profile: List[float]  # effective rank per extracted layer
    layer_entropy_profile: List[float]  # spectral entropy per extracted layer

    # Dimensions
    n_layers: int
    n_extracted: int
    n_tokens: Optional[int] = None

    # Fingerprint — SHA-256 of the canonical feature vector
    fingerprint: str = ""

    def compute_fingerprint(self) -> str:
        """Compute deterministic fingerprint from geometric features."""
        # Canonical feature vector: aggregates + profiles, rounded to 6 decimals
        # Rounding absorbs floating point noise across hardware
        parts = [
            f"{self.norm:.6f}",
            f"{self.norm_per_token:.6f}",
            f"{self.key_rank:.6f}",
            f"{self.key_entropy:.6f}",
        ]
        parts.extend(f"{r:.6f}" for r in self.layer_rank_profile)
        parts.extend(f"{e:.6f}" for e in self.layer_entropy_profile)

        canonical = "|".join(parts)
        return hashlib.sha256(canonical.encode()).hexdigest()


def content_hash(prompt: str, output: str, model_id: str) -> str:
    """Hash input/output/model to bind a MindPrint to specific content."""
    payload = f"{model_id}|{prompt}|{output}"
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def generate_mindprint(
    snapshot: Snapshot,
    model_id: str,
    prompt: str = "",
    output: str = "",
) -> MindPrint:
    """Generate a MindPrint from a CacheScope snapshot.

    Args:
        snapshot: CacheScope Snapshot with per-layer features.
        model_id: HuggingFace model identifier (e.g. "Qwen/Qwen2.5-7B-Instruct").
        prompt: The input prompt (for content binding).
        output: The model's output text (for content binding).

    Returns:
        MindPrint with geometric fingerprint.
    """
    agg = snapshot.aggregate
    layers = snapshot.per_layer

    rank_profile = [lf.key_eff_rank for lf in layers]
    entropy_profile = [lf.key_spectral_entropy for lf in layers]

    c_hash = content_hash(prompt, output, model_id)

    mp = MindPrint(
        model_id=model_id,
        content_hash=c_hash,
        norm=agg.norm,
        norm_per_token=agg.norm_per_token,
        key_rank=agg.key_rank,
        key_entropy=agg.key_entropy,
        layer_rank_profile=rank_profile,
        layer_entropy_profile=entropy_profile,
        n_layers=snapshot.n_layers,
        n_extracted=snapshot.n_extracted,
        n_tokens=snapshot.n_tokens,
    )
    mp.fingerprint = mp.compute_fingerprint()
    return mp
