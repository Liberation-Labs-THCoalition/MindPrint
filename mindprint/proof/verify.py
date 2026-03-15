"""MindPrint verification — compare miner and validator geometric fingerprints.

Verification modes:
  - Exact: fingerprints must match (greedy decode, same hardware)
  - Tolerant: per-feature distance within calibrated thresholds
    (sampled decode, cross-hardware, quantization differences)

The verifier checks:
  1. Content binding — same prompt/output/model claimed
  2. Structural match — same number of layers extracted
  3. Aggregate similarity — 4 Cricket features within tolerance
  4. Profile similarity — per-layer rank/entropy correlation
  5. Fingerprint match — exact hash (strict mode only)
"""

import math
from typing import Dict, List, Optional

from pydantic import BaseModel

from mindprint.proof.mindprint import MindPrint


class VerificationResult(BaseModel):
    """Result of comparing two MindPrints."""

    valid: bool
    mode: str  # "exact" or "tolerant"

    # Per-check results
    content_match: bool
    structure_match: bool
    aggregate_match: bool
    profile_match: bool
    fingerprint_match: bool

    # Detailed distances
    aggregate_distances: Dict[str, float] = {}
    profile_correlation: Optional[float] = None  # Pearson r of rank profiles

    # Anomaly flags
    anomalies: List[str] = []

    confidence: float = 0.0  # 0.0 = definitely dishonest, 1.0 = definitely honest


def _pearson_r(a: List[float], b: List[float]) -> float:
    """Pearson correlation coefficient between two equal-length lists."""
    n = len(a)
    if n < 2:
        return 1.0 if a == b else 0.0

    mean_a = sum(a) / n
    mean_b = sum(b) / n

    cov = sum((ai - mean_a) * (bi - mean_b) for ai, bi in zip(a, b))
    std_a = math.sqrt(sum((ai - mean_a) ** 2 for ai in a))
    std_b = math.sqrt(sum((bi - mean_b) ** 2 for bi in b))

    if std_a < 1e-12 or std_b < 1e-12:
        # One profile is constant — compare means
        return 1.0 if abs(mean_a - mean_b) < 1e-6 else 0.0

    return cov / (std_a * std_b)


def _relative_distance(a: float, b: float) -> float:
    """Relative distance between two values, symmetric."""
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def verify_mindprint(
    miner: MindPrint,
    reference: MindPrint,
    tolerance: float = 0.01,
    profile_r_threshold: float = 0.95,
    strict: bool = False,
) -> VerificationResult:
    """Verify a miner's MindPrint against a validator's reference.

    Args:
        miner: MindPrint from the miner's inference.
        reference: MindPrint from the validator's reference inference.
        tolerance: Maximum relative distance for aggregate features (default 1%).
        profile_r_threshold: Minimum Pearson r for layer profiles (default 0.95).
        strict: If True, require exact fingerprint match.

    Returns:
        VerificationResult with detailed comparison.
    """
    anomalies = []

    # 1. Content binding
    content_match = miner.content_hash == reference.content_hash
    if not content_match:
        anomalies.append("content_hash_mismatch: miner claims different input/output")

    # 2. Structural match
    structure_match = (
        miner.model_id == reference.model_id
        and miner.n_layers == reference.n_layers
        and miner.n_extracted == reference.n_extracted
    )
    if not structure_match:
        if miner.model_id != reference.model_id:
            anomalies.append(f"model_mismatch: {miner.model_id} vs {reference.model_id}")
        if miner.n_layers != reference.n_layers:
            anomalies.append(f"layer_count_mismatch: {miner.n_layers} vs {reference.n_layers}")
        if miner.n_extracted != reference.n_extracted:
            anomalies.append(f"extracted_count_mismatch: {miner.n_extracted} vs {reference.n_extracted}")

    # 3. Aggregate feature distances
    agg_distances = {
        "norm": _relative_distance(miner.norm, reference.norm),
        "norm_per_token": _relative_distance(miner.norm_per_token, reference.norm_per_token),
        "key_rank": _relative_distance(miner.key_rank, reference.key_rank),
        "key_entropy": _relative_distance(miner.key_entropy, reference.key_entropy),
    }

    aggregate_match = all(d <= tolerance for d in agg_distances.values())
    if not aggregate_match:
        for feat, dist in agg_distances.items():
            if dist > tolerance:
                anomalies.append(f"aggregate_{feat}_drift: {dist:.4f} > {tolerance}")

    # 4. Profile correlation
    profile_r = None
    profile_match = True

    if (
        len(miner.layer_rank_profile) == len(reference.layer_rank_profile)
        and len(miner.layer_rank_profile) > 1
    ):
        rank_r = _pearson_r(miner.layer_rank_profile, reference.layer_rank_profile)
        entropy_r = _pearson_r(miner.layer_entropy_profile, reference.layer_entropy_profile)
        profile_r = min(rank_r, entropy_r)
        profile_match = profile_r >= profile_r_threshold

        if not profile_match:
            anomalies.append(
                f"profile_correlation_low: rank_r={rank_r:.4f}, "
                f"entropy_r={entropy_r:.4f} < {profile_r_threshold}"
            )
    elif len(miner.layer_rank_profile) != len(reference.layer_rank_profile):
        profile_match = False
        anomalies.append("profile_length_mismatch")

    # 5. Fingerprint match
    fingerprint_match = miner.fingerprint == reference.fingerprint

    # Overall validity
    if strict:
        valid = fingerprint_match and content_match
    else:
        valid = content_match and structure_match and aggregate_match and profile_match

    # Confidence score
    if not content_match or not structure_match:
        confidence = 0.0
    elif fingerprint_match:
        confidence = 1.0
    else:
        # Weighted combination of aggregate closeness and profile correlation
        agg_score = 1.0 - min(1.0, sum(agg_distances.values()) / (4 * tolerance))
        prof_score = max(0.0, (profile_r - profile_r_threshold) / (1.0 - profile_r_threshold)) if profile_r is not None else 0.5
        confidence = 0.4 * agg_score + 0.6 * prof_score
        confidence = max(0.0, min(1.0, confidence))

    return VerificationResult(
        valid=valid,
        mode="exact" if strict else "tolerant",
        content_match=content_match,
        structure_match=structure_match,
        aggregate_match=aggregate_match,
        profile_match=profile_match,
        fingerprint_match=fingerprint_match,
        aggregate_distances=agg_distances,
        profile_correlation=profile_r,
        anomalies=anomalies,
        confidence=confidence,
    )
