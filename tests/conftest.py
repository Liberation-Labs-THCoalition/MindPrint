"""Test fixtures for MindPrint."""

import pytest

from mindprint.snapshot import AggregateFeatures, LayerFeatures, Snapshot


def make_layer_features(layer_idx: int, seed: float = 1.0) -> LayerFeatures:
    """Create deterministic layer features for testing."""
    s = seed * (layer_idx + 1)
    return LayerFeatures(
        layer_idx=layer_idx,
        key_norm=10.0 * s,
        value_norm=8.0 * s,
        key_mean=0.01 * s,
        value_mean=-0.02 * s,
        key_std=0.5 * s,
        value_std=0.4 * s,
        key_eff_rank=32.0 + layer_idx,
        value_eff_rank=28.0 + layer_idx,
        key_spectral_entropy=0.85 + 0.01 * layer_idx,
        value_spectral_entropy=0.80 + 0.01 * layer_idx,
        key_rank_ratio=0.5 + 0.01 * layer_idx,
        value_rank_ratio=0.44 + 0.01 * layer_idx,
    )


def make_snapshot(seq: int = 1, n_layers: int = 4, seed: float = 1.0) -> Snapshot:
    """Create a deterministic snapshot for testing."""
    layers = [make_layer_features(i, seed) for i in range(n_layers)]
    norm = sum(lf.key_norm for lf in layers)
    n_tokens = 50
    return Snapshot(
        sequence_number=seq,
        per_layer=layers,
        aggregate=AggregateFeatures(
            norm=norm,
            norm_per_token=norm / n_tokens,
            key_rank=sum(lf.key_eff_rank for lf in layers) / n_layers,
            key_entropy=sum(lf.key_spectral_entropy for lf in layers) / n_layers,
        ),
        n_layers=n_layers,
        n_extracted=n_layers,
        n_tokens=n_tokens,
        extraction_time_ms=12.5,
        metadata={"prompt": "test", "seq": seq},
    )
