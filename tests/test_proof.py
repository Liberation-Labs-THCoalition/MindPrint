"""Tests for Proof of Mind — MindPrint generation, verification, and codec."""

import math

from mindprint.proof.mindprint import MindPrint, generate_mindprint, content_hash
from mindprint.proof.verify import VerificationResult, verify_mindprint, _pearson_r
from mindprint.proof.codec import encode_mindprint, decode_mindprint, mindprint_size
from tests.conftest import make_snapshot


# --- MindPrint generation ---

def test_generate_mindprint():
    snap = make_snapshot(seq=1, n_layers=4)
    mp = generate_mindprint(snap, model_id="test/model", prompt="hello", output="world")

    assert mp.model_id == "test/model"
    assert mp.n_layers == 4
    assert mp.n_extracted == 4
    assert len(mp.layer_rank_profile) == 4
    assert len(mp.layer_entropy_profile) == 4
    assert mp.fingerprint != ""
    assert len(mp.fingerprint) == 64  # SHA-256 hex


def test_content_hash_deterministic():
    h1 = content_hash("hello", "world", "model-a")
    h2 = content_hash("hello", "world", "model-a")
    assert h1 == h2


def test_content_hash_sensitive():
    h1 = content_hash("hello", "world", "model-a")
    h2 = content_hash("hello", "world", "model-b")
    h3 = content_hash("goodbye", "world", "model-a")
    assert h1 != h2
    assert h1 != h3


def test_fingerprint_deterministic():
    snap = make_snapshot(seq=1, n_layers=4)
    mp1 = generate_mindprint(snap, model_id="m", prompt="p", output="o")
    mp2 = generate_mindprint(snap, model_id="m", prompt="p", output="o")
    assert mp1.fingerprint == mp2.fingerprint


def test_fingerprint_sensitive_to_geometry():
    snap1 = make_snapshot(seq=1, n_layers=4, seed=1.0)
    snap2 = make_snapshot(seq=1, n_layers=4, seed=2.0)
    mp1 = generate_mindprint(snap1, model_id="m", prompt="p", output="o")
    mp2 = generate_mindprint(snap2, model_id="m", prompt="p", output="o")
    assert mp1.fingerprint != mp2.fingerprint


# --- Verification ---

def test_verify_exact_match():
    snap = make_snapshot(seq=1, n_layers=4)
    mp = generate_mindprint(snap, model_id="m", prompt="p", output="o")
    result = verify_mindprint(mp, mp, strict=True)
    assert result.valid
    assert result.fingerprint_match
    assert result.confidence == 1.0
    assert result.anomalies == []


def test_verify_tolerant_match():
    snap = make_snapshot(seq=1, n_layers=4)
    mp = generate_mindprint(snap, model_id="m", prompt="p", output="o")
    result = verify_mindprint(mp, mp, strict=False)
    assert result.valid
    assert result.confidence == 1.0


def test_verify_different_geometry():
    snap1 = make_snapshot(seq=1, n_layers=4, seed=1.0)
    snap2 = make_snapshot(seq=1, n_layers=4, seed=5.0)  # Very different
    mp1 = generate_mindprint(snap1, model_id="m", prompt="p", output="o")
    mp2 = generate_mindprint(snap2, model_id="m", prompt="p", output="o")
    result = verify_mindprint(mp1, mp2, tolerance=0.01)
    assert not result.valid
    assert not result.fingerprint_match
    assert len(result.anomalies) > 0


def test_verify_content_mismatch():
    snap = make_snapshot(seq=1, n_layers=4)
    mp1 = generate_mindprint(snap, model_id="m", prompt="hello", output="o")
    mp2 = generate_mindprint(snap, model_id="m", prompt="goodbye", output="o")
    result = verify_mindprint(mp1, mp2)
    assert not result.valid
    assert not result.content_match
    assert "content_hash_mismatch" in result.anomalies[0]


def test_verify_model_mismatch():
    snap = make_snapshot(seq=1, n_layers=4)
    mp1 = generate_mindprint(snap, model_id="model-a", prompt="p", output="o")
    mp2 = generate_mindprint(snap, model_id="model-b", prompt="p", output="o")
    result = verify_mindprint(mp1, mp2)
    assert not result.valid
    assert not result.structure_match


def test_verify_wide_tolerance():
    snap1 = make_snapshot(seq=1, n_layers=4, seed=1.0)
    snap2 = make_snapshot(seq=1, n_layers=4, seed=1.05)  # Slightly different
    mp1 = generate_mindprint(snap1, model_id="m", prompt="p", output="o")
    mp2 = generate_mindprint(snap2, model_id="m", prompt="p", output="o")
    result = verify_mindprint(mp1, mp2, tolerance=0.5, profile_r_threshold=0.5)
    assert result.valid


# --- Pearson correlation ---

def test_pearson_perfect():
    assert abs(_pearson_r([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-10


def test_pearson_inverse():
    assert abs(_pearson_r([1, 2, 3, 4], [4, 3, 2, 1]) - (-1.0)) < 1e-10


def test_pearson_uncorrelated():
    r = _pearson_r([1, 2, 1, 2], [1, 1, 2, 2])
    assert abs(r) < 0.5


# --- Codec ---

def test_encode_decode_roundtrip():
    snap = make_snapshot(seq=1, n_layers=4)
    mp = generate_mindprint(snap, model_id="test/model", prompt="p", output="o")

    wire = encode_mindprint(mp)
    decoded = decode_mindprint(wire, model_id="test/model")

    assert decoded.version == mp.version
    assert decoded.content_hash == mp.content_hash
    assert decoded.n_layers == mp.n_layers
    assert decoded.n_extracted == mp.n_extracted
    assert decoded.n_tokens == mp.n_tokens
    assert len(decoded.layer_rank_profile) == len(mp.layer_rank_profile)
    assert len(decoded.layer_entropy_profile) == len(mp.layer_entropy_profile)

    # Float32 roundtrip — check within precision
    for a, b in zip(decoded.layer_rank_profile, mp.layer_rank_profile):
        assert abs(a - b) < 0.01

    for a, b in zip(decoded.layer_entropy_profile, mp.layer_entropy_profile):
        assert abs(a - b) < 0.001


def test_wire_size_compact():
    # 8 extracted layers (stride-4 on 32-layer model)
    size = mindprint_size(8)
    assert size < 200  # Should be well under 200 bytes

    # 32 layers (full extraction)
    size_full = mindprint_size(32)
    assert size_full < 400


def test_encode_size_matches_prediction():
    snap = make_snapshot(seq=1, n_layers=4)
    mp = generate_mindprint(snap, model_id="m", prompt="p", output="o")
    wire = encode_mindprint(mp)
    predicted = mindprint_size(mp.n_extracted)
    assert len(wire) == predicted


def test_decoded_verifies_against_original():
    """Decoded MindPrint should verify against the original."""
    snap = make_snapshot(seq=1, n_layers=4)
    mp = generate_mindprint(snap, model_id="m", prompt="p", output="o")

    wire = encode_mindprint(mp)
    decoded = decode_mindprint(wire, model_id="m")

    # Tolerant verification should pass despite float32 precision loss
    result = verify_mindprint(decoded, mp, tolerance=0.01)
    assert result.valid


# API route tests are in the CacheScope repo where the FastAPI app lives
