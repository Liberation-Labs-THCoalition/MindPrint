"""Tests for Bittensor integration — protocol and verification flow."""

import base64

import bittensor as bt

from mindprint.bittensor.protocol import MindPrintSynapse
from mindprint.proof.mindprint import generate_mindprint
from mindprint.proof.codec import encode_mindprint, decode_mindprint
from mindprint.proof.verify import verify_mindprint
from tests.conftest import make_snapshot


# --- Synapse protocol ---

def test_synapse_creation():
    synapse = MindPrintSynapse(
        query="What is 2+2?",
        model_id="test/model",
        max_tokens=64,
    )
    assert synapse.query == "What is 2+2?"
    assert synapse.model_id == "test/model"
    assert synapse.temperature == 0.0  # Greedy default
    assert synapse.response == ""
    assert synapse.mindprint_b64 == ""


def test_synapse_is_bittensor_synapse():
    synapse = MindPrintSynapse(query="test")
    assert isinstance(synapse, bt.Synapse)
    assert synapse.name == "MindPrintSynapse"


def test_synapse_response_fields():
    synapse = MindPrintSynapse(query="test", model_id="m")
    synapse.response = "The answer is 4."
    synapse.mindprint_b64 = "AAAA"
    synapse.n_tokens = 42
    synapse.verification_passed = True
    synapse.verification_confidence = 0.95

    data = synapse.deserialize()
    assert data["response"] == "The answer is 4."
    assert data["mindprint_b64"] == "AAAA"
    assert data["n_tokens"] == 42


# --- End-to-end miner→validator flow (without GPU) ---

def test_miner_validator_flow_simulated():
    """Simulate the full miner→validator flow using mock snapshots."""
    # Miner side: generate response + MindPrint
    snap = make_snapshot(seq=1, n_layers=4)
    model_id = "test/model"
    prompt = "What is 2+2?"
    output = "The answer is 4."

    mp = generate_mindprint(snap, model_id=model_id, prompt=prompt, output=output)
    wire = encode_mindprint(mp)
    b64 = base64.b64encode(wire).decode()

    # Miner fills synapse
    synapse = MindPrintSynapse(query=prompt, model_id=model_id)
    synapse.response = output
    synapse.mindprint_b64 = b64
    synapse.n_tokens = 50

    # Validator side: decode and verify
    miner_bytes = base64.b64decode(synapse.mindprint_b64)
    miner_mp = decode_mindprint(miner_bytes, model_id=model_id)

    # Validator generates same reference (same snapshot in this simulation)
    ref_mp = generate_mindprint(snap, model_id=model_id, prompt=prompt, output=output)

    result = verify_mindprint(miner_mp, ref_mp, tolerance=0.01)

    assert result.valid
    assert result.content_match
    assert result.structure_match
    assert result.confidence > 0.5


def test_miner_validator_detects_model_swap():
    """Miner claims model-a but uses different geometry (model-b)."""
    snap_a = make_snapshot(seq=1, n_layers=4, seed=1.0)
    snap_b = make_snapshot(seq=1, n_layers=4, seed=5.0)  # Very different geometry

    model_id = "test/model"
    prompt = "test"
    output = "result"

    # Miner uses model-b's geometry but claims model-a
    miner_mp = generate_mindprint(snap_b, model_id=model_id, prompt=prompt, output=output)

    # Validator generates reference from model-a
    ref_mp = generate_mindprint(snap_a, model_id=model_id, prompt=prompt, output=output)

    result = verify_mindprint(miner_mp, ref_mp, tolerance=0.01)

    assert not result.valid
    assert len(result.anomalies) > 0


def test_miner_validator_detects_wrong_layer_count():
    """Miner claims 32-layer model but returns 28-layer geometry."""
    snap_miner = make_snapshot(seq=1, n_layers=3)  # Wrong layer count
    snap_ref = make_snapshot(seq=1, n_layers=4)

    mp_miner = generate_mindprint(snap_miner, model_id="m", prompt="p", output="o")
    mp_ref = generate_mindprint(snap_ref, model_id="m", prompt="p", output="o")

    result = verify_mindprint(mp_miner, mp_ref)

    assert not result.valid
    assert not result.structure_match


def test_model_signature_verification():
    """Test model-signature mode (no reference needed)."""
    from mindprint.bittensor.validator import MindPrintValidator

    validator = MindPrintValidator(model_id="test/model")

    snap = make_snapshot(seq=1, n_layers=4)
    mp = generate_mindprint(snap, model_id="test/model", prompt="p", output="o")

    result = validator._verify_model_signature(mp)
    assert result.valid  # Unknown model passes structural checks
    assert result.mode == "model_signature"
    assert result.confidence == 0.7  # Capped for model-sig mode


def test_model_signature_rejects_zero_norm():
    """Model-signature mode rejects impossible geometry."""
    from mindprint.bittensor.validator import MindPrintValidator
    from mindprint.proof.mindprint import MindPrint

    validator = MindPrintValidator(model_id="test/model")

    mp = MindPrint(
        model_id="test/model",
        content_hash="a" * 32,
        norm=0.0,  # Impossible
        norm_per_token=0.0,
        key_rank=0.5,
        key_entropy=0.85,
        layer_rank_profile=[32.0, 33.0, 34.0, 35.0],
        layer_entropy_profile=[0.85, 0.86, 0.87, 0.88],
        n_layers=4,
        n_extracted=4,
    )

    result = validator._verify_model_signature(mp)
    assert not result.valid
    assert any("norm_zero" in a for a in result.anomalies)
