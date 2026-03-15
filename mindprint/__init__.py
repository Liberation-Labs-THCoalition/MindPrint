"""MindPrint — Verifiable inference via KV-cache geometry.

Proof of Mind protocol for decentralized compute networks.
Generates compact geometric fingerprints (MindPrints) proving
honest model inference. Built for the Bittensor ecosystem.

"Verify what the model thought, not just what it said."

Usage:
    from mindprint import MindPrint, generate_mindprint, verify_mindprint

    # Generate from a CacheScope snapshot or raw cache
    mp = generate_mindprint(snapshot, model_id="Qwen/Qwen2.5-7B-Instruct")

    # Verify
    result = verify_mindprint(miner_print, reference_print)
    assert result.valid
"""

__version__ = "0.1.0"

from mindprint.proof.mindprint import MindPrint, generate_mindprint
from mindprint.proof.verify import VerificationResult, verify_mindprint
from mindprint.proof.codec import encode_mindprint, decode_mindprint
