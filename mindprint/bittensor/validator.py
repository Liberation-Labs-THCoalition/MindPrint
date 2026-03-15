"""Bittensor validator with Proof of Mind verification.

Sends inference queries to miners, receives responses + MindPrints,
and verifies geometric integrity. Supports two verification modes:

1. Full verification: Validator re-runs inference locally and compares
   MindPrints. Maximum security, but requires validator to have the model.

2. Model-signature verification: Validator compares MindPrint against a
   known geometric baseline for the claimed model. Verifies model identity
   without seeing the query content. Lower cost, privacy-preserving.

Usage:
    python -m mindprint.bittensor.validator \
        --model Qwen/Qwen2.5-7B-Instruct \
        --netuid 1 \
        --wallet.name validator \
        --wallet.hotkey default
"""

import argparse
import base64
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import bittensor as bt

from mindprint.bittensor.protocol import MindPrintSynapse
from mindprint.config import CacheScopeConfig
from mindprint.extractor import CacheScopeExtractor
from mindprint.proof.codec import decode_mindprint
from mindprint.proof.mindprint import generate_mindprint
from mindprint.proof.verify import verify_mindprint, VerificationResult

logger = logging.getLogger("mindprint.bittensor.validator")


# Validation prompts — diverse queries to test miner honesty
VALIDATION_PROMPTS = [
    "What is the capital of France?",
    "Explain photosynthesis in one paragraph.",
    "Write a Python function that checks if a number is prime.",
    "What are the three laws of thermodynamics?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "What is the difference between TCP and UDP?",
    "Name five programming languages and their primary use cases.",
    "What causes tides on Earth?",
]


class MindPrintValidator:
    """Bittensor validator that verifies miner inference via KV-cache geometry."""

    def __init__(
        self,
        model=None,
        tokenizer=None,
        model_id: str = "",
        config: Optional[CacheScopeConfig] = None,
        tolerance: float = 0.01,
        profile_r_threshold: float = 0.95,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.config = config or CacheScopeConfig()
        self.extractor = CacheScopeExtractor(self.config) if model else None
        self.tolerance = tolerance
        self.profile_r_threshold = profile_r_threshold

        # Track miner scores
        self.scores: Dict[int, float] = {}  # uid → running score

    def generate_reference(self, query: str, max_tokens: int = 256) -> Tuple[str, str]:
        """Run inference locally to generate a reference MindPrint.

        Returns (response_text, mindprint_b64).
        """
        import torch

        if self.model is None:
            raise RuntimeError("No model loaded — use model-signature mode instead")

        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": query}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = query

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
            )

        generated_ids = output.sequences[0][inputs["input_ids"].shape[-1]:]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        cache_cpu = self.extractor.cache_to_cpu(output.past_key_values)
        snapshot = self.extractor.extract(
            cache_cpu, n_tokens=output.sequences.shape[-1]
        )

        mp = generate_mindprint(
            snapshot,
            model_id=self.model_id,
            prompt=query,
            output=response_text,
        )

        from mindprint.proof.codec import encode_mindprint
        wire = encode_mindprint(mp)
        return response_text, base64.b64encode(wire).decode()

    def verify_miner_response(
        self,
        synapse: MindPrintSynapse,
        reference_b64: Optional[str] = None,
    ) -> VerificationResult:
        """Verify a miner's MindPrint.

        Args:
            synapse: Completed synapse with miner's response + MindPrint.
            reference_b64: Validator's reference MindPrint (for full verification).
                          If None, uses model-signature mode.
        """
        if not synapse.mindprint_b64:
            return VerificationResult(
                valid=False,
                mode="missing",
                content_match=False,
                structure_match=False,
                aggregate_match=False,
                profile_match=False,
                fingerprint_match=False,
                anomalies=["no_mindprint: miner did not provide proof"],
                confidence=0.0,
            )

        try:
            miner_bytes = base64.b64decode(synapse.mindprint_b64)
            miner_mp = decode_mindprint(miner_bytes, model_id=synapse.model_id)
        except Exception as e:
            return VerificationResult(
                valid=False,
                mode="decode_error",
                content_match=False,
                structure_match=False,
                aggregate_match=False,
                profile_match=False,
                fingerprint_match=False,
                anomalies=[f"decode_error: {e}"],
                confidence=0.0,
            )

        if reference_b64:
            # Full verification mode
            ref_bytes = base64.b64decode(reference_b64)
            ref_mp = decode_mindprint(ref_bytes, model_id=synapse.model_id)

            return verify_mindprint(
                miner=miner_mp,
                reference=ref_mp,
                tolerance=self.tolerance,
                profile_r_threshold=self.profile_r_threshold,
            )
        else:
            # Model-signature mode — verify structural consistency
            # without a content-specific reference
            return self._verify_model_signature(miner_mp)

    def _verify_model_signature(self, mp) -> VerificationResult:
        """Verify MindPrint is consistent with claimed model architecture.

        Checks:
          - Layer count matches known architecture
          - Feature values are within plausible ranges
          - Entropy profile shape is consistent with model family
        """
        anomalies = []

        # Known model layer counts
        KNOWN_LAYERS = {
            "Qwen/Qwen2.5-0.5B-Instruct": 24,
            "Qwen/Qwen2.5-7B-Instruct": 28,
            "Qwen/Qwen2.5-14B-Instruct": 48,
            "Qwen/Qwen2.5-32B-Instruct": 64,
            "meta-llama/Llama-3.1-8B-Instruct": 32,
            "mistralai/Mistral-7B-Instruct-v0.3": 32,
            "google/gemma-2-9b-it": 42,
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 22,
        }

        expected_layers = KNOWN_LAYERS.get(mp.model_id)
        structure_match = True
        if expected_layers and mp.n_layers != expected_layers:
            anomalies.append(
                f"layer_count_wrong: {mp.model_id} should have {expected_layers} "
                f"layers, got {mp.n_layers}"
            )
            structure_match = False

        # Plausibility checks on aggregate features
        aggregate_match = True
        if mp.norm <= 0:
            anomalies.append("norm_zero: impossible for real inference")
            aggregate_match = False
        if mp.key_entropy < 0.3 or mp.key_entropy > 1.0:
            anomalies.append(f"entropy_implausible: {mp.key_entropy:.4f} outside [0.3, 1.0]")
            aggregate_match = False
        if mp.key_rank < 1.0:
            anomalies.append(f"rank_implausible: {mp.key_rank:.4f} < 1.0")
            aggregate_match = False

        # Profile shape checks
        profile_match = True
        if len(mp.layer_rank_profile) != mp.n_extracted:
            anomalies.append("profile_length_mismatch")
            profile_match = False

        # Check profile isn't constant (would indicate synthetic/fake data)
        if len(mp.layer_rank_profile) > 1:
            rank_std = _std(mp.layer_rank_profile)
            if rank_std < 0.01:
                anomalies.append(f"rank_profile_constant: std={rank_std:.6f}")
                profile_match = False

        valid = structure_match and aggregate_match and profile_match
        confidence = 0.7 if valid else 0.0  # Model-sig mode caps at 0.7

        return VerificationResult(
            valid=valid,
            mode="model_signature",
            content_match=True,  # Not checked in this mode
            structure_match=structure_match,
            aggregate_match=aggregate_match,
            profile_match=profile_match,
            fingerprint_match=False,  # Can't check without reference
            anomalies=anomalies,
            confidence=confidence,
        )

    def score_miner(self, uid: int, result: VerificationResult) -> float:
        """Update miner score based on verification result.

        Uses exponential moving average to smooth scores over time.
        """
        alpha = 0.1  # EMA smoothing factor

        if result.valid:
            new_score = result.confidence
        else:
            new_score = 0.0

        current = self.scores.get(uid, 0.5)
        self.scores[uid] = alpha * new_score + (1 - alpha) * current
        return self.scores[uid]

    def run_validation_step(
        self,
        dendrite: bt.Dendrite,
        metagraph,
        netuid: int,
    ) -> Dict[int, float]:
        """Run one validation cycle: query miners, verify, score."""
        import random

        # Pick a random validation prompt
        prompt = random.choice(VALIDATION_PROMPTS)

        # Generate reference if we have a local model
        reference_b64 = None
        if self.model is not None:
            _, reference_b64 = self.generate_reference(prompt)

        # Query all miners
        synapse = MindPrintSynapse(
            query=prompt,
            model_id=self.model_id,
            max_tokens=256,
            temperature=0.0,
            seed=42,
        )

        axons = metagraph.axons
        responses = dendrite.query(axons=axons, synapse=synapse, timeout=60)

        # Verify each response
        scores = {}
        for uid, response in enumerate(responses):
            if response is None or not response.response:
                self.score_miner(uid, VerificationResult(
                    valid=False, mode="no_response",
                    content_match=False, structure_match=False,
                    aggregate_match=False, profile_match=False,
                    fingerprint_match=False,
                    anomalies=["no_response"],
                    confidence=0.0,
                ))
                continue

            result = self.verify_miner_response(response, reference_b64)
            score = self.score_miner(uid, result)
            scores[uid] = score

            logger.info(
                f"UID {uid}: valid={result.valid}, "
                f"confidence={result.confidence:.2f}, "
                f"score={score:.3f}, "
                f"anomalies={result.anomalies}"
            )

        return scores


def _std(values: list) -> float:
    """Standard deviation of a list."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def main():
    parser = argparse.ArgumentParser(description="CacheScope Proof of Mind Validator")
    parser.add_argument("--model", default=None, help="HuggingFace model (for full verification)")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--netuid", type=int, default=1)
    parser.add_argument("--wallet-name", default="validator")
    parser.add_argument("--hotkey", default="default")
    parser.add_argument("--network", default="test")
    parser.add_argument("--layer-stride", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--interval", type=int, default=60, help="Seconds between validation rounds")

    args = parser.parse_args()

    config = CacheScopeConfig(layer_stride=args.layer_stride)

    model, tokenizer = None, None
    if args.model:
        sys.path.insert(0, str(Path.home() / "KV-Experiments" / "code"))
        from gpu_utils import load_model
        logger.info(f"Loading reference model: {args.model}")
        model, tokenizer = load_model(args.model, quantize=args.quantize)

    validator = MindPrintValidator(
        model=model,
        tokenizer=tokenizer,
        model_id=args.model or "",
        config=config,
        tolerance=args.tolerance,
    )

    wallet = bt.Wallet(name=args.wallet_name, hotkey=args.hotkey)
    subtensor = bt.Subtensor(network=args.network)
    dendrite = bt.Dendrite(wallet=wallet)
    metagraph = subtensor.metagraph(netuid=args.netuid)

    logger.info(f"Validator running on netuid {args.netuid}")
    logger.info(f"Mode: {'full verification' if model else 'model-signature only'}")

    try:
        while True:
            metagraph.sync()
            scores = validator.run_validation_step(dendrite, metagraph, args.netuid)

            # Set weights on chain
            if scores:
                uids = list(scores.keys())
                weights = [scores[uid] for uid in uids]
                subtensor.set_weights(
                    netuid=args.netuid,
                    wallet=wallet,
                    uids=uids,
                    weights=weights,
                )

            time.sleep(args.interval)
    except KeyboardInterrupt:
        logger.info("Validator stopped")


if __name__ == "__main__":
    main()
