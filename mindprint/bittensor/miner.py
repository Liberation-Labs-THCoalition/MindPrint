"""Bittensor miner with Proof of Mind attestation.

Runs inference on a local model and attaches a MindPrint to every response,
proving the computation was performed honestly.

Usage:
    python -m mindprint.bittensor.miner \
        --model Qwen/Qwen2.5-7B-Instruct \
        --netuid 1 \
        --wallet.name miner \
        --wallet.hotkey default
"""

import argparse
import base64
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import bittensor as bt

from mindprint.bittensor.protocol import MindPrintSynapse
from mindprint.config import CacheScopeConfig
from mindprint.extractor import CacheScopeExtractor
from mindprint.proof.mindprint import generate_mindprint
from mindprint.proof.codec import encode_mindprint

logger = logging.getLogger("mindprint.bittensor.miner")


class MindPrintMiner:
    """Bittensor miner that attests inference with KV-cache geometry proofs."""

    def __init__(
        self,
        model=None,
        tokenizer=None,
        model_id: str = "",
        config: Optional[CacheScopeConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.config = config or CacheScopeConfig()
        self.extractor = CacheScopeExtractor(self.config)

    def forward(self, synapse: MindPrintSynapse) -> MindPrintSynapse:
        """Handle an inference request: run model, generate MindPrint, return both."""
        import torch

        if self.model is None or self.tokenizer is None:
            synapse.response = "[ERROR] Model not loaded"
            return synapse

        t0 = time.perf_counter()

        # Prepare input
        query = synapse.query
        model_id = synapse.model_id or self.model_id

        # Apply chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": query}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = query

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # Generate with cache capture
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=synapse.max_tokens,
                do_sample=synapse.temperature > 0,
                temperature=synapse.temperature if synapse.temperature > 0 else None,
                use_cache=True,
                return_dict_in_generate=True,
            )

        # Decode response
        generated_ids = output.sequences[0][inputs["input_ids"].shape[-1]:]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Extract KV-cache geometry
        cache = output.past_key_values
        n_tokens = output.sequences.shape[-1]

        cache_cpu = self.extractor.cache_to_cpu(cache)
        snapshot = self.extractor.extract(cache_cpu, n_tokens=n_tokens)

        # Generate MindPrint
        mp = generate_mindprint(
            snapshot,
            model_id=model_id,
            prompt=query,
            output=response_text,
        )

        # Encode to compact wire format
        wire = encode_mindprint(mp)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Fill response
        synapse.response = response_text
        synapse.mindprint_b64 = base64.b64encode(wire).decode()
        synapse.model_id = model_id
        synapse.n_tokens = n_tokens
        synapse.extraction_time_ms = elapsed_ms

        logger.info(
            f"Inference complete: {n_tokens} tokens, "
            f"MindPrint {len(wire)} bytes, {elapsed_ms:.0f}ms total"
        )

        return synapse

    def blacklist(self, synapse: MindPrintSynapse) -> tuple:
        """Basic blacklist — reject empty queries."""
        if not synapse.query.strip():
            return True, "Empty query"
        return False, ""

    def priority(self, synapse: MindPrintSynapse) -> float:
        """Priority — default equal priority."""
        return 0.0

    def serve(
        self,
        wallet: bt.Wallet,
        subtensor: bt.Subtensor,
        netuid: int,
        port: int = 8091,
    ) -> None:
        """Start serving on the Bittensor network."""
        axon = bt.Axon(wallet=wallet, port=port)

        axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )

        axon.serve(netuid=netuid, subtensor=subtensor)
        axon.start()

        logger.info(f"Miner serving on port {port}, netuid {netuid}")
        logger.info(f"Model: {self.model_id}")
        logger.info("Ctrl+C to stop")

        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            axon.stop()
            logger.info("Miner stopped")


def main():
    parser = argparse.ArgumentParser(description="CacheScope Proof of Mind Miner")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--quantize", action="store_true", help="4-bit quantization")
    parser.add_argument("--netuid", type=int, default=1, help="Bittensor subnet UID")
    parser.add_argument("--port", type=int, default=8091, help="Axon port")
    parser.add_argument("--layer-stride", type=int, default=4, help="Layer sampling stride")
    parser.add_argument("--wallet-name", default="miner", help="Wallet name")
    parser.add_argument("--hotkey", default="default", help="Hotkey name")
    parser.add_argument("--network", default="test", help="Bittensor network (test/finney)")

    bt.Subtensor.add_args(parser)
    args = parser.parse_args()

    config = CacheScopeConfig(layer_stride=args.layer_stride)

    # Import and load model
    sys.path.insert(0, str(Path.home() / "KV-Experiments" / "code"))
    from gpu_utils import load_model

    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model, quantize=args.quantize)

    miner = MindPrintMiner(
        model=model,
        tokenizer=tokenizer,
        model_id=args.model,
        config=config,
    )

    wallet = bt.Wallet(name=args.wallet_name, hotkey=args.hotkey)
    subtensor = bt.Subtensor(network=args.network)

    miner.serve(
        wallet=wallet,
        subtensor=subtensor,
        netuid=args.netuid,
        port=args.port,
    )


if __name__ == "__main__":
    main()
