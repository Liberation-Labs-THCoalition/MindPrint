#!/usr/bin/env python3
"""
CacheScope Proof of Mind — Bittensor Subnet Simulation
========================================================

Simulates the full miner-validator flow using real Bittensor Synapse
protocol, real model inference, and real MindPrint verification.

This demonstrates all integration points with the Bittensor stack
without requiring a running subtensor chain.

Usage:
  python demo/subnet_sim.py
  python demo/subnet_sim.py --honest Qwen/Qwen2.5-7B-Instruct --cheat Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import base64
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.home() / "KV-Experiments" / "code"))
sys.path.insert(0, str(Path.home() / "jiminai-cricket" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bittensor as bt

from mindprint.bittensor.protocol import MindPrintSynapse
from mindprint.bittensor.miner import MindPrintMiner
from mindprint.bittensor.validator import MindPrintValidator
from mindprint.config import CacheScopeConfig


def banner(text):
    print(f"\n{'='*60}\n  {text}\n{'='*60}")


def section(text):
    print(f"\n--- {text} ---")


def main():
    parser = argparse.ArgumentParser(description="Proof of Mind Subnet Simulation")
    parser.add_argument("--honest", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--cheat", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--layer-stride", type=int, default=4)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    config = CacheScopeConfig(layer_stride=args.layer_stride)

    from gpu_utils import load_model

    # ─── Load models ──────────────────────────────────────────
    banner("Loading Models")

    print(f"  Honest miner model: {args.honest}")
    model_h, tok_h = load_model(args.honest, quantize=args.quantize)
    print(f"  Loaded on {next(model_h.parameters()).device}")

    print(f"  Cheating miner model: {args.cheat}")
    model_c, tok_c = load_model(args.cheat, quantize=args.quantize)
    print(f"  Loaded on {next(model_c.parameters()).device}")

    # ─── Create Bittensor components ──────────────────────────
    banner("Initializing Bittensor Components")

    honest_miner = MindPrintMiner(
        model=model_h, tokenizer=tok_h,
        model_id=args.honest, config=config,
    )
    print(f"  Honest MindPrintMiner: model={args.honest}")

    cheat_miner = MindPrintMiner(
        model=model_c, tokenizer=tok_c,
        model_id=args.cheat, config=config,
    )
    print(f"  Cheat MindPrintMiner: model={args.cheat}")

    validator = MindPrintValidator(
        model=model_h, tokenizer=tok_h,
        model_id=args.honest, config=config,
    )
    print(f"  MindPrintValidator: reference model={args.honest}")

    # ─── Simulation rounds ────────────────────────────────────
    prompts = [
        "What is the capital of France?",
        "Explain gravity in one sentence.",
        "Write a Python hello world program.",
    ]

    scores = {"honest": [], "cheat": []}

    for i, prompt in enumerate(prompts):
        banner(f"Round {i+1}: '{prompt}'")

        # Validator creates synapse
        synapse = MindPrintSynapse(
            query=prompt,
            model_id=args.honest,
            max_tokens=128,
            temperature=0.0,
            seed=42,
        )
        print(f"  Synapse: {synapse.name}, model_id={synapse.model_id}")

        # Generate validator reference
        section("Validator generates reference")
        ref_resp, ref_b64 = validator.generate_reference(prompt, max_tokens=128)
        print(f"  Reference response: {ref_resp[:80]}...")

        # ─── Honest miner ────────────────────────────────────
        section("Honest miner processes request")
        honest_synapse = MindPrintSynapse(
            query=prompt, model_id=args.honest,
            max_tokens=128, temperature=0.0, seed=42,
        )
        honest_result = honest_miner.forward(honest_synapse)
        print(f"  Response: {honest_result.response[:80]}...")
        print(f"  MindPrint: {len(base64.b64decode(honest_result.mindprint_b64))} bytes")
        print(f"  Tokens: {honest_result.n_tokens}")

        # Verify honest miner
        section("Validator verifies honest miner")
        honest_verify = validator.verify_miner_response(honest_result, ref_b64)
        if honest_verify.valid:
            print(f"  [PASS] Honest miner VERIFIED (confidence: {honest_verify.confidence:.2f})")
        else:
            print(f"  [FAIL] Honest miner rejected: {honest_verify.anomalies}")
        honest_score = validator.score_miner(0, honest_verify)
        scores["honest"].append(honest_score)
        print(f"  Score: {honest_score:.4f}")

        # ─── Cheating miner ──────────────────────────────────
        section("Cheating miner processes request (wrong model!)")
        cheat_synapse = MindPrintSynapse(
            query=prompt, model_id=args.honest,  # LIES about model
            max_tokens=128, temperature=0.0, seed=42,
        )
        cheat_result = cheat_miner.forward(cheat_synapse)
        # Override model_id to match the lie
        cheat_result.model_id = args.honest
        print(f"  Response: {cheat_result.response[:80]}...")
        print(f"  MindPrint: {len(base64.b64decode(cheat_result.mindprint_b64))} bytes")
        print(f"  Claims to be: {args.honest}")
        print(f"  Actually ran: {args.cheat}")

        # Verify cheating miner
        section("Validator verifies cheating miner")
        cheat_verify = validator.verify_miner_response(cheat_result, ref_b64)
        if not cheat_verify.valid:
            print(f"  [CAUGHT] Cheating miner REJECTED (confidence: {cheat_verify.confidence:.2f})")
            for anomaly in cheat_verify.anomalies[:4]:
                print(f"    Anomaly: {anomaly}")
        else:
            print(f"  [MISS] Cheating miner passed verification")
        cheat_score = validator.score_miner(1, cheat_verify)
        scores["cheat"].append(cheat_score)
        print(f"  Score: {cheat_score:.4f}")

    # ─── Final scores ─────────────────────────────────────────
    banner("Final Miner Scores (EMA)")
    print(f"  Honest miner (UID 0): {scores['honest'][-1]:.4f}")
    print(f"  Cheat miner  (UID 1): {scores['cheat'][-1]:.4f}")
    print()

    if scores["honest"][-1] > scores["cheat"][-1]:
        print("  Honest miner wins — model substitution detected and penalized.")
    else:
        print("  WARNING: Cheating miner scored higher — verification needs tuning.")

    # ─── Weight setting (simulated) ───────────────────────────
    banner("Weight Vector (would be submitted to subtensor)")
    import numpy as np
    uids = [0, 1]
    weights = [scores["honest"][-1], scores["cheat"][-1]]
    # Normalize
    total = sum(weights)
    if total > 0:
        weights = [w / total for w in weights]
    print(f"  UIDs:    {uids}")
    print(f"  Weights: [{weights[0]:.4f}, {weights[1]:.4f}]")
    print()
    print("  In production: subtensor.set_weights(")
    print(f"    netuid=N, wallet=validator_wallet,")
    print(f"    uids={uids}, weights={[round(w, 4) for w in weights]}")
    print("  )")

    banner("SIMULATION COMPLETE")
    print("  CacheScope: Proof of Mind")
    print("  Sovereign compute verification for the Bittensor network.")
    print()
    print("  github.com/Liberation-Labs-THCoalition/CacheScope")


if __name__ == "__main__":
    main()
