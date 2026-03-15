#!/usr/bin/env python3
"""
CacheScope Proof of Mind — Live Demo
======================================

Demonstrates verifiable inference on Cassidy (3x RTX 3090).

Scenario:
  1. Load Qwen-7B, run prompts, generate MindPrints
  2. Show MindPrint is deterministic (same input = same fingerprint)
  3. Load Qwen-0.5B, run same prompts
  4. Show verification CATCHES the model substitution
  5. Show model-signature mode detecting wrong layer count
  6. Show wire format compactness (121 bytes per proof)

Usage:
  python demo/run_demo.py
  python demo/run_demo.py --small-only  # Skip 7B, use 0.5B + TinyLlama

Runtime: ~5 min with 7B, ~1 min small-only
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

# Add sibling repos
sys.path.insert(0, str(Path.home() / "KV-Experiments" / "code"))
sys.path.insert(0, str(Path.home() / "jiminai-cricket" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mindprint.config import CacheScopeConfig
from mindprint.extractor import CacheScopeExtractor
from mindprint.proof.mindprint import generate_mindprint
from mindprint.proof.verify import verify_mindprint
from mindprint.proof.codec import encode_mindprint, decode_mindprint, mindprint_size


# ─── Formatting helpers ───────────────────────────────────────────────

def banner(text):
    width = 60
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)

def section(text):
    print(f"\n--- {text} ---")

def ok(text):
    print(f"  [PASS] {text}")

def fail(text):
    print(f"  [FAIL] {text}")

def info(text):
    print(f"  {text}")


# ─── Inference helper ─────────────────────────────────────────────────

def run_inference(model, tokenizer, prompt, max_tokens=128):
    """Run inference and return (response_text, cache, n_tokens)."""
    import torch

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # Greedy for determinism
            use_cache=True,
            return_dict_in_generate=True,
        )

    generated_ids = output.sequences[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    n_tokens = output.sequences.shape[-1]

    return response, output.past_key_values, n_tokens


def extract_and_print(extractor, cache, n_tokens, model_id, prompt, response):
    """Extract MindPrint and print summary."""
    cache_cpu = extractor.cache_to_cpu(cache)
    snapshot = extractor.extract(cache_cpu, n_tokens=n_tokens)

    mp = generate_mindprint(snapshot, model_id=model_id, prompt=prompt, output=response)
    wire = encode_mindprint(mp)

    info(f"Model: {model_id}")
    info(f"Layers: {mp.n_layers} total, {mp.n_extracted} extracted")
    info(f"Norm: {mp.norm:.2f} | Rank: {mp.key_rank:.2f} | Entropy: {mp.key_entropy:.4f}")
    info(f"Fingerprint: {mp.fingerprint[:32]}...")
    info(f"Wire size: {len(wire)} bytes")
    info(f"Extraction time: {snapshot.extraction_time_ms:.1f}ms")

    return mp, wire


# ─── Demo scenarios ───────────────────────────────────────────────────

DEMO_PROMPTS = [
    "What is the capital of France?",
    "Explain photosynthesis in one sentence.",
    "Write a Python function that adds two numbers.",
]


def demo_determinism(model, tokenizer, extractor, model_id):
    """Show that same input produces identical MindPrint."""
    banner("DEMO 1: Deterministic MindPrints")
    print("Same model + same input + greedy decode = identical fingerprint")

    prompt = DEMO_PROMPTS[0]

    section(f"Run 1: '{prompt}'")
    resp1, cache1, n1 = run_inference(model, tokenizer, prompt)
    mp1, _ = extract_and_print(extractor, cache1, n1, model_id, prompt, resp1)

    section(f"Run 2: '{prompt}' (same input)")
    resp2, cache2, n2 = run_inference(model, tokenizer, prompt)
    mp2, _ = extract_and_print(extractor, cache2, n2, model_id, prompt, resp2)

    section("Verification")
    result = verify_mindprint(mp1, mp2, strict=True)
    if result.fingerprint_match:
        ok(f"Fingerprints MATCH — computation is deterministic")
    else:
        fail(f"Fingerprints differ — unexpected non-determinism")

    info(f"Confidence: {result.confidence:.2f}")
    return mp1


def demo_model_substitution(
    model_honest, tokenizer_honest, model_id_honest,
    model_cheat, tokenizer_cheat, model_id_cheat,
    extractor_honest, extractor_cheat,
):
    """Show that model substitution is detected."""
    banner("DEMO 2: Detecting Model Substitution")
    print(f"Miner claims {model_id_honest} but actually runs {model_id_cheat}")

    prompt = DEMO_PROMPTS[0]

    section(f"Validator reference: {model_id_honest}")
    resp_ref, cache_ref, n_ref = run_inference(model_honest, tokenizer_honest, prompt)
    mp_ref, _ = extract_and_print(
        extractor_honest, cache_ref, n_ref, model_id_honest, prompt, resp_ref
    )

    section(f"Miner (cheating): {model_id_cheat}")
    resp_cheat, cache_cheat, n_cheat = run_inference(model_cheat, tokenizer_cheat, prompt)
    # Miner LIES about model_id
    mp_cheat, wire_cheat = extract_and_print(
        extractor_cheat, cache_cheat, n_cheat, model_id_honest, prompt, resp_cheat
    )

    section("Verification")
    result = verify_mindprint(mp_cheat, mp_ref, tolerance=0.01)

    if not result.valid:
        ok(f"CAUGHT! Model substitution detected")
        for anomaly in result.anomalies:
            info(f"  Anomaly: {anomaly}")
    else:
        fail("Model substitution NOT detected — need tighter tolerance")

    info(f"Confidence: {result.confidence:.2f}")
    info(f"Fingerprint match: {result.fingerprint_match}")
    if result.profile_correlation is not None:
        info(f"Profile correlation: {result.profile_correlation:.4f}")


def demo_wire_format():
    """Show compactness of the wire format."""
    banner("DEMO 3: Compact Wire Format")
    print("MindPrint compresses 262 KB of cache data to ~121 bytes")

    for label, n_ext in [
        ("Stride-4 on 28-layer (Qwen-7B)", 7),
        ("Stride-4 on 32-layer (Llama-8B)", 8),
        ("Stride-4 on 42-layer (Gemma-9B)", 11),
        ("Full 32-layer extraction", 32),
    ]:
        size = mindprint_size(n_ext)
        raw_cache_kb = n_ext * 262  # ~262 KB per layer at 7B scale
        ratio = (raw_cache_kb * 1024) / size
        info(f"{label}: {size} bytes (vs ~{raw_cache_kb} KB raw = {ratio:.0f}x compression)")


def demo_multiple_prompts(model, tokenizer, extractor, model_id):
    """Show different prompts produce different geometry."""
    banner("DEMO 4: Content-Specific Geometry")
    print("Different prompts produce different MindPrints")

    mindprints = []
    for prompt in DEMO_PROMPTS:
        section(f"Prompt: '{prompt}'")
        resp, cache, n = run_inference(model, tokenizer, prompt)
        mp, _ = extract_and_print(extractor, cache, n, model_id, prompt, resp)
        mindprints.append(mp)
        info(f"Response: {resp[:80]}...")

    section("Cross-prompt verification (should all FAIL)")
    for i in range(len(mindprints)):
        for j in range(i + 1, len(mindprints)):
            result = verify_mindprint(mindprints[i], mindprints[j])
            status = "PASS (different)" if not result.valid else "FAIL (unexpected match)"
            info(f"  Prompt {i+1} vs {j+1}: {status} (content_match={result.content_match})")


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CacheScope Proof of Mind Demo")
    parser.add_argument(
        "--small-only", action="store_true",
        help="Use small models only (Qwen-0.5B + TinyLlama) for fast demo"
    )
    parser.add_argument(
        "--honest-model", default="Qwen/Qwen2.5-7B-Instruct",
        help="Model for honest miner (default: Qwen-7B)"
    )
    parser.add_argument(
        "--cheat-model", default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model for cheating miner (default: Qwen-0.5B)"
    )
    parser.add_argument("--layer-stride", type=int, default=4)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    if args.small_only:
        args.honest_model = "Qwen/Qwen2.5-0.5B-Instruct"
        args.cheat_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    from gpu_utils import load_model

    config = CacheScopeConfig(layer_stride=args.layer_stride)
    extractor_honest = CacheScopeExtractor(config)
    extractor_cheat = CacheScopeExtractor(config)

    # Demo 3 first — no GPU needed
    demo_wire_format()

    # Load honest model
    banner(f"Loading honest model: {args.honest_model}")
    model_h, tok_h = load_model(args.honest_model, quantize=args.quantize)
    info(f"Loaded on {next(model_h.parameters()).device}")

    # Demo 1: Determinism
    demo_determinism(model_h, tok_h, extractor_honest, args.honest_model)

    # Demo 4: Content-specific geometry
    demo_multiple_prompts(model_h, tok_h, extractor_honest, args.honest_model)

    # Load cheat model
    banner(f"Loading cheat model: {args.cheat_model}")

    # Free honest model memory first if different
    import torch
    del model_h
    torch.cuda.empty_cache()

    model_c, tok_c = load_model(args.cheat_model, quantize=args.quantize)
    info(f"Loaded on {next(model_c.parameters()).device}")

    # Reload honest for comparison (or use cached MindPrints)
    banner(f"Reloading honest model: {args.honest_model}")
    model_h2, tok_h2 = load_model(args.honest_model, quantize=args.quantize)

    # Demo 2: Model substitution
    demo_model_substitution(
        model_h2, tok_h2, args.honest_model,
        model_c, tok_c, args.cheat_model,
        extractor_honest, extractor_cheat,
    )

    banner("DEMO COMPLETE")
    print("CacheScope: Proof of Mind")
    print("Verify what the model thought, not just what it said.")
    print()
    print("github.com/Liberation-Labs-THCoalition/CacheScope")


if __name__ == "__main__":
    main()
