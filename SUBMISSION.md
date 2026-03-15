# Hackathon Submission — MindPrint: Proof of Mind

## Track: Sovereign Infrastructure (Bittensor Challenge)

## Summary

MindPrint is a verifiable inference protocol that proves AI models performed honest computation on decentralized infrastructure — without re-running inference or seeing the query content. It extracts a compact geometric fingerprint (a "MindPrint") from a model's KV-cache during inference. Same model + same input + honest computation produces a deterministic 121-byte proof.

Built for the Bittensor ecosystem, it enables validators to verify miner inference integrity at 17,000x compression compared to raw cache data — detecting model substitution, response caching, censorship routing, and computation shortcuts. Honest miners earn, dishonest miners earn nothing.

**This is a new build**, created during this hackathon. The underlying research (KV-cache geometric phenomenology) draws on 5 months of published work across (as of this event) three experimental campaigns and 7 model architectures.

## Threat Model

**What we protect against:**
- **Model substitution**: Provider runs a cheaper model while claiming a premium one. Detected via architecture-specific geometric signatures (layer count, rank/entropy profiles).
- **Censorship via routing**: Provider silently routes sensitive queries to a filtered fine-tune. Detected because censored models produce measurably different refusal geometry (Cohen's d = 0.58–2.05).
- **Response caching**: Provider serves cached responses without running inference. Detected via content-bound MindPrints — different inputs produce different geometry.
- **Computation shortcuts**: Provider skips layers or truncates attention. Detected via per-layer rank/entropy profiles revealing missing computation.

**What we do NOT protect against:**
- Adversarial MindPrint forgery by attackers with full knowledge of verification scheme
- Hardware-level attacks (modified GPU firmware)
- Computation privacy (we verify integrity, not confidentiality)
- Cross-hardware floating point variance (mitigated by tolerant verification mode, not eliminated)

**Trust boundaries:**
- No cryptographic proofs — verification is geometric, using SHA-256 only for fingerprinting and content binding
- Greedy decode is deterministic; sampled decode requires calibrated tolerance thresholds
- Miner and validator must agree on quantization scheme for full verification

Full threat model: [THREAT_MODEL.md](https://github.com/Liberation-Labs-THCoalition/MindPrint/blob/main/THREAT_MODEL.md)

## GitHub

https://github.com/Liberation-Labs-THCoalition/MindPrint

## Documentation

- [README](https://github.com/Liberation-Labs-THCoalition/MindPrint/blob/main/README.md) — Architecture, quick start, API reference
- [EXPLAIN.md](https://github.com/Liberation-Labs-THCoalition/MindPrint/blob/main/EXPLAIN.md) — Plain English explanation (2-minute read)
- [Threat Model](https://github.com/Liberation-Labs-THCoalition/MindPrint/blob/main/THREAT_MODEL.md) — Full security analysis with limitations
- [Demo Scripts](https://github.com/Liberation-Labs-THCoalition/MindPrint/tree/main/demo) — Interactive TUI, economic simulation, scripted demo

## How it touches Bittensor

- Custom `MindPrintSynapse` extending `bt.Synapse` for the miner-validator protocol
- `MindPrintMiner` using `bt.Axon` to serve inference with geometric attestation
- `MindPrintValidator` using `bt.Dendrite` to query miners and verify MindPrints
- Two verification modes: full (re-run inference) and model-signature (verify model identity without seeing query — privacy-preserving)
- EMA-based miner scoring compatible with Bittensor's weight-setting mechanism
- Live subnet (netuid 2) running on local Bittensor chain with funded wallets
- Full Yuma Consensus economic simulation: 41/41/18 emission split, honest miner earns τ2.46 vs cheater τ0.00 after 6 tempos (24,600x reward ratio)

## Current Limitations

1. No adversarial robustness testing against targeted MindPrint forgery
2. Cross-hardware tolerance thresholds not yet calibrated (A100 vs 3090 vs H100)
3. Layer sampling stride is fixed per session, not randomized per query
4. Model-signature mode uses static architecture lookup rather than learned signatures
5. Deception detection signals are strongest above 7B scale — smaller models produce narrower geometric variance

## Patent

Patent Pending — The methods described in this repository, including KV-cache geometric fingerprinting for inference verification, are the subject of a pending patent application. Liberation Labs / THCoalition.
