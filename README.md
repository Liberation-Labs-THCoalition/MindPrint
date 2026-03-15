# MindPrint: Proof of Mind

**Verifiable inference for sovereign compute — prove what the model thought, not just what it said.**

MindPrint is a verification protocol that generates compact geometric fingerprints of AI model inference, proving honest computation without re-running the model or seeing the query content. Built for the [Bittensor](https://bittensor.com/) network's Sovereign Infrastructure track.

## The Problem

In decentralized compute networks, you trust someone else's GPU to run your AI model. They can swap models, cache responses, censor queries, or skip computation — and you'd never know from the output alone.

## The Solution

Every AI model has working memory (KV-cache) with a measurable **geometric shape**. Same model + same input = same shape. Always. MindPrint reads that shape and compresses it into a **121-byte proof** — a MindPrint — that validators can check without re-running inference.

| Metric | Value |
|--------|-------|
| Proof size | **121 bytes** |
| Compression | **17,000x** vs raw cache |
| Extraction time | **23ms** |
| Detection accuracy | **0.999 AUROC** |
| Honest/Cheat reward ratio | **24,600x** in demo |

## How It Touches Bittensor

- Custom `MindPrintSynapse` extending `bt.Synapse`
- `MindPrintMiner` serving inference with geometric attestation via `bt.Axon`
- `MindPrintValidator` verifying MindPrints via `bt.Dendrite`
- EMA-based scoring compatible with Yuma Consensus weight-setting
- Live subnet (netuid 2) on local Bittensor chain
- Full economic simulation: 41% miner / 41% validator / 18% creator emissions

### Two Verification Modes

| Mode | Proves | Cost | Privacy |
|------|--------|------|---------|
| **Full** | Exact computation match | Validator re-runs inference | Validator sees query |
| **Model-signature** | Correct model was used | No GPU needed | Query stays private |

## Threat Model

**Protects against:**
- **Model substitution** — geometry is architecture-specific
- **Response caching** — MindPrints are content-bound
- **Censorship routing** — filtered models have different geometry
- **Computation shortcuts** — per-layer profiles reveal missing layers

**Does NOT protect against:**
- Adversarial forgery by attackers with full scheme knowledge
- Hardware-level attacks
- Cross-hardware float variance (mitigated by tolerance mode)

Full analysis: [THREAT_MODEL.md](THREAT_MODEL.md)

## Demo

```bash
# Economic simulation — honest miners earn, cheaters earn zero
python demo/subnet_full.py --tempos 6

# Interactive TUI — press keys, watch verification happen
python demo/tui.py

# Scripted 4-scenario demo
python demo/run_demo.py --small-only
```

## Architecture

```
mindprint/
  proof/
    mindprint.py    — MindPrint generation, content binding, SHA-256 fingerprint
    verify.py       — Exact + tolerant verification, Pearson profile correlation
    codec.py        — Compact binary wire format (121 bytes)
  bittensor/
    protocol.py     — MindPrintSynapse (extends bt.Synapse)
    miner.py        — Inference + attestation via bt.Axon
    validator.py    — Verification + scoring via bt.Dendrite
  snapshot.py       — Geometric observation data model (12 per-layer features)
```

## Research Foundation

Built on 5 months of published KV-cache geometric phenomenology:
- [KV-Experiments](https://github.com/Liberation-Labs-THCoalition/KV-Experiments) — 2 campaigns, 7 model architectures
- [CacheScope](https://github.com/Liberation-Labs-THCoalition/CacheScope) — Real-time monitoring backend
- [Cricket](https://github.com/Liberation-Labs-THCoalition/jiminai-cricket) — 0.999 AUROC deception classifier

## Project Status

**New build** — created during the Funding the Commons hackathon, March 14-15, 2026. This is the Bittensor Sovereign Infrastructure challenge submission.

## Team

Liberation Labs / THCoalition / JiminAI

## License

MIT
