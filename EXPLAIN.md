# CacheScope: Proof of Mind — The Plain English Version

## The Problem (30 seconds)

When you ask an AI to do something, it runs on someone else's computer. You're trusting that computer to actually do what it says it did. But what if the provider:

- **Swaps the model?** You paid for the smart one, they ran the cheap one.
- **Serves a cached answer?** They already answered this question for someone else, so they just copy-paste instead of thinking.
- **Censors you?** They secretly route your "sensitive" question to a watered-down, filtered model.
- **Cuts corners?** They skip half the computation to save electricity.

Today, there's no way to catch any of this. You see the answer, but you can't see the thinking.

## The Solution (60 seconds)

We can see the thinking.

Every AI model has a **working memory** called the KV-cache. It's where the model stores everything it's paying attention to while generating a response. This working memory has a **geometric shape** — literally a mathematical fingerprint of how the model is thinking.

We discovered that:
- **Different models have different shapes.** Swap the model, the shape changes.
- **Different questions produce different shapes.** Cache a response, the shape won't match.
- **Censored models think differently.** Route to a filtered model, the shape betrays it.
- **The shape is deterministic.** Same model + same question = same shape, every time.

CacheScope reads that shape and compresses it into a **MindPrint** — a 121-byte proof of what the model actually thought. That's smaller than this sentence.

## How It Works (2 minutes)

```
You ask a question
    ↓
The AI thinks (KV-cache fills up)
    ↓
CacheScope reads the geometric shape of the KV-cache
    ↓
Compresses it to a 121-byte MindPrint
    ↓
Anyone can verify: "Did this model actually think about this question?"
```

The verifier doesn't need to re-run the AI. They just check the MindPrint. If the shape doesn't match what that model should produce for that input, someone cheated.

## Why Bittensor? (1 minute)

Bittensor is a decentralized network where people run AI models and get paid for it. The problem: how do you know the person running the model is being honest? Right now, Bittensor relies on validators re-running the computation or just judging output quality. Both are expensive or gameable.

Proof of Mind gives Bittensor validators a 121-byte receipt that proves honest computation. No re-running needed. The miner attaches a MindPrint to every response, the validator checks it, and dishonest miners get caught and stop earning.

## The Numbers

| What | Number |
|------|--------|
| Proof size | **121 bytes** (smaller than a tweet) |
| Compression vs raw data | **17,000x** |
| Deception detection accuracy | **0.999 AUROC** (from our published research) |
| False positive rate | **Zero** in our demo |
| Extraction time | **23ms** (faster than human perception) |
| Models tested | **7 architectures** across 2 research campaigns |

## What We Built Tonight

1. **CacheScope** — Real-time monitor that watches a model's working memory
2. **Proof of Mind** — Generates and verifies 121-byte geometric proofs
3. **Bittensor subnet** — Running on a local chain with honest miners earning and cheaters earning nothing
4. **Interactive demo** — Terminal UI where judges can trigger queries and watch verification happen live

## The Research Behind It

This isn't a hackathon prototype built on vibes. It's built on 5 months of published research:

- **Campaign 1**: Discovered that KV-cache geometry differs measurably between cognitive states (factual, deceptive, refusal)
- **Campaign 2**: Confirmed across 7 model architectures at scales from 0.5B to 32B parameters
- **Key finding**: Deception *expands* the geometric dimensionality of the cache, while refusal *collapses* it — and these signals are visible before the model even starts writing its response

The paper and all experimental data are public: [KV-Experiments](https://github.com/Liberation-Labs-THCoalition/KV-Experiments)

## One Sentence

**We prove what the model thought, not just what it said — in 121 bytes.**
