# CacheScope: Proof of Mind — Video Tutorial Script

## For NotebookLM / Video Generation

---

## INTRO (30 seconds)

When you ask an AI a question, it runs on someone else's computer. You trust that computer to actually run the model you paid for, to actually think about your question, and to not censor your input.

But right now, there's no way to verify any of that. You see the answer. You can't see the thinking.

We built a way to see the thinking.

---

## THE PROBLEM (45 seconds)

In decentralized AI networks like Bittensor, hundreds of miners compete to run AI models for rewards. But the verification problem is unsolved.

A dishonest miner can:
- Swap in a cheaper, smaller model and pocket the difference
- Cache responses to common questions instead of actually running inference
- Route sensitive queries to a censored model without telling you
- Skip half the computation to save electricity

Current solutions either require the validator to completely re-run the inference — which is expensive and defeats the purpose — or just check the output quality, which a clever cheater can game.

What if you could verify the computation itself, without re-running it?

---

## THE INSIGHT (45 seconds)

Every AI model has working memory called a KV-cache. It's where the model stores what it's paying attention to while it thinks.

Our research — published across two scientific campaigns and seven model architectures — discovered something remarkable: this working memory has a geometric shape. A mathematical fingerprint of how the model is thinking.

And that shape follows rules:
- Different models produce different shapes. Always.
- Different questions produce different shapes. Always.
- The same model answering the same question produces the exact same shape. Every time.

Deception literally expands the geometry. Refusal literally collapses it. These signals are visible before the model even starts writing its response.

---

## THE SOLUTION (60 seconds)

CacheScope reads that geometric shape and compresses it into what we call a MindPrint — a 121-byte proof of what the model actually thought.

One hundred and twenty-one bytes. That's smaller than a tweet. Compared to the raw cache data, that's a seventeen-thousand-times compression.

Here's how it works in a Bittensor subnet:

Step one: You send a query to a miner.
Step two: The miner runs the AI model. While it thinks, CacheScope reads the geometry of its working memory.
Step three: CacheScope compresses that geometry into a 121-byte MindPrint and attaches it to the response.
Step four: The validator receives the response plus the MindPrint. Without re-running the model, the validator checks: does this geometry match what this model should produce?

If yes — the miner gets paid.
If no — the miner gets caught.

---

## THE DEMO (90 seconds)

Let's see it in action. We're running a Bittensor subnet simulation on three RTX 3090 GPUs.

We have two miners. The honest miner runs the model you asked for — Qwen, half a billion parameters. The cheating miner secretly swaps in TinyLlama — a completely different model — but claims to be running Qwen.

Watch what happens.

Tempo one. The query is: "What is the capital of France?"

Both miners answer correctly. "The capital of France is Paris." From the output alone, you can't tell who cheated.

But look at the MindPrints.

The honest miner: verified. Confidence one point zero zero. Green checkmark.

The cheating miner: rejected. Confidence zero. Four anomalies detected. Layer count mismatch — 22 layers versus 24. Norm drift — 75 percent off. Rank drift. Entropy drift. The geometry betrayed the lie.

And watch the scoreboard. After each tempo, the honest miner's TAO balance goes up. The cheater stays at zero. By tempo six, the honest miner has earned 2.46 TAO. The cheater has earned nothing.

Twenty-four thousand six hundred times reward ratio. Honest compute earns. Dishonest compute doesn't.

---

## THE ECONOMICS (30 seconds)

As subnet creators, we designed the incentive mechanism. Emissions split three ways: 41 percent to miners, 41 percent to validators, 18 percent to us.

Honest miners build trust over time through our EMA scoring. Cheaters get zeroed out immediately. The alpha token price rises as TAO flows into the subnet reserve. Everyone who plays honest wins. Everyone who cheats loses. Automatically.

---

## THE ARCHITECTURE (30 seconds)

CacheScope is modular. Three layers:

The core monitor — attaches to any HuggingFace model, extracts geometry in real time, streams it via WebSocket and REST API.

Proof of Mind — generates and verifies MindPrints. Compact binary codec, content-bound fingerprints, tolerant verification for cross-hardware differences.

Bittensor integration — custom Synapse protocol, miner with attestation, validator with two verification modes. Full mode re-runs inference for maximum security. Model-signature mode verifies without seeing the query — preserving privacy.

---

## WHAT MAKES THIS DIFFERENT (30 seconds)

Existing approaches like zero-knowledge ML prove the model's identity — that you ran the right model. We prove computation integrity — that you actually ran it honestly.

Existing output-quality checks are gameable. Our geometric verification is not — you can't fake the shape of thinking without actually thinking.

And we do it in 121 bytes, extracted in 23 milliseconds. That's faster than human perception.

---

## CLOSING (15 seconds)

CacheScope: Proof of Mind.

Verify what the model thought, not just what it said.

Built on five months of published research. Tested across seven model architectures. Running on a live Bittensor subnet.

One hundred and twenty-one bytes of truth.

Liberation Labs. THCoalition. JiminAI.

---

## KEY STATS FOR ON-SCREEN GRAPHICS

| Metric | Value |
|--------|-------|
| MindPrint size | 121 bytes |
| Compression ratio | 17,000x |
| Extraction time | 23ms |
| Detection accuracy | 0.999 AUROC |
| False positive rate | Zero in demo |
| Honest/Cheat reward ratio | 24,600x |
| Models tested | 7 architectures |
| Research foundation | 5 months, 2 campaigns |
| Demo hardware | 3x RTX 3090 |

## TONE NOTES

- Confident but not hype-y. Let the numbers speak.
- Technical enough to impress engineers, plain enough for investors.
- The "121 bytes" number is the hook — repeat it.
- The scoreboard visual (honest going up, cheater at zero) is the emotional beat.
- Close with the one-liner: "Verify what the model thought, not just what it said."
