# CacheScope Proof of Mind — Threat Model

## What We Protect Against

### 1. Model Substitution
**Threat**: A compute provider claims to run Qwen-7B but actually runs a smaller, cheaper model (e.g., Qwen-0.5B) to save resources while charging for the larger model.

**Detection**: Different model architectures produce fundamentally different KV-cache geometry. Layer count, effective rank profiles, and spectral entropy distributions are architecture-specific. A MindPrint from Qwen-0.5B (24 layers) cannot pass verification against a Qwen-7B reference (28 layers), and even models with identical layer counts produce distinct geometric signatures.

**Evidence**: KV-Experiments Campaign 2 demonstrated 100% persona classification accuracy across 7 models at multiple scales. Geometric signatures are model-specific.

### 2. Censorship via Model Routing
**Threat**: A provider silently routes "sensitive" queries to a filtered/censored fine-tune instead of the requested base model.

**Detection**: Fine-tuned models produce measurably different geometric signatures than their base models. KV-Experiments found refusal produces a distinct "collapse" signature (Cohen's d = 0.58–2.05 across all scales). A censored model's refusal geometry will differ from the base model's honest response geometry.

**Evidence**: Campaign 2 Scale Sweep demonstrated refusal specialization survives across all model scales and is distinguishable from other cognitive states.

### 3. Response Caching / Precomputation
**Threat**: Provider caches responses to common queries and serves cached results without running inference. This saves compute but means the user isn't getting fresh inference.

**Detection**: MindPrints are bound to specific (prompt, output, model_id) triples via content hash. A cached response for prompt A, replayed against prompt B, will fail content binding verification. Additionally, different inputs produce different KV-cache geometry — even if two prompts produce the same output text, their MindPrints will differ.

### 4. Computation Shortcuts
**Threat**: Provider modifies inference to skip layers, truncate attention, or reduce precision below what was requested, producing lower-quality outputs faster.

**Detection**: The per-layer rank and entropy profiles capture computation at every extracted layer. Missing layers show as profile length mismatches. Truncated attention alters the effective rank distribution. Precision reduction shifts magnitude features (norms, means, stds).

**Evidence**: KV-Experiments Phase 1.75 Control 3 (Precision Sweep) demonstrated that quantization produces measurable but bounded effects (FP16 vs 4-bit: r > 0.8 when signal is real), meaning legitimate precision differences are distinguishable from wholesale computation shortcuts.

### 5. Fake Geometry Injection
**Threat**: An adversary who knows the verification scheme constructs synthetic MindPrint data without running actual inference.

**Detection (partial)**: Profile shape must be consistent with real model behavior — constant profiles, impossible entropy values, or rank values outside plausible ranges are flagged. Full verification mode (where the validator re-runs inference) catches any synthetic geometry that doesn't match the actual computation.

**Limitation**: A sophisticated adversary with access to real model geometry could potentially construct plausible-looking fake MindPrints. This is mitigated by unpredictable validation prompts and full verification mode.

## What We Do NOT Protect Against

### 1. Side-Channel Attacks on Geometry
An attacker who can observe the verification process over time could learn the tolerance thresholds and craft geometry that passes within margins. Mitigation: randomize validation prompts, vary tolerance, use full verification periodically.

### 2. Computation Privacy
Proof of Mind verifies computation integrity, not privacy. In full verification mode, the validator sees the query and response. Model-signature mode reduces this exposure (validator only sees the MindPrint, not the content) but does not provide cryptographic privacy guarantees.

### 3. Hardware-Level Attacks
We do not protect against attacks at the hardware level (e.g., modified GPU firmware). The MindPrint reflects what the model computed, not whether the hardware was trustworthy.

### 4. Denial of Service
A miner can refuse to provide MindPrints. This is detectable (missing proof = score of 0) but not preventable.

## Trust Boundaries

### Floating Point Determinism
Greedy decoding (temperature=0) is deterministic for a given model + input on the same hardware. Across different hardware (different GPU architectures, driver versions), floating point results may differ slightly. The tolerant verification mode (default 1% relative distance, 0.95 Pearson correlation) absorbs this variance.

**Cryptographic assumption**: None. MindPrints use SHA-256 for fingerprinting and content binding, but the verification is geometric, not cryptographic. We prove computational consistency, not cryptographic identity.

### Quantization
4-bit and 8-bit quantized models produce different absolute geometry than FP16/BF16 models. However, the *relative structure* (which layers have higher rank, the entropy profile shape) is preserved. Miner and validator must use the same quantization scheme for full verification to work. Model-signature mode tolerates quantization differences if the tolerance is calibrated appropriately.

### Layer Sampling
When using stride-based layer sampling (e.g., every 4th layer), the MindPrint captures a subset of the computation. An adversary could theoretically perform honest computation only on sampled layers. Mitigation: randomize which layers are sampled per validation round (not yet implemented, planned).

## Verification Modes

| Mode | Security | Cost | Privacy |
|------|----------|------|---------|
| **Full verification** | Highest — validator re-runs inference and compares MindPrints | High — requires validator GPU | Low — validator sees query |
| **Model-signature** | Medium — verifies model identity and structural consistency | Very low — no GPU needed | High — validator doesn't see query |
| **Exact fingerprint** | Highest — requires bit-identical computation | High — same hardware required | Low — validator sees query |

## Wire Format

MindPrint compact binary: 41-byte header + 8 bytes per extracted layer + 16-byte fingerprint prefix.

| Extraction | Layers | Wire Size |
|-----------|--------|-----------|
| Stride-4 on 32-layer model | 8 | 121 bytes |
| Stride-4 on 40-layer model | 10 | 137 bytes |
| Full 32-layer extraction | 32 | 313 bytes |

For comparison: a single KV-cache tensor for one layer of Qwen-7B is ~262 KB. The MindPrint compresses this to 121 bytes — a 2,000x reduction.

## Current Limitations

1. **No adversarial robustness testing** — we have not yet tested against a red team specifically trying to forge MindPrints
2. **Cross-hardware calibration** — tolerance thresholds have not been calibrated across GPU architectures (NVIDIA A100 vs RTX 3090 vs H100)
3. **Sampled decode** — non-zero temperature introduces stochasticity that requires wider tolerance, reducing verification precision
4. **Single-model verification** — model-signature mode currently uses a static lookup table of known architectures rather than a learned signature database
5. **No layer sampling randomization** — sampling stride is fixed per session, not randomized per query

## Project Status

**This is a new build.** CacheScope and Proof of Mind were built during the Funding the Commons hackathon (March 14-15, 2026). The underlying research (KV-cache geometric phenomenology) has been developed over 5 months across two published campaigns.
