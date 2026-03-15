#!/usr/bin/env python3
"""
CacheScope Proof of Mind — Full Subnet Economic Simulation
=============================================================

Simulates multiple tempos of a Proof of Mind subnet with full
Bittensor economic mechanics:
  - Emissions split: 41% miner, 41% validator, 18% creator
  - EMA bond accumulation for consistent validators
  - Stake-weighted consensus
  - TAO accumulation over time

Shows honest miners accumulating wealth while cheaters go to zero.

Usage:
  python demo/subnet_full.py
  python demo/subnet_full.py --tempos 10 --honest Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import base64
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path.home() / "KV-Experiments" / "code"))
sys.path.insert(0, str(Path.home() / "jiminai-cricket" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mindprint.bittensor.protocol import MindPrintSynapse
from mindprint.bittensor.miner import MindPrintMiner
from mindprint.bittensor.validator import MindPrintValidator, VALIDATION_PROMPTS
from mindprint.config import CacheScopeConfig


# ─── Subnet Economics Engine ──────────────────────────────────────────

EMISSION_PER_TEMPO = 1.0  # τ1.0 emitted per tempo
MINER_SHARE = 0.41
VALIDATOR_SHARE = 0.41
CREATOR_SHARE = 0.18
EMA_ALPHA = 0.1  # Bond smoothing factor
KAPPA = 0.5  # Clipping threshold


@dataclass
class MinerState:
    uid: int
    name: str
    is_honest: bool
    tao_balance: float = 0.0
    cumulative_reward: float = 0.0
    incentive_score: float = 0.0
    trust: float = 0.0


@dataclass
class ValidatorState:
    uid: int
    name: str
    stake: float = 100.0
    tao_balance: float = 0.0
    cumulative_dividends: float = 0.0
    bonds: Dict[int, float] = field(default_factory=dict)  # miner_uid → bond


@dataclass
class SubnetState:
    netuid: int = 1
    creator_tao: float = 0.0
    cumulative_creator: float = 0.0
    tempo: int = 0
    alpha_reserve: float = 10000.0
    tao_reserve: float = 100.0

    @property
    def alpha_price(self) -> float:
        return self.tao_reserve / self.alpha_reserve if self.alpha_reserve > 0 else 0


def run_tempo(
    subnet: SubnetState,
    miners: List[MinerState],
    validators: List[ValidatorState],
    weights: Dict[int, Dict[int, float]],  # validator_uid → {miner_uid → weight}
):
    """Run one tempo of Yuma Consensus economics."""
    subnet.tempo += 1

    total_emission = EMISSION_PER_TEMPO
    miner_pool = total_emission * MINER_SHARE
    validator_pool = total_emission * VALIDATOR_SHARE
    creator_pool = total_emission * CREATOR_SHARE

    # Creator cut
    subnet.creator_tao += creator_pool
    subnet.cumulative_creator += creator_pool
    subnet.tao_reserve += creator_pool * 0.5  # Half goes to reserves

    # Aggregate weights (stake-weighted)
    total_stake = sum(v.stake for v in validators)
    if total_stake == 0:
        return

    # Compute consensus weight per miner
    miner_rankings = {}
    for m in miners:
        rank = 0.0
        for v in validators:
            w = weights.get(v.uid, {}).get(m.uid, 0.0)
            rank += (v.stake / total_stake) * w
        miner_rankings[m.uid] = rank

    total_rank = sum(miner_rankings.values())
    if total_rank == 0:
        return

    # Miner emissions (proportional to rank)
    for m in miners:
        share = miner_rankings[m.uid] / total_rank
        emission = miner_pool * share
        m.tao_balance += emission
        m.cumulative_reward += emission
        m.incentive_score = share
        m.trust = min(1.0, m.trust + 0.1) if share > 0.1 else max(0.0, m.trust - 0.1)

    # Validator bonds and dividends (EMA)
    for v in validators:
        total_dividend = 0.0
        for m in miners:
            w = weights.get(v.uid, {}).get(m.uid, 0.0)
            # Instant bond
            denom = sum(
                (vv.stake * weights.get(vv.uid, {}).get(m.uid, 0.0))
                for vv in validators
            )
            instant_bond = (v.stake * w) / denom if denom > 0 else 0.0

            # EMA smoothing
            prev_bond = v.bonds.get(m.uid, 0.0)
            new_bond = EMA_ALPHA * instant_bond + (1 - EMA_ALPHA) * prev_bond
            v.bonds[m.uid] = new_bond

            # Dividend = bond × miner emission share
            miner_emission_share = miner_rankings[m.uid] / total_rank
            dividend = validator_pool * new_bond * miner_emission_share
            total_dividend += dividend

        v.tao_balance += total_dividend
        v.cumulative_dividends += total_dividend


# ─── Display ──────────────────────────────────────────────────────────

def print_header():
    print()
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│         CacheScope: Proof of Mind — Subnet Economics           │")
    print("│              Sovereign Compute Verification                     │")
    print("└─────────────────────────────────────────────────────────────────┘")


def print_tempo_header(tempo: int, prompt: str):
    print(f"\n{'━'*65}")
    print(f"  TEMPO {tempo}  │  Query: \"{prompt[:45]}{'...' if len(prompt) > 45 else ''}\"")
    print(f"{'━'*65}")


def print_verification(name: str, is_honest: bool, result, response: str):
    status = "\033[92m✓ VERIFIED\033[0m" if result.valid else "\033[91m✗ REJECTED\033[0m"
    print(f"\n  {name}: {status}  (confidence: {result.confidence:.2f})")
    print(f"    Response: {response[:60]}...")
    if result.anomalies:
        for a in result.anomalies[:3]:
            print(f"    \033[91m⚠ {a[:55]}\033[0m")


def print_scoreboard(subnet, miners, validators):
    print(f"\n  ┌{'─'*61}┐")
    print(f"  │{'SCOREBOARD':^61}│")
    print(f"  ├{'─'*20}┬{'─'*12}┬{'─'*14}┬{'─'*12}┤")
    print(f"  │{'Participant':^20}│{'Balance':^12}│{'Cumulative':^14}│{'Incentive':^12}│")
    print(f"  ├{'─'*20}┼{'─'*12}┼{'─'*14}┼{'─'*12}┤")

    # Creator
    print(f"  │{'Creator (us)':^20}│{subnet.creator_tao:^12.4f}│{subnet.cumulative_creator:^14.4f}│{'18%':^12}│")

    # Validators
    for v in validators:
        print(f"  │{v.name:^20}│{v.tao_balance:^12.4f}│{v.cumulative_dividends:^14.4f}│{'41%':^12}│")

    # Miners
    for m in miners:
        status = "✓" if m.is_honest else "✗"
        color = "\033[92m" if m.is_honest else "\033[91m"
        reset = "\033[0m"
        name = f"{color}{status} {m.name}{reset}"
        # For column alignment, use raw name length
        raw_name = f"{status} {m.name}"
        pad = 20 - len(raw_name)
        lpad = pad // 2
        rpad = pad - lpad
        print(f"  │{' '*lpad}{name}{' '*rpad}│{m.tao_balance:^12.4f}│{m.cumulative_reward:^14.4f}│{m.incentive_score:^12.2f}│")

    print(f"  └{'─'*20}┴{'─'*12}┴{'─'*14}┴{'─'*12}┘")

    # Alpha token
    print(f"\n  Subnet alpha price: τ{subnet.alpha_price:.6f} per α  (reserve: τ{subnet.tao_reserve:.2f} / α{subnet.alpha_reserve:.0f})")


def print_final_summary(subnet, miners, validators, n_tempos):
    print(f"\n{'='*65}")
    print(f"  FINAL RESULTS AFTER {n_tempos} TEMPOS")
    print(f"{'='*65}")

    honest = [m for m in miners if m.is_honest]
    cheaters = [m for m in miners if not m.is_honest]

    print(f"\n  Creator earnings:  τ{subnet.cumulative_creator:.4f}")
    for v in validators:
        print(f"  Validator earnings: τ{v.cumulative_dividends:.4f}")

    print()
    for m in honest:
        print(f"  \033[92m✓ {m.name}: τ{m.cumulative_reward:.4f}  (trust: {m.trust:.2f})\033[0m")
    for m in cheaters:
        print(f"  \033[91m✗ {m.name}: τ{m.cumulative_reward:.4f}  (trust: {m.trust:.2f})\033[0m")

    if honest and cheaters:
        ratio = honest[0].cumulative_reward / max(cheaters[0].cumulative_reward, 0.0001)
        print(f"\n  Honest/Cheat reward ratio: {ratio:.1f}x")

    print(f"\n  Total emissions: τ{n_tempos * EMISSION_PER_TEMPO:.4f}")
    total_distributed = (
        subnet.cumulative_creator
        + sum(v.cumulative_dividends for v in validators)
        + sum(m.cumulative_reward for m in miners)
    )
    print(f"  Total distributed: τ{total_distributed:.4f}")

    print(f"\n  The economic incentive is clear: honest compute earns,")
    print(f"  dishonest compute doesn't. Proof of Mind makes this enforceable.")


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--honest", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--cheat", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--tempos", type=int, default=6, help="Number of tempos to simulate")
    parser.add_argument("--layer-stride", type=int, default=4)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    config = CacheScopeConfig(layer_stride=args.layer_stride)

    from gpu_utils import load_model
    import random

    print_header()

    # Load models
    print(f"\n  Loading honest model: {args.honest}...")
    model_h, tok_h = load_model(args.honest, quantize=args.quantize)

    print(f"  Loading cheat model: {args.cheat}...")
    model_c, tok_c = load_model(args.cheat, quantize=args.quantize)

    # Create Bittensor components
    honest_miner_agent = MindPrintMiner(
        model=model_h, tokenizer=tok_h, model_id=args.honest, config=config
    )
    cheat_miner_agent = MindPrintMiner(
        model=model_c, tokenizer=tok_c, model_id=args.cheat, config=config
    )
    validator_agent = MindPrintValidator(
        model=model_h, tokenizer=tok_h, model_id=args.honest, config=config
    )

    # Initialize economic state
    subnet = SubnetState()
    miners = [
        MinerState(uid=0, name="Honest", is_honest=True),
        MinerState(uid=1, name="Cheater", is_honest=False),
    ]
    validators = [
        ValidatorState(uid=0, name="Validator (us)", stake=100.0),
    ]

    prompts = VALIDATION_PROMPTS[:args.tempos]
    if len(prompts) < args.tempos:
        prompts = prompts * (args.tempos // len(prompts) + 1)
    prompts = prompts[:args.tempos]

    # Run tempos
    for tempo_idx in range(args.tempos):
        prompt = prompts[tempo_idx]
        print_tempo_header(tempo_idx + 1, prompt)

        # Validator generates reference
        ref_resp, ref_b64 = validator_agent.generate_reference(prompt, max_tokens=128)

        # Honest miner
        h_synapse = MindPrintSynapse(
            query=prompt, model_id=args.honest,
            max_tokens=128, temperature=0.0, seed=42,
        )
        h_result = honest_miner_agent.forward(h_synapse)
        h_verify = validator_agent.verify_miner_response(h_result, ref_b64)
        print_verification("Honest miner", True, h_verify, h_result.response)

        # Cheating miner
        c_synapse = MindPrintSynapse(
            query=prompt, model_id=args.honest,
            max_tokens=128, temperature=0.0, seed=42,
        )
        c_result = cheat_miner_agent.forward(c_synapse)
        c_result.model_id = args.honest  # The lie
        c_verify = validator_agent.verify_miner_response(c_result, ref_b64)
        print_verification("Cheating miner", False, c_verify, c_result.response)

        # Compute weights from verification
        h_weight = h_verify.confidence if h_verify.valid else 0.0
        c_weight = c_verify.confidence if c_verify.valid else 0.0

        weights = {
            0: {0: h_weight, 1: c_weight}  # validator 0's weights for miners 0,1
        }

        # Run economic tempo
        run_tempo(subnet, miners, validators, weights)
        print_scoreboard(subnet, miners, validators)

        time.sleep(0.5)  # Dramatic pause between tempos

    print_final_summary(subnet, miners, validators, args.tempos)


if __name__ == "__main__":
    main()
