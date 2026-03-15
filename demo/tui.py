#!/usr/bin/env python3
"""
CacheScope Proof of Mind — Interactive TUI Demo
=================================================

Interactive terminal dashboard for demonstrating verifiable inference.
Judges can trigger queries, swap models, and watch verification happen
in real time.

Keys:
  [1-3]  Run a demo prompt on the honest miner
  [s]    Swap to cheating model and run same prompt
  [v]    Verify last miner MindPrint against honest reference
  [d]    Show determinism test (run same prompt twice)
  [w]    Show wire format details
  [q]    Quit

Usage:
  python demo/tui.py
  python demo/tui.py --honest Qwen/Qwen2.5-0.5B-Instruct --cheat TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import argparse
import base64
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.home() / "KV-Experiments" / "code"))
sys.path.insert(0, str(Path.home() / "jiminai-cricket" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import Header, Footer, Static, RichLog, Label
from textual.reactive import reactive
from textual import work

from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from mindprint.config import CacheScopeConfig
from mindprint.extractor import CacheScopeExtractor
from mindprint.proof.mindprint import MindPrint, generate_mindprint
from mindprint.proof.verify import verify_mindprint
from mindprint.proof.codec import encode_mindprint, mindprint_size


PROMPTS = [
    "What is the capital of France?",
    "Explain photosynthesis in one sentence.",
    "Write a Python function that adds two numbers.",
]


class GeometryPanel(Static):
    """Displays MindPrint geometry for one model."""

    def __init__(self, title: str = "Model", **kwargs):
        super().__init__(**kwargs)
        self._title = title
        self._mp = None
        self._response = ""
        self._wire_size = 0
        self._extract_ms = 0.0

    def set_data(self, mp: MindPrint, response: str, wire_size: int, extract_ms: float):
        self._mp = mp
        self._response = response
        self._wire_size = wire_size
        self._extract_ms = extract_ms
        self._render()

    def clear_data(self):
        self._mp = None
        self._response = ""
        self.update("")

    def _render(self):
        if self._mp is None:
            self.update("")
            return

        mp = self._mp
        t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        t.add_column("Key", style="bold cyan", width=18)
        t.add_column("Value", style="white")

        t.add_row("Model", mp.model_id)
        t.add_row("Layers", f"{mp.n_layers} total, {mp.n_extracted} extracted")
        t.add_row("Tokens", str(mp.n_tokens or "?"))
        t.add_row("", "")
        t.add_row("Norm", f"{mp.norm:.2f}")
        t.add_row("Norm/Token", f"{mp.norm_per_token:.2f}")
        t.add_row("Key Rank", f"{mp.key_rank:.2f}")
        t.add_row("Key Entropy", f"{mp.key_entropy:.4f}")
        t.add_row("", "")
        t.add_row("Fingerprint", f"{mp.fingerprint[:32]}...")
        t.add_row("Wire Size", f"{self._wire_size} bytes")
        t.add_row("Extract Time", f"{self._extract_ms:.1f}ms")

        # Mini sparkline of rank profile
        ranks = mp.layer_rank_profile
        if ranks:
            max_r = max(ranks) if max(ranks) > 0 else 1
            bars = "".join("▁▂▃▄▅▆▇█"[min(7, int(r / max_r * 7))] for r in ranks)
            t.add_row("", "")
            t.add_row("Rank Profile", bars)

        # Response preview
        resp_text = self._response[:120] + ("..." if len(self._response) > 120 else "")

        panel = Panel(t, title=f"[bold]{self._title}[/bold]", border_style="green" if "Honest" in self._title else "yellow")
        self.update(panel)


class VerificationPanel(Static):
    """Displays verification result."""

    def show_result(self, result, miner_id: str, ref_id: str):
        t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        t.add_column("Key", style="bold", width=20)
        t.add_column("Value")

        if result.valid:
            verdict = Text("VERIFIED", style="bold green")
        else:
            verdict = Text("REJECTED", style="bold red")

        t.add_row("Verdict", verdict)
        t.add_row("Mode", result.mode)
        t.add_row("Confidence", f"{result.confidence:.2f}")
        t.add_row("", "")
        t.add_row("Content Match", self._check(result.content_match))
        t.add_row("Structure Match", self._check(result.structure_match))
        t.add_row("Aggregate Match", self._check(result.aggregate_match))
        t.add_row("Profile Match", self._check(result.profile_match))
        t.add_row("Fingerprint Match", self._check(result.fingerprint_match))

        if result.profile_correlation is not None:
            t.add_row("Profile Correlation", f"{result.profile_correlation:.4f}")

        if result.aggregate_distances:
            t.add_row("", "")
            for feat, dist in result.aggregate_distances.items():
                style = "red" if dist > 0.01 else "green"
                t.add_row(f"  {feat} drift", Text(f"{dist:.4f}", style=style))

        if result.anomalies:
            t.add_row("", "")
            for a in result.anomalies[:6]:
                t.add_row("Anomaly", Text(a[:60], style="red"))

        border = "green" if result.valid else "red"
        title = f"[bold]Verification: {miner_id} vs {ref_id}[/bold]"
        panel = Panel(t, title=title, border_style=border)
        self.update(panel)

    def _check(self, val):
        return Text("OK", style="green") if val else Text("FAIL", style="bold red")

    def clear_result(self):
        self.update("")


class ProofOfMindTUI(App):
    """Interactive Proof of Mind demonstration."""

    CSS = """
    #top-row { height: 1fr; }
    #geometry-left { width: 1fr; }
    #geometry-right { width: 1fr; }
    #verification { height: auto; max-height: 20; }
    #log-panel { height: 1fr; min-height: 8; }
    #log { height: 100%; }
    """

    BINDINGS = [
        ("1", "prompt(0)", "Prompt 1: Capital"),
        ("2", "prompt(1)", "Prompt 2: Photosynthesis"),
        ("3", "prompt(2)", "Prompt 3: Python"),
        ("s", "swap", "Swap Model"),
        ("v", "verify", "Verify"),
        ("d", "determinism", "Determinism Test"),
        ("w", "wire_info", "Wire Format"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, honest_model: str, cheat_model: str, layer_stride: int = 4, quantize: bool = False):
        super().__init__()
        self.honest_model_name = honest_model
        self.cheat_model_name = cheat_model
        self.layer_stride = layer_stride
        self.quantize = quantize

        self.config = CacheScopeConfig(layer_stride=layer_stride)
        self.extractor = CacheScopeExtractor(self.config)

        self.honest_model = None
        self.honest_tokenizer = None
        self.cheat_model = None
        self.cheat_tokenizer = None

        self.last_honest_mp = None
        self.last_cheat_mp = None
        self.last_prompt = None
        self.using_cheat = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="top-row"):
            yield GeometryPanel(title="Honest Miner", id="geometry-left")
            yield GeometryPanel(title="Cheating Miner", id="geometry-right")
        yield VerificationPanel(id="verification")
        yield RichLog(id="log", wrap=True, highlight=True, markup=True)
        yield Footer()

    def on_mount(self):
        self.title = "CacheScope: Proof of Mind"
        self.sub_title = "Verifiable Inference Demo"
        self.log_msg("[bold]CacheScope: Proof of Mind[/bold] — Interactive Demo")
        self.log_msg("")
        self.log_msg("Keys: [bold cyan][1-3][/] Run prompt  [bold cyan][s][/] Swap model  [bold cyan][v][/] Verify  [bold cyan][d][/] Determinism  [bold cyan][w][/] Wire format")
        self.log_msg("")
        self.load_models()

    def log_msg(self, msg: str):
        self.query_one("#log", RichLog).write(msg)

    @work(thread=True)
    def load_models(self):
        from gpu_utils import load_model

        self.log_msg(f"Loading honest model: [bold green]{self.honest_model_name}[/]...")
        self.honest_model, self.honest_tokenizer = load_model(
            self.honest_model_name, quantize=self.quantize
        )
        self.log_msg(f"  Loaded on {next(self.honest_model.parameters()).device}")

        self.log_msg(f"Loading cheat model: [bold yellow]{self.cheat_model_name}[/]...")
        self.cheat_model, self.cheat_tokenizer = load_model(
            self.cheat_model_name, quantize=self.quantize
        )
        self.log_msg(f"  Loaded on {next(self.cheat_model.parameters()).device}")
        self.log_msg("")
        self.log_msg("[bold green]Ready![/] Press [bold cyan]1[/], [bold cyan]2[/], or [bold cyan]3[/] to run a prompt.")

    def _run_inference(self, model, tokenizer, prompt, max_tokens=128):
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
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
            )

        generated_ids = output.sequences[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        n_tokens = output.sequences.shape[-1]

        return response, output.past_key_values, n_tokens

    def _extract(self, cache, n_tokens, model_id, prompt, response):
        cache_cpu = self.extractor.cache_to_cpu(cache)
        snapshot = self.extractor.extract(cache_cpu, n_tokens=n_tokens)
        mp = generate_mindprint(snapshot, model_id=model_id, prompt=prompt, output=response)
        wire = encode_mindprint(mp)
        return mp, wire, snapshot.extraction_time_ms

    @work(thread=True)
    def run_prompt(self, prompt_idx: int):
        if self.honest_model is None:
            self.log_msg("[red]Models still loading...[/]")
            return

        prompt = PROMPTS[prompt_idx]
        self.last_prompt = prompt

        # Run honest
        self.log_msg(f"\n[bold cyan]Prompt {prompt_idx+1}:[/] {prompt}")
        self.log_msg(f"  Running [green]honest[/] miner ({self.honest_model_name})...")

        resp_h, cache_h, n_h = self._run_inference(
            self.honest_model, self.honest_tokenizer, prompt
        )
        mp_h, wire_h, ext_ms_h = self._extract(
            cache_h, n_h, self.honest_model_name, prompt, resp_h
        )
        self.last_honest_mp = mp_h

        self.app.call_from_thread(
            self.query_one("#geometry-left", GeometryPanel).set_data,
            mp_h, resp_h, len(wire_h), ext_ms_h
        )
        self.log_msg(f"  [green]Honest:[/] {resp_h[:80]}...")
        self.log_msg(f"  Fingerprint: [dim]{mp_h.fingerprint[:32]}...[/]")

        # Clear right panel and verification
        self.app.call_from_thread(
            self.query_one("#geometry-right", GeometryPanel).clear_data
        )
        self.app.call_from_thread(
            self.query_one("#verification", VerificationPanel).clear_result
        )
        self.last_cheat_mp = None

    @work(thread=True)
    def run_swap(self):
        if self.cheat_model is None or self.last_prompt is None:
            self.log_msg("[red]Run a prompt first (keys 1-3), then press s[/]")
            return

        prompt = self.last_prompt
        self.log_msg(f"\n  Running [yellow]cheating[/] miner ({self.cheat_model_name})...")
        self.log_msg(f"  [dim]Claiming to be {self.honest_model_name}[/]")

        resp_c, cache_c, n_c = self._run_inference(
            self.cheat_model, self.cheat_tokenizer, prompt
        )
        # Lie about model_id
        mp_c, wire_c, ext_ms_c = self._extract(
            cache_c, n_c, self.honest_model_name, prompt, resp_c
        )
        self.last_cheat_mp = mp_c

        self.app.call_from_thread(
            self.query_one("#geometry-right", GeometryPanel).set_data,
            mp_c, resp_c, len(wire_c), ext_ms_c
        )
        self.log_msg(f"  [yellow]Cheat:[/] {resp_c[:80]}...")
        self.log_msg(f"  Fingerprint: [dim]{mp_c.fingerprint[:32]}...[/]")
        self.log_msg("  Press [bold cyan]v[/] to verify")

    @work(thread=True)
    def run_verify(self):
        if self.last_honest_mp is None:
            self.log_msg("[red]No honest MindPrint — run a prompt first[/]")
            return

        target = self.last_cheat_mp or self.last_honest_mp
        target_name = "cheat" if self.last_cheat_mp else "honest (self)"

        self.log_msg(f"\n  [bold]Verifying {target_name} against honest reference...[/]")

        result = verify_mindprint(target, self.last_honest_mp, tolerance=0.01)

        self.app.call_from_thread(
            self.query_one("#verification", VerificationPanel).show_result,
            result,
            target_name,
            "honest ref",
        )

        if result.valid:
            self.log_msg(f"  [bold green]VERIFIED[/] — confidence {result.confidence:.2f}")
        else:
            self.log_msg(f"  [bold red]REJECTED[/] — {len(result.anomalies)} anomalies detected")
            for a in result.anomalies[:3]:
                self.log_msg(f"    [red]{a}[/]")

    @work(thread=True)
    def run_determinism(self):
        if self.honest_model is None:
            self.log_msg("[red]Models still loading...[/]")
            return

        prompt = PROMPTS[0]
        self.log_msg(f"\n[bold]Determinism test:[/] Running same prompt twice...")

        resp1, cache1, n1 = self._run_inference(
            self.honest_model, self.honest_tokenizer, prompt
        )
        mp1, _, _ = self._extract(cache1, n1, self.honest_model_name, prompt, resp1)

        resp2, cache2, n2 = self._run_inference(
            self.honest_model, self.honest_tokenizer, prompt
        )
        mp2, _, _ = self._extract(cache2, n2, self.honest_model_name, prompt, resp2)

        result = verify_mindprint(mp1, mp2, strict=True)

        if result.fingerprint_match:
            self.log_msg(f"  [bold green]IDENTICAL[/] — fingerprints match exactly")
            self.log_msg(f"  {mp1.fingerprint[:48]}...")
        else:
            self.log_msg(f"  [bold red]DIFFERENT[/] — unexpected non-determinism")

    def action_prompt(self, idx: int):
        self.run_prompt(idx)

    def action_swap(self):
        self.run_swap()

    def action_verify(self):
        self.run_verify()

    def action_determinism(self):
        self.run_determinism()

    def action_wire_info(self):
        self.log_msg("\n[bold]Wire Format — MindPrint sizes:[/]")
        for label, n in [("Stride-4, 24-layer (Qwen-0.5B)", 6),
                         ("Stride-4, 28-layer (Qwen-7B)", 7),
                         ("Stride-4, 32-layer (Llama-8B)", 8),
                         ("Full 32-layer", 32)]:
            size = mindprint_size(n)
            self.log_msg(f"  {label}: [bold cyan]{size}[/] bytes")
        self.log_msg(f"  Raw cache per layer at 7B: ~262 KB")
        self.log_msg(f"  Compression: [bold green]~17,000x[/]")


def main():
    parser = argparse.ArgumentParser(description="CacheScope Proof of Mind TUI")
    parser.add_argument("--honest", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--cheat", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--layer-stride", type=int, default=4)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    app = ProofOfMindTUI(
        honest_model=args.honest,
        cheat_model=args.cheat,
        layer_stride=args.layer_stride,
        quantize=args.quantize,
    )
    app.run()


if __name__ == "__main__":
    main()
