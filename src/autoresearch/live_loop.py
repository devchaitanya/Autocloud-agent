"""
LiveAutoResearch — continuous reward-weight adaptation from live traffic.

Simulates 1 day of Alibaba trace data over 1 hour real-time (24x compression).
Every `interval_minutes`, it:
  1. Snapshots the current live buffer (last N minutes of traffic)
  2. Asks the LLM to propose new reward weights based on current utilisation
  3. Fine-tunes agents from existing checkpoints using the live buffer as workload
  4. If score improves → updates experiment.py + promotes checkpoints
  5. Logs to autoresearch/live_results.tsv

Usage:
    python pipeline.py --mode live \\
      --checkpoint_dir ../outputs/rl_agents \\
      --workload_file  ../outputs/train_Forecaster/day2_processed.npy \\
      --llm_provider   groq \\
      --live_interval  10 \\
      --live_iterations 6
"""
from __future__ import annotations

import os
import sys
import time
import tempfile
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # → src/

from environment.live_buffer import LiveWorkloadBuffer
from autoresearch.engine import (
    _load_program_md, _load_experiment_code,
    _build_prompt, _call_llm, _validate_code, _extract_description,
    EXPERIMENT_FILE,
)
from autoresearch.subprocess_runner import run_trial, FAILURE_SENTINEL

# src/autoresearch/ → up 3 levels → repo root (autocloud_agent/)
_ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LIVE_TSV = os.path.join(_ROOT, "src", "autoresearch", "live_results.tsv")


class LiveAutoResearch:
    """
    Closed-loop AutoResearch that adapts reward weights to live traffic.

    The Alibaba trace is streamed at 24x compression (1 day → 1 hour).
    Every `interval_minutes` real-time:
      - Buffer snapshot → saved as temp .npy → passed to fine-tuning trial
      - LLM gets current utilisation stats in the prompt
      - Agents fine-tune from existing checkpoints (fast, not from scratch)
      - Best config saved to experiment.py, best checkpoints promoted
    """

    def __init__(
        self,
        checkpoint_dir:   str   = "checkpoints",
        workload_file:    Optional[str] = None,
        interval_minutes: int   = 10,
        window_minutes:   int   = 10,
        compression:      float = 24.0,
        trial_steps:      int   = 8000,
        trial_timeout:    int   = 500,
        llm_provider:     str   = "groq",
        llm_model:        Optional[str] = None,
        verbose:          bool  = True,
    ):
        self.checkpoint_dir  = os.path.abspath(checkpoint_dir)
        self.workload_file   = workload_file
        self.interval        = interval_minutes * 60        # seconds
        self.trial_steps     = trial_steps
        self.trial_timeout   = trial_timeout
        self.llm_provider    = llm_provider
        self.llm_model       = llm_model
        self.verbose         = verbose

        self.buffer = LiveWorkloadBuffer(
            window_seconds=window_minutes * 60,
            bin_size=30,
        )
        self.compression = compression

        self.history    = []
        self.best_score = FAILURE_SENTINEL
        self._iter      = 0

        os.makedirs(os.path.join(_ROOT, "src", "autoresearch"), exist_ok=True)
        if not os.path.exists(_LIVE_TSV):
            with open(_LIVE_TSV, "w") as f:
                f.write("iter\tscore\tsla\tcost\tmean_util\tn_samples\tstatus\tdescription\n")

    # ── Main loop ──────────────────────────────────────────────────────

    def run(self, max_iterations: int = 6) -> None:
        real_duration_min = max_iterations * self.interval / 60
        sim_duration_hr   = real_duration_min * self.compression / 60

        print(f"\n{'='*60}")
        print(f"[LiveAR] Live AutoResearch — Continuous Adaptation")
        print(f"[LiveAR] Compression   : {self.compression:.0f}x  "
              f"(1 day → {3600/self.compression/60:.0f} min real-time)")
        print(f"[LiveAR] AR interval   : every {self.interval/60:.0f} min real-time")
        print(f"[LiveAR] Trial mode    : fine-tune from existing checkpoints")
        print(f"[LiveAR] Trial steps   : {self.trial_steps}")
        print(f"[LiveAR] Total runtime : ~{real_duration_min:.0f} min  "
              f"({sim_duration_hr:.1f} h simulated)")
        print(f"{'='*60}\n")

        # Start streaming workload into buffer
        if self.workload_file and os.path.exists(self.workload_file):
            self.buffer.stream_from_npy(
                self.workload_file,
                compression=self.compression,
                loop=True,
                verbose=self.verbose,
            )
        else:
            print("[LiveAR] WARNING: No workload file — using uniform 0.5 util")

        program_md   = _load_program_md()
        current_code = _load_experiment_code()

        print(f"\n[LiveAR] Waiting {self.interval/60:.0f} min for initial data...")

        for i in range(1, max_iterations + 1):
            self._iter = i
            # Wait until interval elapsed and buffer has data
            self._wait_for_interval(first=(i == 1))

            print(f"\n{'='*60}")
            print(f"[LiveAR] Iteration {i}/{max_iterations}")
            print(f"[LiveAR] Buffer : {self.buffer.n_samples} samples  |  "
                  f"Mean util : {self.buffer.mean_util:.1%}  |  "
                  f"Latest : {self.buffer.latest_util:.1%}")
            print(f"{'='*60}")

            if not self.buffer.has_enough_data(min_samples=10):
                print("[LiveAR] Not enough data — skipping iteration")
                continue

            t0 = time.time()

            # Determine traffic regime for explicit LLM guidance
            util = self.buffer.mean_util
            if util < 0.35:
                regime     = "LOW"
                regime_action = (
                    "Traffic is LOW — prioritise COST REDUCTION. "
                    "Increase alpha2 (over-prov penalty) or alpha3 (node count) or beta1 (cost weight). "
                    "Decrease alpha1 or beta2 if SLA is already 100%."
                )
            elif util < 0.60:
                regime     = "MEDIUM"
                regime_action = (
                    "Traffic is MEDIUM — balance SLA and cost. "
                    "Small increase to beta2 (SLA bonus) for headroom, or beta1 (cost) to trim spending."
                )
            else:
                regime     = "HIGH"
                regime_action = (
                    "Traffic is HIGH — prioritise SLA PROTECTION. "
                    "Increase alpha1 (under-prov penalty), beta2 (SLA bonus), gamma2 (SLA violation). "
                    "Decrease beta1 (cost weight) and alpha3 (node count penalty)."
                )

            # Build prompt with live traffic context injected
            live_ctx = (
                f"\n## Live Traffic Context (last {self.buffer.window_seconds//60} min)\n"
                f"Mean CPU utilisation : {self.buffer.mean_util:.1%}\n"
                f"Latest measurement   : {self.buffer.latest_util:.1%}\n"
                f"Buffer samples       : {self.buffer.n_samples}\n"
                f"Traffic regime       : **{regime}**\n\n"
                f"ACTION REQUIRED: {regime_action}\n"
            )
            prompt = _build_prompt(
                current_code, self.history, i, max_iterations,
                program_md + live_ctx,
            )

            # LLM proposes new experiment.py
            raw = _call_llm(prompt, provider=self.llm_provider, model=self.llm_model)
            if raw is None:
                print("[LiveAR] LLM unavailable — skipping")
                continue

            ok, result = _validate_code(raw)
            if not ok:
                print(f"[LiveAR] Validation failed ({result}), retrying...")
                raw2 = _call_llm(prompt, provider=self.llm_provider, model=self.llm_model)
                if raw2:
                    ok, result = _validate_code(raw2)
            if not ok:
                print(f"[LiveAR] Skipping: {result}")
                continue

            proposed_code = result
            desc = _extract_description(current_code, proposed_code)
            print(f"[LiveAR] Proposed: {desc}")

            # Save live buffer snapshot to temp .npy for the subprocess
            workload_npy = self.buffer.as_numpy()

            # Determine where to save improved checkpoints
            promoted_dir = tempfile.mkdtemp(prefix="live_ckpt_")

            score, metrics = run_trial(
                experiment_code=proposed_code,
                total_steps=self.trial_steps,
                seed=i,
                timeout=self.trial_timeout,
                finetune_checkpoint_dir=self.checkpoint_dir,
                output_checkpoint_dir=promoted_dir,
                workload_npy=workload_npy,
            )

            elapsed = time.time() - t0
            sla  = metrics.get("mean_sla", 0.0)
            cost = metrics.get("mean_cost", 0.0)

            if score == FAILURE_SENTINEL:
                err = metrics.get("error", "unknown")
                print(f"[LiveAR] Trial FAILED: {err}")
                self.history.append({
                    "iter": i, "score": -1.0, "sla": 0.0, "cost": 0.0,
                    "kept": False, "description": f"CRASH: {err}",
                })
                self._log(-1.0, 0.0, 0.0, "crash", desc)
                _rm(promoted_dir)
                continue

            kept   = score > self.best_score
            status = "KEEP" if kept else "discard"
            arrow  = "▲" if kept else "▼"
            print(f"[LiveAR] {arrow} score={score:+.4f}  sla={sla:.3f}  "
                  f"cost={cost:.4f}  ft={metrics.get('finetune',False)}  "
                  f"→ {status}  ({elapsed:.0f}s)")

            if kept:
                self.best_score = score
                current_code    = proposed_code
                # Update experiment.py
                with open(EXPERIMENT_FILE, "w") as f:
                    f.write(proposed_code)
                # Promote fine-tuned checkpoints to main dir
                self._promote(promoted_dir)
                print(f"[LiveAR] *** New best: {score:.4f} — "
                      f"experiment.py updated, checkpoints promoted ***")

            _rm(promoted_dir)

            self.history.append({
                "iter": i, "score": score, "sla": sla, "cost": cost,
                "kept": kept, "description": desc,
            })
            self._log(score, sla, cost, status.lower(), desc)

        # Clean up
        self.buffer.stop_stream()
        print(f"\n[LiveAR] Done. Best score: {self.best_score:.4f}")
        print(f"[LiveAR] Log  → {_LIVE_TSV}")
        self._print_table()

    # ── Helpers ────────────────────────────────────────────────────────

    def _wait_for_interval(self, first: bool = False) -> None:
        """Wait self.interval seconds (shorter wait on first iteration)."""
        wait = min(self.interval, 45) if first else self.interval
        deadline = time.time() + wait
        while time.time() < deadline:
            remaining = deadline - time.time()
            if self.buffer.has_enough_data(min_samples=20) and remaining < self.interval * 0.1:
                break
            time.sleep(min(5, remaining))

    def _promote(self, src_dir: str) -> None:
        """Copy *_final.pt from src_dir to self.checkpoint_dir."""
        import os, shutil
        for fname in os.listdir(src_dir):
            if fname.endswith(".pt"):
                shutil.copy2(
                    os.path.join(src_dir, fname),
                    os.path.join(self.checkpoint_dir, fname),
                )
        if self.verbose:
            print(f"[LiveAR] Checkpoints promoted → {self.checkpoint_dir}")

    def _log(self, score, sla, cost, status, desc) -> None:
        with open(_LIVE_TSV, "a") as f:
            f.write(
                f"{self._iter}\t{score:.4f}\t{sla:.4f}\t{cost:.4f}\t"
                f"{self.buffer.mean_util:.3f}\t{self.buffer.n_samples}\t"
                f"{status}\t{desc}\n"
            )

    def _print_table(self) -> None:
        print(f"\n{'iter':>4}  {'score':>8}  {'sla':>6}  {'cost':>8}  "
              f"{'util':>6}  {'status':>8}  description")
        print("-" * 80)
        for h in self.history:
            print(f"{h['iter']:>4}  {h['score']:>8.4f}  {h['sla']:>6.3f}  "
                  f"{h['cost']:>8.4f}  "
                  f"{'KEEP' if h['kept'] else 'discard':>8}  {h['description']}")


def _rm(path: str) -> None:
    """Remove a directory tree, ignoring errors."""
    import shutil
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass
