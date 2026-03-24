"""
Subprocess runner for AutoResearch trials.

Writes the proposed experiment.py to a temp file, runs train.py with
--experiment_file pointing to it, and parses the score from stdout.

Score line format (printed by train.py at the end):
    [AutoResearch] score=0.9512 sla=0.9800 cost=0.2876

Fine-tuning:
    Pass finetune_checkpoint_dir to load existing *_final.pt checkpoints
    before training — agents continue from their current state instead of
    starting from random weights.

Live workload:
    Pass workload_npy (numpy array, shape N×4 or N×1) to use live buffer
    data as the training workload instead of the default synthetic workload.
    The array is saved to a temp .npy file and passed via --workload_file.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, Any, Optional, Tuple

import numpy as np

FAILURE_SENTINEL = float("-inf")

_TRAIN_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "train.py",
)


def run_trial(
    experiment_code:          str,
    total_steps:              int = 3000,
    seed:                     int = 0,
    timeout:                  int = 600,
    python_exe:               str | None = None,
    train_script:             str | None = None,
    finetune_checkpoint_dir:  str | None = None,
    output_checkpoint_dir:    str | None = None,
    workload_npy:             np.ndarray | None = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Run a training trial with the given experiment.py code.

    Args:
        experiment_code:         Complete Python source of the new experiment.py
        total_steps:             Training steps for this trial
        seed:                    Random seed
        timeout:                 Subprocess wall-clock timeout (seconds)
        python_exe:              Python interpreter (defaults to current)
        train_script:            Path to train.py
        finetune_checkpoint_dir: If set, copies *_final.pt from here into the
                                 trial's temp checkpoint dir so agents fine-tune
                                 from existing weights (--load_tag final).
        output_checkpoint_dir:   If set, copies *_final.pt from the trial's
                                 temp dir here after a successful run (so the
                                 live loop can promote improved checkpoints).
        workload_npy:            If set, saves this array as a temp .npy file
                                 and passes --workload_file to train.py (live
                                 buffer data replaces the default workload).

    Returns:
        (score, metrics_dict) — or (FAILURE_SENTINEL, {error: ...}) on failure
    """
    if python_exe is None:
        python_exe = sys.executable
    if train_script is None:
        train_script = _TRAIN_SCRIPT

    # ── Write proposed experiment.py to a temp file ────────────────────
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="ar_experiment_"
    ) as f:
        f.write(experiment_code)
        experiment_file = f.name

    # ── Save live workload to temp .npy if provided ────────────────────
    workload_file = None
    if workload_npy is not None and len(workload_npy) > 0:
        with tempfile.NamedTemporaryFile(
            suffix=".npy", delete=False, prefix="ar_workload_"
        ) as wf:
            workload_file = wf.name
        np.save(workload_file, workload_npy)

    with tempfile.TemporaryDirectory(prefix="ar_ckpt_") as tmpdir:

        # ── Copy existing checkpoints for fine-tuning ──────────────────
        finetune = False
        if finetune_checkpoint_dir and os.path.exists(finetune_checkpoint_dir):
            for fname in os.listdir(finetune_checkpoint_dir):
                if fname.endswith(".pt"):
                    shutil.copy2(
                        os.path.join(finetune_checkpoint_dir, fname),
                        os.path.join(tmpdir, fname),
                    )
            finetune = True

        # ── Build command ──────────────────────────────────────────────
        cmd = [
            python_exe, train_script,
            "--total_steps",     str(total_steps),
            "--seed",            str(seed),
            "--checkpoint_dir",  tmpdir,
            "--experiment_file", experiment_file,
            "--no_verbose",
        ]
        if finetune:
            cmd.extend(["--load_tag", "final"])
        if workload_file:
            cmd.extend(["--workload_file", workload_file])

        # ── Run trial ─────────────────────────────────────────────────
        t0 = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            _cleanup(experiment_file, workload_file)
            return FAILURE_SENTINEL, {"error": f"timeout after {timeout}s"}
        except Exception as e:
            _cleanup(experiment_file, workload_file)
            return FAILURE_SENTINEL, {"error": str(e)}

        elapsed = time.time() - t0

        if result.returncode != 0:
            _cleanup(experiment_file, workload_file)
            return FAILURE_SENTINEL, {
                "error":  f"exit_code={result.returncode}",
                "stderr": result.stderr[-800:],
                "stdout": result.stdout[-400:],
            }

        # ── Parse score line ───────────────────────────────────────────
        # Format: [AutoResearch] score=0.9512 sla=0.9800 cost=0.2876
        score     = FAILURE_SENTINEL
        mean_sla  = 0.0
        mean_cost = 0.0
        for line in result.stdout.splitlines():
            if "[AutoResearch] score=" in line:
                try:
                    for p in line.split():
                        if p.startswith("score="):
                            score = float(p.split("=")[1])
                        elif p.startswith("sla="):
                            mean_sla = float(p.split("=")[1])
                        elif p.startswith("cost="):
                            mean_cost = float(p.split("=")[1])
                except (ValueError, IndexError):
                    pass

        if score == FAILURE_SENTINEL:
            # Fallback: try metrics JSON
            metrics_path = os.path.join(tmpdir, "training_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    m = json.load(f)
                sla_rates = m.get("sla_rates", [])
                costs     = m.get("costs", [])
                if sla_rates:
                    half      = max(1, len(sla_rates) // 2)
                    mean_sla  = float(sum(sla_rates[-half:]) / half)
                    mean_cost = float(sum(costs[-half:]) / half) if costs else 0.0
                    score     = mean_sla - 0.1 * mean_cost
            else:
                _cleanup(experiment_file, workload_file)
                return FAILURE_SENTINEL, {"error": "no score line and no metrics file"}

        if score != score:   # NaN check
            _cleanup(experiment_file, workload_file)
            return FAILURE_SENTINEL, {"error": "NaN score"}

        # ── Promote checkpoints if requested ───────────────────────────
        if output_checkpoint_dir and result.returncode == 0:
            os.makedirs(output_checkpoint_dir, exist_ok=True)
            for fname in os.listdir(tmpdir):
                if fname.endswith(".pt"):
                    shutil.copy2(
                        os.path.join(tmpdir, fname),
                        os.path.join(output_checkpoint_dir, fname),
                    )

    _cleanup(experiment_file, workload_file)
    return score, {
        "mean_sla":  mean_sla,
        "mean_cost": mean_cost,
        "elapsed_s": elapsed,
        "stdout":    result.stdout[-1000:],
        "finetune":  finetune,
    }


def _cleanup(*paths) -> None:
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.unlink(p)
            except OSError:
                pass
