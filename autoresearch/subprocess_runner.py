"""
Subprocess runner for AutoResearch trials.

Writes the proposed experiment.py to a temp file, runs train.py with
--experiment_file pointing to it, and parses the score from stdout.

Score line format (printed by train.py at the end):
    [AutoResearch] score=0.9512 sla=0.9800 cost=0.2876
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Dict, Any, Tuple

FAILURE_SENTINEL = float("-inf")

_TRAIN_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "train.py",
)


def run_trial(
    experiment_code: str,
    total_steps:     int = 3000,
    seed:            int = 0,
    timeout:         int = 600,
    python_exe:      str | None = None,
    train_script:    str | None = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Run a training trial with the given experiment.py code.

    Args:
        experiment_code: Complete Python source of the new experiment.py
        total_steps:     Training steps for this trial
        seed:            Random seed
        timeout:         Subprocess wall-clock timeout (seconds)
        python_exe:      Python interpreter (defaults to current)
        train_script:    Path to train.py

    Returns:
        (score, metrics_dict) — or (FAILURE_SENTINEL, {error: ...}) on failure
    """
    if python_exe is None:
        python_exe = sys.executable
    if train_script is None:
        train_script = _TRAIN_SCRIPT

    # Write proposed experiment.py to a temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="ar_experiment_"
    ) as f:
        f.write(experiment_code)
        experiment_file = f.name

    # Use a temp dir for checkpoints so trials don't interfere
    with tempfile.TemporaryDirectory(prefix="ar_ckpt_") as tmpdir:
        cmd = [
            python_exe, train_script,
            "--total_steps",     str(total_steps),
            "--seed",            str(seed),
            "--checkpoint_dir",  tmpdir,
            "--experiment_file", experiment_file,
            "--no_verbose",
        ]

        t0 = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            os.unlink(experiment_file)
            return FAILURE_SENTINEL, {"error": f"timeout after {timeout}s"}
        except Exception as e:
            os.unlink(experiment_file)
            return FAILURE_SENTINEL, {"error": str(e)}
        finally:
            try:
                os.unlink(experiment_file)
            except OSError:
                pass

        elapsed = time.time() - t0

        if result.returncode != 0:
            return FAILURE_SENTINEL, {
                "error":  f"exit_code={result.returncode}",
                "stderr": result.stderr[-800:],
                "stdout": result.stdout[-400:],
            }

        # Parse the score line from stdout
        # Format: [AutoResearch] score=0.9512 sla=0.9800 cost=0.2876
        score    = FAILURE_SENTINEL
        mean_sla = 0.0
        mean_cost = 0.0
        for line in result.stdout.splitlines():
            if "[AutoResearch] score=" in line:
                try:
                    parts = line.split()
                    for p in parts:
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
                return FAILURE_SENTINEL, {"error": "no score line and no metrics file"}

        if score != score:   # NaN check
            return FAILURE_SENTINEL, {"error": "NaN score"}

        return score, {
            "mean_sla":  mean_sla,
            "mean_cost": mean_cost,
            "elapsed_s": elapsed,
            "stdout":    result.stdout[-1000:],
        }
