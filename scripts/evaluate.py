#!/usr/bin/env python3
"""
AutoCloud-Agent — Evaluation CLI
=================================

Usage:
    cd autocloud_agent/
    pip install -e .                       # one-time package install
    python scripts/evaluate.py             # auto-detect everything
    python scripts/evaluate.py --checkpoint_dir outputs/rl_agents --n_episodes 5
    python scripts/evaluate.py --workload outputs/train_Forecaster/day2_processed.npy
"""
from __future__ import annotations

import argparse
import json

from autocloud.config.settings import DEFAULT_CONFIG
from autocloud.config.paths import ArtifactPaths
from autocloud.evaluation.evaluator import Evaluator


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate AutoCloud-Agent vs baselines")
    p.add_argument("--checkpoint_dir", type=str, default=None,
                   help="Override checkpoint directory (auto-detected if omitted)")
    p.add_argument("--workload", type=str, default=None,
                   help="Path to .npy workload trace (auto-detected if omitted)")
    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--output", type=str, default="evaluation_results.json")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Resolve artifact paths automatically ───────────────────────
    paths = ArtifactPaths(
        checkpoint_dir=args.checkpoint_dir,
        workload_file=args.workload,
    )
    paths.validate_checkpoints()
    workload_fn = paths.make_workload_fn()

    print(f"{'=' * 60}")
    print(f"  AutoCloud-Agent Evaluation")
    print(f"  Checkpoints : {paths.checkpoint_dir}")
    print(f"  Workload    : {paths.workload_file or 'synthetic'}")
    print(f"  Forecaster  : {paths.forecaster_path or 'none'}")
    print(f"  Episodes    : {args.n_episodes} × {len(args.seeds)} seeds")
    print(f"{'=' * 60}\n")

    evaluator = Evaluator(
        config=DEFAULT_CONFIG,
        checkpoint_dir=paths.checkpoint_dir,
        n_episodes=args.n_episodes,
        seeds=args.seeds,
        verbose=True,
        workload_fn=workload_fn,
    )

    results = evaluator.evaluate_all()
    evaluator.print_table(results)
    evaluator.save_results(results, args.output)
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
