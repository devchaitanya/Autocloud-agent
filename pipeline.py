"""
AutoCloud-Agent — Local Pipeline
=================================
Run this after downloading trained checkpoints from Kaggle.

Modes:
    eval          — evaluate trained agents vs all 7 baselines
    autoresearch  — run Karpathy-style LLM reward tuning
    all           — eval then autoresearch

Usage:
    python pipeline.py                          # evaluate only
    python pipeline.py --mode eval
    python pipeline.py --mode autoresearch --llm_provider groq
    python pipeline.py --mode autoresearch --llm_provider ollama
    python pipeline.py --mode all --llm_provider groq

Setup (one-time):
    pip install simpy gymnasium numpy torch anthropic  # core
    pip install groq                                   # if using Groq
    pip install google-generativeai                    # if using Gemini
    # or: curl -fsSL https://ollama.com/install.sh | sh && ollama pull llama3.2:3b

Checkpoints expected at: ./checkpoints/so_actor_final.pt (etc.)
Workload data (optional): ./data/day1_processed.npy
Forecaster weights (opt): ./checkpoints/forecaster_weights.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ #
# Args
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="AutoCloud-Agent local pipeline")
    p.add_argument("--mode",            type=str, default="eval",
                   choices=["eval", "autoresearch", "all", "live"],
                   help="What to run (default: eval)")
    p.add_argument("--checkpoint_dir",  type=str, default="checkpoints",
                   help="Directory with trained .pt files")
    p.add_argument("--workload_file",   type=str, default=None,
                   help="Path to .npy workload trace (e.g. data/day1_processed.npy)")
    p.add_argument("--forecaster_path", type=str, default=None,
                   help="Path to forecaster_weights.pt (optional)")
    p.add_argument("--n_episodes",      type=int, default=10,
                   help="Episodes per method per seed for evaluation")
    p.add_argument("--seeds",           type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--output",          type=str, default="evaluation_results.json")
    # AutoResearch args
    p.add_argument("--llm_provider",    type=str, default="groq",
                   choices=["groq", "ollama", "anthropic", "gemini"],
                   help="LLM provider for AutoResearch")
    p.add_argument("--llm_model",       type=str, default=None,
                   help="Model name (default: provider default)")
    p.add_argument("--ar_iterations",   type=int, default=4,
                   help="AutoResearch iterations (default: 4)")
    p.add_argument("--ar_steps",        type=int, default=2000,
                   help="Training steps per AutoResearch trial (default: 2000)")
    # Live adaptation args
    p.add_argument("--live_interval",   type=int, default=10,
                   help="Minutes between live AutoResearch iterations (default: 10)")
    p.add_argument("--live_iterations", type=int, default=6,
                   help="Number of live iterations (default: 6 → 1 hour with 10-min interval)")
    p.add_argument("--live_steps",      type=int, default=8000,
                   help="Fine-tuning steps per live trial (default: 8000)")
    p.add_argument("--compression",     type=float, default=24.0,
                   help="Workload compression ratio: 24 = 1 day in 1 hour (default: 24)")
    return p.parse_args()


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _find_workload(workload_file):
    """Try to find workload .npy automatically if not specified."""
    if workload_file and os.path.exists(workload_file):
        return workload_file
    # Auto-detect common locations
    candidates = [
        "data/day1_processed.npy",
        "data/day2_processed.npy",
        "checkpoints/day1_processed.npy",
        "checkpoints/day2_processed.npy",
    ]
    for c in candidates:
        if os.path.exists(c):
            print(f"[pipeline] Auto-detected workload: {c}")
            return c
    return None


def _make_workload_fn(npy_path):
    """Build workload_fn from a processed .npy trace file."""
    data  = np.load(npy_path)
    rates = np.clip(data[:, 0], 0.1, 1.0)
    n     = len(rates)
    def fn(sim_time: float) -> float:
        return float(rates[int(sim_time / 30.0) % n])
    print(f"[pipeline] Workload loaded: {npy_path}  ({n} steps = {n*30/3600:.1f} h)")
    return fn


def _load_forecaster(forecaster_path, device="cpu"):
    """Load MCDropoutForecaster from weights file."""
    if forecaster_path is None:
        # Auto-detect
        for c in ["checkpoints/forecaster_weights.pt", "forecaster_weights.pt"]:
            if os.path.exists(c):
                forecaster_path = c
                break
    if forecaster_path is None or not os.path.exists(forecaster_path):
        print("[pipeline] No forecaster weights found — running without forecaster")
        return None
    import torch
    from forecaster.transformer_model import WorkloadTransformer
    from forecaster.mc_dropout import MCDropoutForecaster
    model = WorkloadTransformer(
        input_dim=4, d_model=64, n_heads=4, d_ff=256, n_layers=2,
        dropout=0.2, seq_len=20, n_horizons=4,
    )
    model.load_state_dict(torch.load(forecaster_path, map_location=device))
    model.to(device)
    forecaster = MCDropoutForecaster(model, k_samples=30, device=device)
    print(f"[pipeline] Forecaster loaded: {forecaster_path}")
    return forecaster


# ------------------------------------------------------------------ #
# Mode: eval
# ------------------------------------------------------------------ #

def run_eval(args):
    from configs.default_config import DEFAULT_CONFIG
    from evaluation.evaluator import Evaluator

    workload_path = _find_workload(args.workload_file)
    workload_fn   = _make_workload_fn(workload_path) if workload_path else None

    print(f"\n{'='*60}")
    print(f"  Evaluating: {args.n_episodes} episodes × {len(args.seeds)} seeds × 8 methods")
    print(f"  Checkpoints: {args.checkpoint_dir}")
    print(f"  Workload: {workload_path or 'synthetic (no .npy found)'}")
    print(f"{'='*60}\n")

    evaluator = Evaluator(
        config=DEFAULT_CONFIG,
        checkpoint_dir=args.checkpoint_dir,
        n_episodes=args.n_episodes,
        seeds=args.seeds,
        verbose=True,
        workload_fn=workload_fn,
    )

    results = evaluator.evaluate_all()
    evaluator.print_table(results)
    evaluator.save_results(results, args.output)
    print(f"\n[pipeline] Results saved → {args.output}")
    return results


# ------------------------------------------------------------------ #
# Mode: autoresearch
# ------------------------------------------------------------------ #

def run_autoresearch(args):
    from autoresearch.engine import AutoResearchEngine

    # Check API key for cloud providers
    key_env = {
        "groq":      "GROQ_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini":    "GEMINI_API_KEY",
        "ollama":    None,
    }
    if key_env[args.llm_provider] and not os.environ.get(key_env[args.llm_provider]):
        print(f"\n[pipeline] ERROR: {key_env[args.llm_provider]} not set.")
        if args.llm_provider == "groq":
            print("  → Free key at https://console.groq.com  then: export GROQ_API_KEY=gsk_...")
        elif args.llm_provider == "gemini":
            print("  → Free key at https://aistudio.google.com  then: export GEMINI_API_KEY=...")
        elif args.llm_provider == "anthropic":
            print("  → export ANTHROPIC_API_KEY=sk-ant-...")
        return None

    print(f"\n{'='*60}")
    print(f"  AutoResearch — Karpathy-style LLM reward tuning")
    print(f"  Provider : {args.llm_provider}  |  Model: {args.llm_model or 'default'}")
    print(f"  Trials   : {args.ar_iterations}  |  Steps/trial: {args.ar_steps}")
    print(f"  Target   : experiment.py  (reward weights + PPO params)")
    print(f"{'='*60}\n")

    engine = AutoResearchEngine(
        n_iterations=args.ar_iterations,
        total_steps=args.ar_steps,
        seed=0,
        trial_timeout=360,
        verbose=True,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
    )

    best_score, best_code = engine.run()
    print(f"\n[pipeline] AutoResearch done. Best score: {best_score:.4f}")
    print(f"[pipeline] Best config saved → experiment.py")
    print(f"[pipeline] Trial log         → autoresearch/results.tsv")
    return best_score


# ------------------------------------------------------------------ #
# Mode: live
# ------------------------------------------------------------------ #

def run_live(args):
    from autoresearch.live_loop import LiveAutoResearch

    workload_path = _find_workload(args.workload_file)

    print(f"\n{'='*60}")
    print(f"  Live AutoResearch — Continuous Adaptation")
    print(f"  Workload     : {workload_path or 'NOT FOUND'}")
    print(f"  Checkpoints  : {args.checkpoint_dir}")
    print(f"  Compression  : {args.compression:.0f}x  "
          f"(1 day → {3600/args.compression/60:.0f} min real-time)")
    print(f"  AR interval  : every {args.live_interval} min")
    print(f"  Iterations   : {args.live_iterations}  "
          f"(~{args.live_interval * args.live_iterations} min total)")
    print(f"  Trial steps  : {args.live_steps} (fine-tune from checkpoints)")
    print(f"  LLM provider : {args.llm_provider}")
    print(f"{'='*60}\n")

    if not workload_path:
        print("[pipeline] ERROR: No workload file found. Pass --workload_file.")
        return

    key_env = {"groq": "GROQ_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
                "gemini": "GEMINI_API_KEY", "ollama": None}
    if key_env.get(args.llm_provider) and not os.environ.get(key_env[args.llm_provider]):
        print(f"[pipeline] ERROR: {key_env[args.llm_provider]} not set.")
        return

    loop = LiveAutoResearch(
        checkpoint_dir=args.checkpoint_dir,
        workload_file=workload_path,
        interval_minutes=args.live_interval,
        compression=args.compression,
        trial_steps=args.live_steps,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        verbose=True,
    )
    loop.run(max_iterations=args.live_iterations)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    args = parse_args()

    print(f"\n AutoCloud-Agent Pipeline")
    print(f" Mode: {args.mode}")

    if args.mode in ("eval", "all"):
        run_eval(args)

    if args.mode in ("autoresearch", "all"):
        run_autoresearch(args)

    if args.mode == "live":
        run_live(args)

    print("\n[pipeline] Done.")


if __name__ == "__main__":
    main()
