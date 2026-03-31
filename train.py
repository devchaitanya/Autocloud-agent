"""
AutoCloud-Agent training entry point.
Called by AutoResearch subprocess_runner for fast trials.

Usage:
    python train.py --total_steps 50000 --seed 0
    python train.py --total_steps 5000  --seed 0 --device cpu
    python train.py --eval_only --checkpoint_dir checkpoints

For full local operations (evaluate, autoresearch, etc.) use pipeline.py.
"""
from __future__ import annotations

import argparse
import os
import json
import numpy as np

from autocloud.config.settings import DEFAULT_CONFIG
from autocloud.config.paths import ArtifactPaths
from autocloud.training.ippo_trainer import IPPOTrainer


def parse_args():
    p = argparse.ArgumentParser(description="Train AutoCloud-Agent (I-PPO)")
    p.add_argument("--total_steps",     type=int, default=50_000)
    p.add_argument("--seed",            type=int, default=0)
    p.add_argument("--device",          type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--checkpoint_dir",  type=str, default="checkpoints")
    p.add_argument("--forecaster_path", type=str, default=None)
    p.add_argument("--load_tag",        type=str, default=None)
    p.add_argument("--log_interval",    type=int, default=1000)
    p.add_argument("--eval_only",       action="store_true")
    p.add_argument("--experiment_file", type=str, default=None,
                   help="Path to experiment.py defining get_config() (AutoResearch)")
    p.add_argument("--workload_file",   type=str, default=None,
                   help="Path to .npy workload trace (live buffer or Alibaba data)")
    p.add_argument("--verbose",         action="store_true", default=True)
    p.add_argument("--no_verbose",      action="store_false", dest="verbose")
    return p.parse_args()


def load_forecaster(path, device):
    paths = ArtifactPaths(forecaster_path=path)
    return paths.load_forecaster(device=device)


def quick_eval(trainer, n_episodes=5):
    env = trainer.env
    sla_rates, costs, ep_returns = [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_return, ep_sla, ep_steps = 0.0, 0, 0
        while not done:
            action = trainer.select_action(obs, env)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += sum(info["rewards"].values())
            m = info["metrics"]
            ep_sla   += float(m["p95_latency"] < trainer.config.reward.sla_latency_ms or m["p95_latency"] == 0)
            ep_steps += 1
        sla_rates.append(ep_sla / max(ep_steps, 1))
        costs.append(info["metrics"]["total_cost"])
        ep_returns.append(ep_return)
    return {
        "mean_return":   float(np.mean(ep_returns)),
        "mean_sla_rate": float(np.mean(sla_rates)),
        "mean_cost":     float(np.mean(costs)),
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # ── Config: experiment.py override (AutoResearch) ──────────────
    config = DEFAULT_CONFIG
    if args.experiment_file and os.path.exists(args.experiment_file):
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("experiment", args.experiment_file)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            config = mod.get_config()
            if args.verbose:
                print(f"[info] Config loaded from {args.experiment_file}")
        except Exception as e:
            print(f"[warn] Failed to load experiment_file: {e} — using DEFAULT_CONFIG")

    forecaster = load_forecaster(args.forecaster_path, args.device)

    # ── Workload function ──────────────────────────────────────────────
    workload_fn = None
    if args.workload_file and os.path.exists(args.workload_file):
        data  = np.load(args.workload_file)
        rates = np.clip(data[:, 0] if data.ndim > 1 else data, 0.1, 1.0)
        n     = len(rates)
        def workload_fn(sim_time: float) -> float:
            return float(rates[int(sim_time / 30.0) % n])

    trainer = IPPOTrainer(
        config=config,
        seed=args.seed,
        device=args.device,
        forecaster=forecaster,
        workload_fn=workload_fn,
        verbose=args.verbose,
        log_interval=args.log_interval,
    )

    if args.load_tag:
        trainer.load(args.checkpoint_dir, tag=args.load_tag)
        print(f"[info] Loaded: {args.checkpoint_dir}/{args.load_tag}")

    if args.eval_only:
        print("\n=== Quick Eval (5 episodes) ===")
        results = quick_eval(trainer, n_episodes=5)
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
        return

    # ── Train ─────────────────────────────────────────────────────
    print(f"\n=== Training: {args.total_steps} steps | seed={args.seed} | device={args.device} ===\n")
    metrics = trainer.train(total_steps=args.total_steps, checkpoint_dir=args.checkpoint_dir)
    trainer.save(args.checkpoint_dir, tag="final")

    n_ep = len(metrics.so_returns)
    if n_ep > 0:
        w = min(20, n_ep)
        mean_sla  = float(np.mean(metrics.sla_rates[-w:]))
        mean_cost = float(np.mean(metrics.costs[-w:]))
        print(f"\n=== Done | Episodes: {n_ep} | SLA: {mean_sla:.2%} | Cost: {mean_cost:.3f} ===")

        os.makedirs(args.checkpoint_dir, exist_ok=True)
        with open(os.path.join(args.checkpoint_dir, "training_metrics.json"), "w") as f:
            json.dump({
                "total_steps": args.total_steps, "seed": args.seed, "episodes": n_ep,
                "sla_rates": metrics.sla_rates[-100:], "costs": metrics.costs[-100:],
                "so_losses": metrics.so_losses[-100:], "sch_losses": metrics.sch_losses[-100:],
            }, f, indent=2)

        # Score line parsed by AutoResearch subprocess_runner
        score = mean_sla - 0.1 * mean_cost
        print(f"[AutoResearch] score={score:.4f} sla={mean_sla:.4f} cost={mean_cost:.4f}")


if __name__ == "__main__":
    main()
