"""
Evaluation harness — compare 5 methods across 3 seeds.

Methods:
  1. AutoCloud-Agent (I-PPO, 3 agents)
  2. SingleAgentPPO
  3. ThresholdReactive
  4. ThresholdPredictive
  5. StaticN

Metrics (per-episode, averaged over seeds):
  - SLA rate:         fraction of steps where P95 latency < sla_threshold_ms
  - Cost efficiency:  1 - actual_cost / max_cost  (higher = cheaper)
  - Mean CPU util:    average CPU utilization across active nodes
  - Node stability:   1 - std(node_count) / mean(node_count)  (higher = more stable)

Usage:
    python -m evaluation.evaluator [--checkpoint_dir checkpoints] [--n_episodes 10] [--seeds 0 1 2]
"""
from __future__ import annotations

import argparse
import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from autocloud.config.settings import DEFAULT_CONFIG, Config
from autocloud.simulator.cloud_env import CloudEnv
from autocloud.evaluation.baselines import (StaticN, ThresholdReactive,
                                SingleAgentPPO, KubernetesHPA,
                                AWSTargetTracking, MPCController)
from autocloud.inference.runner import InferenceRunner


# ------------------------------------------------------------------ #
# Episode runner
# ------------------------------------------------------------------ #

def run_episode(policy, env: CloudEnv, config: Config) -> Dict[str, float]:
    """
    Run one episode with the given policy.
    Returns a dict of per-episode metrics.
    """
    obs, _ = env.reset()
    if hasattr(policy, "reset"):
        policy.reset()
    done = False

    sla_steps = 0
    total_steps = 0
    total_cost = 0.0
    cpu_utils = []
    node_counts = []
    latencies = []

    while not done:
        action = policy.select_action(obs, env)

        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        m = info["metrics"]
        sla_ok = float(
            m["p95_latency"] < config.reward.sla_latency_ms or m["p95_latency"] == 0
        )
        sla_steps  += sla_ok
        total_steps += 1
        total_cost  += m.get("step_cost", 0.0)

        cpu_utils.append(m.get("mean_cpu_util", 0.0))
        node_counts.append(m.get("n_active", 0))
        latencies.append(m.get("p95_latency", 0.0))

    sla_rate = sla_steps / max(total_steps, 1)

    # Cost efficiency: fraction of max possible cost NOT spent
    max_cost = config.sim.n_max * 0.40 * config.sim.episode_steps * 30.0 / 3600.0
    cost_eff = 1.0 - min(total_cost / max(max_cost, 1e-6), 1.0)

    mean_cpu   = float(np.mean(cpu_utils)) if cpu_utils else 0.0
    stability  = 1.0 - (
        float(np.std(node_counts)) / float(np.mean(node_counts) + 1e-6)
        if node_counts else 0.0
    )
    stability  = max(0.0, min(1.0, stability))

    return {
        "sla_rate":        sla_rate,
        "cost_efficiency": cost_eff,
        "mean_cpu_util":   mean_cpu,
        "node_stability":  stability,
        "total_cost":      total_cost,
        "n_episodes":      1,
        "node_count_trace": node_counts,   # list of n_active per step (for time-series plots)
        "cpu_util_trace":   cpu_utils,     # list of mean CPU per step
    }


# ------------------------------------------------------------------ #
# Evaluator
# ------------------------------------------------------------------ #

class Evaluator:
    def __init__(
        self,
        config: Config = DEFAULT_CONFIG,
        checkpoint_dir: str = "checkpoints",
        n_episodes: int = 10,
        seeds: List[int] = None,
        verbose: bool = True,
        workload_fn=None,
    ):
        self.config         = config
        self.checkpoint_dir = checkpoint_dir
        self.n_episodes     = n_episodes
        self.seeds          = seeds if seeds is not None else [0, 1, 2]
        self.verbose        = verbose
        self.workload_fn    = workload_fn

    # Policy factories

    def _validate_autocloud_checkpoints(self, tag: str = "final") -> None:
        required = [
            f"so_actor_{tag}.pt", f"so_critic_{tag}.pt",
            f"con_actor_{tag}.pt", f"con_critic_{tag}.pt",
            f"sch_actor_{tag}.pt", f"sch_critic_{tag}.pt",
        ]
        missing = [
            name for name in required
            if not os.path.exists(os.path.join(self.checkpoint_dir, name))
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing RL checkpoints in '{self.checkpoint_dir}': {missing}"
            )

    def _make_autocloud(self, seed: int):
        return InferenceRunner(
            checkpoint_dir=self.checkpoint_dir,
            config=self.config,
            device="cpu",
        )

    def _make_single_ppo(self, seed: int) -> SingleAgentPPO:
        agent = SingleAgentPPO(device="cpu")
        return agent

    def _make_threshold_reactive(self) -> ThresholdReactive:
        return ThresholdReactive()

    def _make_static_n(self) -> StaticN:
        return StaticN(n_nodes=10)

    # Evaluation runner

    def _eval_policy(self, policy_name: str, policy, seed: int) -> Dict[str, float]:
        """Run n_episodes of the policy and return averaged metrics."""
        env = CloudEnv(config=self.config, seed=seed, workload_fn=self.workload_fn)
        all_metrics = []
        for ep in range(self.n_episodes):
            m = run_episode(policy, env, self.config)
            all_metrics.append(m)

        avg = {}
        trace_keys = {"node_count_trace", "cpu_util_trace"}
        for key in all_metrics[0]:
            if key in trace_keys:
                # Keep the last episode's trace (representative sample)
                avg[key] = all_metrics[-1][key]
            else:
                vals = [m[key] for m in all_metrics]
                avg[key] = float(np.mean(vals))
        return avg

    def evaluate_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all 5 methods across all seeds.

        Returns:
            results[method_name] = {
                "mean": {metric: value},
                "std":  {metric: value},
                "per_seed": [{metric: value}, ...]
            }
        """
        methods = {
            "AutoCloud-Agent":     lambda s: self._make_autocloud(s),
            # Industry-standard autoscalers
            "KubernetesHPA":       lambda s: KubernetesHPA(),
            "AWSTargetTracking":   lambda s: AWSTargetTracking(),
            # Strongest non-RL baseline
            "MPCController":       lambda s: MPCController(),
            # Classic rule-based
            "ThresholdReactive":   lambda s: self._make_threshold_reactive(),
            # Deep RL ablation (single-agent vs multi-agent)
            "SingleAgentPPO":      lambda s: self._make_single_ppo(s),
            # Do-nothing lower bound
            "StaticN":             lambda s: self._make_static_n(),
        }

        results: Dict[str, Dict] = {}

        for method_name, factory in methods.items():
            if self.verbose:
                print(f"\n[Eval] {method_name} ...")

            seed_metrics = []
            for seed in self.seeds:
                if self.verbose:
                    print(f"  seed={seed} ...", end="", flush=True)
                policy = factory(seed)
                m = self._eval_policy(method_name, policy, seed)
                seed_metrics.append(m)
                if self.verbose:
                    print(f" SLA={m['sla_rate']:.2%} cost_eff={m['cost_efficiency']:.3f}")

            # Aggregate (traces are lists — keep last seed's trace, skip std)
            mean_m = {}
            std_m  = {}
            trace_keys = {"node_count_trace", "cpu_util_trace"}
            for key in seed_metrics[0]:
                if key in trace_keys:
                    mean_m[key] = seed_metrics[-1][key]
                    std_m[key]  = []
                else:
                    vals = [m[key] for m in seed_metrics]
                    mean_m[key] = float(np.mean(vals))
                    std_m[key]  = float(np.std(vals))

            results[method_name] = {
                "mean":     mean_m,
                "std":      std_m,
                "per_seed": seed_metrics,
            }

        return results

    def print_table(self, results: Dict[str, Dict]) -> None:
        """Print a formatted results table (scalar metrics only)."""
        metrics = ["sla_rate", "cost_efficiency", "mean_cpu_util", "node_stability"]
        col_w = 22

        header = f"{'Method':<22}" + "".join(f"{m:<{col_w}}" for m in metrics)
        print("\n" + "=" * (22 + col_w * len(metrics)))
        print(header)
        print("-" * (22 + col_w * len(metrics)))

        for method, data in results.items():
            m_dict  = data["mean"]
            sd_dict = data["std"]
            row = f"{method:<22}"
            for k in metrics:
                val = m_dict.get(k, 0.0)
                std = sd_dict.get(k, 0.0)
                if k == "sla_rate":
                    cell = f"{val:.1%} ± {std:.1%}"
                else:
                    cell = f"{val:.3f} ± {std:.3f}"
                row += f"{cell:<{col_w}}"
            print(row)

        print("=" * (22 + col_w * len(metrics)))

    def save_results(self, results: Dict, path: str = "evaluation_results.json") -> None:
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        if self.verbose:
            print(f"\nResults saved to {path}")


# ------------------------------------------------------------------ #
# CLI entry point
# ------------------------------------------------------------------ #

def main():
    p = argparse.ArgumentParser(description="Evaluate AutoCloud-Agent vs baselines")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--n_episodes",     type=int, default=10)
    p.add_argument("--seeds",          type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--output",         type=str, default="evaluation_results.json")
    args = p.parse_args()

    evaluator = Evaluator(
        config=DEFAULT_CONFIG,
        checkpoint_dir=args.checkpoint_dir,
        n_episodes=args.n_episodes,
        seeds=args.seeds,
        verbose=True,
    )

    results = evaluator.evaluate_all()
    evaluator.print_table(results)
    evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()
