"""
AutoCloud-Agent Stress Test
===========================
Tests the agent on 4 high-value peak scenarios sliced from / inspired by
the Alibaba 2018 cluster trace.

Scenario 1 — Standard Ramp-Up        (Day 2, real data, steps 600-1000)
Scenario 2 — Early Shock             (Days 5 & 6 pattern, synthetic)
Scenario 3 — Sustained Choppy Plateau(Days 3 & 4 pattern, synthetic)
Scenario 4 — Deep Trough + Recovery  (Day 7 pattern, synthetic)

Usage:
    python stress_test.py \\
      --checkpoint_dir ../outputs/rl_agents \\
      --workload_file  ../outputs/train_Forecaster/day2_processed.npy
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from configs.default_config import DEFAULT_CONFIG
from environment.cloud_env import CloudEnv
from training.ippo_trainer import IPPOTrainer
from training.baselines import (KubernetesHPA, ThresholdReactive, StaticN,
                                AWSTargetTracking, MPCController, PIDDerivative, BurstAwareScaler)


# ─── Scenario builders ────────────────────────────────────────────────────────

def _make_fn(rates: np.ndarray, bin_size: int = 30):
    """Wrap a 1-D rates array into a CloudEnv-compatible workload_fn."""
    rates = np.clip(rates, 0.05, 1.0).astype(np.float32)
    n = len(rates)
    def fn(sim_time: float) -> float:
        return float(rates[int(sim_time / bin_size) % n])
    return fn


def scenario1_ramp_up(day2_path: str):
    """
    Day 2, steps 600–1000 — steady climb to 0.74, then plateau.
    Tests: baseline generalisation to unseen data.
    """
    data  = np.load(day2_path)
    rates = data[600:1000, 0]          # 400 steps ≈ 3.3 h real trace
    return _make_fn(rates), f"mean={rates.mean():.2f}  peak={rates.max():.2f}"


def scenario2_early_shock(seed: int = 0):
    """
    Days 5 & 6 pattern — massive CPU spike in first 150 steps, then moderate.
    Tests: Forecaster uncertainty, Safety Coordinator, reactive ScaleOut.
    """
    rng  = np.random.default_rng(seed)
    n    = 400
    base = np.full(n, 0.25)

    # Early spike: steps 50-150
    spike = np.zeros(n)
    spike[50:150] = 0.65 + 0.20 * rng.random(100)   # 0.65–0.85 burst

    # Moderate tail: steps 150-400
    tail = np.zeros(n)
    tail[150:] = 0.30 + 0.15 * rng.random(n - 150)

    rates = base + spike + tail + rng.normal(0, 0.02, n)
    return _make_fn(rates), f"mean={rates.mean():.2f}  peak={rates.max():.2f}  shock@step50-150"


def scenario3_choppy_plateau(seed: int = 1):
    """
    Days 3 & 4 pattern — sustained high-volatility oscillation 0.35–0.75.
    Tests: Consolidation anti-thrashing (don't drain then immediately regret).
    """
    rng  = np.random.default_rng(seed)
    n    = 650          # steps 150-800
    t    = np.arange(n)

    # Base plateau around 0.50
    base = 0.50 + 0.08 * np.sin(2 * np.pi * t / 40)   # 40-step cycle
    # Fast turbulence layered on top
    turb = 0.12 * np.sin(2 * np.pi * t / 12) + 0.06 * np.sin(2 * np.pi * t / 7)
    noise = rng.normal(0, 0.025, n)

    rates = base + turb + noise
    return _make_fn(rates), f"mean={rates.mean():.2f}  peak={rates.max():.2f}  chop_period~40steps"


def scenario4_trough_recovery(seed: int = 2):
    """
    Day 7 pattern — deep low-util trough (steps 0-800), then slow recovery.
    Tests: aggressive bin-packing at low load + smooth scale-up on recovery.
    """
    rng   = np.random.default_rng(seed)
    n     = 1380         # steps 1500-2880

    # Trough: first 800 steps very light (0.10-0.22)
    trough = 0.14 + 0.06 * rng.random(800)

    # Recovery: steps 800-1380 gradual ramp from 0.20 to 0.45
    ramp = np.linspace(0.20, 0.45, 580) + rng.normal(0, 0.025, 580)

    rates = np.concatenate([trough, ramp])
    return _make_fn(rates), f"mean={rates.mean():.2f}  trough={trough.mean():.2f}  recovery_peak={ramp.max():.2f}"


# ─── Episode runner ────────────────────────────────────────────────────────────

def run_episode(policy, config, workload_fn, seed: int = 42):
    env = CloudEnv(config=config, seed=seed, workload_fn=workload_fn)
    obs, _ = env.reset()
    if hasattr(policy, 'reset'):
        policy.reset()
    done = False
    sla_steps, total_steps = 0, 0
    while not done:
        if hasattr(policy, 'select_action'):
            action = policy.select_action(obs, env)
        else:
            a_so,  _, _ = policy.so_agent.act(obs)
            a_con, _, _ = policy.con_agent.act(obs, env.get_active_mask())
            a_sch, _, _ = policy.sch_agent.act(obs)
            action = {'scaleout': int(a_so), 'consolidation': a_con, 'scheduling': int(a_sch)}
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        m = info['metrics']
        sla_steps   += int(m['p95_latency'] < config.reward.sla_latency_ms or m['p95_latency'] == 0)
        total_steps += 1
    sla_rate = sla_steps / max(total_steps, 1)
    cost_eff = 1.0 - info['metrics']['total_cost'] / 10.0   # normalised
    cost_eff = float(np.clip(cost_eff, 0, 1))
    stability = info['metrics'].get('node_stability', float(info['metrics'].get('n_active', 10)) / 20)
    cpu_util  = info['metrics'].get('mean_cpu_util', 0.0)
    return sla_rate, cost_eff, cpu_util, stability


def evaluate_scenario(name, profile_str, workload_fn, trainer, config, seeds=(0, 1, 2)):
    """Run AutoCloud-Agent + all baselines on a scenario, return result dict."""
    policies = {
        'AutoCloud-Agent':   trainer,
        # Industry-standard
        'KubernetesHPA':     KubernetesHPA(),
        'AWSTargetTracking': AWSTargetTracking(),
        # Classical control
        'PIDerivative':      PIDDerivative(),
        'MPCController':     MPCController(),
        # Burst-aware heuristic
        'BurstAwareScaler':  BurstAwareScaler(),
        'ThresholdReactive': ThresholdReactive(),
        # Do-nothing
        'StaticN(10)':       StaticN(10),
    }
    results = {}
    for pol_name, pol in policies.items():
        sla_list, ceff_list, cpu_list, stab_list = [], [], [], []
        for s in seeds:
            sla, ceff, cpu, stab = run_episode(pol, config, workload_fn, seed=s)
            sla_list.append(sla)
            ceff_list.append(ceff)
            cpu_list.append(cpu)
            stab_list.append(stab)
        results[pol_name] = {
            'sla':  (np.mean(sla_list),  np.std(sla_list)),
            'ceff': (np.mean(ceff_list), np.std(ceff_list)),
            'cpu':  (np.mean(cpu_list),  np.std(cpu_list)),
            'stab': (np.mean(stab_list), np.std(stab_list)),
        }
    return results


def print_scenario(name, profile_str, what_it_tests, results):
    print(f"\n{'='*72}")
    print(f"  {name}")
    print(f"  Profile : {profile_str}")
    print(f"  Tests   : {what_it_tests}")
    print(f"{'='*72}")
    print(f"  {'Method':<22} {'SLA Rate':>12} {'Cost Eff':>12} {'CPU Util':>12} {'Stability':>12}")
    print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for method, r in results.items():
        bold = "  **" if method == 'AutoCloud-Agent' else "    "
        print(
            f"{bold}{method:<22}"
            f"  {r['sla'][0]:>8.1%}±{r['sla'][1]:.1%}"
            f"  {r['ceff'][0]:>7.3f}±{r['ceff'][1]:.3f}"
            f"  {r['cpu'][0]:>8.1%}±{r['cpu'][1]:.1%}"
            f"  {r['stab'][0]:>8.3f}±{r['stab'][1]:.3f}"
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir",  default="checkpoints")
    p.add_argument("--workload_file",   default=None,
                   help="Path to day2_processed.npy (needed for Scenario 1)")
    p.add_argument("--stress_dir",      default=None,
                   help="Dir with real .npy windows from extract_stress_windows.ipynb")
    p.add_argument("--seeds",           type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--scenarios",       type=int, nargs="+", default=[1, 2, 3, 4],
                   help="Which scenarios to run (default: all)")
    return p.parse_args()


def main():
    args = parse_args()
    config = DEFAULT_CONFIG

    print("\n AutoCloud-Agent Stress Test")
    print(f" Checkpoints : {args.checkpoint_dir}")
    print(f" Seeds       : {args.seeds}")
    print(f" Scenarios   : {args.scenarios}")

    # Load trained agents
    trainer = IPPOTrainer(config=config, seed=0, device='cpu', verbose=False)
    ckpt_path = os.path.join(args.checkpoint_dir, 'so_actor_final.pt')
    if os.path.exists(ckpt_path):
        trainer.load(args.checkpoint_dir, tag='final')
        print(f" Checkpoints loaded from {args.checkpoint_dir}\n")
    else:
        print(f" WARNING: no checkpoints at {args.checkpoint_dir} — using random weights\n")

    def _real_or_synthetic(real_name, synthetic_fn):
        """Use real .npy from stress_dir if available, otherwise synthetic."""
        if args.stress_dir:
            p = os.path.join(args.stress_dir, real_name)
            if os.path.exists(p):
                data  = np.load(p)
                rates = data[:, 0] if data.ndim > 1 else data
                tag   = f"REAL  mean={rates.mean():.2f}  peak={rates.max():.2f}"
                return _make_fn(rates), tag
        fn, tag = synthetic_fn()
        return fn, "SYNTHETIC  " + tag

    scenarios = {
        1: {
            'name': "Scenario 1 — Standard Ramp-Up (Day 2, steps 600-1000)",
            'tests': "Baseline generalisation to unseen data; SLA should stay 100%",
            'build': lambda: scenario1_ramp_up(args.workload_file)
                             if args.workload_file and os.path.exists(args.workload_file)
                             else _real_or_synthetic("s1_rampup_day2_600_1000.npy",
                                                     lambda: (None, "SKIPPED")),
        },
        2: {
            'name': "Scenario 2 — Early Shock (Days 5 & 6, steps 0-400)",
            'tests': "Forecaster uncertainty spike + reactive ScaleOut under surprise burst",
            'build': lambda: _real_or_synthetic("s2_earlyshock_day5_0_400.npy",
                                                scenario2_early_shock),
        },
        3: {
            'name': "Scenario 3 — Sustained Choppy Plateau (Days 3 & 4, steps 150-800)",
            'tests': "Consolidation anti-thrashing through rapid oscillating load",
            'build': lambda: _real_or_synthetic("s3_choppy_day3_150_800.npy",
                                                scenario3_choppy_plateau),
        },
        4: {
            'name': "Scenario 4 — Deep Trough + Recovery (Day 7, steps 1500-2880)",
            'tests': "Aggressive bin-packing at low load + smooth scale-up on recovery",
            'build': lambda: _real_or_synthetic("s4_trough_day7_1500_2880.npy",
                                                scenario4_trough_recovery),
        },
    }

    all_results = {}
    for idx in args.scenarios:
        s = scenarios[idx]
        fn, profile_str = s['build']()
        if fn is None:
            print(f"\n[Stress] {s['name']} — {profile_str}")
            continue
        print(f"\n[Stress] Running {s['name']} ...")
        results = evaluate_scenario(
            s['name'], profile_str, fn, trainer, config, seeds=args.seeds
        )
        all_results[idx] = results
        print_scenario(s['name'], profile_str, s['tests'], results)

    # Summary table
    print(f"\n\n{'='*72}")
    print("  SUMMARY — AutoCloud-Agent across all scenarios")
    print(f"{'='*72}")
    print(f"  {'Scenario':<42} {'SLA':>8} {'CostEff':>9} {'CPUUtil':>9} {'Stability':>10}")
    print(f"  {'-'*42} {'-'*8} {'-'*9} {'-'*9} {'-'*10}")
    labels = {
        1: "S1 Ramp-Up     (Day2 real, high load)",
        2: "S2 Early Shock (surprise burst t=50)",
        3: "S3 Choppy      (anti-thrash test)",
        4: "S4 Trough      (cost squeeze + recovery)",
    }
    for idx, res in all_results.items():
        r = res['AutoCloud-Agent']
        print(
            f"  {labels[idx]:<42}"
            f"  {r['sla'][0]:>6.1%}"
            f"  {r['ceff'][0]:>8.3f}"
            f"  {r['cpu'][0]:>8.1%}"
            f"  {r['stab'][0]:>9.3f}"
        )
    print()


if __name__ == "__main__":
    main()
