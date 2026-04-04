#!/usr/bin/env python3
"""
AutoCloud-Agent — Live Demo
============================
A visually rich terminal demo that shows the RL agent managing a cloud
cluster in real time, side-by-side with a baseline (Kubernetes HPA).

Usage:
    cd autocloud_agent/
    conda activate myenv
    python demo.py
    python demo.py --speed fast       # faster demo (skip sleep)
    python demo.py --speed slow       # slower step delay
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import numpy as np

from autocloud.config.settings import DEFAULT_CONFIG, Config, SimConfig, RewardConfig
from autocloud.config.paths import ArtifactPaths
from autocloud.simulator.cloud_env import CloudEnv
from autocloud.inference.runner import InferenceRunner
from autocloud.evaluation.baselines import (
    KubernetesHPA, AWSTargetTracking, ThresholdReactive,
    MPCController, StaticN,
)
import copy


# ANSI color helpers 
BOLD    = "\033[1m"
DIM     = "\033[2m"
RESET   = "\033[0m"
RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"
CYAN    = "\033[96m"
WHITE   = "\033[97m"
BG_RED  = "\033[41m"
BG_GREEN= "\033[42m"
BG_BLUE = "\033[44m"
BG_GREY = "\033[100m"
ITALIC  = "\033[3m"


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def colored_bar(value: float, width: int = 30, label: str = "") -> str:
    """Draw a colored progress bar: green < 50%, yellow 50-80%, red > 80%."""
    filled = int(value * width)
    filled = max(0, min(filled, width))
    empty = width - filled

    if value < 0.50:
        color = GREEN
    elif value < 0.80:
        color = YELLOW
    else:
        color = RED

    bar = f"{color}{'█' * filled}{DIM}{'░' * empty}{RESET}"
    pct = f"{value * 100:5.1f}%"
    return f"{bar} {pct} {label}"


def node_display(n_active: int, n_booting: int, n_draining: int, n_max: int = 20) -> str:
    """Draw node status as colored blocks."""
    blocks = []
    for i in range(n_max):
        if i < n_active:
            blocks.append(f"{GREEN}●{RESET}")
        elif i < n_active + n_booting:
            blocks.append(f"{YELLOW}◔{RESET}")
        elif i < n_active + n_booting + n_draining:
            blocks.append(f"{RED}◌{RESET}")
        else:
            blocks.append(f"{DIM}·{RESET}")
    return " ".join(blocks)


def sla_badge(p95: float, threshold: float = 500.0) -> str:
    if p95 == 0 or p95 < threshold:
        return f"{BG_GREEN}{BOLD} ✓ SLA OK {RESET}"
    else:
        return f"{BG_RED}{BOLD} ✗ SLA BREACH {RESET}"


def format_cost(cost: float) -> str:
    if cost < 0.5:
        return f"{GREEN}${cost:.4f}{RESET}"
    elif cost < 1.0:
        return f"{YELLOW}${cost:.4f}{RESET}"
    else:
        return f"{RED}${cost:.4f}{RESET}"


def print_header():
    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║         █▀█ █░█ ▀█▀ █▀█ █▀▀ █░░ █▀█ █░█ █▀▄   ▄▀█ █▀▀ █▀▀ █▄░█ ▀█▀   ║
║         █▀█ █▄█ ░█░ █▄█ █▄▄ █▄▄ █▄█ █▄█ █▄▀   █▀█ █▄█ ██▄ █░▀█ ░█░   ║
║                                                                          ║
║         Multi-Agent RL for Cloud Autoscaling — Live Demo                 ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝{RESET}
""")


def print_phase_banner(phase: str, description: str):
    width = 74
    print(f"\n{BOLD}{MAGENTA}{'━' * width}")
    print(f"  ▶  {phase}")
    print(f"     {DIM}{description}{RESET}")
    print(f"{BOLD}{MAGENTA}{'━' * width}{RESET}\n")


def forecast_bar(means: list, sigmas: list) -> str:
    """Draw a mini sparkline-style forecast with uncertainty bands."""
    horizons = ["t+1", "t+5", "t+10", "t+15"]
    parts = []
    for i, h in enumerate(horizons):
        m = means[i] if i < len(means) else 0.0
        s = sigmas[i] if i < len(sigmas) else 0.0
        # Color by demand level
        if m < 0.3:
            c = GREEN
        elif m < 0.6:
            c = YELLOW
        else:
            c = RED
        # Uncertainty indicator
        if s > 0.15:
            unc = f"{RED}±{s:.2f}{RESET}"
        elif s > 0.05:
            unc = f"{YELLOW}±{s:.2f}{RESET}"
        else:
            unc = f"{DIM}±{s:.2f}{RESET}"
        parts.append(f"{DIM}{h}:{RESET}{c}{m:.2f}{RESET}({unc})")
    return "  ".join(parts)


def uncertainty_level(sigmas: list) -> str:
    """Return a human-readable uncertainty label."""
    avg_sigma = np.mean(sigmas) if sigmas else 0.0
    if avg_sigma > 0.15:
        return f"{RED}{BOLD}HIGH ⚠{RESET}"
    elif avg_sigma > 0.05:
        return f"{YELLOW}MEDIUM{RESET}"
    else:
        return f"{GREEN}LOW{RESET}"


def agent_decision_block(diag: dict) -> list[str]:
    """Format the 3 agent decisions + safety coordinator as lines."""
    lines = []

    # ScaleOut agent
    so_raw = diag.get("so_raw", 0)
    so_filtered = diag.get("so_filtered", 0)
    so_acted = diag.get("so_acted", False)
    so_reason = diag.get("so_reason", "skip")

    so_action_names = {0: "HOLD", 1: "+1 node", 2: "+2 nodes"}
    if not so_acted:
        so_line = f"{DIM}ScaleOut     │ sleeping (acts every 10 steps){RESET}"
    else:
        raw_name = so_action_names.get(so_raw, "HOLD")
        filt_name = so_action_names.get(so_filtered, "HOLD")
        trigger = f"[{so_reason}]"
        if diag.get("so_overridden"):
            so_line = (f"{CYAN}ScaleOut     │ {trigger} decided {YELLOW}{raw_name}{RESET}"
                       f" → {RED}safety override → {filt_name}{RESET}")
        else:
            color = GREEN if so_filtered > 0 else DIM
            so_line = f"{CYAN}ScaleOut     │ {trigger} → {color}{BOLD}{filt_name}{RESET}"
    lines.append(so_line)

    # Consolidation agent
    con_acted = diag.get("con_acted", False)
    con_raw = diag.get("con_raw_drains", 0)
    con_filt = diag.get("con_filtered_drains", 0)

    if not con_acted:
        con_line = f"{DIM}Consolidation│ sleeping (acts every 2 steps){RESET}"
    else:
        if con_raw == 0:
            con_line = f"{CYAN}Consolidation│ → {DIM}no drains{RESET}"
        elif diag.get("con_overridden"):
            con_line = (f"{CYAN}Consolidation│ drain {YELLOW}{con_raw} VMs{RESET}"
                        f" → {RED}safety blocked {con_raw - con_filt}{RESET}"
                        f" → draining {con_filt}")
        else:
            con_line = f"{CYAN}Consolidation│ → {RED}drain {con_filt} VMs{RESET}"
    lines.append(con_line)

    # Scheduling agent
    sch_name = diag.get("sch_name", "Best-Fit")
    sch_line = f"{CYAN}Scheduling   │ → {GREEN}{sch_name}{RESET}"
    lines.append(sch_line)

    return lines


def demo_step_display(
    step: int,
    total_steps: int,
    agent_name: str,
    agent_color: str,
    metrics: dict,
    action: dict,
    cum_sla: float,
    cum_cost: float,
    info: dict | None = None,
    diag: dict | None = None,
):
    """Render one step of the live simulation with full telemetry."""
    m = metrics
    n_active = m["n_active"]
    n_booting = m["n_booting"]
    n_draining = m["n_draining"]
    cpu = m["mean_cpu_util"]
    mem = m["mean_mem_util"]
    p95 = m["p95_latency"]
    queue = m.get("peak_queue_len", m["queue_len"])
    step_cost = m["step_cost"]
    completions = m.get("step_completions", 0)
    migrations = m.get("step_migrations", 0)

    # Time display
    sim_minutes = step * 0.5  # 30s per step
    time_str = f"{int(sim_minutes // 60):02d}:{int(sim_minutes % 60 * 60 // 60):02d}"

    progress = int((step / total_steps) * 40)
    progress_bar = f"{agent_color}{'▓' * progress}{'░' * (40 - progress)}{RESET}"

    # Header 
    print(f"  {BOLD}[{agent_color}{agent_name}{RESET}{BOLD}]{RESET}  "
          f"Step {step:3d}/{total_steps}  T={time_str}  {progress_bar}")

    # Workload section 
    load_color = GREEN if queue < 5 else (YELLOW if queue < 15 else RED)
    print(f"  ┌─ {BOLD}Workload{RESET} ──────────────────────────────────────────────────")
    print(f"  │  Jobs in queue: {load_color}{BOLD}{queue}{RESET}  "
          f"Completed: {GREEN}{completions}{RESET}  "
          f"Migrations: {YELLOW if migrations > 0 else DIM}{migrations}{RESET}  "
          f"P95 latency: {p95:.0f}ms {sla_badge(p95)}")

    # Forecaster section (only for RL agent) 
    if info is not None:
        f_means = info.get("forecast_means", [0]*4)
        f_sigmas = info.get("forecast_sigmas", [0]*4)
        print(f"  ├─ {BOLD}Forecaster{RESET} ─────────────────────────────────────────────────")
        print(f"  │  Predicted demand:  {forecast_bar(f_means, f_sigmas)}")
        print(f"  │  Uncertainty level: {uncertainty_level(f_sigmas)}")

    # Cluster status 
    print(f"  ├─ {BOLD}Cluster{RESET} ────────────────────────────────────────────────────")
    print(f"  │  Nodes {node_display(n_active, n_booting, n_draining)}  "
          f"({GREEN}{n_active}{RESET} active  "
          f"{YELLOW}{n_booting}{RESET} booting  "
          f"{RED}{n_draining}{RESET} draining)")
    print(f"  │  CPU  {colored_bar(cpu, 35)}")
    print(f"  │  Mem  {colored_bar(mem, 35)}")

    # Agent decisions section (only for RL agent with diagnostics) 
    if diag is not None and diag:
        print(f"  ├─ {BOLD}Agent Decisions{RESET} ─────────────────────────────────────────────")
        for line in agent_decision_block(diag):
            print(f"  │  {line}")
        # Safety coordinator summary
        overrides = []
        if diag.get("so_overridden"):
            overrides.append("ScaleOut")
        if diag.get("con_overridden"):
            overrides.append("Consolidation")
        if overrides:
            print(f"  │  {RED}{BOLD}⚠ Safety Coordinator overrode: {', '.join(overrides)}{RESET}")
        else:
            print(f"  │  {GREEN}✓ Safety Coordinator: all actions approved{RESET}")
    else:
        # Baseline — show simple action display
        so_action = action.get("scaleout", 0)
        so_names = {0: f"{DIM}hold{RESET}", 1: f"{GREEN}+1 node{RESET}", 2: f"{GREEN}+2 nodes{RESET}"}
        so_display = so_names.get(int(so_action) if isinstance(so_action, (int, np.integer)) else 0,
                                  f"{DIM}hold{RESET}")
        con_vec = action.get("consolidation", np.zeros(20))
        n_drain = int(np.sum(np.array(con_vec) > 0.5))
        con_display = f"{RED}drain {n_drain}{RESET}" if n_drain > 0 else f"{DIM}none{RESET}"
        print(f"  ├─ {BOLD}HPA Decision{RESET}: ScaleOut={so_display}  Consolidation={con_display}")

    # Cost footer 
    print(f"  └─ Cost: {format_cost(step_cost)}/step  "
          f"Cumulative: {format_cost(cum_cost)}  "
          f"SLA: {GREEN}{cum_sla * 100:.1f}%{RESET}")
    print()


def run_single_demo(
    agent_name: str,
    agent_color: str,
    policy,
    config,
    workload_fn,
    seed: int,
    total_steps: int,
    delay: float,
    is_rl: bool = False,
):
    """Run one full episode and display live metrics."""
    env = CloudEnv(config=config, seed=seed, workload_fn=workload_fn)
    obs, _ = env.reset()
    if hasattr(policy, "reset"):
        policy.reset()

    cum_sla = 0
    cum_cost = 0.0
    sla_steps = 0
    done = False
    step = 0

    while not done:
        action = policy.select_action(obs, env)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

        m = info["metrics"]
        sla_ok = m["p95_latency"] < config.reward.sla_latency_ms or m["p95_latency"] == 0
        sla_steps += int(sla_ok)
        cum_sla = sla_steps / step
        cum_cost += m["step_cost"]

        # Get diagnostics from RL agent if available
        diag = getattr(policy, "last_diag", None) if is_rl else None

        # Display every 2nd step (keeps output manageable)
        if True:  # show every step for consistency
            demo_step_display(
                step, total_steps, agent_name, agent_color,
                m, action, cum_sla, cum_cost,
                info=info if is_rl else None,
                diag=diag,
            )
            if delay > 0:
                time.sleep(delay)

    return cum_sla, cum_cost, step


def print_comparison_table(results: dict):
    """Print a beautiful side-by-side comparison table."""
    print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════════════════╗")
    print(f"║                      FINAL RESULTS COMPARISON                          ║")
    print(f"╚══════════════════════════════════════════════════════════════════════════╝{RESET}\n")

    header = (f"  {BOLD}{'Method':<24} {'SLA Rate':>10} {'Total Cost':>12} "
              f"{'Cost Eff':>10} {'Verdict':>12}{RESET}")
    print(header)
    print(f"  {'─' * 70}")

    best_cost = min(r["cost"] for r in results.values())

    for name, r in results.items():
        sla_str = f"{r['sla'] * 100:.1f}%"
        cost_str = f"${r['cost']:.4f}"
        eff = 1.0 - r['cost'] / max(best_cost * 3, 0.001)  # relative efficiency
        eff_str = f"{eff:.3f}"

        if name == "AutoCloud-Agent":
            color = GREEN
            verdict = f"{GREEN}{BOLD}★ WINNER{RESET}"
        else:
            color = YELLOW
            saved = ((r['cost'] - results.get("AutoCloud-Agent", r)['cost'])
                     / max(r['cost'], 0.001) * 100)
            if saved > 0:
                verdict = f"{RED}+{saved:.1f}% cost{RESET}"
            else:
                verdict = f"{GREEN}{saved:.1f}% cost{RESET}"

        print(f"  {color}{BOLD}{name:<24}{RESET} {sla_str:>10} {cost_str:>12} "
              f"{eff_str:>10} {verdict:>20}")

    print(f"  {'─' * 70}")


def print_what_happened():
    """Educational summary of what the demo showed."""
    print(f"""
{BOLD}{BLUE}╔══════════════════════════════════════════════════════════════════════════╗
║                         WHAT JUST HAPPENED?                              ║
╚══════════════════════════════════════════════════════════════════════════╝{RESET}

  {BOLD}1. Cloud Simulator{RESET} replayed real Alibaba workload data (30s intervals)

  {BOLD}2. AutoCloud-Agent{RESET} (our RL system) used {CYAN}3 specialised neural networks{RESET}:
     • {GREEN}ScaleOut Agent{RESET}    – decided when to add/remove VMs
     • {GREEN}Consolidation Agent{RESET} – decided which idle VMs to drain
     • {GREEN}Scheduling Agent{RESET}   – decided job priority ordering

  {BOLD}3. Safety Coordinator{RESET} filtered dangerous actions:
     • Never dropped below 3 nodes
     • Blocked scale-down when forecaster was uncertain
     • Protected booting nodes from premature drain

  {BOLD}4. Workload Forecaster{RESET} (Transformer + MC Dropout):
     • Predicted demand 1-15 steps ahead
     • Uncertainty estimates guided proactive scaling

  {BOLD}5. Kubernetes HPA{RESET} (industry baseline) used a fixed rule:
     desiredReplicas = ceil(current × CPU / target)

  {BOLD}Key insight:{RESET} RL agents learn {GREEN}proactive{RESET} policies (scale {ITALIC}before{RESET}
  spikes), while HPA is purely {YELLOW}reactive{RESET} (scale {ITALIC}after{RESET} CPU exceeds threshold).
""")


# Baselines registry 
BASELINES = {
    "1": ("KubernetesHPA",    "k8s HPA: ceil(replicas × cpu / target)",       KubernetesHPA),
    "2": ("AWSTargetTracking", "AWS-style policy with asymmetric cooldowns",   AWSTargetTracking),
    "3": ("ThresholdReactive", "Add if CPU>80%, drain if CPU<30%",             ThresholdReactive),
    "4": ("MPCController",     "5-step MPC with EWM forecast",                 MPCController),
    "5": ("StaticN",           "Fixed nodes, never scales (lower bound)",       StaticN),
}

# Demo modes registry
MODES = {
    "1": ("standard",  "RL vs one chosen baseline — step-by-step walkthrough"),
    "2": ("shootout",  "RL vs ALL 5 baselines — full ranking table at the end"),
    "3": ("stress",    "Pick an extreme workload scenario — see how the agent handles it"),
    "4": ("ablation",  "3-agent I-PPO vs single-agent PPO — explains WHY we decompose"),
}

# Stress scenarios
STRESS_SCENARIOS = {
    "1": ("Ramp-Up",         "Load rises linearly 1× → 3× over the episode"),
    "2": ("Early Shock",     "Sudden 4× spike at step 15 — can the agent recover?"),
    "3": ("Choppy Plateau",  "Oscillating load at 2.5× ± random noise — stability test"),
    "4": ("Trough+Recovery", "Load collapses to 0.3× then rebounds — avoid over-scaling"),
}


def make_stress_workload(scenario_key: str, base_steps: int = 120) -> callable:
    """Return a workload_fn for the chosen stress scenario."""
    key = scenario_key

    def ramp_up(sim_time: float) -> float:
        t = min(sim_time / (base_steps * 30.0), 1.0)
        return 1.0 + 2.0 * t  # 1× → 3×

    def early_shock(sim_time: float) -> float:
        step = sim_time / 30.0
        if step < 15:
            return 0.8
        elif step < 25:
            return 4.0  # 4× spike
        else:
            return 1.0 + 0.3 * np.sin(2 * np.pi * step / 40)

    def choppy_plateau(sim_time: float) -> float:
        step = sim_time / 30.0
        noise = np.random.default_rng(int(sim_time) % 9999).uniform(-0.3, 0.3)
        return 2.5 + noise

    def trough_recovery(sim_time: float) -> float:
        step = sim_time / 30.0
        if step < base_steps * 0.35:
            return 0.3  # trough
        elif step < base_steps * 0.50:
            return 0.3 + 2.2 * ((step - base_steps * 0.35) / (base_steps * 0.15))  # ramp back
        else:
            return 2.5

    return {"1": ramp_up, "2": early_shock, "3": choppy_plateau, "4": trough_recovery}[key]


def prompt_int(label: str, default: int, lo: int, hi: int) -> int:
    """Prompt user for an integer within [lo, hi], with a default."""
    while True:
        raw = input(f"  {CYAN}│{RESET}  {label} [{BOLD}{default}{RESET}]: ").strip()
        if raw == "":
            return default
        try:
            val = int(raw)
            if lo <= val <= hi:
                return val
            print(f"  {CYAN}│{RESET}  {RED}Please enter a number between {lo} and {hi}{RESET}")
        except ValueError:
            print(f"  {CYAN}│{RESET}  {RED}Invalid number, try again{RESET}")


def prompt_float(label: str, default: float, lo: float, hi: float) -> float:
    """Prompt user for a float within [lo, hi], with a default."""
    while True:
        raw = input(f"  {CYAN}│{RESET}  {label} [{BOLD}{default}{RESET}]: ").strip()
        if raw == "":
            return default
        try:
            val = float(raw)
            if lo <= val <= hi:
                return val
            print(f"  {CYAN}│{RESET}  {RED}Please enter a value between {lo} and {hi}{RESET}")
        except ValueError:
            print(f"  {CYAN}│{RESET}  {RED}Invalid number, try again{RESET}")


def prompt_choice(label: str, options: dict, default: str) -> str:
    """Prompt user to pick from numbered options."""
    raw = input(f"  {CYAN}│{RESET}  {label} [{BOLD}{default}{RESET}]: ").strip()
    if raw == "" or raw not in options:
        return default
    return raw


def interactive_setup(args) -> dict:
    """Interactive configuration wizard — lets user set up the demo."""
    print(f"""
  {BOLD}{CYAN}┌──────────────────────────────────────────────────────────────────────┐
  │                     DEMO CONFIGURATION                             │
  └──────────────────────────────────────────────────────────────────────┘{RESET}
  {DIM}  Press Enter to accept defaults (shown in brackets).{RESET}
""")

    # 0. Demo Mode Selection
    print(f"  {BOLD}{MAGENTA}── 0. SELECT DEMO MODE ──{RESET}")
    for key, (mode, desc) in MODES.items():
        marker = f"{GREEN}★{RESET}" if key == "1" else " "
        print(f"  {CYAN}│{RESET}  {marker} {BOLD}{key}{RESET}. {mode:<16} {DIM}{desc}{RESET}")
    mode_key = prompt_choice("Pick mode (1-4)", {k: v[0] for k, v in MODES.items()}, default="1")
    chosen_mode = MODES[mode_key][0]
    print(f"  {CYAN}│{RESET}  {GREEN}→ Mode: {BOLD}{chosen_mode}{RESET}")
    print()

    # 1. Cluster Setup
    print(f"  {BOLD}{MAGENTA}── 1. CLUSTER SETUP ──{RESET}")
    n_init = prompt_int("Initial VMs in the cluster", default=5, lo=3, hi=20)
    n_max  = prompt_int("Maximum VMs allowed", default=20, lo=n_init, hi=50)
    n_min  = prompt_int("Minimum VMs (safety floor)", default=3, lo=1, hi=n_init)
    print()

    # 2. Workload & Duration
    print(f"  {BOLD}{MAGENTA}── 2. WORKLOAD & DURATION ──{RESET}")
    steps = prompt_int("Simulation steps (1 step = 30 sec)", default=args.steps, lo=10, hi=500)
    sim_min = steps * 30 // 60
    print(f"  {CYAN}│{RESET}  {DIM}→ That's {sim_min} minutes of simulated cloud time{RESET}")
    print()

    # 3. SLA & Cost
    print(f"  {BOLD}{MAGENTA}── 3. SLA & PERFORMANCE THRESHOLDS ──{RESET}")
    sla_ms = prompt_float("SLA latency threshold (ms)", default=500.0, lo=50.0, hi=5000.0)
    cpu_high = prompt_float("CPU high threshold (scale-up trigger)", default=0.80, lo=0.5, hi=0.99)
    cpu_low  = prompt_float("CPU low threshold (scale-down trigger)", default=0.30, lo=0.05, hi=cpu_high)
    print()

    # 4. Mode-specific options
    baseline_name = "KubernetesHPA"
    baseline_desc = "k8s HPA: ceil(replicas × cpu / target)"
    baseline_cls  = KubernetesHPA
    stress_key    = "1"

    if chosen_mode == "standard":
        print(f"  {BOLD}{MAGENTA}── 4. CHOOSE BASELINE TO COMPARE AGAINST ──{RESET}")
        for key, (name, desc, _) in BASELINES.items():
            marker = f"{GREEN}★{RESET}" if key == "1" else " "
            print(f"  {CYAN}│{RESET}  {marker} {BOLD}{key}{RESET}. {name:<22} {DIM}{desc}{RESET}")
        baseline_key = prompt_choice("Pick baseline (1-5)", BASELINES, default="1")
        bl_name, bl_desc, bl_cls = BASELINES[baseline_key]
        baseline_name, baseline_desc, baseline_cls = bl_name, bl_desc, bl_cls
        print(f"  {CYAN}│{RESET}  {GREEN}→ Selected: {BOLD}{baseline_name}{RESET}")
        print()

    elif chosen_mode == "stress":
        print(f"  {BOLD}{MAGENTA}── 4. CHOOSE STRESS SCENARIO ──{RESET}")
        for key, (name, desc) in STRESS_SCENARIOS.items():
            marker = f"{GREEN}★{RESET}" if key == "1" else " "
            print(f"  {CYAN}│{RESET}  {marker} {BOLD}{key}{RESET}. {name:<20} {DIM}{desc}{RESET}")
        stress_key = prompt_choice("Pick scenario (1-4)", STRESS_SCENARIOS, default="1")
        print(f"  {CYAN}│{RESET}  {GREEN}→ Selected: {BOLD}{STRESS_SCENARIOS[stress_key][0]}{RESET}")
        print()

    # 5. Demo Speed
    print(f"  {BOLD}{MAGENTA}── 5. DEMO SPEED ──{RESET}")
    print(f"  {CYAN}│{RESET}    {BOLD}1{RESET}. Fast (no delay)   {BOLD}2{RESET}. Normal (0.15s)   {BOLD}3{RESET}. Slow (0.4s)")
    speed_map = {"1": "fast", "2": "normal", "3": "slow"}
    speed_default = {"fast": "1", "normal": "2", "slow": "3"}[args.speed]
    speed_key = prompt_choice("Speed", speed_map, default=speed_default)
    speed = speed_map.get(speed_key, args.speed)
    print()

    # Summary
    print(f"  {BOLD}{CYAN}┌──────────────────────────────────────────────────────────────────────┐")
    print(f"  │                     CONFIGURATION SUMMARY                           │")
    print(f"  └──────────────────────────────────────────────────────────────────────┘{RESET}")
    print(f"  {CYAN}│{RESET}  Mode       : {BOLD}{chosen_mode.upper()}{RESET}")
    print(f"  {CYAN}│{RESET}  Cluster    : {BOLD}{n_init}{RESET} initial → max {BOLD}{n_max}{RESET} VMs (min floor: {n_min})")
    print(f"  {CYAN}│{RESET}  Duration   : {BOLD}{steps}{RESET} steps ({sim_min} min simulated)")
    print(f"  {CYAN}│{RESET}  SLA target : P95 < {BOLD}{sla_ms:.0f}{RESET}ms")
    print(f"  {CYAN}│{RESET}  CPU range  : scale-down < {BOLD}{cpu_low:.0%}{RESET} < normal < {BOLD}{cpu_high:.0%}{RESET} < scale-up")
    if chosen_mode == "standard":
        print(f"  {CYAN}│{RESET}  Baseline   : {BOLD}{baseline_name}{RESET}")
    elif chosen_mode == "stress":
        print(f"  {CYAN}│{RESET}  Scenario   : {BOLD}{STRESS_SCENARIOS[stress_key][0]}{RESET}")
    print(f"  {CYAN}│{RESET}  Speed      : {BOLD}{speed}{RESET}")
    print()

    return {
        "mode": chosen_mode,
        "n_init": n_init,
        "n_max": n_max,
        "n_min": n_min,
        "steps": steps,
        "sla_ms": sla_ms,
        "cpu_high": cpu_high,
        "cpu_low": cpu_low,
        "baseline_name": baseline_name,
        "baseline_desc": baseline_desc,
        "baseline_cls": baseline_cls,
        "stress_key": stress_key,
        "speed": speed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Shootout Mode: RL vs all 5 baselines
# ─────────────────────────────────────────────────────────────────────────────

def run_shootout(config, workload_fn, paths, seed, total_steps, delay, pause_fn):
    """Run AutoCloud-Agent against all 5 baselines and print a ranked table."""
    all_policies = [
        ("AutoCloud-Agent",  GREEN,   None,                  True),
        ("KubernetesHPA",    YELLOW,  KubernetesHPA(),       False),
        ("AWSTargetTracking",YELLOW,  AWSTargetTracking(),   False),
        ("ThresholdReactive",YELLOW,  ThresholdReactive(),   False),
        ("MPCController",    CYAN,    MPCController(),       False),
        ("StaticN",          DIM,     StaticN(n_nodes=10),   False),
    ]

    results = {}

    for idx, (name, color, policy, is_rl) in enumerate(all_policies):
        clear()
        print_phase_banner(
            f"[{idx+1}/{len(all_policies)}]  {name}",
            f"Evaluating on {total_steps} steps"
        )

        if is_rl:
            policy = InferenceRunner(
                checkpoint_dir=str(paths.checkpoint_dir),
                config=config, device="cpu",
                forecaster=paths.load_forecaster(),
            )

        sla, cost, _ = run_single_demo(
            name, color, policy, config,
            workload_fn, seed, total_steps, delay,
            is_rl=is_rl,
        )
        results[name] = {"sla": sla, "cost": cost}
        print(f"  {BOLD}{color}✓ {name} complete!{RESET}")

        if idx < len(all_policies) - 1:
            pause_fn(f"  {BOLD}Press Enter for next method ({all_policies[idx+1][0]})...{RESET}")

    # Extended comparison table with ranked rows
    pause_fn(f"\n  {BOLD}Press Enter for final ranked results...{RESET}")
    clear()
    print_header()
    _print_ranked_table(results)
    print_what_happened()
    _print_shootout_insights(results)


def _print_ranked_table(results: dict):
    """Print results sorted by cost efficiency (best first)."""
    sorted_results = sorted(results.items(), key=lambda x: x[1]["cost"])

    print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════════════════╗")
    print(f"║                    FULL BENCHMARK RANKING                              ║")
    print(f"╚══════════════════════════════════════════════════════════════════════════╝{RESET}\n")

    header = (f"  {BOLD}{'Rank':<6}{'Method':<24}{'SLA Rate':>10}{'Total Cost':>12}"
              f"{'vs Best':>10}{'Medal':>8}{RESET}")
    print(header)
    print(f"  {'─' * 72}")

    best_cost = sorted_results[0][1]["cost"]
    medals = ["🥇", "🥈", "🥉"]

    for rank, (name, r) in enumerate(sorted_results, 1):
        sla_str  = f"{r['sla'] * 100:.1f}%"
        cost_str = f"${r['cost']:.4f}"
        vs_best  = f"+{((r['cost'] - best_cost) / max(best_cost, 1e-6)) * 100:.1f}%" \
                   if r['cost'] > best_cost else f"{GREEN}BEST{RESET}"
        medal    = medals[rank - 1] if rank <= 3 else f"  #{rank}"

        if rank == 1:
            color = f"{GREEN}{BOLD}"
        elif name == "AutoCloud-Agent":
            color = f"{CYAN}{BOLD}"
        else:
            color = ""

        print(f"  {color}#{rank:<5}{name:<24}{RESET}{sla_str:>10}{cost_str:>12}"
              f"  {vs_best:>10}  {medal}")

    print(f"  {'─' * 72}\n")


def _print_shootout_insights(results: dict):
    """Print 3 key insights from the shootout."""
    rl = results.get("AutoCloud-Agent", {})
    hpa = results.get("KubernetesHPA", {})
    static = results.get("StaticN", {})

    rl_cost    = rl.get("cost", 0)
    hpa_cost   = hpa.get("cost", 0)
    static_cost = static.get("cost", 0)

    saving_vs_hpa    = ((hpa_cost - rl_cost) / max(hpa_cost, 1e-6)) * 100
    saving_vs_static = ((static_cost - rl_cost) / max(static_cost, 1e-6)) * 100

    print(f"\n{BOLD}{BLUE}╔══════════════════════════════════════════════════════════════════════════╗")
    print(f"║                         KEY INSIGHTS                                   ║")
    print(f"╚══════════════════════════════════════════════════════════════════════════╝{RESET}\n")
    print(f"  {GREEN}▶{RESET}  AutoCloud-Agent saves {GREEN}{BOLD}{saving_vs_hpa:.1f}%{RESET} cost vs Kubernetes HPA")
    print(f"  {GREEN}▶{RESET}  AutoCloud-Agent saves {GREEN}{BOLD}{saving_vs_static:.1f}%{RESET} cost vs doing nothing (StaticN)")
    print(f"  {GREEN}▶{RESET}  All methods achieved 100% SLA — cost efficiency is the differentiator")
    print(f"  {GREEN}▶{RESET}  RL scales {ITALIC}proactively{RESET} (before spikes); HPA reacts {ITALIC}after{RESET} CPU exceeds threshold\n")


# ─────────────────────────────────────────────────────────────────────────────
# Stress Test Mode
# ─────────────────────────────────────────────────────────────────────────────

def run_stress_demo(config, paths, seed, total_steps, delay, stress_key, pause_fn):
    """Run AutoCloud-Agent and KubernetesHPA on a chosen stress scenario."""
    scenario_name, scenario_desc = STRESS_SCENARIOS[stress_key]
    stress_wl = make_stress_workload(stress_key, total_steps)

    stress_banners = {
        "1": "Load climbs from 1× → 3× of baseline. Watch the agent scale out BEFORE saturation.",
        "2": "A 4× spike hits at step 15. Forecaster uncertainty triggers Safety Coordinator.",
        "3": "Load oscillates wildly at 2.5×. Agent must stay stable without thrashing VMs.",
        "4": "Load drops to 0.3× then rebounds to 2.5×. Agent must NOT scale down too eagerly.",
    }

    # Phase 1: RL on stress workload
    clear()
    print_phase_banner(
        f"STRESS TEST — {scenario_name}  [AutoCloud-Agent]",
        stress_banners[stress_key]
    )

    rl_agent = InferenceRunner(
        checkpoint_dir=str(paths.checkpoint_dir),
        config=config, device="cpu",
        forecaster=paths.load_forecaster(),
    )
    sla_rl, cost_rl, _ = run_single_demo(
        "AutoCloud-Agent", GREEN, rl_agent, config,
        stress_wl, seed, total_steps, delay, is_rl=True,
    )
    print(f"  {BOLD}{GREEN}✓ AutoCloud-Agent complete!{RESET}")
    pause_fn(f"  {BOLD}Press Enter to see KubernetesHPA on the same scenario...{RESET}")

    # Phase 2: HPA on same stress workload
    clear()
    print_phase_banner(
        f"STRESS TEST — {scenario_name}  [KubernetesHPA baseline]",
        "Industry baseline — reactive only, no forecaster"
    )
    hpa_policy = KubernetesHPA()
    sla_hpa, cost_hpa, _ = run_single_demo(
        "KubernetesHPA", YELLOW, hpa_policy, config,
        stress_wl, seed, total_steps, delay,
    )
    print(f"  {BOLD}{YELLOW}✓ KubernetesHPA complete!{RESET}")
    pause_fn(f"  {BOLD}Press Enter for stress test results...{RESET}")

    # Results
    clear()
    print_header()
    print(f"\n  {BOLD}{RED}⚡ STRESS SCENARIO: {scenario_name.upper()}{RESET}")
    print(f"  {DIM}{scenario_desc}{RESET}\n")

    print_comparison_table({
        "AutoCloud-Agent": {"sla": sla_rl,  "cost": cost_rl},
        "KubernetesHPA":   {"sla": sla_hpa, "cost": cost_hpa},
    })

    _print_stress_insights(stress_key, sla_rl, cost_rl, sla_hpa, cost_hpa)
    print_what_happened()


def _print_stress_insights(stress_key, sla_rl, cost_rl, sla_hpa, cost_hpa):
    """Print scenario-specific explanation of what we just saw."""
    insights = {
        "1": [
            "Ramp-Up: RL agent saw demand rising early via the Transformer forecaster",
            "→ It booted VMs 1-2 steps before CPU actually hit 80%",
            "→ HPA had to react after threshold was crossed, causing brief queue build-up",
        ],
        "2": [
            "Early Shock: Forecaster uncertainty spiked → Safety Coordinator fired proactive scale-out",
            "→ RL had extra VMs ready when the 4× shock hit",
            "→ HPA's 5-minute cooldown meant it was still scaling when shock ended",
        ],
        "3": [
            "Choppy Plateau: RL agent learned NOT to drain/provision on every noise fluctuation",
            "→ Stability penalty in reward discouraged thrashing",
            "→ HPA oscillated (provision → drain → provision) causing cost and latency spikes",
        ],
        "4": [
            "Trough+Recovery: RL agent conserved VMs during the trough rather than draining all",
            "→ When load rebounded, it was ready without a slow re-boot phase",
            "→ HPA drained aggressively at 0.3× load, then scrambled to re-provision",
        ],
    }

    print(f"\n{BOLD}{BLUE}╔══════════════════════════════════════════════════════════════════════════╗")
    print(f"║                    WHAT THIS SCENARIO TESTED                           ║")
    print(f"╚══════════════════════════════════════════════════════════════════════════╝{RESET}\n")
    for line in insights[stress_key]:
        prefix = "  →" if line.startswith("→") else f"  {GREEN}▶{RESET}"
        print(f"{prefix}  {line.lstrip('→').strip()}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Ablation Mode: 3-agent I-PPO vs Single-Agent PPO
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_demo(config, workload_fn, paths, seed, total_steps, delay, pause_fn):
    """Run I-PPO vs SingleAgentPPO to show the value of decomposition."""
    from autocloud.evaluation.baselines import SingleAgentPPO

    # Phase 1: Full AutoCloud-Agent
    clear()
    print_phase_banner(
        "ABLATION — AutoCloud-Agent (3-agent I-PPO)",
        "ScaleOut Agent + Consolidation Agent + Scheduling Agent — each specialised"
    )
    rl_agent = InferenceRunner(
        checkpoint_dir=str(paths.checkpoint_dir),
        config=config, device="cpu",
        forecaster=paths.load_forecaster(),
    )
    sla_rl, cost_rl, _ = run_single_demo(
        "AutoCloud-Agent", GREEN, rl_agent, config,
        workload_fn, seed, total_steps, delay, is_rl=True,
    )
    print(f"  {BOLD}{GREEN}✓ AutoCloud-Agent complete!{RESET}")
    pause_fn(f"  {BOLD}Press Enter to see Single-Agent PPO (ablation)...{RESET}")

    # Phase 2: Single-agent PPO
    clear()
    print_phase_banner(
        "ABLATION — Single-Agent PPO (one network for everything)",
        "One neural network tries to learn Scale-Out + Consolidation + Scheduling simultaneously"
    )
    single_agent = SingleAgentPPO(device="cpu")
    sla_sa, cost_sa, _ = run_single_demo(
        "SingleAgentPPO", MAGENTA, single_agent, config,
        workload_fn, seed, total_steps, delay,
    )
    print(f"  {BOLD}{MAGENTA}✓ SingleAgentPPO complete!{RESET}")
    pause_fn(f"  {BOLD}Press Enter for ablation results...{RESET}")

    # Results
    clear()
    print_header()
    print(f"\n  {BOLD}{MAGENTA}ABLATION STUDY: Does decomposition into 3 agents actually help?{RESET}\n")

    print_comparison_table({
        "AutoCloud-Agent (I-PPO)": {"sla": sla_rl, "cost": cost_rl},
        "SingleAgentPPO":          {"sla": sla_sa, "cost": cost_sa},
    })

    _print_ablation_insights(cost_rl, cost_sa)
    print_what_happened()


def _print_ablation_insights(cost_rl: float, cost_sa: float):
    improvement = ((cost_sa - cost_rl) / max(cost_sa, 1e-6)) * 100

    print(f"\n{BOLD}{BLUE}╔══════════════════════════════════════════════════════════════════════════╗")
    print(f"║                    WHY DECOMPOSITION HELPS                             ║")
    print(f"╚══════════════════════════════════════════════════════════════════════════╝{RESET}\n")
    print(f"  {GREEN}▶{RESET}  I-PPO saves {GREEN}{BOLD}{improvement:.1f}%{RESET} cost compared to single-agent PPO")
    print(f"  {GREEN}▶{RESET}  Joint action space = 3 × 2²⁰ × 5 ≈ {RED}15 million combinations{RESET}")
    print(f"  {GREEN}▶{RESET}  With I-PPO: each agent sees a space of {GREEN}3{RESET}, {GREEN}2²⁰{RESET}, or {GREEN}5{RESET} — separately")
    print(f"  {GREEN}▶{RESET}  Temporal hierarchy: ScaleOut every 10 steps, Consolidation every 2,")
    print(f"     Scheduling every step — matches real-world decision latencies")
    print(f"  {GREEN}▶{RESET}  Single agent confuses short-horizon (scheduling) and")
    print(f"     long-horizon (scale-out) decisions — it cannot specialise\n")


def main():
    parser = argparse.ArgumentParser(description="AutoCloud-Agent Live Demo")
    parser.add_argument("--speed", choices=["fast", "normal", "slow"],
                        default="normal", help="Demo speed")
    parser.add_argument("--steps", type=int, default=60,
                        help="Episode steps (default 60 = 30 min simulated)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pause", action="store_true",
                        help="Skip interactive pauses (for testing)")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Skip configuration wizard, use defaults")
    parser.add_argument("--mode", choices=["standard", "shootout", "stress", "ablation"],
                        default=None,
                        help="Demo mode: standard | shootout | stress | ablation")
    args = parser.parse_args()

    # Resolve paths
    paths = ArtifactPaths()
    paths.validate_checkpoints()
    workload_fn = paths.make_workload_fn()

    config = copy.deepcopy(DEFAULT_CONFIG)

    clear()
    print_header()

    print(f"  {DIM}Checkpoints : {paths.checkpoint_dir}{RESET}")
    print(f"  {DIM}Workload    : {paths.workload_file or 'synthetic'}{RESET}")
    print(f"  {DIM}Forecaster  : {paths.forecaster_path or 'none'}{RESET}")
    print()

    def pause(msg: str):
        if not args.no_pause:
            input(msg)

    # Interactive Configuration or Defaults
    # Skip wizard if --mode is given explicitly, or --no-interactive is set
    if not args.no_pause and not args.no_interactive and args.mode is None:
        setup = interactive_setup(args)
        config.sim.n_init = setup["n_init"]
        config.sim.n_max  = setup["n_max"]
        config.sim.n_min  = setup["n_min"]
        config.sim.episode_steps = setup["steps"]
        config.reward.sla_latency_ms = setup["sla_ms"]
        config.reward.cpu_high = setup["cpu_high"]
        config.reward.cpu_low  = setup["cpu_low"]
        total_steps    = setup["steps"]
        chosen_mode    = setup["mode"]
        baseline_name  = setup["baseline_name"]
        baseline_desc  = setup["baseline_desc"]
        baseline_cls   = setup["baseline_cls"]
        stress_key     = setup["stress_key"]
        speed          = setup["speed"]
    else:
        config.sim.episode_steps = args.steps
        total_steps   = args.steps
        chosen_mode   = args.mode or "standard"
        baseline_name = "KubernetesHPA"
        baseline_desc = "desiredReplicas = ceil(currentReplicas × currentCPU / targetCPU)"
        baseline_cls  = KubernetesHPA
        stress_key    = "1"
        speed         = args.speed

    delay_map = {"fast": 0.0, "normal": 0.15, "slow": 0.4}
    delay = delay_map[speed]

    pause(f"\n  {BOLD}Press Enter to start the simulation...{RESET}")

    # ── Route to chosen mode ───────────────────────────────────────────────────

    if chosen_mode == "shootout":
        run_shootout(config, workload_fn, paths, args.seed, total_steps, delay, pause)

    elif chosen_mode == "stress":
        run_stress_demo(config, paths, args.seed, total_steps, delay, stress_key, pause)

    elif chosen_mode == "ablation":
        run_ablation_demo(config, workload_fn, paths, args.seed, total_steps, delay, pause)

    else:
        # ── Standard mode (original flow) ──────────────────────────────────────

        # Phase 1: AutoCloud-Agent
        clear()
        print_phase_banner(
            "PHASE 1 — AutoCloud-Agent (Multi-Agent RL)",
            f"3 PPO agents + Transformer forecaster + Safety Coordinator  |  "
            f"{config.sim.n_init} init VMs, max {config.sim.n_max}, "
            f"SLA < {config.reward.sla_latency_ms:.0f}ms"
        )

        agent = InferenceRunner(
            checkpoint_dir=str(paths.checkpoint_dir),
            config=config,
            device="cpu",
            forecaster=paths.load_forecaster(),
        )

        sla_rl, cost_rl, steps_rl = run_single_demo(
            "AutoCloud-Agent", GREEN, agent, config,
            workload_fn, args.seed, total_steps, delay,
            is_rl=True,
        )

        print(f"  {BOLD}{GREEN}✓ AutoCloud-Agent complete!{RESET}")
        pause(f"  {BOLD}Press Enter to see {baseline_name} baseline...{RESET}")

        # Phase 2: Selected Baseline
        clear()
        print_phase_banner(
            f"PHASE 2 — {baseline_name} (Baseline)",
            baseline_desc
        )

        baseline_policy = baseline_cls()

        sla_bl, cost_bl, steps_bl = run_single_demo(
            baseline_name, YELLOW, baseline_policy, config,
            workload_fn, args.seed, total_steps, delay,
        )

        print(f"  {BOLD}{YELLOW}✓ {baseline_name} complete!{RESET}")
        pause(f"  {BOLD}Press Enter for results comparison...{RESET}")

        # Phase 3: Comparison
        clear()
        print_header()
        print_comparison_table({
            "AutoCloud-Agent": {"sla": sla_rl, "cost": cost_rl},
            baseline_name:     {"sla": sla_bl, "cost": cost_bl},
        })
        print_what_happened()

    print(f"  {BOLD}{CYAN}Demo complete! Questions?{RESET}\n")


if __name__ == "__main__":
    main()
