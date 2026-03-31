# AutoCloud-Agent

**Multi-Agent Reinforcement Learning for Cloud Autoscaling**

Three specialised RL agents learn to manage a cloud cluster — adding servers, draining idle ones, and prioritising jobs — while a Transformer forecaster predicts demand and a Safety Coordinator prevents dangerous actions.

Tested against 6 SOTA baselines (Kubernetes HPA, AWS Target Tracking, MPC, etc.) on real **Alibaba Cluster Trace 2018** data.

---

## Prerequisites

- **Python 3.11+** (tested with conda environment)
- **PyTorch ≥ 2.0** (CPU is sufficient for inference/evaluation; GPU needed only for training)
- **Packages:** gymnasium, simpy, pandas, numpy (all installed automatically)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/devchaitanya/Autocloud-agent.git
cd Autocloud-agent/autocloud_agent/

# Create conda environment (recommended)
conda create -n myenv python=3.11 -y
conda activate myenv

# Install the package (editable mode — includes all dependencies)
pip install -e .
```

This installs the `autocloud` Python package and all dependencies from `pyproject.toml`.

---

## Quick Start

```bash
python demo.py                   # interactive live demo
python scripts/evaluate.py       # full evaluation (7 methods × 3 seeds)
python stress_test.py            # 4 peak-load stress scenarios
```

---

## How to Run

### 1. Live Demo (`demo.py`)

An interactive terminal demo that runs AutoCloud-Agent and Kubernetes HPA side-by-side, with coloured output, progress bars, and a comparison table at the end. **Best for presenting to an instructor.**

```bash
# Default: 60 steps, normal speed, interactive pauses between phases
python demo.py

# Fast mode (shorter delays between steps)
python demo.py --speed fast

# Slow mode (longer delays — good for explaining each step live)
python demo.py --speed slow

# Skip "Press Enter to continue" pauses (non-interactive / recording)
python demo.py --no-pause

# Shorter demo (10 steps instead of 60)
python demo.py --steps 10

# Combine flags
python demo.py --speed fast --steps 30 --no-pause --seed 123
```

| Flag | Default | Description |
|------|---------|-------------|
| `--speed` | `normal` | `fast` / `normal` / `slow` — controls delay between steps |
| `--steps` | `60` | Number of simulation steps per phase |
| `--seed` | `42` | Random seed for reproducibility |
| `--no-pause` | off | Skip interactive "Press Enter" prompts |

**What you'll see:**
- **Phase 1:** AutoCloud-Agent running (node status dots, CPU/memory bars, SLA badge, live cost)
- **Phase 2:** Kubernetes HPA baseline on the same workload
- **Phase 3:** Side-by-side comparison table + educational summary of how the RL agent works

### 2. Full Evaluation (`scripts/evaluate.py`)

Runs all 7 methods (AutoCloud-Agent + 6 baselines) across multiple episodes and seeds. Results saved to JSON.

```bash
# Default: 10 episodes × 3 seeds, auto-detect checkpoints
python scripts/evaluate.py

# Custom number of episodes and seeds
python scripts/evaluate.py --n_episodes 5 --seeds 0 1 2

# Specify paths manually (if auto-detection doesn't find them)
python scripts/evaluate.py --checkpoint_dir checkpoints/ --workload path/to/day2_processed.npy

# Save results to a custom file
python scripts/evaluate.py --output my_results.json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint_dir` | auto-detect | Path to directory with `.pt` checkpoint files |
| `--workload` | auto-detect | Path to workload `.npy` file |
| `--n_episodes` | `10` | Episodes per method per seed |
| `--seeds` | `0 1 2` | Random seeds to average over |
| `--output` | `evaluation_results.json` | Output file for results |

### 3. Stress Test (`stress_test.py`)

Tests robustness under 4 extreme workload patterns (ramp-up, sudden spike, choppy plateau, trough-then-recovery).

```bash
# Run all 4 scenarios
python stress_test.py

# Run specific scenarios only (1-4)
python stress_test.py --scenarios 1 3

# Custom seeds
python stress_test.py --seeds 0 1 2 3 4
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint_dir` | auto-detect | Path to checkpoint directory |
| `--workload_file` | auto-detect | Path to workload `.npy` file |
| `--seeds` | `0 1 2` | Random seeds |
| `--scenarios` | `1 2 3 4` | Which scenarios to run (1=ramp, 2=shock, 3=choppy, 4=trough) |

### 4. Training (`train.py` — GPU recommended)

Training is best done on **Kaggle** using the provided notebooks (free T4 GPU). For local training:

```bash
# Train with default settings (50k steps, CPU)
python train.py

# Train on GPU
python train.py --device cuda --total_steps 100000

# Resume from checkpoint
python train.py --load_tag final --checkpoint_dir checkpoints/

# Evaluation-only mode (no training, just evaluate loaded checkpoints)
python train.py --eval_only --load_tag final
```

| Flag | Default | Description |
|------|---------|-------------|
| `--total_steps` | `50000` | Total training steps |
| `--seed` | `0` | Random seed |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--checkpoint_dir` | `checkpoints` | Where to save/load checkpoints |
| `--forecaster_path` | auto-detect | Path to `forecaster_weights.pt` |
| `--load_tag` | none | Load checkpoints with this tag (e.g., `final`) |
| `--eval_only` | off | Skip training, just evaluate |
| `--workload_file` | auto-detect | Path to workload `.npy` data |
| `--log_interval` | `1000` | Print metrics every N steps |

### Checkpoint & Data Auto-Discovery

You don't need to specify paths manually. The system auto-discovers files from:
1. `autocloud_agent/checkpoints/` (6 `.pt` files)
2. `outputs/rl_agents/` (fallback)
3. `outputs/train_Forecaster/` (workload data + forecaster weights)

---

## Results

Evaluated on real Alibaba cluster trace data (5 episodes × 3 seeds):

| Method | SLA Rate | Cost Efficiency | CPU Utilisation | Stability |
|--------|----------|-----------------|-----------------|-----------|
| **AutoCloud-Agent (Ours)** | **100%** | **0.962** | **55.2%** | 0.889 |
| MPCController | 100% | 0.962 | 55.2% | 0.941 |
| ThresholdReactive | 100% | 0.955 | 48.3% | 0.822 |
| KubernetesHPA | 100% | 0.930 | 31.2% | 0.842 |
| AWSTargetTracking | 100% | 0.928 | 29.5% | 0.876 |
| SingleAgentPPO | 100% | 0.924 | 41.3% | 0.794 |
| StaticN (do-nothing) | 100% | 0.938 | 33.3% | 1.000 |

**Key findings:**
- AutoCloud-Agent matches the best classical method (MPC) on cost and CPU efficiency
- **3.4% cheaper** than Kubernetes HPA, **77% better CPU utilisation** than AWS Target Tracking
- Multi-agent I-PPO beats single-agent PPO across all metrics (confirming the value of decomposition)

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Observation (215-dim)                │
│  120 node features + 80 job features + 15 globals    │
└────────┬────────────────┬────────────────┬───────────┘
         │                │                │
    ┌────▼────┐     ┌─────▼─────┐    ┌────▼────┐
    │ScaleOut │     │Consolidat.│    │Scheduling│
    │ Agent   │     │  Agent    │    │  Agent   │
    │(every   │     │(every     │    │(every    │
    │ 10 step)│     │ 2 steps)  │    │  step)   │
    └────┬────┘     └─────┬─────┘    └────┬────┘
         │                │               │
    ┌────▼────────────────▼───────────────▼────┐
    │          Safety Coordinator               │
    │  5 filters: boot-protect, N_min floor,    │
    │  uncertainty hold, anti-overlap,          │
    │  proactive scale-out                      │
    └─────────────────┬────────────────────────┘
                      │
              ┌───────▼───────┐
              │Cloud Simulator│  ← Alibaba trace
              │  (Gymnasium)  │     workload data
              └───────────────┘
```

The **Workload Forecaster** (Transformer + MC Dropout) predicts demand 1–15 steps ahead and provides uncertainty estimates that feed into both the observation and the Safety Coordinator.

---

## Project Structure

```
autocloud_agent/
├── demo.py                 ← Live demo (RL agent vs Kubernetes HPA)
├── stress_test.py          ← 4 stress scenarios
├── train.py                ← Training entry point
├── pyproject.toml          ← Package config (pip install -e .)
├── design_doc.tex/.pdf     ← LaTeX design document
│
├── scripts/
│   └── evaluate.py         ← CLI evaluation (auto-detects checkpoints)
│
├── autocloud/              ← Installable Python package
│   ├── config/
│   │   ├── settings.py     ← All hyperparameters (dataclasses)
│   │   └── paths.py        ← Auto-discovers checkpoints & data files
│   ├── simulator/
│   │   ├── cloud_env.py    ← Gymnasium environment (obs=215, actions=3)
│   │   ├── engine.py       ← SimPy discrete-event simulation
│   │   ├── node.py         ← VM model (BOOTING→ACTIVE→DRAINING→TERMINATED)
│   │   ├── job.py          ← Job dataclass
│   │   └── workload.py     ← Alibaba trace loader + synthetic fallback
│   ├── agents/
│   │   ├── ppo.py          ← Base PPO algorithm (GAE, clipping, entropy)
│   │   ├── scaleout.py     ← ScaleOut agent (Discrete(3))
│   │   ├── consolidation.py← Consolidation agent (MultiBinary(20))
│   │   ├── scheduling.py   ← Scheduling agent (Discrete(5), weight-tied)
│   │   └── loader.py       ← Load all 3 agents from checkpoints
│   ├── forecaster/
│   │   ├── transformer_model.py  ← WorkloadTransformer (2-layer, 4-head)
│   │   └── mc_dropout.py   ← MC Dropout uncertainty (30 forward passes)
│   ├── coordinator/
│   │   └── safety.py       ← 5-filter Safety Coordinator
│   ├── inference/
│   │   └── runner.py       ← InferenceRunner (ties everything together)
│   ├── evaluation/
│   │   ├── evaluator.py    ← Multi-seed evaluation harness
│   │   └── baselines.py    ← 6 SOTA baselines
│   └── training/
│       ├── ippo_trainer.py ← I-PPO training loop
│       └── ema_normalizer.py ← EMA reward normalisation
│
├── notebooks/              ← Kaggle training notebooks
│   ├── train_forecaster.ipynb
│   ├── train_rl_agents.ipynb
│   ├── results.ipynb
│   └── demo.ipynb
│
└── tests/
    ├── test_simulator.py
    ├── test_ppo.py
    ├── test_coordinator.py
    └── test_transformer.py
```

---

## Training (GPU required — use Kaggle)

1. **Train Forecaster:** Run `notebooks/train_forecaster.ipynb` on Kaggle (~20 min on T4 GPU)
   → saves `forecaster_weights.pt` + `day2_processed.npy`

2. **Train RL Agents:** Run `notebooks/train_rl_agents.ipynb` on Kaggle (~30 min on T4 GPU)
   → saves 6 checkpoint files (`so_actor_final.pt`, etc.)

3. **Download** the `outputs/` folder from Kaggle and evaluate locally (no GPU needed).

---

## Dataset

Real data from **Alibaba Cluster Trace 2018** — 4,023 machines, 7 days of CPU/memory measurements.
- Binned into 30-second intervals → 2,880 snapshots per day
- Day 1 used for training; Days 2–7 for testing
- Dataset: [github.com/alibaba/clusterdata](https://github.com/alibaba/clusterdata)

---

## Baselines

| Category | Method | Description |
|----------|--------|-------------|
| **Industry** | KubernetesHPA | k8s HPA formula with 10% dead-band |
| **Industry** | AWSTargetTracking | AWS policy with asymmetric cooldowns |
| **Control theory** | MPCController | 5-step MPC with EWM forecast (AWARE baseline) |
| **Rule-based** | ThresholdReactive | CPU > 80% → add; CPU < 30% → drain |
| **RL ablation** | SingleAgentPPO | One agent for all 3 actions (shows I-PPO advantage) |
| **Lower bound** | StaticN | Fixed 10 nodes, never scales |
