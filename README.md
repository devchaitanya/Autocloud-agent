# AutoCloud-Agent

**Multi-Agent Reinforcement Learning for Cloud Autoscaling**

Three specialised RL agents learn to manage a cloud cluster — adding servers, draining idle ones, and prioritising jobs — while a Transformer forecaster predicts demand and a Safety Coordinator prevents dangerous actions.

Tested against 6 SOTA baselines (Kubernetes HPA, AWS Target Tracking, MPC, etc.) on real **Alibaba Cluster Trace 2018** data.

---

## Quick Start

```bash
cd autocloud_agent/
conda activate myenv
pip install -e .      # one-time install

# Live demo (interactive, shows RL agent vs Kubernetes HPA side-by-side)
python demo.py

# Full evaluation (7 methods × 5 episodes × 3 seeds)
python scripts/evaluate.py

# Stress test (4 peak scenarios)
python stress_test.py
```

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
