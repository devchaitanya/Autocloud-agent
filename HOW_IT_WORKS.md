# How AutoCloud-Agent Works

A plain-English explanation of every moving part — written for someone who knows basic ML but not reinforcement learning.

---

## 1. The Problem

A cloud provider runs a cluster of virtual machines (VMs). Users submit jobs that consume CPU and memory. The provider must decide **every 30 seconds**:

- **Scale-out:** Should we boot new VMs or shut down idle ones?
- **Consolidation:** Which VMs should be drained (moving their jobs elsewhere) to save cost?
- **Scheduling:** When a job arrives, which VM should it go to?

If you scale too aggressively → waste money on idle servers.
If you scale too conservatively → jobs get dropped (SLA violation).

**Goal:** Minimise cost while keeping SLA rate at 100%.

---

## 2. Why Three Agents?

These three decisions operate at different time scales:

| Agent | Action Space | Decision Frequency | Why Separate |
|-------|-------------|-------------------|--------------|
| **ScaleOut** | Discrete(3): add / hold / remove | Every 10 steps | Capacity planning is slow (VMs take 60s to boot) |
| **Consolidation** | MultiBinary(20): drain flags per VM | Every 2 steps | Migration has medium latency |
| **Scheduling** | Discrete(5): which scheduling rule | Every step | Job dispatch must be instant |

A single agent trying to learn all three simultaneously would need to explore a much larger joint action space. By decomposing, each agent has a focused, smaller problem. This is **Independent PPO (I-PPO)** — each agent has its own actor-critic network but they share the same reward signal and environment.

---

## 3. The Observation Space (215 dimensions)

Every 30 seconds, the environment produces a vector of 215 numbers:

```
obs[0:120]   → Node features (20 nodes × 6 features each)
                - CPU utilisation, memory utilisation, number of running jobs,
                  state (booting/active/draining), load trend, time alive

obs[120:200] → Job queue features (20 jobs × 4 features each)
                - CPU request, memory request, duration, priority

obs[200:215] → Global features (15 values)
                - total active nodes, cluster CPU/memory utilisation,
                  SLA rate, queued jobs, forecast demand (5 horizons),
                  forecast uncertainty
```

Each feature is normalised to [0, 1] so the neural network trains stably.

---

## 4. The Simulator

The simulator (`autocloud/simulator/`) is a Gymnasium environment backed by a SimPy discrete-event engine:

1. **Workload replay:** Reads real Alibaba trace data (`day2_processed.npy`), one 30-second snapshot at a time
2. **Node lifecycle:** Each VM transitions through BOOTING (60s) → ACTIVE → DRAINING → TERMINATED
3. **Job placement:** Jobs are assigned to nodes based on the Scheduling agent's chosen rule (Best-Fit, First-Fit, Round-Robin, Least-Loaded, Most-Loaded)
4. **SLA checking:** A job is "dropped" if it cannot be placed within its deadline

**Reward function** (shared by all 3 agents):
```
reward = -α × (active_nodes / max_nodes)    ← cost penalty
         -β × sla_violations                 ← SLA penalty
         +γ × mean_cpu_utilisation           ← efficiency bonus
         -δ × |active_nodes - prev_active|   ← stability penalty
```

The agents learn to maximise this jointly.

---

## 5. The Forecaster

The **Workload Forecaster** predicts CPU demand 1–15 steps ahead so agents can act proactively:

- Architecture: 2-layer Transformer encoder (4 attention heads, d_model=64)
- Input: last 30 snapshots of cluster CPU/memory (window = 30 × 30s = 15 min)
- Output: predicted demand for next 15 steps
- **Uncertainty:** MC Dropout (30 forward passes with dropout ON at inference) → mean prediction + standard deviation

The 5 forecast horizons and the uncertainty estimate are injected into `obs[200:215]` (global features).

Trained separately on Day 1 of the Alibaba trace (`notebooks/train_forecaster.ipynb`).

---

## 6. The Safety Coordinator

Between the agents' raw actions and the environment, a **Safety Coordinator** (`autocloud/coordinator/safety.py`) applies 5 rule-based filters:

| # | Filter | What It Does |
|---|--------|-------------|
| 1 | **Boot-protect** | Prevents draining a VM that is still booting (< 60s old) |
| 2 | **N_min floor** | Ensures at least N_min=3 VMs are always active |
| 3 | **Uncertainty hold** | If forecast uncertainty > threshold, blocks scale-down (wait for clarity) |
| 4 | **Anti-overlap** | Prevents the ScaleOut agent from adding and Consolidation from draining in the same step |
| 5 | **Proactive scale-out** | If forecast predicts demand spike in next 5 steps, forces a scale-out even if the ScaleOut agent said "hold" |

These filters are **not learned** — they are hard-coded safety constraints. They override agent actions when needed.

---

## 7. Training Pipeline

Training is done on Kaggle (free T4 GPU). Two phases:

### Phase A: Train Forecaster (~20 minutes)
```
notebooks/train_forecaster.ipynb
```
- Loads Day 1 Alibaba trace → sliding window dataset → trains Transformer → saves `forecaster_weights.pt`

### Phase B: Train RL Agents (~30 minutes)
```
notebooks/train_rl_agents.ipynb
```
- Uses `autocloud/training/ippo_trainer.py`
- 500 episodes, γ=0.99, learning rate=3×10⁻⁴, clip ε=0.2
- Each episode: simulate 1 day (2,880 steps × 30s = 24h)
- EMA reward normalisation (`ema_normalizer.py`) for stable training
- Saves 6 checkpoints: `{so,con,sch}_{actor,critic}_final.pt`

### Phase C: Download & Evaluate Locally (CPU only)
```bash
pip install -e .
python scripts/evaluate.py          # 7 methods × 5 episodes × 3 seeds
python demo.py                      # interactive live demo
python stress_test.py               # 4 peak-load scenarios
```

---

## 8. PPO Algorithm — Simple Explanation

PPO (Proximal Policy Optimisation) is the RL algorithm each agent uses. Here's the intuition:

1. **Actor** network outputs a probability distribution over actions
2. **Critic** network estimates "how good is this state?" (value function)
3. After collecting a batch of (state, action, reward) tuples:
   - Compute **advantage** = "was this action better or worse than average?" (using GAE)
   - Update the actor to make good actions more likely, bad actions less likely
   - **Clipping**: Don't change probabilities too much in one step (ratio stays in [1-ε, 1+ε])
   - Also add an **entropy bonus** to encourage exploration early on

Each of our 3 agents has its own actor + critic. They don't share parameters but they share the same reward.

---

## 9. Evaluation and Baselines

We compare AutoCloud-Agent against 6 methods:

| Method | Type | How It Decides |
|--------|------|---------------|
| **StaticN** | Lower bound | Fixed 10 nodes, never changes |
| **ThresholdReactive** | Rule-based | Add if CPU > 80%, drain if CPU < 30% |
| **KubernetesHPA** | Industry | k8s HPA formula: `ceil(current × cpu_util / target_util)` with 10% dead-band |
| **AWSTargetTracking** | Industry | AWS-style policy with asymmetric scale-up (1 min) / scale-down (5 min) cooldowns |
| **MPCController** | Model-based | 5-step horizon MPC using exponentially-weighted moving average forecast |
| **SingleAgentPPO** | RL ablation | One PPO agent making all 3 decisions (proves I-PPO decomposition helps) |

Evaluation protocol: 5 episodes × 3 random seeds, Alibaba Day 2 workload.

### Results

| Method | SLA | Cost Eff. | CPU Util. | Stability |
|--------|-----|-----------|-----------|-----------|
| **AutoCloud-Agent** | **100%** | **0.962** | **55.2%** | 0.889 |
| MPCController | 100% | 0.962 | 55.2% | 0.941 |
| ThresholdReactive | 100% | 0.955 | 48.3% | 0.822 |
| StaticN | 100% | 0.938 | 33.3% | 1.000 |
| KubernetesHPA | 100% | 0.930 | 31.2% | 0.842 |
| AWSTargetTracking | 100% | 0.928 | 29.5% | 0.876 |
| SingleAgentPPO | 100% | 0.924 | 41.3% | 0.794 |

AutoCloud-Agent matches the best classical method (MPC) while being **reactive** — MPC requires a hand-tuned forecast model, while our agent learns purely from experience.

---

## 10. Stress Testing

`stress_test.py` tests robustness under 4 extreme workload patterns:

| Scenario | Pattern | What It Tests |
|----------|---------|---------------|
| **Ramp-up** | Load rises linearly 1× → 3× over 500 steps | Can the agent pre-emptively scale before saturation? |
| **Early shock** | Normal → sudden 4× spike at step 50 | Recovery speed after unexpected burst |
| **Choppy plateau** | Oscillating high load (2.5× ± random noise) | Stability under noisy signals |
| **Trough + recovery** | Load drops to 0.3× then rebounds to 2.5× | Does the agent avoid premature scale-down? |

---

## 11. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **I-PPO over single agent** | 3-way decomposition reduces each agent's action space → faster convergence (+4% cost efficiency in eval) |
| **Temporal hierarchy** | ScaleOut every 10 steps, Consolidation every 2, Scheduling every 1 — matches real-world latencies |
| **Safety Coordinator** | RL can explore unsafe actions; hard constraints prevent catastrophic decisions |
| **MC Dropout uncertainty** | Cheap uncertainty estimate (no ensemble needed) → enables uncertainty-aware safety filter |
| **SimPy engine** | Discrete-event simulation is more realistic than fixed-step — events (boot, drain, job finish) happen at their real times |

---

## 12. Running Locally

```bash
# Prerequisites
conda activate myenv
pip install -e .                     # installs autocloud package

# Interactive demo (best for showing to someone)
python demo.py                       # default: 60 steps, normal speed
python demo.py --speed fast          # faster for quick shows
python demo.py --no-pause            # skip "press Enter" pauses

# Full evaluation
python scripts/evaluate.py           # 7 methods, 5 eps × 3 seeds (~5 min)

# Stress test
python stress_test.py                # 4 scenarios (~2 min)

# Training (requires GPU — use Kaggle notebooks instead)
python train.py                      # runs IPPOTrainer locally
```

Checkpoints are auto-discovered from `checkpoints/` or `../outputs/rl_agents/`.
Workload data is auto-discovered from `../outputs/train_Forecaster/`.

---

## 13. File Reference

| File | Purpose |
|------|---------|
| `autocloud/simulator/cloud_env.py` | Gymnasium environment |
| `autocloud/simulator/engine.py` | SimPy simulation core |
| `autocloud/agents/ppo.py` | PPO algorithm implementation |
| `autocloud/agents/{scaleout,consolidation,scheduling}.py` | 3 specialised agents |
| `autocloud/forecaster/transformer_model.py` | Workload Transformer |
| `autocloud/forecaster/mc_dropout.py` | MC Dropout uncertainty |
| `autocloud/coordinator/safety.py` | 5-filter Safety Coordinator |
| `autocloud/inference/runner.py` | InferenceRunner (ties all components) |
| `autocloud/evaluation/baselines.py` | 6 SOTA baselines |
| `autocloud/evaluation/evaluator.py` | Multi-seed evaluation harness |
| `autocloud/training/ippo_trainer.py` | I-PPO training loop |
| `autocloud/config/settings.py` | All hyperparameters |
| `autocloud/config/paths.py` | Auto-discovers checkpoints & data |
| `demo.py` | Interactive live demo |
| `stress_test.py` | 4-scenario stress test |
| `scripts/evaluate.py` | CLI evaluation entry point |
| `design_doc.tex` | LaTeX design document |
