# AutoCloud-Agent — How It Works

A complete explanation of every component: what it does, why it exists, and how it connects to everything else.

---

## Big Picture

```
Alibaba Trace (real cloud data)
         │
         ▼
  WorkloadTransformer          ← trains on historical CPU/mem patterns
  (Forecaster)                 ← predicts next 1/5/10/15 steps
         │
         ▼ forecast (means + uncertainty)
                    ┌─────────────────────────────────────┐
                    │           CloudEnv (Gymnasium)       │
                    │                                      │
                    │   ┌─── SimPy Simulator ───────────┐  │
                    │   │  Poisson arrivals              │  │
                    │   │  LogNormal service times       │  │
                    │   │  M/G/c queue                   │  │
                    │   │  Node: Boot→Active→Drain→Dead  │  │
                    │   └───────────────────────────────-┘  │
                    │          observation (215-dim)        │
                    └──────────────┬──────────────────────-─┘
                                   │
              ┌────────────────────┼──────────────────────┐
              │                    │                       │
              ▼                    ▼                       ▼
     ScaleOutAgent        ConsolidationAgent       SchedulingAgent
     (acts every 10       (acts every 2 steps)     (acts every step)
      steps or on
      emergency)
              │                    │                       │
              └────────────────────┼───────────────────────┘
                                   │
                                   ▼
                          SafetyCoordinator
                          (4 safety filters)
                                   │
                                   ▼
                          CloudEnv.step(action)
                                   │
                          rewards + next observation
                                   │
                          PPO update (independently
                          per agent when buffer fills)
```

And on top of all this:

```
AutoResearch Engine
  │
  ├── reads experiment.py   (current reward weights + PPO params)
  ├── sends to LLM          (Claude / Llama / Gemini)
  ├── LLM rewrites file     (proposes better reward weights)
  ├── runs train.py trial   (2000 steps, ~2 min)
  ├── score = SLA − 0.1×cost
  └── KEEP if better, DISCARD if not  (Karpathy-style)
```

---

## Component 1: The Simulation (`environment/simulator.py`)

### What it is
A **discrete-event simulation** of a cloud cluster using SimPy. SimPy is a Python library where you define processes that `yield` timeouts — it's like asyncio but for simulation time, not real time.

### The queue model: M/G/c
- **M** = Markovian (Poisson) arrivals — inter-arrival time is exponentially distributed
- **G** = General service times — we use LogNormal(μ=2, σ=1), which mimics real job distributions (many short jobs, occasional long ones)
- **c** = Number of servers (active nodes), which changes dynamically

### How a job lives
```
1. _arrivals() generates a job every ~0.5s (Poisson process)
2. Job joins queue
3. _try_dispatch() finds the least-loaded ACTIVE node with enough CPU+RAM
4. _serve_job() runs as a SimPy process → yields timeout(service_time)
5. On completion: resources freed, next queued job dispatched
6. If node drains during service: job is interrupted, re-queued with is_migrated=True
   (migrated jobs go to front of queue and keep remaining service time)
```

### Node lifecycle
```
BOOTING (45s) → ACTIVE → DRAINING (30s grace period) → TERMINATED
```

During BOOTING: node costs money but accepts no jobs.
During DRAINING: node costs money, accepts no new jobs, finishes current jobs.
After grace period: any remaining jobs are forcibly migrated.

### What each step looks like
Each RL step = 30 seconds of simulation time:
1. Simulator runs `env.run(until=now + 30s)`
2. During those 30s, hundreds of arrivals happen and complete
3. After step: we compute P95 latency, CPU util, cost, queue length

---

## Component 2: The Observation Space (`environment/cloud_env.py`)

Every step the environment returns a **215-dimensional vector** to all three agents:

```
obs[0:120]   — 20 nodes × 6 features each
              [cpu_util, mem_util, age_norm, is_booting, is_active, is_draining]

obs[120:200] — 20 queued jobs × 4 features each
              [size_norm, wait_norm, priority_norm, deadline_urgency]

obs[200:215] — 15 global features
              [active_count/20, mean_cpu, mean_mem, queue_len/50,
               migrations/10, forecast_mean×4, forecast_sigma×4,
               booting_count/20, draining_count/20]
```

All values are clipped to [0, 1].

The **forecast features** (obs[205:215]) come from the Transformer forecaster — they tell the agents what workload is predicted 1/5/10/15 steps ahead, and how uncertain the prediction is.

---

## Component 3: The Three Agents

### Agent 1 — ScaleOutAgent (`agents/scaleout_agent.py`)

**Job**: Decide how many new nodes to provision.

**Action space**: Discrete(3) → {0, +1, +2} new nodes

**Temporal hierarchy**: Acts every **10 steps** (every 5 minutes of sim time), OR immediately if:
- Mean CPU > 90%, OR
- Queue length > 80% of max

This interrupt mechanism prevents the cluster from running out of capacity.

**Network**: Actor + Critic, both MLP [512, 256, 128]

**Reward**:
```
r_scaleout = -α1 × max(0, cpu - 0.8)   # penalize high CPU
           - α2 × max(0, 0.3 - cpu)    # penalize low CPU (waste)
           - α3 × (n_active / 20)      # penalize running many nodes
           - α4 × (action > 0)         # penalize each boot event
```

### Agent 2 — ConsolidationAgent (`agents/consolidation_agent.py`)

**Job**: Decide which nodes to drain (shut down gracefully).

**Action space**: MultiBinary(20) → binary vector, one bit per node slot

**Temporal hierarchy**: Acts every **2 steps** (every 60s of sim time).

**Network**: Actor + Critic, both MLP [512, 256, 128]

**Reward**:
```
r_consolidation = -β1 × step_cost       # penalize dollar cost
                + β2 × sla_met          # reward meeting SLA (P95 < 500s)
                - β3 × migrations       # penalize forced migrations
                - β4 × max(0, mem-0.85) # penalize memory pressure
```

### Agent 3 — SchedulingAgent (`agents/scheduling_agent.py`)

**Job**: Re-order the job queue by assigning priority buckets (0–4).

**Action space**: The agent assigns a priority to each of the top-20 queued jobs.

**Temporal hierarchy**: Acts every **step**.

**Network**: More complex — uses a pointer-network style architecture:
- Global MLP processes the full observation: 135 → 128 → 128
- Job MLP processes each job's features: 4 → 64
- Score MLP combines them: 192 → 64 → 5 (priority bucket)

**Reward**:
```
r_scheduling = -γ1 × (mean_wait / 300)        # penalize long waits
             - γ2 × max(0, p95/500 - 1)       # penalize SLA violation
```

---

## Component 4: Safety Coordinator (`coordinator/safety_coordinator.py`)

### Why it exists

Without the coordinator, the Consolidation agent can make dangerous proposals:
- Drain a node that just finished booting (wasted boot cost)
- Drain so many nodes that the cluster falls below minimum capacity
- Drain during high forecast uncertainty (risky if demand spikes)
- Drain while simultaneously booting new nodes (contradictory)

The coordinator runs **4 filters in sequence** before any drain is applied:

```
Filter 1 — Boot Protection
  └── Remove any node younger than (boot_time + 60s warmup)
      Reason: fresh nodes have 0% CPU and look idle, but are prime drain targets for the wrong reasons

Filter 2 — N_min Floor
  └── If draining would leave < 3 active nodes:
      Remove highest-CPU nodes from drain_set until floor is met
      Reason: the cluster must keep minimum capacity

Filter 3 — Uncertainty Suppression
  └── If forecast uncertainty σ_{t+5} > 0.3:
      Clear the entire drain_set
      Reason: if we don't know what demand is coming, be conservative

Filter 4 — Simultaneous Scale-Out Suppression
  └── If ScaleOut just fired (action > 0):
      Clear the entire drain_set
      Reason: don't drain while booting — new capacity hasn't arrived yet
```

Scale-Out actions are **never blocked** — adding capacity is always safe.
Scheduling actions are **always passed through** unchanged — priority assignment is always safe.

---

## Component 5: The Forecaster (`forecaster/`)

### Why forecast?

The agents can see current state but cannot see the future. The forecaster gives them a glimpse of what workload is coming, so they can act proactively instead of reactively.

### Architecture: WorkloadTransformer

A **Transformer encoder** trained on the Alibaba cluster trace:
- Input: 20-step window of [CPU_util, mem_util, arrival_rate, queue_len]
- Output: quantile predictions (q10, q50, q90) at horizons t+1, t+5, t+10, t+15

### MC Dropout for Uncertainty

Instead of just predicting a point estimate, we use **Monte Carlo Dropout**:
```
Keep model in train() mode → dropout active
Run K=30 forward passes on same input
means  = average of 30 q50 predictions
sigmas = variance across 30 predictions  ← epistemic uncertainty
```

High sigma means "the forecaster is unsure" → Coordinator Filter 3 fires → no draining.

---

## Component 6: I-PPO Training (`training/ippo_trainer.py`)

### What I-PPO means

**Independent PPO**: Each agent has its own PPO instance, its own replay buffer, and updates independently. They share the environment but learn separately.

Alternative would be MAPPO (Multi-Agent PPO) with a shared critic — we keep it independent for simplicity and stability.

### Temporal hierarchy in training

```
global_step=0:  so acts, con acts, sch acts → store all three
global_step=1:  so skips, con skips, sch acts → store only sch
global_step=2:  so skips, con acts, sch acts → store con + sch
...
global_step=10: so acts, con skips, sch acts → store so + sch
```

Critical rule: **only store experience on steps when an agent actually acted**. Storing no-op steps would teach the agent that doing nothing is always the policy.

### Buffer flushes are independent

Each agent updates whenever its buffer fills (2048 transitions):
- SchedulingAgent fills ~10× faster than ScaleOut (acts every step vs every 10)
- Each agent's PPO update is triggered independently

### Reward normalization (EMA)

Raw rewards are normalized before storing:
```
normalized_reward = (raw_reward - ema_mean) / (ema_std + ε)
```
This prevents one agent's reward scale from dominating another's learning.

### Entropy annealing

Entropy coefficient starts at 0.01 and decays to 0.001 over 100k steps:
```
entropy = 0.01 × (1 - progress) + 0.001 × progress
```
Early training: high entropy → more exploration.
Late training: low entropy → exploit learned policy.

### Seed randomization

Each episode resets with a different random seed (drawn from the trainer's RNG). This is crucial for **generalization** — the agent learns to handle many different workload patterns, not just one fixed scenario.

---

## Component 7: Evaluation (`evaluation/evaluator.py`)

### 8 methods compared

| Method | Description |
|--------|-------------|
| **AutoCloud-Agent** | Our I-PPO system (3 agents + forecaster + coordinator) |
| **KubernetesHPA** | Kubernetes Horizontal Pod Autoscaler formula |
| **PIController** | Classical proportional-integral controller |
| **ARIMAPredictive** | Holt's double exponential smoothing for prediction |
| **SingleAgentPPO** | Standard PPO with a single flat policy |
| **ThresholdReactive** | Scale up if CPU > 80%, scale down if CPU < 30% |
| **ThresholdPredictive** | Same but uses exponential smoothing of CPU trend |
| **StaticN** | Fixed 10 nodes, never scales |

### Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| SLA Rate | steps where P95 < 500s / total steps | Fraction of time meeting latency SLA |
| Cost Efficiency | 1 − actual_cost / max_possible_cost | Higher = cheaper |
| Mean CPU Util | average CPU across active nodes | How well-loaded the cluster is |
| Node Stability | 1 − std(node_count) / mean(node_count) | How much the cluster oscillates |

---

## Component 8: AutoResearch (`autoresearch/`)

### The Karpathy Inspiration

Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) lets an LLM autonomously improve a GPT training script overnight:
1. LLM modifies `train.py` (architecture, optimizer, hyperparams — anything)
2. Runs for exactly 5 minutes (fixed time budget)
3. Evaluates `val_bpb` (validation bits per byte)
4. Keeps if better, reverts if not
5. Loops forever — "wake up to 100 experiments"

### How we apply it here

Instead of modifying `train.py` (which is complex), we define a single simple file the LLM modifies: **`experiment.py`**.

```
experiment.py defines get_config() → all reward weights + PPO params
```

The loop:

```
Iteration 1:
  LLM reads experiment.py (baseline values)
  LLM reasons: "beta2=2.0 seems low for SLA bonus, try 3.0"
  LLM writes new experiment.py with beta2=3.0
  subprocess: python train.py --experiment_file tmp_exp.py --total_steps 2000
  score = 0.9620
  baseline was 0.9500 → KEEP → update experiment.py

Iteration 2:
  LLM reads experiment.py (now has beta2=3.0)
  LLM reasons: "alpha1=2.0 might not be aggressive enough, try 3.0"
  LLM writes new experiment.py with alpha1=3.0
  subprocess: python train.py --experiment_file tmp_exp.py --total_steps 2000
  score = 0.9480
  best was 0.9620 → DISCARD → experiment.py stays as beta2=3.0

...and so on for N iterations
```

### The key files

```
program.md      ← human writes this (research goal, what's tunable, strategy)
experiment.py   ← LLM modifies this (the "train.py" analog from Karpathy)
autoresearch/
  engine.py     ← runs the loop, calls LLM, applies keep/discard logic
  subprocess_runner.py  ← runs train.py in a subprocess, returns score
  results.tsv   ← log of all trials (like Karpathy's results.tsv)
```

### What the LLM prompt looks like

```
You are an autonomous RL researcher improving AutoCloud-Agent.

## Research Program
[contents of program.md]

## Current experiment.py (last kept version)
[full Python code]

## Trial History
[KEEP] iter=1 score=+0.9620 sla=0.962 cost=0.0000 | +config.reward.beta2 = 3.0
[DISC] iter=2 score=+0.9480 sla=0.948 cost=0.0000 | +config.reward.alpha1 = 3.0

## Your Task — Iteration 3/4
Propose ONE focused change. Return ONLY the complete Python file.
```

### Why reward weights, not architecture?

1. The agents already use a well-sized network ([512, 256, 128])
2. PPO hyperparameters (lr=3e-4, γ=0.99) are well-established defaults
3. **Reward weights are project-specific** — there's no textbook answer for β2
4. Changing β2 from 2.0 to 3.0 means "SLA compliance is 50% more important than cost" — this has a clear behavioral interpretation the LLM can reason about

---

## Data Flow: End-to-End

```
1. Alibaba trace loaded → WorkloadTransformer trained → saved as forecaster_weights.pt

2. IPPOTrainer starts:
   - CloudEnv created (wraps SimPy simulator)
   - 3 PPO agents created (so, con, sch)
   - SafetyCoordinator created
   - Each step:
       a. MCDropoutForecaster.predict(obs) → means, sigmas
       b. env.inject_forecast(means, sigmas)
       c. so_agent.act(obs) every 10 steps
       d. con_agent.act(obs, mask) every 2 steps
       e. sch_agent.act(obs) every step
       f. coordinator.resolve(a_so, a_con, a_sch, ...) → safe actions
       g. env.step(safe_actions) → next_obs, rewards, info
       h. Store experience in each agent's buffer
       i. When buffer full → PPO update
   - Save checkpoint every 10k steps

3. Evaluator loads checkpoint → runs 10 episodes × 3 seeds × 8 methods
   → prints comparison table

4. AutoResearch:
   - engine.py builds prompt (program.md + experiment.py + history)
   - LLM returns new experiment.py
   - subprocess_runner.py: python train.py --experiment_file tmp.py --steps 2000
   - parse score from stdout
   - keep or discard, log to results.tsv
```

---

## Free LLM Options (No Anthropic Key Needed)

### Option 1 — Ollama (recommended, completely local)

No API key. Runs on your machine. Free forever.

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (Llama 3.2 3B is small and fast enough for this)
ollama pull llama3.2:3b

# It starts a local server at http://localhost:11434
```

In `engine.py`, the engine now accepts `llm_provider="ollama"`.

### Option 2 — Groq (free API key, cloud-based, very fast)

Free tier: ~14,400 requests/day with Llama 3.1 70B.
Sign up at https://console.groq.com (no credit card needed).

```bash
pip install groq
export GROQ_API_KEY=gsk_your_key_here
```

In `engine.py`, use `llm_provider="groq"`.

### Option 3 — Google Gemini (free API key)

Free tier: 60 requests/minute with gemini-1.5-flash.
Sign up at https://aistudio.google.com.

```bash
pip install google-generativeai
export GEMINI_API_KEY=your_key_here
```

In `engine.py`, use `llm_provider="gemini"`.

---

---

## Running the Project

### Complete Workflow

```
Step 1 (Kaggle — GPU needed)
  train_forecaster.ipynb   → forecaster_weights.pt + day1_processed.npy

Step 2 (Kaggle — GPU needed)
  train_rl_agents.ipynb    → checkpoints/so_actor_final.pt (+ 5 other .pt files)

Step 3 (Local — CPU fine)
  Download from Kaggle outputs:
    checkpoints/so_actor_final.pt
    checkpoints/con_actor_final.pt
    checkpoints/sch_actor_final.pt
    checkpoints/so_critic_final.pt
    checkpoints/con_critic_final.pt
    checkpoints/sch_critic_final.pt
    checkpoints/training_metrics.json
    checkpoints/forecaster_weights.pt   (optional)
    day1_processed.npy                  (optional, put in data/)

Step 4 (Local — any machine)
  python pipeline.py                              → evaluate 8 methods
  python pipeline.py --mode autoresearch          → LLM reward tuning
  jupyter notebook notebooks/demo.ipynb           → full demo
```

### Install dependencies (one-time)

```bash
pip install simpy gymnasium numpy torch matplotlib pandas

# For AutoResearch with Groq (free, recommended):
pip install groq
export GROQ_API_KEY=gsk_...     # from console.groq.com

# For AutoResearch with Ollama (local, no key):
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b
ollama serve &
```

### `pipeline.py` — the main local entry point

```bash
# Evaluate trained agents vs all 7 baselines (default)
python pipeline.py
python pipeline.py --n_episodes 10 --seeds 0 1 2

# Evaluate with real Alibaba workload
python pipeline.py --workload_file data/day1_processed.npy

# Run AutoResearch (LLM tunes reward weights in experiment.py)
python pipeline.py --mode autoresearch --llm_provider groq
python pipeline.py --mode autoresearch --llm_provider ollama
python pipeline.py --mode autoresearch --ar_iterations 6 --ar_steps 3000

# Eval + AutoResearch in one go
python pipeline.py --mode all --llm_provider groq
```

### `train.py` — used by AutoResearch internally

You don't normally call this directly. AutoResearch calls it in a subprocess.
If you want to do a quick local training run (e.g., fine-tune with new reward weights):

```bash
python train.py --total_steps 5000 --seed 0              # synthetic workload, CPU ~5min
python train.py --total_steps 5000 --experiment_file experiment.py
```

### Notebooks

| Notebook | Where to run | What it does |
|----------|-------------|--------------|
| `train_forecaster.ipynb` | Kaggle (GPU) | Train Transformer forecaster on Alibaba trace |
| `train_rl_agents.ipynb` | Kaggle (GPU) | Train 3 I-PPO agents for 300k steps |
| `results.ipynb` | Local | Evaluate + compare vs baselines |
| `multiday_eval.ipynb` | Kaggle | Test generalization across all 7 days |
| `demo.ipynb` | Local | Full demo: simulation viz + comparison + AutoResearch |

---

## Files Reference

```
autocloud_agent/
│
├── program.md                    ← Research directives (human edits this)
├── experiment.py                 ← Config the LLM modifies (Karpathy's "train.py")
├── train.py                      ← Training entry point
│
├── configs/
│   └── default_config.py         ← All Config dataclasses (SimConfig, PPOConfig, RewardConfig, ...)
│
├── environment/
│   ├── simulator.py              ← SimPy M/G/c queue, node lifecycle, job dispatch
│   ├── cloud_env.py              ← Gymnasium wrapper: obs/action spaces, reward computation
│   ├── node.py                   ← Node dataclass: BOOTING/ACTIVE/DRAINING/TERMINATED
│   ├── job.py                    ← Job dataclass: arrival, service, migration
│   └── workload.py               ← Alibaba trace loader + SyntheticWorkload
│
├── agents/
│   ├── ppo.py                    ← Shared PPO implementation (actor, critic, buffer, update)
│   ├── scaleout_agent.py         ← Discrete(3) action, MLP [512,256,128]
│   ├── consolidation_agent.py    ← MultiBinary(20) action, MLP [512,256,128]
│   └── scheduling_agent.py       ← Per-job priority, pointer-network style
│
├── coordinator/
│   └── safety_coordinator.py     ← 4-filter safety gate for consolidation actions
│
├── forecaster/
│   ├── transformer_model.py      ← WorkloadTransformer (quantile predictions)
│   ├── mc_dropout.py             ← MCDropoutForecaster (K=30 passes → mean + sigma)
│   └── trainer.py                ← Forecaster training loop
│
├── training/
│   ├── ippo_trainer.py           ← I-PPO training loop with temporal hierarchy
│   ├── baselines.py              ← 7 baseline policies
│   └── ema_normalizer.py         ← EMA reward normalization
│
├── autoresearch/
│   ├── engine.py                 ← Karpathy-style loop: LLM reads code → proposes → keep/discard
│   ├── subprocess_runner.py      ← Runs train.py in subprocess, parses score
│   ├── schema.py                 ← (legacy) JSON param validation
│   └── results.tsv               ← Trial log (auto-generated)
│
├── evaluation/
│   └── evaluator.py              ← Evaluates all 8 methods, prints table
│
└── notebooks/
    ├── train_forecaster.ipynb    ← Step 1: train the Transformer forecaster
    ├── train_rl_agents.ipynb     ← Step 2: train I-PPO agents
    ├── results.ipynb             ← Step 3: evaluate and compare
    ├── multiday_eval.ipynb       ← Step 4: test generalization across 7 days
    └── demo.ipynb                ← Full demo: visualization + AutoResearch
```
