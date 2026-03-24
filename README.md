# AutoCloud-Agent

Hierarchical Multi-Agent Reinforcement Learning for autonomous cloud resource management.

Three independent PPO agents — **ScaleOut**, **Consolidation**, and **Scheduling** — co-operate inside a SimPy discrete-event cloud simulator, guided by a Transformer workload forecaster and a 4-filter Safety Coordinator. A Karpathy-style AutoResearch loop uses an LLM to continuously tune reward weights based on live traffic.

---

## Architecture

```
Alibaba Trace → WorkloadTransformer (MC Dropout)
                        │ forecast (mean + uncertainty)
                        ▼
              ┌─── CloudEnv (SimPy M/G/c) ───┐
              │  ScaleOut   (every 10 steps)  │
              │  Consolidation (every 2 steps)│  ← I-PPO
              │  Scheduling   (every step)    │
              └──────────┬───────────────────┘
                         ▼
               SafetyCoordinator (4 filters)
                         ▼
                   env.step(action)
```

| Component | Description |
|-----------|-------------|
| **ScaleOut Agent** | Discrete(3) — provision 0/1/2 nodes, acts on 10-step cadence or CPU emergency |
| **Consolidation Agent** | MultiBinary(20) — drain idle nodes, filtered by Safety Coordinator |
| **Scheduling Agent** | Per-job priority reordering (pointer-network style) |
| **Safety Coordinator** | Boot-protect · N_min floor · Uncertainty suppression · Scale-out suppression |
| **Forecaster** | Transformer encoder → quantile predictions at t+1/5/10/15 with MC Dropout uncertainty |
| **AutoResearch** | LLM rewrites `experiment.py` each iteration, runs trial, keeps or discards |
| **Live Adaptation** | Streams live traffic into rolling buffer → fine-tunes agents every N minutes |

---

## Results

Evaluated across 3 seeds × 10 episodes against 7 baselines on Alibaba 2018 cluster trace (Day 2):

| Method | SLA Rate | Cost Efficiency | CPU Utilisation | Node Stability |
|--------|----------|-----------------|-----------------|----------------|
| **AutoCloud-Agent** | **100.0%** | **0.962** | **55.2%** | **0.912** |
| ThresholdPredictive | 100.0% | 0.961 | 54.6% | 0.910 |
| PIController | 100.0% | 0.956 | 51.4% | 0.718 |
| ThresholdReactive | 100.0% | 0.955 | 48.3% | 0.822 |
| ARIMAPredictive | 100.0% | 0.954 | 48.0% | 0.816 |
| StaticN (10 nodes) | 100.0% | 0.938 | 33.3% | 1.000 |
| KubernetesHPA | 100.0% | 0.930 | 31.2% | 0.842 |
| SingleAgentPPO | 100.0% | 0.924 | 41.0% | 0.787 |

SLA threshold: P95 latency < 500ms. Training: 300k steps on Alibaba Day 1 trace (Kaggle T4 GPU).
AutoCloud-Agent leads on cost efficiency and node stability while maximising CPU utilisation.

---

## Quickstart

### 1 — Install

```bash
pip install simpy gymnasium torch numpy matplotlib pandas
pip install groq        # for AutoResearch (free API key at console.groq.com)
```

### 2 — Train (Kaggle, GPU recommended)

Open in order on Kaggle:
1. `notebooks/train_forecaster.ipynb` — trains Transformer forecaster
2. `notebooks/train_rl_agents.ipynb` — trains 3 I-PPO agents (300k steps, ~3h on T4)

Download `outputs/` folder to your local machine.

### 3 — Evaluate locally

```bash
python pipeline.py \
  --checkpoint_dir ../outputs/rl_agents \
  --workload_file  ../outputs/train_Forecaster/day2_processed.npy
# Output: evaluation_results.json + printed comparison table
```

### 4 — AutoResearch (offline reward tuning)

```bash
export GROQ_API_KEY=gsk_...        # free key from console.groq.com
python pipeline.py --mode autoresearch --llm_provider groq \
  --checkpoint_dir ../outputs/rl_agents \
  --workload_file  ../outputs/train_Forecaster/day2_processed.npy \
  --ar_steps 25000 --ar_iterations 4
```

The LLM reads `experiment.py` + trial history, proposes reward weight changes, runs a trial, keeps or discards — Karpathy-style.

### 5 — Live Adaptation (continuous tuning from live traffic)

```bash
python pipeline.py --mode live --llm_provider groq \
  --checkpoint_dir ../outputs/rl_agents \
  --workload_file  ../outputs/train_Forecaster/day2_processed.npy \
  --live_interval 3 --live_iterations 5 --live_steps 6000
```

Streams the Alibaba Day 2 trace at **24x compression** (1 day plays in 1 hour).
Every 3 minutes real-time (~72 min of simulated traffic), the LLM sees current
utilisation stats and proposes reward weight updates. Agents are fine-tuned from
existing checkpoints — not retrained from scratch. Best config is promoted live.

For a full 1-hour run (6 iterations × 10 min):
```bash
python pipeline.py --mode live --llm_provider groq \
  --checkpoint_dir ../outputs/rl_agents \
  --workload_file  ../outputs/train_Forecaster/day2_processed.npy \
  --live_interval 10 --live_iterations 6 --live_steps 8000
```

### 6 — Demo notebook

```bash
jupyter notebook notebooks/demo.ipynb
```

Covers: live simulation visualisation · baseline comparison charts · AutoResearch loop.

---

## Project Structure

```
autocloud_agent/
├── pipeline.py        ← main entry point  (eval / autoresearch / live)
├── train.py           ← training script   (called by AutoResearch internally)
├── experiment.py      ← single file the LLM modifies (reward weights + PPO params)
├── program.md         ← AutoResearch research directives (human-editable)
│
├── src/               ← all source code
│   ├── agents/        ← ScaleOut, Consolidation, Scheduling agents + shared PPO
│   ├── configs/       ← Config dataclasses (SimConfig, PPOConfig, RewardConfig)
│   ├── coordinator/   ← SafetyCoordinator (4-filter hierarchical safety gate)
│   ├── environment/   ← SimPy simulator, CloudEnv, workload loader, LiveWorkloadBuffer
│   ├── evaluation/    ← Evaluator (8 methods × 3 seeds × metrics)
│   ├── forecaster/    ← WorkloadTransformer + MCDropoutForecaster
│   ├── training/      ← IPPOTrainer, 7 baselines, EMA normaliser
│   └── autoresearch/  ← LLM engine, subprocess runner, live adaptation loop
│
└── notebooks/
    ├── train_forecaster.ipynb   ← Kaggle: train Transformer forecaster
    ├── train_rl_agents.ipynb    ← Kaggle: train 3 I-PPO agents (300k steps)
    ├── results.ipynb            ← plot learning curves + baseline comparison
    ├── multiday_eval.ipynb      ← evaluate across all 7 Alibaba days
    └── demo.ipynb               ← live visualisation + AutoResearch demo
```

---

## Dataset

[Alibaba 2018 Cluster Trace](https://github.com/alibaba/clusterdata) — 4023 machines, 7 days.
The workload loader (`environment/workload.py`) preprocesses CPU/mem utilization into 30-second bins.
Agents are trained on Day 1 (~25% avg util) and evaluated on Day 2 (30–60% util, unseen distribution).

---

## Key Design Decisions

- **Temporal hierarchy**: agents act at different timescales (1 / 2 / 10 steps) matching real cluster control loops
- **Independent buffers**: each agent's PPO update fires when its own buffer fills — scheduling fills 10× faster than scale-out
- **Safety first**: coordinator filters run before every action; scale-out is never blocked, consolidation is heavily gated
- **MC Dropout uncertainty**: forecaster uncertainty feeds directly into coordinator — high σ suppresses all drain actions
- **Live adaptation**: rolling buffer streams real traffic; LLM detects utilisation shifts and adjusts reward weights in minutes

---

## AutoResearch

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

**Offline mode** (`--mode autoresearch`): LLM reads `experiment.py` + history, proposes a full rewrite,
runs a fresh trial, keeps or discards based on `score = SLA_rate − 0.1 × cost`.

**Live mode** (`--mode live`): Same loop but driven by a rolling buffer of live traffic measurements.
The LLM receives current CPU utilisation stats and adjusts weights accordingly — high traffic shifts
weights toward SLA protection, low traffic toward cost reduction. Agents fine-tune from existing
checkpoints rather than training from scratch, so each iteration takes minutes not hours.

Works with Groq (free), Ollama (local), Gemini (free tier), or Anthropic.

---

## License

MIT
