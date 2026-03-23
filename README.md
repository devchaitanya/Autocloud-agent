# AutoCloud-Agent

Hierarchical Multi-Agent Reinforcement Learning for autonomous cloud resource management.

Three independent PPO agents — **ScaleOut**, **Consolidation**, and **Scheduling** — co-operate inside a SimPy discrete-event cloud simulator, guided by a Transformer workload forecaster and a 4-filter Safety Coordinator. A Karpathy-style AutoResearch loop uses an LLM to autonomously tune reward weights overnight.

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
| **AutoResearch** | LLM rewrites `experiment.py` each iteration, runs fast trial, keeps or discards |

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
AutoCloud-Agent achieves the best cost efficiency and node stability while maximising CPU utilisation.

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

Download outputs: `checkpoints/` folder + `day1_processed.npy`

### 3 — Evaluate locally

```bash
python pipeline.py
# Output: evaluation_results.json + printed comparison table
```

### 4 — AutoResearch (LLM reward tuning)

```bash
export GROQ_API_KEY=gsk_...        # free key from console.groq.com
python pipeline.py --mode autoresearch --llm_provider groq

# Or fully local with Ollama (no API key):
ollama pull llama3.2:3b && ollama serve &
python pipeline.py --mode autoresearch --llm_provider ollama
```

The LLM reads `experiment.py`, proposes reward weight changes, runs a 2000-step trial, and keeps or discards — Karpathy-style.

### 5 — Demo notebook

```bash
jupyter notebook notebooks/demo.ipynb
```

Covers: live simulation visualization · baseline comparison charts · AutoResearch loop.

---

## Project Structure

```
autocloud_agent/
├── pipeline.py              ← main local entry point (eval + autoresearch)
├── train.py                 ← training script (called by AutoResearch internally)
├── experiment.py            ← single file the LLM modifies (reward weights + PPO params)
├── program.md               ← AutoResearch research directives (human-editable)
│
├── agents/                  ← ScaleOut, Consolidation, Scheduling agents + shared PPO
├── environment/             ← SimPy simulator, CloudEnv (Gymnasium), workload loader
├── coordinator/             ← SafetyCoordinator (4-filter hierarchical safety gate)
├── forecaster/              ← WorkloadTransformer + MCDropoutForecaster
├── training/                ← IPPOTrainer, baselines (7 methods)
├── autoresearch/            ← Karpathy-style LLM engine + subprocess runner
├── evaluation/              ← Evaluator (8 methods × 3 seeds × metrics)
├── configs/                 ← Config dataclasses (SimConfig, PPOConfig, RewardConfig)
└── notebooks/
    ├── train_forecaster.ipynb
    ├── train_rl_agents.ipynb
    ├── results.ipynb
    ├── multiday_eval.ipynb
    └── demo.ipynb
```

---

## Dataset

[Alibaba 2018 Cluster Trace](https://github.com/alibaba/clusterdata) — 4023 machines, 7 days.
The workload loader (`environment/workload.py`) preprocesses CPU/mem utilization into 30-second bins.

---

## Key Design Decisions

- **Temporal hierarchy**: agents act at different timescales (1 / 2 / 10 steps) matching real cluster control loops
- **Independent buffers**: each agent's PPO update fires when its own buffer fills — scheduling fills 10× faster than scale-out
- **Safety first**: coordinator filters run before every action; scale-out is never blocked, consolidation is heavily gated
- **MC Dropout uncertainty**: forecaster uncertainty feeds directly into coordinator — high σ suppresses all drain actions
- **Seed randomization**: each training episode uses a different random seed for domain randomization and generalization

---

## AutoResearch

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

The LLM is given the current `experiment.py` code + trial history, proposes a rewrite, and the result is kept or discarded based on `score = SLA_rate − 0.1 × cost`. Works with Groq (free), Ollama (local), Gemini (free tier), or Anthropic.

---

## License

MIT
