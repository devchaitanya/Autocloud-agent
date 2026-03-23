# AutoCloud-Agent AutoResearch

Autonomous hyperparameter research for a hierarchical multi-agent RL cloud resource manager.

## The System

Three PPO agents control a SimPy cloud cluster:
- **ScaleOut** — decides how many new nodes to provision (acts every 10 steps)
- **Consolidation** — decides which idle nodes to drain (acts every 2 steps)
- **Scheduling** — assigns job priorities in the queue (acts every step)

A **SafetyCoordinator** filters all consolidation actions through 4 safety gates before execution.

## The Single Modifiable File

`experiment.py` — this is the ONLY file the agent modifies. It defines `get_config()` which returns a `Config` object with all tunable values.

**What you CAN change in experiment.py:**
- PPO hyperparameters: `lr`, `gamma`, `gae_lambda`, `clip_eps`, `entropy_coef`, `entropy_coef_min`, `update_epochs`, `buffer_size`
- Reward weights: `alpha1-4` (ScaleOut), `beta1-4` (Consolidation), `gamma1-2` (Scheduling)
- Coordinator threshold: `sigma_high`

**What you CANNOT change:** anything outside experiment.py.

## Reward Structure

```
r_scaleout     = -α1×max(0, cpu-0.8)     # penalize CPU > 80% (under-provisioned)
               - α2×max(0, 0.3-cpu)      # penalize CPU < 30% (over-provisioned)
               - α3×(n_active/20)        # penalize running too many nodes
               - α4×scale_event          # penalize every boot event

r_consolidation = -β1×step_cost          # penalize dollar cost per step
                + β2×sla_met             # reward when P95 latency < 500s
                - β3×migrations          # penalize forced job migrations
                - β4×max(0, mem-0.85)    # penalize memory > 85%

r_scheduling   = -γ1×(mean_wait/300)     # penalize long queue waits
               - γ2×max(0, p95/500-1)    # penalize SLA violations
```

## The Score

```
score = mean_SLA_rate - 0.1 × mean_cost
```

- `mean_SLA_rate`: fraction of steps with P95 latency < 500s (1.0 = perfect)
- `mean_cost`: total dollar cost over the episode
- **Higher score = better.** Typical range: 0.85 – 1.00

## Running an Experiment

```bash
cd /path/to/autocloud_agent
python train.py --total_steps 3000 --seed 0 --checkpoint_dir /tmp/exp_TAG \
                --experiment_file experiment.py --no_verbose
```

The final output line contains the score:
```
[AutoResearch] score=0.9512 sla=0.9800 cost=0.2876
```

## The Experiment Loop

```
LOOP:
1. Read current experiment.py
2. Read trial history
3. Propose a change (modify get_config values)
4. Write new experiment.py
5. Run training (3000 steps, ~2 min on CPU)
6. Parse score
7. If score improved → KEEP (log status=keep)
8. If score same or worse → REVERT to previous experiment.py (log status=discard)
9. Repeat
```

## Logging

Append each trial to `autoresearch/results.tsv` (tab-separated):
```
iter    score   sla_rate    cost    status  description
1       0.9500  0.9500      0.00    keep    baseline
2       0.9620  0.9620      0.00    keep    increased beta2 SLA bonus to 3.0
3       0.9480  0.9480      0.00    discard decreased lr to 1e-4, SLA dropped
```

## Research Strategy

Start with reward weights (most impactful, least understood):
1. `beta2` (SLA bonus) — increasing this pushes agents to prioritize latency over cost
2. `alpha1` (under-provisioning penalty) — increasing this makes ScaleOut more aggressive
3. `gamma2` (SLA violation penalty) — increasing this makes Scheduling more latency-aware

Then try PPO hyperparameters if reward tuning plateaus:
- Increase `entropy_coef` if the policy seems stuck (low exploration)
- Decrease `lr` if training is unstable
- Increase `buffer_size` for more stable gradients

## Simplicity Criterion

All else equal, simpler is better. A 0.001 score improvement that changes 5 values is not worth it.
Prefer single-value changes so the effect is interpretable.

## NEVER STOP

Once the experiment loop begins, do NOT pause to ask the human if you should continue.
Loop autonomously until manually interrupted.
