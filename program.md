# AutoCloud-Agent AutoResearch

Autonomous reward-weight research for a hierarchical multi-agent RL cloud resource manager.
**Primary goal: minimise cost while keeping SLA (P95 latency < 500ms) at 100%.**

## The System

Three PPO agents control a SimPy cloud cluster:
- **ScaleOut** — decides how many new nodes to provision (acts every 10 steps)
- **Consolidation** — decides which idle nodes to drain (acts every 2 steps)
- **Scheduling** — assigns job priorities in the queue (acts every step)

A **SafetyCoordinator** filters all consolidation actions through 4 safety gates before execution.

## The Single Modifiable File

`experiment.py` — this is the ONLY file you modify. It defines `get_config()` returning a `Config` object.

**What you CAN change:**
- Reward weights: `alpha1-4` (ScaleOut), `beta1-4` (Consolidation), `gamma1-2` (Scheduling)
- PPO hyperparameters: `lr`, `gamma`, `gae_lambda`, `clip_eps`, `entropy_coef`, `buffer_size`
- Coordinator threshold: `sigma_high`

**What you CANNOT change:** anything outside experiment.py.

## Reward Structure

```
r_scaleout      = -α1×max(0, cpu-0.8)      # penalize CPU > 80%  (under-provisioned → SLA risk)
                - α2×max(0, 0.3-cpu)        # penalize CPU < 30%  (over-provisioned  → wasted cost)
                - α3×(n_active/20)           # penalize running too many nodes
                - α4×scale_event            # penalize every boot event

r_consolidation = -β1×step_cost            # penalize dollar cost per step
                + β2×sla_met               # reward when P95 latency < 500ms
                - β3×migrations            # penalize forced job migrations
                - β4×max(0, mem-0.85)      # penalize memory > 85%

r_scheduling    = -γ1×(mean_wait/300)      # penalize long queue waits
                - γ2×max(0, p95/500-1)     # penalize SLA violations
```

## The Score

```
score = mean_SLA_rate - 0.1 × mean_cost
```

- `mean_SLA_rate`: fraction of steps with P95 latency < 500ms (target: 1.0)
- `mean_cost`: total dollar cost over the episode (lower = better)
- **Higher score = better.** Typical range: 0.85 – 1.00

## Traffic-Aware Reward Strategy

**This is the core idea: adjust reward weights based on current traffic.**

### LOW traffic (mean_util < 35%)
Goal: aggressively cut cost — consolidate idle nodes, avoid over-provisioning.
```
↑ alpha2   (over-provisioning penalty)  — punish running too many idle nodes
↑ alpha3   (node count penalty)         — push ScaleOut to provision fewer nodes
↑ beta1    (cost penalty weight)        — make Consolidation more cost-aggressive
↓ alpha1   (under-prov penalty)         — relax SLA guard (traffic is low, SLA is safe)
↓ beta2    (SLA bonus)                  — less need to reward SLA when it's trivially met
```

### MEDIUM traffic (35% ≤ mean_util < 60%)
Goal: balance — maintain SLA, moderate cost savings.
```
Keep defaults or make small adjustments.
↑ beta2   slightly (SLA bonus) to keep latency headroom
↑ beta1   slightly (cost) to avoid unnecessary provisioning
```

### HIGH traffic (mean_util ≥ 60%)
Goal: protect SLA — provision aggressively, do not drain nodes.
```
↑ alpha1   (under-prov penalty)  — punish CPU > 80% harshly
↑ beta2    (SLA met bonus)       — reward latency compliance heavily
↑ gamma2   (SLA violation penalty) — make Scheduling prioritize latency
↓ beta1    (cost penalty)        — relax cost pressure, SLA is the priority
↓ alpha3   (node count penalty)  — allow more nodes to be active
```

## Practical Rules

1. **Always check the live traffic context** in the prompt before proposing a change.
2. **Change ONE weight at a time** — keeps the effect interpretable.
3. **Match the regime** — if util < 35%, push cost weights; if util > 60%, push SLA weights.
4. **If SLA_rate < 1.0**, immediately increase `alpha1`, `beta2`, `gamma2` regardless of cost.
5. **If cost is high with SLA_rate = 1.0**, increase `beta1` or `alpha3` to drive consolidation.
6. **Discard any change that drops SLA_rate below 0.95** — cost savings don't matter if SLA fails.

## Research Priority Order

1. `beta1` (consolidation cost penalty) — most direct lever for cost reduction
2. `alpha2` (over-provisioning penalty) — stops ScaleOut from over-provisioning at low load
3. `alpha3` (node count penalty) — scales with cluster size
4. `beta2` (SLA bonus) — critical at high traffic
5. `gamma2` (SLA violation penalty) — scheduling-level SLA guard
6. PPO hyperparameters — only if reward tuning has plateaued

## Simplicity Criterion

A 0.001 score improvement that changes 5 values is not worth it.
Prefer single-value changes so the effect is interpretable and reversible.

## NEVER STOP

Once the loop begins, do NOT pause to ask. Loop autonomously until interrupted.
