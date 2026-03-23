"""
AutoResearch Experiment — THIS IS THE ONLY FILE THE AGENT MODIFIES.

Defines get_config() which returns the Config used for training.
Everything in get_config() is fair game: PPO hyperparameters, reward
weights, and coordinator threshold.

The agent reads this file, proposes changes, and rewrites it each iteration.
If the score improves the new version is kept; otherwise it is reverted.
"""
import copy
from configs.default_config import Config, DEFAULT_CONFIG


def get_config() -> Config:
    """Return the experimental config. Modify the values below freely."""
    config = copy.deepcopy(DEFAULT_CONFIG)

    # ── PPO Hyperparameters ────────────────────────────────────────────
    config.ppo.lr                = 3e-4    # Adam learning rate
    config.ppo.gamma             = 0.99   # discount factor
    config.ppo.gae_lambda        = 0.95   # GAE lambda (bias-variance tradeoff)
    config.ppo.clip_eps          = 0.2    # PPO clip epsilon
    config.ppo.entropy_coef      = 0.01   # entropy bonus (exploration)
    config.ppo.entropy_coef_min  = 0.001  # entropy floor after annealing
    config.ppo.update_epochs     = 4      # PPO epochs per buffer flush
    config.ppo.buffer_size       = 2048   # rollout buffer size per agent

    # ── ScaleOut Reward Weights ────────────────────────────────────────
    # r_scaleout = -α1×max(0,cpu-0.8) - α2×max(0,0.3-cpu)
    #              - α3×(n_active/20) - α4×scale_event
    config.reward.alpha1 = 2.0   # under-provisioning penalty (cpu > 80%)
    config.reward.alpha2 = 1.0   # over-provisioning penalty  (cpu < 30%)
    config.reward.alpha3 = 0.5   # running cost pressure
    config.reward.alpha4 = 0.3   # boot-event penalty

    # ── Consolidation Reward Weights ───────────────────────────────────
    # r_consolidation = -β1×step_cost + β2×sla_met
    #                   - β3×migrations - β4×max(0,mem-0.85)
    config.reward.beta1  = 1.0   # dollar cost penalty
    config.reward.beta2  = 2.0   # SLA met bonus  (p95 latency < 500s)
    config.reward.beta3  = 0.5   # forced migration penalty
    config.reward.beta4  = 1.0   # memory pressure penalty

    # ── Scheduling Reward Weights ──────────────────────────────────────
    # r_scheduling = -γ1×(mean_wait/300) - γ2×max(0, p95/500-1)
    config.reward.gamma1 = 1.0   # mean queue wait penalty
    config.reward.gamma2 = 2.0   # SLA violation penalty

    # ── Safety Coordinator ─────────────────────────────────────────────
    # Drain actions are suppressed when forecast uncertainty σ_t5 > sigma_high
    config.coordinator.sigma_high = 0.3

    return config
