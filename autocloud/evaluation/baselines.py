"""
Baseline policies for comparison against AutoCloud-Agent.

All baselines expose the same interface:
    select_action(obs, env) -> Dict[str, Any]

Baselines (SOTA):
  1. StaticN              — fixed N=10 nodes, do-nothing lower bound
  2. ThresholdReactive    — CPU threshold with cooldown (classic rule-based)
  3. KubernetesHPA        — industry-standard k8s Horizontal Pod Autoscaler
  4. AWSTargetTracking    — AWS Auto Scaling Target Tracking policy
  5. MPCController        — Model Predictive Control (strongest non-RL baseline)
  6. SingleAgentPPO       — single-agent RL ablation (shows value of I-PPO)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from autocloud.simulator.cloud_env import CloudEnv, OBS_DIM
from autocloud.agents.ppo import PPO, build_mlp, orthogonal_init
from torch.distributions import Categorical, Bernoulli


N_MAX  = 20
K_JOBS = 20


# ═══════════════════════════════════════════════════════════════════
# 1. Static-N  (do-nothing lower bound)
# ═══════════════════════════════════════════════════════════════════

class StaticN:
    """
    Fixed N=10 nodes. No scaling, no draining, default priority for all jobs.
    Serves as the 'do nothing' baseline.
    """
    def __init__(self, n_nodes: int = 10):
        self.n_nodes = n_nodes

    def select_action(self, obs: np.ndarray, env: CloudEnv) -> Dict:
        return {
            "scaleout":      0,
            "consolidation": np.zeros(N_MAX, dtype=np.float32),
            "scheduling":    2,   # neutral priority
        }

    def reset(self): pass


# ═══════════════════════════════════════════════════════════════════
# 2. Threshold-Reactive  (classic rule-based)
# ═══════════════════════════════════════════════════════════════════

class ThresholdReactive:
    """
    Rule-based reactive autoscaler:
      - Scale up (+1 node) if mean CPU > cpu_high AND cooldown elapsed
      - Drain lowest-util node if mean CPU < cpu_low AND cooldown elapsed
      - 5-minute cooldown between scaling events (matches AWS Auto Scaling default)
    """
    def __init__(
        self,
        cpu_high: float = 0.80,
        cpu_low:  float = 0.30,
        cooldown_steps: int = 10,   # 10 steps × 30s = 5 min
    ):
        self.cpu_high = cpu_high
        self.cpu_low  = cpu_low
        self.cooldown_steps = cooldown_steps
        self._last_scale_step = -cooldown_steps

    def select_action(self, obs: np.ndarray, env: CloudEnv) -> Dict:
        metrics = env.get_sim_metrics()
        cpu     = metrics["mean_cpu_util"]
        step    = env._step_count

        scaleout      = 0
        consolidation = np.zeros(N_MAX, dtype=np.float32)

        cooldown_ok = (step - self._last_scale_step) >= self.cooldown_steps

        if cooldown_ok:
            if cpu > self.cpu_high:
                scaleout = 1
                self._last_scale_step = step
            elif cpu < self.cpu_low:
                slot = self._lowest_util_slot(obs, env)
                if slot is not None:
                    consolidation[slot] = 1.0
                    self._last_scale_step = step

        return {
            "scaleout":      scaleout,
            "consolidation": consolidation,
            "scheduling":    2,
        }

    @staticmethod
    def _lowest_util_slot(obs: np.ndarray, env: CloudEnv) -> Optional[int]:
        from autocloud.simulator.node import NodeState
        live_nodes = [n for n in env.sim.nodes if n.state != NodeState.TERMINATED]
        active = [(i, n) for i, n in enumerate(live_nodes[:N_MAX])
                  if n.state == NodeState.ACTIVE]
        if not active:
            return None
        return min(active, key=lambda x: x[1].cpu_util)[0]

    def reset(self):
        self._last_scale_step = -self.cooldown_steps


# ═══════════════════════════════════════════════════════════════════
# 3. Kubernetes HPA  (industry standard)
# ═══════════════════════════════════════════════════════════════════

class KubernetesHPA:
    """
    Replicates the Kubernetes Horizontal Pod Autoscaler algorithm.

    Official formula (k8s docs):
        desiredReplicas = ceil(currentReplicas × currentMetric / desiredMetric)

    Dead band: only scale if ratio > 1.1 (scale-up) or < 0.9 (scale-down)
    to avoid thrashing — matches the default k8s tolerance of 0.1.

    Scale-down stabilisation: 5-min window (10 steps) before draining,
    matching the default --horizontal-pod-autoscaler-downscale-stabilization.

    Reference: https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
    """
    def __init__(
        self,
        target_cpu: float = 0.50,
        tolerance: float = 0.10,
        cooldown_steps: int = 10,
        n_min: int = 3,
        n_max: int = 20,
    ):
        self.target_cpu     = target_cpu
        self.tolerance      = tolerance
        self.cooldown_steps = cooldown_steps
        self.n_min          = n_min
        self.n_max          = n_max
        self._last_scaledown_step = -cooldown_steps

    def select_action(self, obs: np.ndarray, env: CloudEnv) -> Dict:
        from autocloud.simulator.node import NodeState
        metrics  = env.get_sim_metrics()
        cpu      = metrics["mean_cpu_util"]
        step     = env._step_count

        live_nodes = [n for n in env.sim.nodes if n.state != NodeState.TERMINATED]
        n_active   = max(sum(1 for n in live_nodes if n.state == NodeState.ACTIVE), 1)

        ratio   = cpu / max(self.target_cpu, 1e-6)
        desired = int(np.ceil(n_active * ratio))
        desired = int(np.clip(desired, self.n_min, self.n_max))

        scaleout      = 0
        consolidation = np.zeros(N_MAX, dtype=np.float32)

        if ratio > (1.0 + self.tolerance) and desired > n_active:
            scaleout = min(desired - n_active, 2)
        elif ratio < (1.0 - self.tolerance) and desired < n_active:
            if (step - self._last_scaledown_step) >= self.cooldown_steps:
                slot = ThresholdReactive._lowest_util_slot(obs, env)
                if slot is not None:
                    consolidation[slot] = 1.0
                    self._last_scaledown_step = step

        return {"scaleout": scaleout, "consolidation": consolidation, "scheduling": 2}

    def reset(self):
        self._last_scaledown_step = -self.cooldown_steps


# ═══════════════════════════════════════════════════════════════════
# 4. AWS Target Tracking  (industry standard)
# ═══════════════════════════════════════════════════════════════════

class AWSTargetTracking:
    """
    Replicates AWS Auto Scaling Target Tracking policy.

    Differences from KubernetesHPA:
      - Scale-out fires as soon as ratio > target (no dead-band)
      - Scale-in is protected by a longer stabilisation window (default 15 min)
        to avoid premature scale-in during brief dips.
      - Capacity is always rounded UP on scale-out (safety bias).
      - Metric aggregation uses max over the last N steps (not current point),
        matching AWS's 1-minute aggregated CloudWatch metric.

    Reference: https://docs.aws.amazon.com/autoscaling/ec2/userguide/as-scaling-target-tracking.html
    """
    def __init__(
        self,
        target_cpu: float = 0.60,
        scaleout_cooldown: int = 4,
        scalein_cooldown: int = 30,
        aggregation_window: int = 2,
        n_min: int = 3,
        n_max: int = 20,
    ):
        self.target_cpu          = target_cpu
        self.scaleout_cooldown   = scaleout_cooldown
        self.scalein_cooldown    = scalein_cooldown
        self.aggregation_window  = aggregation_window
        self.n_min               = n_min
        self.n_max               = n_max
        self._last_scaleout_step = -scaleout_cooldown
        self._last_scalein_step  = -scalein_cooldown
        self._cpu_history: list  = []

    def select_action(self, obs: np.ndarray, env: CloudEnv) -> Dict:
        from autocloud.simulator.node import NodeState
        metrics = env.get_sim_metrics()
        cpu     = metrics["mean_cpu_util"]
        step    = env._step_count

        self._cpu_history.append(cpu)
        if len(self._cpu_history) > self.aggregation_window:
            self._cpu_history.pop(0)
        agg_cpu = max(self._cpu_history)

        live_nodes = [n for n in env.sim.nodes if n.state != NodeState.TERMINATED]
        n_active   = max(sum(1 for n in live_nodes if n.state == NodeState.ACTIVE), 1)

        desired = int(np.ceil(n_active * agg_cpu / max(self.target_cpu, 1e-6)))
        desired = int(np.clip(desired, self.n_min, self.n_max))

        scaleout      = 0
        consolidation = np.zeros(N_MAX, dtype=np.float32)

        if desired > n_active and (step - self._last_scaleout_step) >= self.scaleout_cooldown:
            scaleout = min(desired - n_active, 2)
            self._last_scaleout_step = step
        elif desired < n_active and (step - self._last_scalein_step) >= self.scalein_cooldown:
            slot = ThresholdReactive._lowest_util_slot(obs, env)
            if slot is not None:
                consolidation[slot] = 1.0
                self._last_scalein_step = step

        return {"scaleout": scaleout, "consolidation": consolidation, "scheduling": 2}

    def reset(self):
        self._last_scaleout_step = -self.scaleout_cooldown
        self._last_scalein_step  = -self.scalein_cooldown
        self._cpu_history        = []


# ═══════════════════════════════════════════════════════════════════
# 5. MPC Controller  (strongest non-RL baseline)
# ═══════════════════════════════════════════════════════════════════

class MPCController:
    """
    Model Predictive Control autoscaler.

    Forecasts CPU utilisation over horizon H using Holt's double-exponential
    smoothing, then picks the minimum node count that keeps forecast CPU
    below the SLA target.

    MPC is a key baseline in the cloud-autoscaling literature:
      - AWARE (USENIX ATC 2023) uses MPC as its strongest non-RL baseline
      - Roy et al. (2011) compare MPC vs EMA-threshold for cloud scale-out

    References:
      Garcia et al. (1989) "Model Predictive Control: Theory and Practice"
      AWARE: Qiu et al. (2023) USENIX ATC
    """
    def __init__(
        self,
        horizon: int = 5,
        alpha: float = 0.3,
        beta: float  = 0.1,
        sla_target: float = 0.75,
        w_sla:  float = 3.0,
        w_cost: float = 1.0,
        cooldown_steps: int = 4,
        n_min: int = 3,
        n_max: int = 20,
    ):
        self.horizon        = horizon
        self.alpha          = alpha
        self.beta           = beta
        self.sla_target     = sla_target
        self.w_sla          = w_sla
        self.w_cost         = w_cost
        self.cooldown_steps = cooldown_steps
        self.n_min          = n_min
        self.n_max          = n_max
        self._level: Optional[float] = None
        self._trend: float  = 0.0
        self._last_step: int = -cooldown_steps

    def _forecast_horizon(self, cpu: float) -> list:
        if self._level is None:
            self._level = cpu
        prev_level  = self._level
        self._level = self.alpha * cpu + (1 - self.alpha) * (self._level + self._trend)
        self._trend = self.beta * (self._level - prev_level) + (1 - self.beta) * self._trend
        return [
            float(np.clip(self._level + h * self._trend, 0.0, 1.0))
            for h in range(1, self.horizon + 1)
        ]

    def _mpc_optimal_n(self, forecasts: list, n_active: int) -> int:
        cpu_peak = max(forecasts)
        n_needed = int(np.ceil(cpu_peak / max(self.sla_target, 1e-6)))
        n_needed = int(np.clip(n_needed, self.n_min, self.n_max))
        return n_needed

    def select_action(self, obs: np.ndarray, env: CloudEnv) -> Dict:
        from autocloud.simulator.node import NodeState
        metrics  = env.get_sim_metrics()
        cpu      = metrics["mean_cpu_util"]
        step     = env._step_count

        live_nodes = [n for n in env.sim.nodes if n.state != NodeState.TERMINATED]
        n_active   = max(sum(1 for n in live_nodes if n.state == NodeState.ACTIVE), 1)

        forecasts = self._forecast_horizon(cpu)
        n_opt     = self._mpc_optimal_n(forecasts, n_active)

        scaleout      = 0
        consolidation = np.zeros(N_MAX, dtype=np.float32)

        if (step - self._last_step) >= self.cooldown_steps:
            if n_opt > n_active:
                scaleout = min(n_opt - n_active, 2)
                self._last_step = step
            elif n_opt < n_active:
                slot = ThresholdReactive._lowest_util_slot(obs, env)
                if slot is not None:
                    consolidation[slot] = 1.0
                    self._last_step = step

        return {"scaleout": scaleout, "consolidation": consolidation, "scheduling": 2}

    def reset(self):
        self._level     = None
        self._trend     = 0.0
        self._last_step = -self.cooldown_steps


# ═══════════════════════════════════════════════════════════════════
# 6. Single-Agent PPO  (RL ablation — shows value of multi-agent)
# ═══════════════════════════════════════════════════════════════════

class SingleAgentActor(nn.Module):
    """Joint actor: shared backbone → three action heads."""
    def __init__(self, obs_dim: int = OBS_DIM):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128),    nn.LayerNorm(128), nn.ReLU(),
        )
        self.head_so  = nn.Linear(128, 3)
        self.head_con = nn.Linear(128, N_MAX)
        self.head_sch = nn.Linear(128, 5)
        orthogonal_init(self.backbone, gain=np.sqrt(2))
        for h in [self.head_so, self.head_con, self.head_sch]:
            orthogonal_init(nn.Sequential(h), gain=0.01)

    def forward(self, obs):
        h = self.backbone(obs)
        return self.head_so(h), self.head_con(h), self.head_sch(h)


class SingleAgentCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM):
        super().__init__()
        self.net = build_mlp(obs_dim, [256, 128], 1, layernorm=True)
        orthogonal_init(self.net, gain=np.sqrt(2))

    def forward(self, obs):
        return self.net(obs)


class SingleAgentPPO(PPO):
    """One PPO agent controlling all three action dimensions jointly."""
    def __init__(self, obs_dim: int = OBS_DIM, device: str = "cpu", **ppo_kwargs):
        actor  = SingleAgentActor(obs_dim)
        critic = SingleAgentCritic(obs_dim)
        super().__init__(actor=actor, critic=critic, device=device, **ppo_kwargs)
        self._device = device

    @torch.no_grad()
    def act(self, obs: np.ndarray):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self._device)
        logits_so, logits_con, logits_sch = self.actor(obs_t)

        dist_so  = Categorical(logits=logits_so)
        dist_con = Bernoulli(logits=logits_con)
        dist_sch = Categorical(logits=logits_sch)

        a_so  = dist_so.sample()
        a_con = dist_con.sample()
        a_sch = dist_sch.sample()

        lp_so  = dist_so.log_prob(a_so)
        lp_con = dist_con.log_prob(a_con).sum()
        lp_sch = dist_sch.log_prob(a_sch)
        log_prob = (lp_so + lp_con + lp_sch).item()

        value = self.critic(obs_t).squeeze().item()

        action = np.concatenate([
            a_so.cpu().numpy(),
            a_con.squeeze(0).cpu().numpy(),
            a_sch.cpu().numpy(),
        ])
        return action, log_prob, value

    def select_action(self, obs: np.ndarray, env: CloudEnv) -> Dict:
        action_vec, _, _ = self.act(obs)
        so  = int(action_vec[0])
        con = action_vec[1:1 + N_MAX]
        sch = int(action_vec[1 + N_MAX])
        return {"scaleout": so, "consolidation": con, "scheduling": sch}

    def _get_dist(self, obs):
        logits_so, logits_con, logits_sch = self.actor(obs)
        return (Categorical(logits=logits_so),
                Bernoulli(logits=logits_con),
                Categorical(logits=logits_sch))

    def _log_prob(self, dist, action, mask=None):
        dist_so, dist_con, dist_sch = dist
        a_so  = action[:, 0].long()
        a_con = action[:, 1:1 + N_MAX]
        a_sch = action[:, 1 + N_MAX].long()
        lp = (dist_so.log_prob(a_so)
              + dist_con.log_prob(a_con).sum(dim=-1)
              + dist_sch.log_prob(a_sch))
        return lp

    def _entropy(self, dist, mask=None):
        dist_so, dist_con, dist_sch = dist
        return (dist_so.entropy().mean()
                + dist_con.entropy().mean()
                + dist_sch.entropy().mean())

    def reset(self): pass
