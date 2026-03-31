"""
Baseline policies for comparison against AutoCloud-Agent.

All baselines expose the same interface:
    select_action(obs, env) -> Dict[str, Any]

Baselines:
  1. StaticN              — fixed N=10 nodes, no scaling ever
  2. ThresholdReactive    — CPU threshold on current metrics, 5-min cooldown
  3. ThresholdPredictive  — same thresholds applied to LSTM forecast (t+5)
  4. SingleAgentPPO       — one PPO agent with joint action space
  5. KubernetesHPA        — industry-standard ratio-based autoscaler (k8s formula)
  6. PIController         — Proportional-Integral control theory autoscaler
  7. ARIMAPredictive      — EMA double-smoothing forecast + threshold (ARIMA-lite)
  8. AWSTargetTracking    — AWS Auto Scaling Target Tracking policy (2019–present)
  9. MPCController        — Model Predictive Control, 5-step EWMA lookahead
 10. PIDDerivative        — Full PID (adds derivative term to PIController)
 11. BurstAwareScaler     — Sliding-window burst detection + proactive scale-out
"""
from __future__ import annotations

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.cloud_env import CloudEnv, OBS_DIM
from agents.ppo import PPO, build_mlp, orthogonal_init
from torch.distributions import Categorical, Bernoulli


N_MAX  = 20
K_JOBS = 20


# ═══════════════════════════════════════════════════════════════════
# 1. Static-N
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
# 2. Threshold-Reactive
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
                # Drain lowest-utilization ACTIVE node
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
        from environment.node import NodeState
        live_nodes = [n for n in env.sim.nodes if n.state != NodeState.TERMINATED]
        active = [(i, n) for i, n in enumerate(live_nodes[:N_MAX])
                  if n.state == NodeState.ACTIVE]
        if not active:
            return None
        return min(active, key=lambda x: x[1].cpu_util)[0]

    def reset(self):
        self._last_scale_step = -self.cooldown_steps


# ═══════════════════════════════════════════════════════════════════
# 3. Threshold-Predictive
# ═══════════════════════════════════════════════════════════════════

class ThresholdPredictive:
    """
    Same thresholds as ThresholdReactive but applied to the t+5 forecast
    (obs[206] = forecast_mean_t5, the 7th global feature at index 200+6).
    Demonstrates value of forecasting without RL.
    """
    FORECAST_T5_IDX = 206   # obs[200+6] = forecast mean at t+5

    def __init__(
        self,
        cpu_high: float = 0.80,
        cpu_low:  float = 0.30,
        cooldown_steps: int = 10,
    ):
        self.cpu_high = cpu_high
        self.cpu_low  = cpu_low
        self.cooldown_steps = cooldown_steps
        self._last_scale_step = -cooldown_steps

    def select_action(self, obs: np.ndarray, env: CloudEnv) -> Dict:
        forecast_cpu = float(obs[self.FORECAST_T5_IDX])
        step = env._step_count

        scaleout      = 0
        consolidation = np.zeros(N_MAX, dtype=np.float32)

        cooldown_ok = (step - self._last_scale_step) >= self.cooldown_steps

        if cooldown_ok:
            if forecast_cpu > self.cpu_high:
                scaleout = 1
                self._last_scale_step = step
            elif forecast_cpu < self.cpu_low:
                slot = ThresholdReactive._lowest_util_slot(obs, env)
                if slot is not None:
                    consolidation[slot] = 1.0
                    self._last_scale_step = step

        return {
            "scaleout":      scaleout,
            "consolidation": consolidation,
            "scheduling":    2,
        }

    def reset(self):
        self._last_scale_step = -self.cooldown_steps


# ═══════════════════════════════════════════════════════════════════
# 4. Single-Agent PPO
# ═══════════════════════════════════════════════════════════════════

class SingleAgentActor(nn.Module):
    """
    Joint actor: outputs logits for all 3 action dimensions simultaneously.
    Shared backbone → three heads.
    """
    def __init__(self, obs_dim: int = OBS_DIM):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128),    nn.LayerNorm(128), nn.ReLU(),
        )
        self.head_so  = nn.Linear(128, 3)     # scaleout: Discrete(3)
        self.head_con = nn.Linear(128, N_MAX)  # consolidation: MultiBinary(20)
        self.head_sch = nn.Linear(128, 5)      # scheduling: Discrete(5)
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
    """
    One PPO agent controlling all three action dimensions jointly.
    Combined reward = sum of all three per-agent rewards.
    """
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
        # action shape: (batch, 1 + N_MAX + 1)
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


# ═══════════════════════════════════════════════════════════════════
# 5. Kubernetes HPA
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
        from environment.node import NodeState
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
# 6. PI Controller
# ═══════════════════════════════════════════════════════════════════

class PIController:
    """
    Proportional-Integral (PI) control law for autoscaling.

    References:
      - Hellerstein et al. (2004) "Feedback Control of Computing Systems"
      - Padala et al. (2009) "Autocontrol: Automated control of application"
    """
    def __init__(
        self,
        target_cpu: float = 0.60,
        kp: float = 2.0,
        ki: float = 0.5,
        dt: float = 1.0,
        max_integral: float = 3.0,
        cooldown_steps: int = 6,
        n_min: int = 3,
        n_max: int = 20,
    ):
        self.target_cpu     = target_cpu
        self.kp             = kp
        self.ki             = ki
        self.dt             = dt
        self.max_integral   = max_integral
        self.cooldown_steps = cooldown_steps
        self.n_min          = n_min
        self.n_max          = n_max
        self._integral      = 0.0
        self._last_step     = -cooldown_steps

    def select_action(self, obs: np.ndarray, env: CloudEnv) -> Dict:
        from environment.node import NodeState
        metrics  = env.get_sim_metrics()
        cpu      = metrics["mean_cpu_util"]
        step     = env._step_count

        live_nodes = [n for n in env.sim.nodes if n.state != NodeState.TERMINATED]
        n_active   = max(sum(1 for n in live_nodes if n.state == NodeState.ACTIVE), 1)

        error = self.target_cpu - cpu
        self._integral = float(np.clip(
            self._integral + error * self.dt,
            -self.max_integral, self.max_integral
        ))
        u = self.kp * error + self.ki * self._integral
        desired = int(np.clip(round(n_active * (1.0 - u)), self.n_min, self.n_max))

        scaleout      = 0
        consolidation = np.zeros(N_MAX, dtype=np.float32)

        if (step - self._last_step) >= self.cooldown_steps:
            if desired > n_active:
                scaleout = min(desired - n_active, 2)
                self._last_step = step
            elif desired < n_active:
                slot = ThresholdReactive._lowest_util_slot(obs, env)
                if slot is not None:
                    consolidation[slot] = 1.0
                    self._last_step = step

        return {"scaleout": scaleout, "consolidation": consolidation, "scheduling": 2}

    def reset(self):
        self._integral  = 0.0
        self._last_step = -self.cooldown_steps


# ═══════════════════════════════════════════════════════════════════
# 7. ARIMA Predictive (Double EMA / Holt's method)
# ═══════════════════════════════════════════════════════════════════

class ARIMAPredictive:
    """
    Statistical forecasting autoscaler using Double Exponential Smoothing
    (Holt's method) — an ARIMA(0,1,1) approximation.

    References:
      - Roy et al. (2011) "Efficient autoscaling in the cloud using
        predictive models for workload forecasting"
      - Caron et al. (2010) "Auto-scaling, load balancing and monitoring"
    """
    def __init__(
        self,
        alpha: float = 0.3,
        beta:  float = 0.1,
        horizon: int = 5,
        cpu_high: float = 0.80,
        cpu_low:  float = 0.30,
        cooldown_steps: int = 10,
    ):
        self.alpha          = alpha
        self.beta           = beta
        self.horizon        = horizon
        self.cpu_high       = cpu_high
        self.cpu_low        = cpu_low
        self.cooldown_steps = cooldown_steps
        self._level         = None
        self._trend         = 0.0
        self._last_step     = -cooldown_steps

    def _forecast(self, cpu: float) -> float:
        if self._level is None:
            self._level = cpu
            return cpu
        prev_level  = self._level
        self._level = self.alpha * cpu + (1 - self.alpha) * (self._level + self._trend)
        self._trend = self.beta * (self._level - prev_level) + (1 - self.beta) * self._trend
        return float(np.clip(self._level + self.horizon * self._trend, 0.0, 1.0))

    def select_action(self, obs: np.ndarray, env: CloudEnv) -> Dict:
        metrics      = env.get_sim_metrics()
        cpu          = metrics["mean_cpu_util"]
        step         = env._step_count
        forecast_cpu = self._forecast(cpu)

        scaleout      = 0
        consolidation = np.zeros(N_MAX, dtype=np.float32)

        if (step - self._last_step) >= self.cooldown_steps:
            if forecast_cpu > self.cpu_high:
                scaleout = 1
                self._last_step = step
            elif forecast_cpu < self.cpu_low:
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
# 8. AWS Target Tracking
# ═══════════════════════════════════════════════════════════════════

class AWSTargetTracking:
    """
    Replicates AWS Auto Scaling Target Tracking policy.

    Differences from KubernetesHPA:
      - Scale-out fires as soon as ratio > target (no dead-band)
      - Scale-in is protected by a longer stabilisation window (default 15 min)
        to avoid premature scale-in during brief dips — AWS documentation
        explicitly notes this asymmetry to guard against thrashing.
      - Capacity is always rounded UP on scale-out (safety bias).
      - Metric aggregation uses max over the last N steps (not current point),
        matching AWS's 1-minute aggregated CloudWatch metric.

    References:
      AWS Auto Scaling User Guide — "Target Tracking Scaling Policies"
      https://docs.aws.amazon.com/autoscaling/ec2/userguide/as-scaling-target-tracking.html
    """
    def __init__(
        self,
        target_cpu: float = 0.60,
        scaleout_cooldown: int = 4,    # 4 steps × 30s = 2 min (AWS default: 300s)
        scalein_cooldown: int = 30,    # 30 steps × 30s = 15 min (AWS default: 300s scale-in stabilisation)
        aggregation_window: int = 2,   # use max over last 2 steps (AWS: 1-min aggregation)
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
        from environment.node import NodeState
        metrics = env.get_sim_metrics()
        cpu     = metrics["mean_cpu_util"]
        step    = env._step_count

        # Rolling max over aggregation window (mimics CloudWatch 1-min max)
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

        # Scale-out: faster cooldown, fires when desired > current
        if desired > n_active and (step - self._last_scaleout_step) >= self.scaleout_cooldown:
            scaleout = min(desired - n_active, 2)
            self._last_scaleout_step = step

        # Scale-in: longer stabilisation window before draining
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
# 9. MPC Controller
# ═══════════════════════════════════════════════════════════════════

class MPCController:
    """
    Model Predictive Control autoscaler.

    At each step, forecasts CPU utilisation over a horizon H using Holt's
    double-exponential smoothing (same forecast model as ARIMAPredictive)
    then solves the following 1-D optimisation analytically:

        n* = argmin_{n ∈ [n_min, n_max]} Σ_{h=1}^{H} [
                 w_sla  × max(0, cpu_h / n_norm - sla_target)²
               + w_cost × n / n_max
             ]

    where cpu_h is the forecast at horizon h and n_norm = n / n_ref.
    Because the cost is convex and separable, the optimum is the smallest
    n such that cpu_peak / n_norm ≤ sla_target, clipped to [n_min, n_max].

    MPC is a key baseline in the cloud-autoscaling literature:
      - AWARE (USENIX ATC 2023) uses MPC as its strongest non-RL baseline
      - Hellerstein et al. (2004) frame PID as a degenerate 1-step MPC
      - Roy et al. (2011) compare MPC vs EMA-threshold for cloud scale-out

    References:
      Garcia et al. (1989) "Model Predictive Control: Theory and Practice"
      AWARE: Qiu et al. (2023) USENIX ATC — autoscaling in production clouds
    """
    def __init__(
        self,
        horizon: int = 5,
        alpha: float = 0.3,
        beta: float  = 0.1,
        sla_target: float = 0.75,   # fraction of n_max above which SLA is at risk
        w_sla:  float = 3.0,        # SLA violation weight in MPC cost
        w_cost: float = 1.0,        # provisioning cost weight in MPC cost
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
        """Update EWM state and return list of forecasts for h=1..H."""
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
        """
        Analytical MPC solution: find smallest n that minimises
        SLA risk over the forecast horizon, subject to cost penalty.
        """
        cpu_peak = max(forecasts)    # worst-case demand in horizon

        # Minimum nodes to keep cpu_peak / n_util_per_node ≤ sla_target
        # n_util_per_node ≈ cpu_peak / n  →  n ≥ cpu_peak / sla_target
        n_needed = int(np.ceil(cpu_peak / max(self.sla_target, 1e-6)))
        n_needed = int(np.clip(n_needed, self.n_min, self.n_max))

        # Cost pressure: if current count already satisfies SLA, prefer
        # reducing to n_needed (but don't go below it).
        return n_needed

    def select_action(self, obs: np.ndarray, env: CloudEnv) -> Dict:
        from environment.node import NodeState
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
# 10. PID Derivative Controller
# ═══════════════════════════════════════════════════════════════════

class PIDDerivative:
    """
    Full Proportional-Integral-Derivative (PID) autoscaler.

    Extends PIController with a derivative term that anticipates trend:
        u = Kp × e + Ki × ∫e dt + Kd × de/dt

    The D-term reacts to the *rate of change* of CPU utilisation, allowing
    the controller to begin scaling before the error grows large — analogous
    to a car driver pressing the brakes earlier when seeing the car ahead
    brake, not only when their own speed is too high.

    The derivative is low-pass filtered (first-order filter with τ_d) to
    avoid amplifying measurement noise — standard practice in industrial PID
    (Åström & Hägglund, 2006).

    References:
      Åström & Hägglund (2006) "Advanced PID Control", ISA Press
      Padala et al. (2009) "Autocontrol: Automated Control of Applications"
      Li et al. (2020) "PID control for cloud autoscaling: a survey"
    """
    def __init__(
        self,
        target_cpu: float = 0.60,
        kp: float = 2.0,
        ki: float = 0.3,
        kd: float = 1.5,        # derivative gain
        tau_d: float = 0.1,     # derivative filter time constant
        dt: float = 1.0,
        max_integral: float = 3.0,
        cooldown_steps: int = 4,
        n_min: int = 3,
        n_max: int = 20,
    ):
        self.target_cpu     = target_cpu
        self.kp             = kp
        self.ki             = ki
        self.kd             = kd
        self.tau_d          = tau_d
        self.dt             = dt
        self.max_integral   = max_integral
        self.cooldown_steps = cooldown_steps
        self.n_min          = n_min
        self.n_max          = n_max
        self._integral      = 0.0
        self._prev_error    = 0.0
        self._deriv_filt    = 0.0   # filtered derivative
        self._last_step     = -cooldown_steps

    def select_action(self, obs: np.ndarray, env: CloudEnv) -> Dict:
        from environment.node import NodeState
        metrics  = env.get_sim_metrics()
        cpu      = metrics["mean_cpu_util"]
        step     = env._step_count

        live_nodes = [n for n in env.sim.nodes if n.state != NodeState.TERMINATED]
        n_active   = max(sum(1 for n in live_nodes if n.state == NodeState.ACTIVE), 1)

        error          = self.target_cpu - cpu
        self._integral = float(np.clip(
            self._integral + error * self.dt,
            -self.max_integral, self.max_integral
        ))

        # Low-pass filtered derivative to suppress noise
        raw_deriv       = (error - self._prev_error) / self.dt
        alpha_d         = self.dt / (self.tau_d + self.dt)
        self._deriv_filt = alpha_d * raw_deriv + (1 - alpha_d) * self._deriv_filt
        self._prev_error = error

        u = self.kp * error + self.ki * self._integral + self.kd * self._deriv_filt
        desired = int(np.clip(round(n_active * (1.0 - u)), self.n_min, self.n_max))

        scaleout      = 0
        consolidation = np.zeros(N_MAX, dtype=np.float32)

        if (step - self._last_step) >= self.cooldown_steps:
            if desired > n_active:
                scaleout = min(desired - n_active, 2)
                self._last_step = step
            elif desired < n_active:
                slot = ThresholdReactive._lowest_util_slot(obs, env)
                if slot is not None:
                    consolidation[slot] = 1.0
                    self._last_step = step

        return {"scaleout": scaleout, "consolidation": consolidation, "scheduling": 2}

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0
        self._deriv_filt = 0.0
        self._last_step  = -self.cooldown_steps


# ═══════════════════════════════════════════════════════════════════
# 11. Burst-Aware Scaler
# ═══════════════════════════════════════════════════════════════════

class BurstAwareScaler:
    """
    Proactive burst detection and pre-emptive scaling.

    Maintains a sliding window of CPU measurements and fires a scale-out
    event when it detects an accelerating trend — defined as the slope of
    the linear regression over the last `window` steps exceeding a threshold.
    This mimics the "proactive burst scaling" approach described in:

      Lorido-Botran et al. (2014) "A Review of Auto-scaling Techniques for
      Elastic Applications in Cloud Environments", JGITM

    Specifically implements:
      1. Slope detection: OLS regression over last `window` steps
      2. Pre-emptive scale-out: if slope > slope_threshold, scale NOW
         (before CPU reaches the ceiling), provisioning 1 extra node
      3. Scale-in: conservative — only when both current AND trend CPU < low
      4. Burst guard: once a burst is detected, blocks scale-in for
         `burst_lockout_steps` steps to hold capacity warm

    This is a meaningful comparison because it isolates the value of RL
    (AutoCloud-Agent) vs a hand-crafted burst detection heuristic.
    """
    def __init__(
        self,
        window: int = 8,              # sliding window for trend estimation
        slope_threshold: float = 0.03, # CPU/step rate that triggers burst mode
        cpu_high: float = 0.75,
        cpu_low: float = 0.25,
        burst_lockout_steps: int = 8,  # hold capacity after burst for N steps
        cooldown_steps: int = 4,
        n_min: int = 3,
        n_max: int = 20,
    ):
        self.window              = window
        self.slope_threshold     = slope_threshold
        self.cpu_high            = cpu_high
        self.cpu_low             = cpu_low
        self.burst_lockout_steps = burst_lockout_steps
        self.cooldown_steps      = cooldown_steps
        self.n_min               = n_min
        self.n_max               = n_max
        self._history: list      = []
        self._last_scale_step    = -cooldown_steps
        self._last_burst_step    = -burst_lockout_steps

    def _cpu_trend_slope(self) -> float:
        """OLS slope (CPU per step) over the sliding window."""
        n = len(self._history)
        if n < 3:
            return 0.0
        x = np.arange(n, dtype=float)
        y = np.array(self._history, dtype=float)
        # Least-squares slope: β = cov(x,y) / var(x)
        xm, ym = x.mean(), y.mean()
        denom = ((x - xm) ** 2).sum()
        if denom < 1e-9:
            return 0.0
        return float(((x - xm) * (y - ym)).sum() / denom)

    def select_action(self, obs: np.ndarray, env: CloudEnv) -> Dict:
        from environment.node import NodeState
        metrics = env.get_sim_metrics()
        cpu     = metrics["mean_cpu_util"]
        step    = env._step_count

        self._history.append(cpu)
        if len(self._history) > self.window:
            self._history.pop(0)

        slope    = self._cpu_trend_slope()
        burst    = slope > self.slope_threshold or cpu > self.cpu_high
        in_guard = (step - self._last_burst_step) < self.burst_lockout_steps

        live_nodes = [n for n in env.sim.nodes if n.state != NodeState.TERMINATED]
        n_active   = max(sum(1 for n in live_nodes if n.state == NodeState.ACTIVE), 1)

        scaleout      = 0
        consolidation = np.zeros(N_MAX, dtype=np.float32)
        cooldown_ok   = (step - self._last_scale_step) >= self.cooldown_steps

        if burst and cooldown_ok and n_active < self.n_max:
            # Pre-emptive scale-out: provision 1 node ahead of the peak
            scaleout = 1
            self._last_scale_step = step
            self._last_burst_step = step

        elif not burst and not in_guard and cpu < self.cpu_low and cooldown_ok:
            # Conservative scale-in: only when both current AND trend are low
            slot = ThresholdReactive._lowest_util_slot(obs, env)
            if slot is not None:
                consolidation[slot] = 1.0
                self._last_scale_step = step

        return {"scaleout": scaleout, "consolidation": consolidation, "scheduling": 2}

    def reset(self):
        self._history         = []
        self._last_scale_step = -self.cooldown_steps
        self._last_burst_step = -self.burst_lockout_steps
