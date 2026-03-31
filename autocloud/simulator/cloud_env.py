"""
CloudEnv: Gymnasium-compatible environment wrapping the SimPy cloud simulator.

Observation space: Box(0, 1, shape=(215,))
  [0:120]   — per-node features (6 × 20, zero-padded)
  [120:200] — per-job features  (4 × 20, zero-padded)
  [200:215] — global features   (15)

Action space: Dict(
  scaleout      = Discrete(3)          # {0, +1, +2}
  consolidation = MultiBinary(20)      # per-node drain flag
  scheduling    = MultiBinary(20)      # per-job priority bump (0..1 simplified)
)

Per-agent rewards are returned in info['rewards'] to keep the main
return Gymnasium-compliant (scalar reward = sum of all three).
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple

from .engine import CloudSimulator
from .workload import SyntheticWorkload
from .node import NodeState, NODE_TYPES
from autocloud.config.settings import Config, DEFAULT_CONFIG


# Observation dimension constants
N_MAX = 20
K_JOBS = 20
NODE_FEAT = 6       # per-node: [cpu_util, mem_util, age_norm, is_booting, is_active, is_draining]
JOB_FEAT = 4        # per-job:  [size_norm, wait_norm, priority_norm, deadline_urgency]
GLOBAL_FEAT = 15
OBS_DIM = N_MAX * NODE_FEAT + K_JOBS * JOB_FEAT + GLOBAL_FEAT   # = 120 + 80 + 15 = 215

assert OBS_DIM == 215, f"OBS_DIM mismatch: {OBS_DIM}"

# Global feature layout (15 dims):
# 0  - active node count (norm)
# 1  - mean CPU util
# 2  - mean mem util
# 3  - queue length (norm)
# 4  - migration count (norm)
# 5  - forecast mean t+1
# 6  - forecast mean t+5
# 7  - forecast mean t+10
# 8  - forecast mean t+15
# 9  - forecast sigma t+1
# 10 - forecast sigma t+5
# 11 - forecast sigma t+10
# 12 - forecast sigma t+15
# 13 - booting node count (norm)
# 14 - draining node count (norm)


class CloudEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: Config = DEFAULT_CONFIG,
        seed: Optional[int] = None,
        workload_fn=None,
    ):
        super().__init__()
        self.config = config
        self.sim_cfg = config.sim
        self.rew_cfg = config.reward
        self._seed = seed

        # Observation and action spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Dict({
            "scaleout":      spaces.Discrete(3),                # 0, 1, 2
            "consolidation": spaces.MultiBinary(N_MAX),         # per-node drain
            "scheduling":    spaces.Discrete(5),                # priority bucket (applied to top-k jobs)
        })

        # Simulator
        self._rng = np.random.default_rng(seed)
        if workload_fn is None:
            self._workload = SyntheticWorkload(np.random.default_rng(seed if seed is not None else 42))
            workload_fn = self._workload
        self.sim = CloudSimulator(
            rng=np.random.default_rng(seed if seed is not None else 0),
            n_init=self.sim_cfg.n_init,
            n_max=self.sim_cfg.n_max,
            n_min=self.sim_cfg.n_min,
            step_duration=self.sim_cfg.step_duration,
            grace_period=self.sim_cfg.grace_period,
            warmup_period=self.sim_cfg.warmup_period,
            lognormal_mu=self.sim_cfg.lognormal_mu,
            lognormal_sigma=self.sim_cfg.lognormal_sigma,
            k_jobs=self.sim_cfg.k_jobs,
            workload_fn=workload_fn,
        )

        # Episode tracking
        self._step_count: int = 0
        self._prev_n_active: int = self.sim_cfg.n_init
        self._last_scaleout_step: int = -100
        self._prev_cpu: float = 0.0   # for cpu_rising detection

        # Forecaster output (injected externally each step)
        self.forecast_means: np.ndarray = np.zeros(4, dtype=np.float32)
        self.forecast_sigmas: np.ndarray = np.zeros(4, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        self.sim.reset(seed=self._seed)
        self._step_count = 0
        self._prev_n_active = self.sim_cfg.n_init
        self._last_scaleout_step = -100
        self._prev_cpu = 0.0
        self.forecast_means = np.zeros(4, dtype=np.float32)
        self.forecast_sigmas = np.zeros(4, dtype=np.float32)

        # Let simulator warm up by advancing past initial boot times
        self.sim.env.run(until=max(n.node_type.boot_time for n in self.sim.nodes) + 1.0)

        obs = self._build_observation()
        return obs, {}

    def step(self, action: Dict) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        scaleout_action = int(action["scaleout"])
        consolidation_action = np.asarray(action["consolidation"], dtype=np.float32)
        # scheduling may be a scalar int (baselines) or a per-job array (SchedulingAgent)
        scheduling_action = action["scheduling"]

        # 1. Apply scheduling priority to top-k queued jobs
        self._apply_scheduling(scheduling_action)

        # 2. Apply scale-out
        nodes_provisioned = 0
        if scaleout_action > 0:
            predicted_demand = float(self.forecast_means[1]) if self.forecast_means[1] > 0 else 0.5
            node_type = self.sim.node_type_for_demand(predicted_demand)
            for _ in range(scaleout_action):
                n = self.sim.provision_node(node_type)
                if n is not None:
                    nodes_provisioned += 1
            if nodes_provisioned > 0:
                self._last_scaleout_step = self._step_count

        # 3. Apply consolidation (via coordinator)
        #    Coordinator is applied externally in ippo_trainer; here we apply directly
        #    if called standalone (e.g., for baselines or testing).
        drainable_ids = self.sim.get_drainable_node_ids(N_MAX)
        n_active_before = len([n for n in self.sim.nodes if n.state == NodeState.ACTIVE])

        drain_count = 0
        for slot_idx, drain_flag in enumerate(consolidation_action[:len(drainable_ids)]):
            if drain_flag > 0.5:
                node_id = drainable_ids[slot_idx]
                # Safety: don't drain below N_min
                if (n_active_before - drain_count) > self.sim_cfg.n_min:
                    drained = self.sim.drain_node(node_id)
                    if drained:
                        drain_count += 1

        # 4. Advance simulation
        target_time = self.sim.env.now + self.sim_cfg.step_duration
        self.sim.step(target_time)
        self._step_count += 1

        # 5. Collect metrics
        metrics = self.sim.get_metrics()

        # 6. Compute per-agent rewards (pass uncertainty + rising-CPU signal)
        sigma_t5   = float(self.forecast_sigmas[1])
        cpu_rising = metrics["mean_cpu_util"] > self._prev_cpu
        rewards    = self._compute_rewards(metrics, scaleout_action, sigma_t5, cpu_rising)
        self._prev_cpu = metrics["mean_cpu_util"]

        # 7. Build observation
        obs = self._build_observation()

        # 8. Episode termination
        terminated = False
        truncated = self._step_count >= self.sim_cfg.episode_steps

        self._prev_n_active = metrics["n_active"]

        # Combined scalar reward (Gymnasium-compliant main return)
        total_reward = float(sum(rewards.values()))

        info = {
            "rewards": rewards,
            "metrics": metrics,
            "step": self._step_count,
            "forecast_means": self.forecast_means.tolist(),
            "forecast_sigmas": self.forecast_sigmas.tolist(),
        }

        return obs, total_reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Observation builder
    # ------------------------------------------------------------------ #

    def _build_observation(self) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        current_time = self.sim.env.now

        # [0:120] Per-node features (N_MAX × NODE_FEAT)
        node_feats = self.sim.get_node_features(current_time, N_MAX)  # (20, 6)
        obs[0:120] = node_feats.flatten()

        # [120:200] Per-job features (K_JOBS × JOB_FEAT)
        job_feats = self.sim.get_job_features(K_JOBS, current_time)   # (20, 4)
        obs[120:200] = job_feats.flatten()

        # [200:215] Global features (15)
        metrics = self.sim.get_metrics()
        g = np.zeros(GLOBAL_FEAT, dtype=np.float32)
        g[0]  = metrics["n_active"] / N_MAX
        g[1]  = metrics["mean_cpu_util"]
        g[2]  = metrics["mean_mem_util"]
        g[3]  = min(metrics["queue_len"] / 50.0, 1.0)
        g[4]  = min(metrics["step_migrations"] / 10.0, 1.0)
        g[5]  = float(np.clip(self.forecast_means[0], 0, 1))
        g[6]  = float(np.clip(self.forecast_means[1], 0, 1))
        g[7]  = float(np.clip(self.forecast_means[2], 0, 1))
        g[8]  = float(np.clip(self.forecast_means[3], 0, 1))
        g[9]  = float(np.clip(self.forecast_sigmas[0], 0, 1))
        g[10] = float(np.clip(self.forecast_sigmas[1], 0, 1))
        g[11] = float(np.clip(self.forecast_sigmas[2], 0, 1))
        g[12] = float(np.clip(self.forecast_sigmas[3], 0, 1))
        g[13] = metrics["n_booting"] / N_MAX
        g[14] = metrics["n_draining"] / N_MAX
        obs[200:215] = g

        obs = np.clip(obs, 0.0, 1.0)
        return obs

    # ------------------------------------------------------------------ #
    # Reward computation
    # ------------------------------------------------------------------ #

    def _compute_rewards(
        self,
        metrics: dict,
        scaleout_action: int,
        sigma_t5: float = 0.0,
        cpu_rising: bool = False,
    ) -> Dict[str, float]:
        r = self.rew_cfg
        cpu = metrics["mean_cpu_util"]
        n_active = metrics["n_active"]
        cost = metrics["step_cost"]
        migrations = metrics["step_migrations"]
        mem = metrics["mean_mem_util"]
        mean_wait = metrics["mean_wait"]
        p95 = metrics["p95_latency"]
        sla_ms = r.sla_latency_ms

        # SLA met: p95 latency < threshold
        sla_met = float(p95 < sla_ms) if p95 > 0 else 1.0

        # Scale-Out reward
        # alpha5 bonus: when forecaster is uncertain AND load is rising, reward
        # proactive scaling so the agent learns to act early under surprise load.
        uncertainty_bonus = r.alpha5 * sigma_t5 * float(cpu_rising)
        r_scaleout = (
            - r.alpha1 * max(0.0, cpu - r.cpu_high)
            - r.alpha2 * max(0.0, r.cpu_low - cpu)
            - r.alpha3 * (n_active / N_MAX)
            - r.alpha4 * float(scaleout_action > 0)
            + uncertainty_bonus
        )

        # Consolidation reward
        r_consolidation = (
            - r.beta1 * cost
            + r.beta2 * sla_met
            - r.beta3 * migrations
            - r.beta4 * max(0.0, mem - r.mem_high)
        )

        # Scheduling reward (wait time in seconds, normalize by 300s)
        wait_norm = mean_wait / 300.0
        p95_norm = p95 / sla_ms if sla_ms > 0 else 0.0
        r_scheduling = (
            - r.gamma1 * wait_norm
            - r.gamma2 * max(0.0, p95_norm - 1.0)
        )

        return {
            "scaleout": float(r_scaleout),
            "consolidation": float(r_consolidation),
            "scheduling": float(r_scheduling),
        }

    # ------------------------------------------------------------------ #
    # Scheduling action application
    # ------------------------------------------------------------------ #

    def _apply_scheduling(self, scheduling_action) -> None:
        """Apply scheduling priority to queued jobs.

        scheduling_action may be:
          - int / scalar: same bucket applied to all top-k jobs (baselines)
          - (K_JOBS,) array: per-slot priority bucket (SchedulingAgent)
        """
        top_k_ids = self.sim.get_top_k_job_ids(K_JOBS)
        if isinstance(scheduling_action, (int, np.integer)) or (
            isinstance(scheduling_action, np.ndarray) and scheduling_action.ndim == 0
        ):
            bucket = int(scheduling_action)
            overrides = {job_id: bucket for job_id in top_k_ids}
        else:
            # Per-job array — slot i corresponds to top_k_ids[i]
            arr = np.asarray(scheduling_action, dtype=np.int32)
            overrides = {
                job_id: int(arr[i])
                for i, job_id in enumerate(top_k_ids)
                if i < len(arr)
            }
        self.sim.apply_scheduling_action(overrides)

    # ------------------------------------------------------------------ #
    # Helpers for coordinator / trainer
    # ------------------------------------------------------------------ #

    def get_node_slot_ids(self) -> list:
        """Return list of node_ids in slot order (same as obs[0:120])."""
        return self.sim.get_drainable_node_ids(N_MAX)

    def get_active_mask(self) -> np.ndarray:
        return self.sim.get_active_node_mask(N_MAX)

    def inject_forecast(
        self, means: np.ndarray, sigmas: np.ndarray
    ) -> None:
        """Called by trainer each step to update forecast features in obs."""
        self.forecast_means = np.array(means, dtype=np.float32)
        self.forecast_sigmas = np.array(sigmas, dtype=np.float32)

    def get_sim_metrics(self) -> dict:
        return self.sim.get_metrics()

    def get_sim_time(self) -> float:
        return self.sim.env.now

    def render(self) -> None:
        m = self.sim.get_metrics()
        print(
            f"Step {self._step_count:4d} | "
            f"Active: {m['n_active']:2d} Boot: {m['n_booting']} Drain: {m['n_draining']} | "
            f"CPU: {m['mean_cpu_util']:.2f} Mem: {m['mean_mem_util']:.2f} | "
            f"Queue: {m['queue_len']:3d} | "
            f"P95: {m['p95_latency']:.1f}ms | "
            f"Cost: ${m['total_cost']:.4f}"
        )

    def should_interrupt_scaleout(self) -> bool:
        """Returns True if Scale-Out agent should act immediately (emergency)."""
        m = self.sim.get_metrics()
        queue_norm = min(m["queue_len"] / 50.0, 1.0)
        return m["mean_cpu_util"] > 0.9 or queue_norm > 0.8
