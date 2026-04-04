"""
Inference runner — clean evaluation loop with temporal hierarchy + safety coordinator.

Loads trained agents from checkpoints and provides a unified select_action()
interface that mirrors the training loop's temporal hierarchy:

  ScaleOut:      acts every 10 steps OR on interrupt (CPU > 90% / queue > 80%)
  Consolidation: acts every 2 steps
  Scheduling:    acts every step
  All actions pass through SafetyCoordinator filters

Usage:
    from autocloud.inference import InferenceRunner

    runner = InferenceRunner("checkpoints/")
    env    = CloudEnv()
    obs, _ = env.reset()
    runner.reset()
    done   = False
    while not done:
        action = runner.select_action(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Optional

from autocloud.agents.loader import load_agents
from autocloud.coordinator.safety import SafetyCoordinator
from autocloud.simulator.cloud_env import CloudEnv
from autocloud.config.settings import Config, DEFAULT_CONFIG


class InferenceRunner:
    """
    Drop-in inference-only replacement for IPPOTrainer.

    Unlike IPPOTrainer, carries no training state (no buffers, no EMA
    normalizers, no training env).  Ideal for evaluation and production.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        config: Config = DEFAULT_CONFIG,
        device: str = "cpu",
        tag: str = "final",
        forecaster=None,
    ):
        self.config = config
        self.seq_len = config.forecast.seq_len  # 20
        self.so, self.con, self.sch = load_agents(
            checkpoint_dir, tag=tag, device=device, config=config,
        )
        self.coordinator = SafetyCoordinator(
            n_min=config.sim.n_min,
            warmup_period=config.sim.warmup_period,
            sigma_high=config.coordinator.sigma_high,
        )
        self.forecaster = forecaster
        self._eval_step = 0
        # Rolling workload history for forecaster: (seq_len, 4)
        self._history: list = []
        # Diagnostics for the last select_action call (used by demo)
        self.last_diag: dict = {}

    def reset(self) -> None:
        """Reset per-episode step counter and forecast history."""
        self._eval_step = 0
        self._history = []
        self.last_diag = {}

    def _extract_forecast_features(self, obs: np.ndarray, env: CloudEnv) -> np.ndarray:
        """Extract the 4 workload features the Transformer forecaster expects."""
        cpu_util = float(obs[201])               # mean_cpu_util (global[1])
        queue_norm = float(obs[203])             # queue_len_norm (global[3])
        sim_time = env.get_sim_time()
        hour_of_day = (sim_time % 86400) / 86400.0
        return np.array([cpu_util, cpu_util, queue_norm, hour_of_day], dtype=np.float32)

    def _get_forecast_window(self, obs: np.ndarray, env: CloudEnv) -> np.ndarray:
        """Build the (seq_len, 4) window for the forecaster, zero-padding if short."""
        feats = self._extract_forecast_features(obs, env)
        self._history.append(feats)
        if len(self._history) > self.seq_len:
            self._history = self._history[-self.seq_len:]
        window = np.zeros((self.seq_len, 4), dtype=np.float32)
        n = len(self._history)
        window[-n:] = np.array(self._history[-n:])
        return window

    def select_action(self, obs: np.ndarray, env: CloudEnv) -> Dict:
        """
        Full inference step: forecast → temporal hierarchy → coordinator → action dict.
        """
        diag: dict = {"step": self._eval_step}

        # Inject forecast if forecaster is attached
        if self.forecaster is not None:
            window = self._get_forecast_window(obs, env)
            means, sigmas = self.forecaster.predict(window)
            env.inject_forecast(means, sigmas)
            diag["forecast_means"] = means.tolist() if hasattr(means, "tolist") else list(means)
            diag["forecast_sigmas"] = sigmas.tolist() if hasattr(sigmas, "tolist") else list(sigmas)
        else:
            diag["forecast_means"] = env.forecast_means.tolist()
            diag["forecast_sigmas"] = env.forecast_sigmas.tolist()

        ep_step = self._eval_step

        # ScaleOut: every 10 steps or on emergency interrupt
        so_acted = ep_step % 10 == 0 or env.should_interrupt_scaleout()
        diag["so_acted"] = so_acted
        diag["so_reason"] = "periodic" if ep_step % 10 == 0 else ("interrupt" if so_acted else "skip")
        if so_acted:
            a_so, _, _ = self.so.act(obs)
        else:
            a_so = 0
        diag["so_raw"] = int(np.asarray(a_so).flat[0])

        # Consolidation: every 2 steps
        con_acted = ep_step % 2 == 0
        diag["con_acted"] = con_acted
        if con_acted:
            a_con, _, _ = self.con.act(obs, env.get_active_mask())
        else:
            a_con = np.zeros(20, dtype=np.float32)
        diag["con_raw_drains"] = int(np.sum(np.array(a_con) > 0.5))

        # Scheduling: every step
        a_sch, _, _ = self.sch.act(obs)
        _a_sch_int = int(np.asarray(a_sch).flat[0])
        diag["sch_raw"] = _a_sch_int
        sch_names = ["Best-Fit", "First-Fit", "Round-Robin", "Least-Loaded", "Most-Loaded"]
        diag["sch_name"] = sch_names[_a_sch_int] if _a_sch_int < len(sch_names) else f"rule-{_a_sch_int}"

        # Safety coordinator
        cpu = env.sim.get_metrics().get("mean_cpu_util", 0.0)
        queue_len = env.sim.get_metrics().get("queue_len", 0)
        a_so_f, a_con_f, a_sch_f = self.coordinator.resolve(
            a_scaleout=a_so,
            consolidation_vec=a_con,
            a_scheduling=a_sch,
            nodes=env.sim.nodes,
            current_time=env.get_sim_time(),
            sigma_t5=float(env.forecast_sigmas[1]),
            cpu_rising=cpu > env._prev_cpu,
            cpu_delta=cpu - env._prev_cpu,
            queue_len=queue_len,
            mean_cpu=cpu,
        )

        def _int(x):
            """Safely convert any scalar (tensor, ndarray, int) to Python int."""
            return int(np.asarray(x).flat[0])

        # Record safety overrides
        diag["so_filtered"] = _int(a_so_f)
        diag["so_overridden"] = _int(a_so_f) != _int(a_so)
        diag["con_filtered_drains"] = int(np.sum(np.array(a_con_f) > 0.5))
        diag["con_overridden"] = diag["con_filtered_drains"] != diag["con_raw_drains"]
        diag["sch_filtered"] = _int(a_sch_f)

        self._eval_step += 1
        self.last_diag = diag

        return {
            "scaleout": a_so_f,
            "consolidation": a_con_f,
            "scheduling": a_sch_f,
        }
