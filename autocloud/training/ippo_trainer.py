"""
Joint I-PPO training loop with temporal hierarchy.

Temporal hierarchy:
  Scale-Out:    acts every 10 steps OR on interrupt (CPU > 90% / queue > 80%)
  Consolidation: acts every 2 steps
  Scheduling:   acts every step

Critical rule: only store experience for an agent on steps when it actually acted.
No-op placeholder steps (a=0 for scaleout when it doesn't fire) are NOT stored —
storing them would train the agent to believe the no-op is always the policy choice.

Buffer flush: each agent updates independently when its buffer hits buffer_size.
Agents fill at very different rates: scheduling fills ~10× faster than scaleout.
"""
from __future__ import annotations

import os
import numpy as np
import torch
from typing import Optional, Dict, List
from dataclasses import dataclass, field

from autocloud.simulator.cloud_env import CloudEnv
from autocloud.agents.scaleout import ScaleOutAgent
from autocloud.agents.consolidation import ConsolidationAgent
from autocloud.agents.scheduling import SchedulingAgent
from autocloud.coordinator.safety import SafetyCoordinator
from autocloud.training.ema_normalizer import AgentRewardTracker
from autocloud.config.settings import Config, DEFAULT_CONFIG


@dataclass
class TrainingMetrics:
    """Accumulated metrics for logging."""
    step: int = 0
    episode: int = 0
    so_returns:  List[float] = field(default_factory=list)
    con_returns: List[float] = field(default_factory=list)
    sch_returns: List[float] = field(default_factory=list)
    so_losses:   List[float] = field(default_factory=list)
    con_losses:  List[float] = field(default_factory=list)
    sch_losses:  List[float] = field(default_factory=list)
    sla_rates:   List[float] = field(default_factory=list)
    costs:       List[float] = field(default_factory=list)


class IPPOTrainer:
    def __init__(
        self,
        config: Config = DEFAULT_CONFIG,
        seed: int = 0,
        device: str = "cpu",
        forecaster=None,        # MCDropoutForecaster instance (optional)
        workload_fn=None,       # callable(sim_time) -> rate multiplier
        verbose: bool = True,
        log_interval: int = 1000,
    ):
        self.config   = config
        self.seed     = seed
        self.device   = device
        self.verbose  = verbose
        self.log_interval = log_interval

        ppo_kwargs = dict(
            lr=config.ppo.lr,
            gamma=config.ppo.gamma,
            gae_lambda=config.ppo.gae_lambda,
            clip_eps=config.ppo.clip_eps,
            entropy_coef=config.ppo.entropy_coef,
            vf_coef=config.ppo.vf_coef,
            max_grad_norm=config.ppo.max_grad_norm,
            minibatch_size=config.ppo.minibatch_size,
            update_epochs=config.ppo.update_epochs,
            buffer_size=config.ppo.buffer_size,
            device=device,
        )

        self.so_agent  = ScaleOutAgent(**ppo_kwargs)
        self.con_agent = ConsolidationAgent(**ppo_kwargs)
        self.sch_agent = SchedulingAgent(**ppo_kwargs)

        self.coordinator = SafetyCoordinator(
            n_min=config.sim.n_min,
            warmup_period=config.sim.warmup_period,
            sigma_high=config.coordinator.sigma_high,
        )

        self.so_tracker  = AgentRewardTracker(config.ppo.ema_window)
        self.con_tracker = AgentRewardTracker(config.ppo.ema_window)
        self.sch_tracker = AgentRewardTracker(config.ppo.ema_window)

        self.env = CloudEnv(config=config, seed=seed, workload_fn=workload_fn)
        self.forecaster = forecaster
        self.metrics = TrainingMetrics()
        self.rng = np.random.default_rng(seed)

        # Entropy annealing
        self._entropy_coef = config.ppo.entropy_coef
        self._entropy_min  = config.ppo.entropy_coef_min
        self._entropy_decay_steps = config.ppo.entropy_decay_steps

    # ------------------------------------------------------------------ #
    # Main training loop
    # ------------------------------------------------------------------ #

    def train(self, total_steps: int, checkpoint_dir: Optional[str] = None,
              seed_randomize: bool = True) -> TrainingMetrics:
        obs, _ = self.env.reset(seed=self.seed)

        # Episode state
        ep_so_return  = 0.0
        ep_con_return = 0.0
        ep_sch_return = 0.0
        ep_sla_steps  = 0
        ep_total_steps = 0

        # Last observations for each agent (used for buffer flush bootstrap)
        last_so_obs  = obs.copy()
        last_con_obs = obs.copy()
        last_sch_obs = obs.copy()

        global_step = 0

        while global_step < total_steps:
            # ── Entropy annealing ──────────────────────────────────────
            frac = min(global_step / max(self._entropy_decay_steps, 1), 1.0)
            cur_entropy = self._entropy_coef * (1 - frac) + self._entropy_min * frac
            self.so_agent.entropy_coef  = cur_entropy
            self.con_agent.entropy_coef = cur_entropy
            self.sch_agent.entropy_coef = cur_entropy

            # ── Inject forecast into env ───────────────────────────────
            if self.forecaster is not None:
                means, sigmas = self.forecaster.predict(obs)
                self.env.inject_forecast(means, sigmas)
                sigma_t5 = float(sigmas[1])
            else:
                means  = np.zeros(4, dtype=np.float32)
                sigmas = np.zeros(4, dtype=np.float32)
                sigma_t5 = 0.0

            ep_step = self.metrics.step % self.config.sim.episode_steps

            # ── Scale-Out: acts every 10 steps OR on interrupt ─────────
            so_acts_this_step = (
                ep_step % 10 == 0
                or self.env.should_interrupt_scaleout()
            )
            if so_acts_this_step:
                a_so, lp_so, v_so = self.so_agent.act(obs)
                last_so_obs = obs.copy()
            else:
                a_so, lp_so, v_so = 0, 0.0, 0.0

            # ── Consolidation: acts every 2 steps ─────────────────────
            con_acts_this_step = (ep_step % 2 == 0)
            if con_acts_this_step:
                active_mask = self.env.get_active_mask()
                a_con, lp_con, v_con = self.con_agent.act(obs, active_mask)
                last_con_obs = obs.copy()
            else:
                a_con  = np.zeros(20, dtype=np.float32)
                lp_con, v_con = 0.0, 0.0

            # ── Scheduling: acts every step ────────────────────────────
            a_sch, lp_sch, v_sch = self.sch_agent.act(obs)
            last_sch_obs = obs.copy()

            # ── Coordinator: apply safety filters ─────────────────────
            curr_cpu = self.env.sim.get_metrics().get("mean_cpu_util", 0.0)
            a_so_final, a_con_final, a_sch_final = self.coordinator.resolve(
                a_scaleout=a_so,
                consolidation_vec=a_con,
                a_scheduling=a_sch,
                nodes=self.env.sim.nodes,
                current_time=self.env.get_sim_time(),
                sigma_t5=sigma_t5,
                cpu_rising=curr_cpu > self.env._prev_cpu,
                cpu_delta=curr_cpu - self.env._prev_cpu,
            )

            # ── Env step ───────────────────────────────────────────────
            action = {
                "scaleout":      a_so_final,
                "consolidation": a_con_final,
                "scheduling":    a_sch_final,
            }
            next_obs, _, terminated, truncated, info = self.env.step(action)
            rewards = info["rewards"]
            done = terminated or truncated

            # ── Store experience (only on acting steps) ────────────────
            if so_acts_this_step:
                r_so_norm = self.so_tracker.normalize(rewards["scaleout"])
                self.so_agent.store(obs, a_so, lp_so, r_so_norm, v_so, done)
            else:
                # Still update EMA stats even on no-op steps
                self.so_tracker.ema.update(rewards["scaleout"])

            if con_acts_this_step:
                r_con_norm = self.con_tracker.normalize(rewards["consolidation"])
                active_mask = self.env.get_active_mask()
                self.con_agent.store(obs, a_con, lp_con, r_con_norm, v_con, done,
                                     mask=active_mask)
            else:
                self.con_tracker.ema.update(rewards["consolidation"])

            r_sch_norm = self.sch_tracker.normalize(rewards["scheduling"])
            job_mask   = self._get_job_mask(obs)
            self.sch_agent.store(obs, a_sch, lp_sch, r_sch_norm, v_sch, done,
                                  mask=job_mask)

            # ── Raw return tracking ────────────────────────────────────
            ep_so_return  += rewards["scaleout"]
            ep_con_return += rewards["consolidation"]
            ep_sch_return += rewards["scheduling"]

            m = info["metrics"]
            sla_met = float(m["p95_latency"] < self.config.reward.sla_latency_ms or m["p95_latency"] == 0)
            ep_sla_steps  += sla_met
            ep_total_steps += 1

            # ── Independent buffer flushes ─────────────────────────────
            if self.so_agent.should_update():
                so_metrics = self.so_agent.update(next_obs, done)
                if so_metrics:
                    self.metrics.so_losses.append(so_metrics.get("pg_loss", 0))

            if self.con_agent.should_update():
                con_metrics = self.con_agent.update(next_obs, done)
                if con_metrics:
                    self.metrics.con_losses.append(con_metrics.get("pg_loss", 0))

            if self.sch_agent.should_update():
                sch_metrics = self.sch_agent.update(next_obs, done)
                if sch_metrics:
                    self.metrics.sch_losses.append(sch_metrics.get("pg_loss", 0))

            obs = next_obs
            global_step += 1
            self.metrics.step += 1

            # ── Episode end ────────────────────────────────────────────
            if done:
                self.metrics.episode += 1
                self.metrics.so_returns.append(ep_so_return)
                self.metrics.con_returns.append(ep_con_return)
                self.metrics.sch_returns.append(ep_sch_return)
                sla_rate = ep_sla_steps / max(ep_total_steps, 1)
                self.metrics.sla_rates.append(sla_rate)
                self.metrics.costs.append(m["total_cost"])

                if self.verbose and self.metrics.episode % 5 == 0:
                    self._log_episode(sla_rate, ep_so_return, ep_con_return, ep_sch_return)

                # Reset episode accumulators
                ep_so_return = ep_con_return = ep_sch_return = 0.0
                ep_sla_steps = ep_total_steps = 0
                next_seed = int(self.rng.integers(0, 10000)) if seed_randomize else None
                obs, _ = self.env.reset(seed=next_seed)

            # ── Periodic logging ───────────────────────────────────────
            if self.verbose and global_step % self.log_interval == 0:
                self._log_step(global_step, total_steps)

            # ── Checkpoint ────────────────────────────────────────────
            if checkpoint_dir and global_step % 10_000 == 0:
                self.save(checkpoint_dir, tag=f"step_{global_step}")

        return self.metrics

    # ------------------------------------------------------------------ #
    # Save / load
    # ------------------------------------------------------------------ #

    def save(self, directory: str, tag: str = "final") -> None:
        os.makedirs(directory, exist_ok=True)
        torch.save(self.so_agent.actor.state_dict(),
                   os.path.join(directory, f"so_actor_{tag}.pt"))
        torch.save(self.so_agent.critic.state_dict(),
                   os.path.join(directory, f"so_critic_{tag}.pt"))
        torch.save(self.con_agent.actor.state_dict(),
                   os.path.join(directory, f"con_actor_{tag}.pt"))
        torch.save(self.con_agent.critic.state_dict(),
                   os.path.join(directory, f"con_critic_{tag}.pt"))
        torch.save(self.sch_agent.actor.state_dict(),
                   os.path.join(directory, f"sch_actor_{tag}.pt"))
        torch.save(self.sch_agent.critic.state_dict(),
                   os.path.join(directory, f"sch_critic_{tag}.pt"))
        if self.verbose:
            print(f"Checkpoint saved: {directory}/{tag}")

    def load(self, directory: str, tag: str = "final") -> None:
        def _load(module, fname):
            path = os.path.join(directory, fname)
            if os.path.exists(path):
                module.load_state_dict(torch.load(path, map_location=self.device))
        _load(self.so_agent.actor,   f"so_actor_{tag}.pt")
        _load(self.so_agent.critic,  f"so_critic_{tag}.pt")
        _load(self.con_agent.actor,  f"con_actor_{tag}.pt")
        _load(self.con_agent.critic, f"con_critic_{tag}.pt")
        _load(self.sch_agent.actor,  f"sch_actor_{tag}.pt")
        _load(self.sch_agent.critic, f"sch_critic_{tag}.pt")

    # ------------------------------------------------------------------ #
    # Inference interface (mirrors training temporal hierarchy + coordinator)
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset per-episode step counter for inference."""
        self._eval_step = 0

    def select_action(self, obs: np.ndarray, env: "CloudEnv") -> Dict:
        """
        Inference-time action selection that mirrors the training loop:
          - so_agent  acts every 10 steps
          - con_agent acts every 2 steps
          - sch_agent acts every step
          - Actions go through SafetyCoordinator
        """
        if not hasattr(self, '_eval_step'):
            self._eval_step = 0

        ep_step = self._eval_step

        if ep_step % 10 == 0 or env.should_interrupt_scaleout():
            a_so, _, _ = self.so_agent.act(obs)
        else:
            a_so = 0

        if ep_step % 2 == 0:
            active_mask = env.get_active_mask()
            a_con, _, _ = self.con_agent.act(obs, active_mask)
        else:
            a_con = np.zeros(20, dtype=np.float32)

        a_sch, _, _ = self.sch_agent.act(obs)

        current_cpu = env.sim.get_metrics().get("mean_cpu_util", 0.0)
        a_so_final, a_con_final, a_sch_final = self.coordinator.resolve(
            a_scaleout=a_so,
            consolidation_vec=a_con,
            a_scheduling=a_sch,
            nodes=env.sim.nodes,
            current_time=env.get_sim_time(),
            sigma_t5=float(env.forecast_sigmas[1]),
            cpu_rising=current_cpu > env._prev_cpu,
            cpu_delta=current_cpu - env._prev_cpu,
        )

        self._eval_step += 1

        return {
            "scaleout":      a_so_final,
            "consolidation": a_con_final,
            "scheduling":    a_sch_final,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_job_mask(obs: np.ndarray) -> np.ndarray:
        """(20,) mask: 1 if job slot has any nonzero feature."""
        job_feats = obs[120:200].reshape(20, 4)
        return (job_feats.sum(axis=-1) > 0).astype(np.float32)

    def _log_episode(self, sla_rate, r_so, r_con, r_sch):
        print(
            f"Ep {self.metrics.episode:4d} | "
            f"SLA={sla_rate:.2%} | "
            f"r_so={r_so:7.2f} r_con={r_con:7.2f} r_sch={r_sch:7.2f}"
        )

    def _log_step(self, step, total):
        n = len(self.metrics.so_losses)
        so_loss  = np.mean(self.metrics.so_losses[-10:])  if n > 0 else 0
        sch_loss = np.mean(self.metrics.sch_losses[-10:]) if self.metrics.sch_losses else 0
        print(
            f"Step {step:6d}/{total} | "
            f"Ep={self.metrics.episode} | "
            f"so_loss={so_loss:.4f} sch_loss={sch_loss:.4f}"
        )
