"""
Reusable Independent PPO engine.

Takes actor and critic as constructor arguments — owns no network architectures.
All three I-PPO agents share this class with their own instances.

Key design:
- RolloutBuffer stores only steps where the agent actually acted (no no-op padding).
- GAE computed at update time from stored (obs, action, log_prob, reward, value, done).
- Separate EMA normalizer per agent (injected via reward_normalizer).
- Padding masks stored per-step for consolidation/scheduling agents.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Callable
from dataclasses import dataclass, field


# ------------------------------------------------------------------ #
# Rollout buffer
# ------------------------------------------------------------------ #

@dataclass
class Transition:
    obs: np.ndarray
    action: np.ndarray          # scalar or vector depending on agent
    log_prob: float
    reward: float               # EMA-normalized reward
    value: float
    done: bool
    mask: Optional[np.ndarray] = None   # padding mask for consolidation/scheduling


class RolloutBuffer:
    def __init__(self, buffer_size: int = 2048):
        self.buffer_size = buffer_size
        self.transitions: list[Transition] = []

    def store(
        self,
        obs: np.ndarray,
        action,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        self.transitions.append(Transition(
            obs=obs.copy(),
            action=np.atleast_1d(np.array(action, dtype=np.float32)),
            log_prob=float(log_prob),
            reward=float(reward),
            value=float(value),
            done=bool(done),
            mask=mask.copy() if mask is not None else None,
        ))

    def is_full(self) -> bool:
        return len(self.transitions) >= self.buffer_size

    def clear(self) -> None:
        self.transitions = []

    def __len__(self) -> int:
        return len(self.transitions)


# ------------------------------------------------------------------ #
# PPO engine
# ------------------------------------------------------------------ #

class PPO:
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        minibatch_size: int = 64,
        update_epochs: int = 4,
        buffer_size: int = 2048,
        device: str = "cpu",
    ):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.minibatch_size = minibatch_size
        self.update_epochs = update_epochs
        self.device = device

        # Shared optimizer for actor + critic
        self.optimizer = torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()),
            lr=lr,
        )

        self.buffer = RolloutBuffer(buffer_size)
        self.total_updates = 0

    def store(self, obs, action, log_prob, reward, value, done, mask=None):
        self.buffer.store(obs, action, log_prob, reward, value, done, mask)

    def should_update(self) -> bool:
        return self.buffer.is_full()

    def update(self, last_obs: np.ndarray, last_done: bool) -> dict:
        """
        Compute GAE advantages and run PPO update.
        Returns dict of training metrics.
        """
        transitions = self.buffer.transitions
        n = len(transitions)
        if n == 0:
            return {}

        # Compute last value for bootstrap
        with torch.no_grad():
            obs_t = torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
            last_value = self.critic(obs_t).squeeze().item()
        if last_done:
            last_value = 0.0

        # Compute GAE advantages
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else transitions[t + 1].value
            next_done = 1.0 if (t == n - 1 and last_done) else float(transitions[t + 1].done if t < n - 1 else last_done)
            delta = transitions[t].reward + self.gamma * next_value * (1.0 - next_done) - transitions[t].value
            gae = delta + self.gamma * self.gae_lambda * (1.0 - next_done) * gae
            advantages[t] = gae

        returns = advantages + np.array([t.value for t in transitions], dtype=np.float32)

        # Normalize advantages
        adv_mean, adv_std = advantages.mean(), advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # Stack tensors
        obs_arr    = np.array([t.obs for t in transitions], dtype=np.float32)
        act_arr    = np.array([t.action for t in transitions], dtype=np.float32)
        logp_arr   = np.array([t.log_prob for t in transitions], dtype=np.float32)
        masks_arr  = [t.mask for t in transitions]

        obs_t    = torch.FloatTensor(obs_arr).to(self.device)
        act_t    = torch.FloatTensor(act_arr).to(self.device)
        logp_old = torch.FloatTensor(logp_arr).to(self.device)
        adv_t    = torch.FloatTensor(advantages).to(self.device)
        ret_t    = torch.FloatTensor(returns).to(self.device)

        # Build mask tensor (or None)
        if masks_arr[0] is not None:
            masks_t = torch.FloatTensor(np.array(masks_arr, dtype=np.float32)).to(self.device)
        else:
            masks_t = None

        # PPO update for `update_epochs` epochs
        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_ent     = 0.0
        n_updates = 0

        indices = np.arange(n)
        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.minibatch_size):
                end = min(start + self.minibatch_size, n)
                idx = indices[start:end]

                mb_obs  = obs_t[idx]
                mb_act  = act_t[idx]
                mb_logp = logp_old[idx]
                mb_adv  = adv_t[idx]
                mb_ret  = ret_t[idx]
                mb_mask = masks_t[idx] if masks_t is not None else None

                # Forward pass
                new_dist = self._get_dist(mb_obs)
                new_logp = self._log_prob(new_dist, mb_act, mb_mask)
                entropy  = self._entropy(new_dist, mb_mask)
                value    = self.critic(mb_obs).squeeze(-1)

                # Ratio and clipped surrogate loss
                ratio = torch.exp(new_logp - mb_logp)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                vf_loss = 0.5 * ((value - mb_ret) ** 2).mean()

                # Total loss
                loss = pg_loss + self.vf_coef * vf_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_ent     += entropy.item()
                n_updates += 1

        self.buffer.clear()
        self.total_updates += 1

        return {
            "pg_loss":  total_pg_loss / max(n_updates, 1),
            "vf_loss":  total_vf_loss / max(n_updates, 1),
            "entropy":  total_ent     / max(n_updates, 1),
        }

    # ------------------------------------------------------------------ #
    # Subclass hooks — overridden by each agent
    # ------------------------------------------------------------------ #

    def _get_dist(self, obs: torch.Tensor):
        """Return action distribution from actor. Must be overridden."""
        raise NotImplementedError

    def _log_prob(self, dist, action: torch.Tensor, mask=None) -> torch.Tensor:
        """Return log prob of action under dist. Must be overridden."""
        raise NotImplementedError

    def _entropy(self, dist, mask=None) -> torch.Tensor:
        """Return mean entropy. Must be overridden."""
        raise NotImplementedError


# ------------------------------------------------------------------ #
# Shared MLP builder
# ------------------------------------------------------------------ #

def build_mlp(input_dim: int, hidden_dims: list, output_dim: int,
              activation=nn.ReLU, layernorm: bool = True) -> nn.Sequential:
    layers = []
    in_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        if layernorm:
            layers.append(nn.LayerNorm(h))
        layers.append(activation())
        in_dim = h
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


def orthogonal_init(module: nn.Module, gain: float = np.sqrt(2)) -> nn.Module:
    """Apply orthogonal initialization to all Linear layers."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            nn.init.constant_(m.bias, 0.0)
    return module
