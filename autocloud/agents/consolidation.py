"""
Consolidation Agent — drains only (per-node binary MultiBinary(20)).

Actor:  MLP [215 → 256 → 128 → 20]  independent Bernoulli logits per node
Critic: MLP [215 → 256 → 128 → 1]

Padding mask: log_prob is summed only over active node slots.
Inactive slots (zero-padded obs) are masked to "keep" (drain=0) regardless
of what the actor outputs — gradient flows only through active slots.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from typing import Tuple, Optional

from .ppo import PPO, build_mlp, orthogonal_init

OBS_DIM = 215
N_MAX   = 20


class ConsolidationActor(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, n_max: int = N_MAX):
        super().__init__()
        self.net = build_mlp(obs_dim, [512, 256, 128], n_max, layernorm=True)
        orthogonal_init(self.net, gain=np.sqrt(2))
        orthogonal_init(self.net[-1:], gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)   # logits (batch, n_max)


class ConsolidationCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM):
        super().__init__()
        self.net = build_mlp(obs_dim, [512, 256, 128], 1, layernorm=True)
        orthogonal_init(self.net, gain=np.sqrt(2))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ConsolidationAgent(PPO):
    def __init__(self, obs_dim: int = OBS_DIM, n_max: int = N_MAX,
                 device: str = "cpu", **ppo_kwargs):
        actor  = ConsolidationActor(obs_dim, n_max)
        critic = ConsolidationCritic(obs_dim)
        super().__init__(actor=actor, critic=critic, device=device, **ppo_kwargs)
        self.n_max = n_max

    @torch.no_grad()
    def act(self, obs: np.ndarray, active_mask: Optional[np.ndarray] = None
            ) -> Tuple[np.ndarray, float, float]:
        """
        Args:
            obs:         (215,) observation vector
            active_mask: (n_max,) binary mask — 1 for live node slots

        Returns:
            action   — (n_max,) binary array; inactive slots forced to 0
            log_prob — float (sum over active slots only)
            value    — float
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        logits = self.actor(obs_t)                      # (1, n_max)
        dist   = Bernoulli(logits=logits)
        action = dist.sample()                           # (1, n_max)

        # Zero out inactive slots
        if active_mask is not None:
            mask_t = torch.FloatTensor(active_mask).unsqueeze(0).to(self.device)
            action = action * mask_t

        # log_prob: sum over active slots
        log_prob_per_node = dist.log_prob(action)        # (1, n_max)
        if active_mask is not None:
            log_prob = (log_prob_per_node * mask_t).sum().item()
        else:
            log_prob = log_prob_per_node.sum().item()

        value = self.critic(obs_t).squeeze().item()
        return action.squeeze(0).cpu().numpy(), log_prob, value

    # PPO hooks

    def _get_dist(self, obs: torch.Tensor) -> Bernoulli:
        return Bernoulli(logits=self.actor(obs))

    def _log_prob(self, dist: Bernoulli, action: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sum log_prob over active node slots."""
        lp = dist.log_prob(action)    # (batch, n_max)
        if mask is not None:
            lp = lp * mask
        return lp.sum(dim=-1)         # (batch,)

    def _entropy(self, dist: Bernoulli,
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Mean entropy over active slots."""
        ent = dist.entropy()          # (batch, n_max)
        if mask is not None:
            ent = ent * mask
            return ent.sum(dim=-1).mean()
        return ent.mean()
