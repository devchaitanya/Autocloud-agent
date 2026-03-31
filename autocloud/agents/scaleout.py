"""
Scale-Out Agent — provisions only (actions: {0, +1, +2}).

Actor:  MLP [215 → 256 → 128 → 3]  Categorical output
Critic: MLP [215 → 256 → 128 → 1]  Value estimate
"""
from __future__ import annotations


import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Tuple, Optional

from .ppo import PPO, build_mlp, orthogonal_init

OBS_DIM = 215


class ScaleOutActor(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM):
        super().__init__()
        self.net = build_mlp(obs_dim, [512, 256, 128], 3, layernorm=True)
        orthogonal_init(self.net, gain=np.sqrt(2))
        # Re-init final layer with small gain
        orthogonal_init(self.net[-1:], gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)   # logits (batch, 3)


class ScaleOutCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM):
        super().__init__()
        self.net = build_mlp(obs_dim, [512, 256, 128], 1, layernorm=True)
        orthogonal_init(self.net, gain=np.sqrt(2))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)   # (batch, 1)


class ScaleOutAgent(PPO):
    def __init__(self, obs_dim: int = OBS_DIM, device: str = "cpu", **ppo_kwargs):
        actor  = ScaleOutActor(obs_dim)
        critic = ScaleOutCritic(obs_dim)
        super().__init__(actor=actor, critic=critic, device=device, **ppo_kwargs)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> Tuple[int, float, float]:
        """
        Returns:
            action   — int in {0, 1, 2}
            log_prob — float
            value    — float (critic estimate)
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        logits = self.actor(obs_t)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value    = self.critic(obs_t).squeeze().item()
        return action.item(), log_prob.item(), value

    # PPO hooks

    def _get_dist(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.actor(obs))

    def _log_prob(self, dist: Categorical, action: torch.Tensor, mask=None) -> torch.Tensor:
        return dist.log_prob(action.squeeze(-1).long())

    def _entropy(self, dist: Categorical, mask=None) -> torch.Tensor:
        return dist.entropy().mean()
