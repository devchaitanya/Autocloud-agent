"""
Scheduling Agent — per-job priority scoring network.

Architecture (weight-tied across K jobs):
  MLP_global [135 → 64]   encodes non-job state (node + global features)
  MLP_job    [4   → 32]   encodes per-job features (shared weights)
  MLP_score  [96  → 32 → 5]  scoring head per job → 5 priority buckets

Observation slicing (215-dim):
  obs[0:120]   — per-node features (120 dims)
  obs[120:200] — per-job features  (80 dims = 20 jobs × 4 features)
  obs[200:215] — global features   (15 dims)

MLP_global input: obs[0:120] + obs[200:215] = 135 dims
MLP_job input:    obs[120:200].reshape(20, 4) per job

Padding mask: jobs with all-zero feature rows are inactive;
their log_prob contribution is masked to 0.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Tuple, Optional

from .ppo import PPO, orthogonal_init

OBS_DIM      = 215
N_MAX        = 20
K_JOBS       = 20
N_NODE_FEATS = 120   # obs[0:120]
N_JOB_FEATS  = 80    # obs[120:200]
N_GLOB_FEATS = 15    # obs[200:215]
GLOBAL_DIM   = N_NODE_FEATS + N_GLOB_FEATS  # 135
JOB_DIM      = 4
N_PRIORITIES = 5


class SchedulingNetwork(nn.Module):
    """Per-job scoring network with shared (weight-tied) MLP_job."""

    def __init__(self):
        super().__init__()
        # Encodes node + global state
        self.mlp_global = nn.Sequential(
            nn.Linear(GLOBAL_DIM, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        # Shared per-job encoder
        self.mlp_job = nn.Sequential(
            nn.Linear(JOB_DIM, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        # Scoring head: [128 + 64 = 192] → 5 logits
        self.mlp_score = nn.Sequential(
            nn.Linear(192, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, N_PRIORITIES),
        )
        orthogonal_init(self.mlp_global, gain=np.sqrt(2))
        orthogonal_init(self.mlp_job,    gain=np.sqrt(2))
        orthogonal_init(self.mlp_score,  gain=np.sqrt(2))
        orthogonal_init(self.mlp_score[-1:], gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, 215)
        Returns:
            logits: (batch, K_JOBS, N_PRIORITIES)
        """
        batch = obs.shape[0]

        # Split observation
        node_feats   = obs[:, :N_NODE_FEATS]             # (batch, 120)
        job_feats    = obs[:, N_NODE_FEATS:N_NODE_FEATS + N_JOB_FEATS]   # (batch, 80)
        global_feats = obs[:, N_NODE_FEATS + N_JOB_FEATS:]               # (batch, 15)

        # Global context
        global_ctx = self.mlp_global(
            torch.cat([node_feats, global_feats], dim=-1)   # (batch, 135)
        )  # (batch, 64)

        # Per-job encoding: reshape then apply shared MLP
        job_feats_3d = job_feats.view(batch, K_JOBS, JOB_DIM)   # (batch, 20, 4)
        job_enc = self.mlp_job(job_feats_3d)                      # (batch, 20, 32)

        # Concatenate global context (broadcast) with each job encoding
        global_exp = global_ctx.unsqueeze(1).expand(-1, K_JOBS, -1)  # (batch, 20, 128)
        combined = torch.cat([global_exp, job_enc], dim=-1)           # (batch, 20, 192)

        # Score each job
        logits = self.mlp_score(combined)   # (batch, 20, 5)
        return logits


class SchedulingCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256),    nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128),    nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        orthogonal_init(self.net, gain=np.sqrt(2))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class SchedulingAgent(PPO):
    def __init__(self, obs_dim: int = OBS_DIM, k_jobs: int = K_JOBS,
                 device: str = "cpu", **ppo_kwargs):
        actor  = SchedulingNetwork()
        critic = SchedulingCritic(obs_dim)
        super().__init__(actor=actor, critic=critic, device=device, **ppo_kwargs)
        self.k_jobs = k_jobs

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Returns:
            action   — (k_jobs,) int array of priority buckets {0..4}
            log_prob — float (sum over active jobs)
            value    — float
        """
        obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        logits = self.actor(obs_t)              # (1, K_JOBS, 5)
        dist   = Categorical(logits=logits)
        action = dist.sample()                  # (1, K_JOBS)

        # Job mask: job slot is active if any feature is nonzero
        job_feats = obs[N_NODE_FEATS:N_NODE_FEATS + N_JOB_FEATS].reshape(K_JOBS, JOB_DIM)
        job_mask  = (job_feats.sum(axis=-1) > 0).astype(np.float32)   # (K_JOBS,)

        mask_t    = torch.FloatTensor(job_mask).unsqueeze(0).to(self.device)  # (1, K_JOBS)
        log_prob_per_job = dist.log_prob(action)   # (1, K_JOBS)
        log_prob  = (log_prob_per_job * mask_t).sum().item()

        value = self.critic(obs_t).squeeze().item()
        return action.squeeze(0).cpu().numpy(), log_prob, value

    # PPO hooks

    def _get_dist(self, obs: torch.Tensor) -> Categorical:
        logits = self.actor(obs)        # (batch, K_JOBS, 5)
        return Categorical(logits=logits)

    def _log_prob(self, dist: Categorical, action: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        action: (batch, K_JOBS) stored as float — cast to long
        mask:   (batch, K_JOBS) job activity mask
        """
        act_long = action.long()
        lp = dist.log_prob(act_long)    # (batch, K_JOBS)
        if mask is not None:
            lp = lp * mask
        return lp.sum(dim=-1)           # (batch,)

    def _entropy(self, dist: Categorical,
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        ent = dist.entropy()            # (batch, K_JOBS)
        if mask is not None:
            ent = ent * mask
            return ent.sum(dim=-1).mean()
        return ent.mean()
