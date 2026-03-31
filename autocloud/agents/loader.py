"""
Load trained I-PPO agent checkpoints into inference-ready PPO wrappers.

Usage:
    from autocloud.agents.loader import load_agents
    so, con, sch = load_agents("checkpoints/")
"""
from __future__ import annotations

import os
import torch
from typing import Tuple

from autocloud.agents.scaleout import ScaleOutAgent
from autocloud.agents.consolidation import ConsolidationAgent
from autocloud.agents.scheduling import SchedulingAgent
from autocloud.config.settings import Config, DEFAULT_CONFIG


def load_agents(
    checkpoint_dir: str,
    tag: str = "final",
    device: str = "cpu",
    config: Config = DEFAULT_CONFIG,
) -> Tuple[ScaleOutAgent, ConsolidationAgent, SchedulingAgent]:
    """
    Instantiate the three I-PPO agents and load saved weights.

    Raises FileNotFoundError if any required .pt file is missing.
    """
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

    so_agent = ScaleOutAgent(**ppo_kwargs)
    con_agent = ConsolidationAgent(**ppo_kwargs)
    sch_agent = SchedulingAgent(**ppo_kwargs)

    def _load(module: torch.nn.Module, filename: str) -> None:
        path = os.path.join(checkpoint_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        module.load_state_dict(
            torch.load(path, map_location=device)
        )

    _load(so_agent.actor,   f"so_actor_{tag}.pt")
    _load(so_agent.critic,  f"so_critic_{tag}.pt")
    _load(con_agent.actor,  f"con_actor_{tag}.pt")
    _load(con_agent.critic, f"con_critic_{tag}.pt")
    _load(sch_agent.actor,  f"sch_actor_{tag}.pt")
    _load(sch_agent.critic, f"sch_critic_{tag}.pt")

    return so_agent, con_agent, sch_agent
