"""
AutoResearch Experiment — THIS IS THE ONLY FILE THE AGENT MODIFIES.

Defines get_config() which returns the Config used for training.
Everything in get_config() is fair game: PPO hyperparameters, reward
weights, and coordinator threshold.

The agent reads this file, proposes changes, and rewrites it each iteration.
If the score improves the new version is kept; otherwise it is reverted.
"""

import copy
from configs.default_config import Config, DEFAULT_CONFIG


def get_config() -> Config:
    """Return the experimental config. Modify the values below freely."""
    config = copy.deepcopy(DEFAULT_CONFIG)

    # Increase gamma2 (SLA violation penalty) to discourage Scheduling from allowing SLA violations
    config.reward.gamma2  = 2.0
    
    return config