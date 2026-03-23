"""
EMA reward normalizer.

Two separate trackers per agent:
  1. ema.normalize(r)  → used for PPO loss (stabilizes gradient scale)
  2. raw_return        → accumulated separately, NEVER normalized
                         used for evaluation metrics and curriculum checks

Critical: EMA normalizer updates its running stats on every reward sample
regardless of whether the agent acted that step.
"""
from __future__ import annotations


class EMANormalizer:
    """Exponential Moving Average normalization for rewards."""

    def __init__(self, window: int = 1000, epsilon: float = 1e-8):
        # EMA smoothing factor: alpha = 2 / (window + 1)
        self.alpha = 2.0 / (window + 1)
        self.epsilon = epsilon
        self._mean = 0.0
        self._var  = 1.0
        self._initialized = False

    def update(self, x: float) -> None:
        """Update running mean and variance with new sample."""
        if not self._initialized:
            self._mean = x
            self._var  = 1.0
            self._initialized = True
        else:
            delta = x - self._mean
            self._mean = self._mean + self.alpha * delta
            self._var  = (1.0 - self.alpha) * (self._var + self.alpha * delta ** 2)

    def normalize(self, x: float) -> float:
        """Update stats and return normalized value."""
        self.update(x)
        std = max(self._var ** 0.5, self.epsilon)
        return (x - self._mean) / std

    def reset(self) -> None:
        self._mean = 0.0
        self._var  = 1.0
        self._initialized = False

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        return max(self._var ** 0.5, self.epsilon)


class AgentRewardTracker:
    """
    Combines EMA normalization (for PPO) with raw return tracking (for eval).

    Usage in training loop:
        r_norm = tracker.normalize(r_t)   # feeds into buffer.store()
        # raw_return is accumulated automatically inside normalize()
    """

    def __init__(self, ema_window: int = 1000):
        self.ema = EMANormalizer(window=ema_window)
        self._episode_raw_return: float = 0.0
        self._total_raw_return:   float = 0.0
        self._episode_steps:      int   = 0

    def normalize(self, reward: float) -> float:
        """Normalize reward for PPO and track raw return."""
        self._episode_raw_return += reward
        self._total_raw_return   += reward
        self._episode_steps      += 1
        return self.ema.normalize(reward)

    def end_episode(self) -> float:
        """Call at episode boundary. Returns raw episode return and resets."""
        ret = self._episode_raw_return
        self._episode_raw_return = 0.0
        self._episode_steps      = 0
        return ret

    def reset(self) -> None:
        self.ema.reset()
        self._episode_raw_return = 0.0
        self._total_raw_return   = 0.0
        self._episode_steps      = 0

    @property
    def episode_raw_return(self) -> float:
        return self._episode_raw_return

    @property
    def total_raw_return(self) -> float:
        return self._total_raw_return
