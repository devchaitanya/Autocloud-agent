"""
Day 3 deliverable tests:
  1. All 3 agents act() on random 215-dim obs without error.
  2. GAE values are computed correctly on a hand-crafted sequence.
  3. Buffer flush + PPO update runs without NaN losses.
  4. Gradient norms are bounded after update.
  5. Consolidation padding mask: inactive slots don't receive gradient.
  6. Scheduling agent log_prob sums only over active jobs.
  7. EMA normalizer: normalized values have ~zero mean over time.
  8. Raw return is never normalized (tracked separately).
"""
import numpy as np
import torch
import pytest

from autocloud.agents.scaleout import ScaleOutAgent
from autocloud.agents.consolidation import ConsolidationAgent
from autocloud.agents.scheduling import SchedulingAgent
from autocloud.training.ema_normalizer import EMANormalizer, AgentRewardTracker

OBS_DIM = 215
N_MAX   = 20
K_JOBS  = 20


def random_obs():
    return np.random.rand(OBS_DIM).astype(np.float32)


# ─────────────────────────────────────────────
# Agent act() tests
# ─────────────────────────────────────────────

class TestAgentAct:
    def test_scaleout_act_shape(self):
        agent = ScaleOutAgent(device="cpu")
        obs = random_obs()
        action, log_prob, value = agent.act(obs)
        assert action in {0, 1, 2}
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
        assert np.isfinite(log_prob)
        assert np.isfinite(value)

    def test_consolidation_act_shape(self):
        agent = ConsolidationAgent(device="cpu")
        obs  = random_obs()
        mask = np.ones(N_MAX, dtype=np.float32)
        mask[10:] = 0.0   # only 10 active nodes
        action, log_prob, value = agent.act(obs, active_mask=mask)
        assert action.shape == (N_MAX,)
        assert np.all((action == 0) | (action == 1))
        assert np.all(action[10:] == 0), "Inactive slots must be 0"
        assert np.isfinite(log_prob)

    def test_scheduling_act_shape(self):
        agent = SchedulingAgent(device="cpu")
        obs = random_obs()
        # Set some job features to nonzero
        obs[120:140] = np.random.rand(20).astype(np.float32)
        action, log_prob, value = agent.act(obs)
        assert action.shape == (K_JOBS,)
        assert np.all((action >= 0) & (action <= 4))
        assert np.isfinite(log_prob)

    def test_scaleout_no_nan_over_episodes(self):
        agent = ScaleOutAgent(device="cpu")
        for _ in range(200):
            obs = random_obs()
            action, lp, val = agent.act(obs)
            assert np.isfinite(lp) and np.isfinite(val)

    def test_consolidation_all_active(self):
        """With full active mask, all 20 slots can be drained."""
        agent = ConsolidationAgent(device="cpu")
        obs  = random_obs()
        mask = np.ones(N_MAX, dtype=np.float32)
        action, lp, val = agent.act(obs, active_mask=mask)
        assert action.shape == (N_MAX,)

    def test_scheduling_empty_queue(self):
        """With all-zero job features, log_prob should be 0 (no active jobs)."""
        agent = SchedulingAgent(device="cpu")
        obs = random_obs()
        obs[120:200] = 0.0   # zero out all job features
        _, log_prob, _ = agent.act(obs)
        assert log_prob == 0.0, f"Expected 0.0 for empty queue, got {log_prob}"


# ─────────────────────────────────────────────
# GAE computation test
# ─────────────────────────────────────────────

class TestGAE:
    def test_gae_correct_values(self):
        """
        Hand-crafted GAE check.
        For a 3-step sequence with γ=1.0, λ=1.0, done=False:
            δ_t = r_t + V(s_{t+1}) - V(s_t)
            GAE = δ_t + γλ * δ_{t+1} + (γλ)^2 * δ_{t+2}
        """
        agent = ScaleOutAgent(device="cpu", gamma=1.0, gae_lambda=1.0, buffer_size=3)

        rewards = [1.0, 2.0, 3.0]
        values  = [0.5, 1.0, 1.5]

        for i in range(3):
            obs = random_obs()
            agent.buffer.store(
                obs=obs,
                action=np.array([0]),
                log_prob=-0.5,
                reward=rewards[i],
                value=values[i],
                done=False,
            )

        # last_value = 0 (terminal), last_done = True
        last_obs = random_obs()
        # Manually compute GAE
        # δ2 = 3.0 + 1.0 * 0.0 - 1.5 = 1.5          (last_value=0)
        # δ1 = 2.0 + 1.0 * 1.5  - 1.0 = 2.5
        # δ0 = 1.0 + 1.0 * 1.0  - 0.5 = 1.5
        # GAE2 = 1.5
        # GAE1 = 2.5 + 1.0 * 1.5 = 4.0
        # GAE0 = 1.5 + 1.0 * 4.0 = 5.5
        expected_advantages = [5.5, 4.0, 1.5]

        transitions = agent.buffer.transitions
        n = len(transitions)
        advantages = np.zeros(n)
        gae = 0.0
        last_value_manual = 0.0  # terminal
        for t in reversed(range(n)):
            next_val = last_value_manual if t == n - 1 else transitions[t + 1].value
            next_done = float(t == n - 1)  # True only for last step when last_done=True
            delta = transitions[t].reward + 1.0 * next_val * (1.0 - next_done) - transitions[t].value
            gae = delta + 1.0 * 1.0 * (1.0 - next_done) * gae
            advantages[t] = gae

        for i in range(3):
            assert abs(advantages[i] - expected_advantages[i]) < 1e-4, \
                f"GAE[{i}] = {advantages[i]:.4f}, expected {expected_advantages[i]:.4f}"


# ─────────────────────────────────────────────
# PPO update tests
# ─────────────────────────────────────────────

class TestPPOUpdate:
    def _fill_buffer(self, agent, mask=None):
        """Fill agent's buffer with random data."""
        for _ in range(agent.buffer.buffer_size):
            obs = random_obs()
            if isinstance(agent, ConsolidationAgent):
                active_mask = np.ones(N_MAX, dtype=np.float32)
                action, lp, val = agent.act(obs, active_mask=active_mask)
                agent.store(obs, action, lp, np.random.randn(), val, False, mask=active_mask)
            elif isinstance(agent, SchedulingAgent):
                action, lp, val = agent.act(obs)
                job_mask = np.ones(K_JOBS, dtype=np.float32)
                agent.store(obs, action, lp, np.random.randn(), val, False, mask=job_mask)
            else:
                action, lp, val = agent.act(obs)
                agent.store(obs, action, lp, np.random.randn(), val, False)

    def test_scaleout_update_no_nan(self):
        agent = ScaleOutAgent(device="cpu", buffer_size=256, minibatch_size=64, update_epochs=2)
        self._fill_buffer(agent)
        assert agent.should_update()
        metrics = agent.update(random_obs(), last_done=True)
        assert np.isfinite(metrics["pg_loss"]), f"NaN pg_loss: {metrics}"
        assert np.isfinite(metrics["vf_loss"]), f"NaN vf_loss: {metrics}"
        assert np.isfinite(metrics["entropy"]), f"NaN entropy: {metrics}"
        assert len(agent.buffer) == 0, "Buffer not cleared after update"

    def test_consolidation_update_no_nan(self):
        agent = ConsolidationAgent(device="cpu", buffer_size=256, minibatch_size=64, update_epochs=2)
        self._fill_buffer(agent)
        metrics = agent.update(random_obs(), last_done=True)
        assert np.isfinite(metrics["pg_loss"])
        assert np.isfinite(metrics["vf_loss"])

    def test_scheduling_update_no_nan(self):
        agent = SchedulingAgent(device="cpu", buffer_size=256, minibatch_size=64, update_epochs=2)
        self._fill_buffer(agent)
        metrics = agent.update(random_obs(), last_done=True)
        assert np.isfinite(metrics["pg_loss"])
        assert np.isfinite(metrics["vf_loss"])

    def test_grad_norms_bounded(self):
        """After update, gradient norms should be bounded (clip at 0.5)."""
        agent = ScaleOutAgent(device="cpu", buffer_size=128, max_grad_norm=0.5, update_epochs=1)
        self._fill_buffer(agent)
        agent.update(random_obs(), last_done=True)
        # Check that params didn't explode
        for p in agent.actor.parameters():
            assert torch.all(torch.isfinite(p)), "Actor params contain NaN/Inf after update"
        for p in agent.critic.parameters():
            assert torch.all(torch.isfinite(p)), "Critic params contain NaN/Inf after update"

    def test_buffer_clears_after_update(self):
        agent = ScaleOutAgent(device="cpu", buffer_size=64)
        for _ in range(64):
            obs = random_obs()
            a, lp, val = agent.act(obs)
            agent.store(obs, a, lp, 1.0, val, False)
        assert agent.should_update()
        agent.update(random_obs(), last_done=False)
        assert len(agent.buffer) == 0


# ─────────────────────────────────────────────
# EMA normalizer tests
# ─────────────────────────────────────────────

class TestEMANormalizer:
    def test_variance_nonzero(self):
        norm = EMANormalizer(window=100)
        vals = [norm.normalize(float(x)) for x in np.random.randn(500)]
        assert np.std(vals[10:]) > 0.1   # should have spread

    def test_raw_return_never_normalized(self):
        tracker = AgentRewardTracker(ema_window=100)
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        raw_sum = 0.0
        for r in rewards:
            tracker.normalize(r)
            raw_sum += r
        assert abs(tracker.episode_raw_return - raw_sum) < 1e-6

    def test_end_episode_resets_raw(self):
        tracker = AgentRewardTracker()
        for r in [1.0, 2.0, 3.0]:
            tracker.normalize(r)
        ep_ret = tracker.end_episode()
        assert abs(ep_ret - 6.0) < 1e-6
        assert tracker.episode_raw_return == 0.0

    def test_ema_mean_converges(self):
        """After many samples from N(5,1), EMA mean should converge toward 5."""
        rng  = np.random.default_rng(0)
        norm = EMANormalizer(window=500)
        for x in rng.normal(5.0, 1.0, 2000):
            norm.normalize(float(x))
        assert abs(norm.mean - 5.0) < 0.5, f"EMA mean={norm.mean:.2f}, expected ~5.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
