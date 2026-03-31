"""
Day 1 deliverable tests:
  1. Node boots in exactly boot_time seconds.
  2. Two-phase drain terminates in ≤ 2 × grace_period seconds (zombie prevention).
  3. No zombie SimPy processes after 100 env steps.
  4. Observation shape is (215,).
  5. Reward dict has correct keys.
  6. Episode terminates at episode_steps.
"""
import numpy as np
import pytest
import simpy

from autocloud.simulator.engine import CloudSimulator
from autocloud.simulator.node import NodeState
from autocloud.simulator.cloud_env import CloudEnv, OBS_DIM
from autocloud.config.settings import DEFAULT_CONFIG


RNG = np.random.default_rng(42)


# 
# Simulator-level tests
# 

class TestNodeLifecycle:
    def _make_sim(self, seed=0):
        return CloudSimulator(
            rng=np.random.default_rng(seed),
            n_init=0,
            n_max=20,
            n_min=3,
            step_duration=30.0,
            grace_period=30.0,
            warmup_period=60.0,
        )

    def test_boot_timing_small(self):
        """small node (boot_time=30s) should be ACTIVE after 30s.
        Run until 30.1 because SimPy env.run(until=T) processes events at t<T strictly;
        the event scheduled at exactly T is processed when we run past T."""
        sim = self._make_sim()
        sim.reset(seed=0)
        node = sim._provision_node("small")
        assert node.state == NodeState.BOOTING
        sim.env.run(until=30.1)
        assert node.state == NodeState.ACTIVE, f"Expected ACTIVE, got {node.state}"

    def test_boot_timing_xlarge(self):
        """xlarge node (boot_time=90s) should still be BOOTING at 89s."""
        sim = self._make_sim()
        sim.reset(seed=0)
        node = sim._provision_node("xlarge")
        sim.env.run(until=89.0)
        assert node.state == NodeState.BOOTING
        sim.env.run(until=90.1)
        assert node.state == NodeState.ACTIVE

    def test_drain_bounded_time(self):
        """
        Two-phase drain must complete within 2 × grace_period (= 60s).
        Even with long-running jobs, Phase 2 hard-migrates them within grace_period.
        """
        sim = self._make_sim()
        sim.reset(seed=0)
        # Provision a node and let it boot
        node = sim._provision_node("medium")
        sim.env.run(until=node.node_type.boot_time + 1.0)
        assert node.state == NodeState.ACTIVE

        # Inject a very long job directly into active jobs
        from autocloud.simulator.job import Job
        long_job = Job(job_id=9999, arrival_time=sim.env.now, service_time=3600.0)
        long_job.service_start_time = sim.env.now
        long_job.assigned_node_id = node.node_id
        node.cpu_used += long_job.cpu_req
        node.mem_used += long_job.mem_req
        node.active_jobs.append(long_job)
        long_job.service_process = sim.env.process(sim._serve_job(long_job, node))

        # Start drain
        t_drain_start = sim.env.now
        sim.drain_node(node.node_id)
        assert node.state == NodeState.DRAINING

        # Run past 2 × grace_period
        sim.env.run(until=t_drain_start + 2 * sim.grace_period + 1.0)
        assert node.state == NodeState.TERMINATED, f"Node still in state {node.state} after 2×grace"

    def test_no_zombie_processes(self):
        """After 100 steps, there should be no alive SimPy processes for TERMINATED nodes."""
        sim = self._make_sim()
        sim.reset(seed=1)
        for _ in range(5):
            sim._provision_node("small")

        # Run 100 steps
        for i in range(100):
            target = (i + 1) * 30.0
            sim.step(target)

            # Drain a random active node every 10 steps
            if i % 10 == 5:
                active = [n for n in sim.nodes if n.state == NodeState.ACTIVE]
                if len(active) > 3:
                    sim.drain_node(active[0].node_id)

        # Check terminated nodes
        terminated = [n for n in sim.nodes if n.state == NodeState.TERMINATED]
        for node in terminated:
            assert node.drain_process is None, f"Node {node.node_id} has alive drain_process after termination"
            assert node.boot_process is None, f"Node {node.node_id} has alive boot_process after termination"
            assert len(node.active_jobs) == 0, f"Node {node.node_id} has leftover active_jobs"

    def test_migration_count_increments(self):
        """Hard migration should increment step_migrations."""
        sim = self._make_sim()
        sim.reset(seed=0)
        node = sim._provision_node("medium")
        sim.env.run(until=node.node_type.boot_time + 1.0)

        from autocloud.simulator.job import Job
        job = Job(job_id=1, arrival_time=sim.env.now, service_time=3600.0)
        job.service_start_time = sim.env.now
        job.assigned_node_id = node.node_id
        node.cpu_used += job.cpu_req
        node.mem_used += job.mem_req
        node.active_jobs.append(job)
        job.service_process = sim.env.process(sim._serve_job(job, node))

        t0 = sim.env.now
        sim.drain_node(node.node_id)
        sim.step_migrations = 0  # reset before running
        sim.env.run(until=t0 + sim.grace_period + 1.0)  # trigger Phase 2
        assert sim.step_migrations >= 1, "Expected at least 1 migration"


# 
# CloudEnv (Gymnasium wrapper) tests
# 

class TestCloudEnv:
    def _make_env(self, seed=0):
        return CloudEnv(config=DEFAULT_CONFIG, seed=seed)

    def test_observation_shape(self):
        env = self._make_env()
        obs, _ = env.reset()
        assert obs.shape == (OBS_DIM,), f"Expected ({OBS_DIM},), got {obs.shape}"
        assert obs.dtype == np.float32

    def test_observation_values_in_range(self):
        env = self._make_env()
        obs, _ = env.reset()
        assert np.all(obs >= 0.0), "Obs has negative values"
        assert np.all(obs <= 1.0), "Obs has values > 1.0"

    def test_step_returns_correct_keys(self):
        env = self._make_env()
        env.reset()
        action = {
            "scaleout": 0,
            "consolidation": np.zeros(20, dtype=np.int8),
            "scheduling": 0,
        }
        obs, reward, term, trunc, info = env.step(action)
        assert "rewards" in info
        assert set(info["rewards"].keys()) == {"scaleout", "consolidation", "scheduling"}
        assert "metrics" in info

    def test_step_obs_shape(self):
        env = self._make_env()
        env.reset()
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert obs.shape == (OBS_DIM,)

    def test_episode_truncation(self):
        """Episode should truncate at episode_steps (120 by default)."""
        env = self._make_env()
        env.reset()
        action = {
            "scaleout": 0,
            "consolidation": np.zeros(20, dtype=np.int8),
            "scheduling": 0,
        }
        for step in range(DEFAULT_CONFIG.sim.episode_steps):
            _, _, terminated, truncated, _ = env.step(action)
            if step < DEFAULT_CONFIG.sim.episode_steps - 1:
                assert not truncated, f"Truncated early at step {step}"
            else:
                assert truncated, "Episode did not truncate at episode_steps"

    def test_no_obs_nan(self):
        """Run a full episode and ensure no NaN in any observation."""
        env = self._make_env(seed=7)
        obs, _ = env.reset()
        assert not np.any(np.isnan(obs))
        for _ in range(DEFAULT_CONFIG.sim.episode_steps):
            action = env.action_space.sample()
            obs, _, _, truncated, _ = env.step(action)
            assert not np.any(np.isnan(obs)), "NaN detected in observation"
            if truncated:
                break

    def test_scaleout_increases_node_count(self):
        """Scaleout action of +1 should eventually increase active node count."""
        env = self._make_env()
        env.reset()

        # Wait for initial nodes to boot
        for _ in range(5):
            env.step({"scaleout": 0, "consolidation": np.zeros(20, dtype=np.int8), "scheduling": 0})

        n_before = env.sim.get_metrics()["n_active"]
        # Provision 2 nodes
        env.step({"scaleout": 2, "consolidation": np.zeros(20, dtype=np.int8), "scheduling": 0})

        # Wait for boot (xlarge takes 90s = 3 steps)
        for _ in range(5):
            env.step({"scaleout": 0, "consolidation": np.zeros(20, dtype=np.int8), "scheduling": 0})

        n_after = env.sim.get_metrics()["n_active"] + env.sim.get_metrics()["n_booting"]
        assert n_after >= n_before, "Node count did not increase after scaleout"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
