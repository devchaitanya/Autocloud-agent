"""
SimPy cloud simulator.

Key design decisions:
- M/G/c queue: Poisson arrivals, LogNormal service times.
- Two-phase VM drain:
    Phase 1 (soft): node stops accepting new jobs; in-flight jobs complete naturally.
    Phase 2 (hard): any job still running is forcibly migrated: re-enqueued with
                    MIGRATED=True so it gets dispatched before new arrivals.
- Zombie prevention: drain always terminates within 2 × grace_period seconds.
- All SimPy Process references are cleared on node termination.
"""
from __future__ import annotations

import simpy
import numpy as np
from typing import List, Optional, Dict, Callable
from collections import deque

from .node import Node, NodeState, NODE_TYPES
from .job import Job


class CloudSimulator:
    def __init__(
        self,
        rng: np.random.Generator,
        n_init: int = 5,
        n_max: int = 20,
        n_min: int = 3,
        step_duration: float = 30.0,
        grace_period: float = 30.0,
        warmup_period: float = 60.0,
        lognormal_mu: float = 2.0,
        lognormal_sigma: float = 1.0,
        base_arrival_rate: float = 2.0,   # jobs/second at baseline
        k_jobs: int = 20,
        workload_fn: Optional[Callable[[float], float]] = None,
    ):
        self.rng = rng
        self.n_init = n_init
        self.n_max = n_max
        self.n_min = n_min
        self.step_duration = step_duration
        self.grace_period = grace_period
        self.warmup_period = warmup_period
        self.lognormal_mu = lognormal_mu
        self.lognormal_sigma = lognormal_sigma
        self.base_arrival_rate = base_arrival_rate
        self.k_jobs = k_jobs
        self.workload_fn = workload_fn  # callable(sim_time) -> arrival rate multiplier

        # SimPy environment (re-created on reset)
        self.env: Optional[simpy.Environment] = None

        # Cluster state
        self.nodes: List[Node] = []
        self._next_node_id: int = 0
        self._next_job_id: int = 0

        # Job queue: deque sorted by (is_migrated desc, priority desc, arrival_time asc)
        # We maintain it as a plain deque and sort on dispatch.
        self.queue: deque = deque()

        # Metrics for current step
        self.step_completions: int = 0
        self.step_migrations: int = 0
        self.step_wait_times: List[float] = []
        self.step_sojourn_times: List[float] = []
        self.total_cost: float = 0.0

        # Running processes
        self._arrival_process: Optional[simpy.Process] = None

        # Completed jobs tracking (for p95 latency)
        self._recent_sojourns: deque = deque(maxlen=200)

    # ------------------------------------------------------------------ #
    # Reset / init
    # ------------------------------------------------------------------ #

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.env = simpy.Environment()
        self.nodes = []
        self._next_node_id = 0
        self._next_job_id = 0
        self.queue = deque()
        self.step_completions = 0
        self.step_migrations = 0
        self.step_wait_times = []
        self.step_sojourn_times = []
        self.total_cost = 0.0
        self._recent_sojourns = deque(maxlen=200)

        # Boot initial nodes (all "medium")
        for _ in range(self.n_init):
            self._provision_node("medium")

        # Start arrival process
        self._arrival_process = self.env.process(self._arrivals())

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #

    def step(self, target_time: float) -> None:
        """Advance simulation to target_time, resetting per-step metrics first."""
        self.step_completions = 0
        self.step_migrations = 0
        self.step_wait_times = []
        self.step_sojourn_times = []
        self.env.run(until=target_time)
        self._accrue_cost()

    # ------------------------------------------------------------------ #
    # Provisioning
    # ------------------------------------------------------------------ #

    def provision_node(self, node_type_name: str = "medium") -> Optional[Node]:
        active = self._active_count()
        if active >= self.n_max:
            return None
        return self._provision_node(node_type_name)

    def _provision_node(self, node_type_name: str) -> Node:
        nt = NODE_TYPES[node_type_name]
        node = Node(
            node_id=self._next_node_id,
            node_type=nt,
            state=NodeState.BOOTING,
            boot_start_time=self.env.now if self.env else 0.0,
        )
        self._next_node_id += 1
        self.nodes.append(node)
        node.boot_process = self.env.process(self._boot_node(node))
        return node

    def _boot_node(self, node: Node):
        yield self.env.timeout(node.node_type.boot_time)
        if node.state == NodeState.BOOTING:
            node.state = NodeState.ACTIVE
            node.boot_process = None
            self._try_dispatch()

    # ------------------------------------------------------------------ #
    # Drain (two-phase)
    # ------------------------------------------------------------------ #

    def drain_node(self, node_id: int) -> bool:
        node = self._get_node(node_id)
        if node is None or node.state != NodeState.ACTIVE:
            return False
        node.state = NodeState.DRAINING
        node.drain_start_time = self.env.now
        node.drain_process = self.env.process(self._drain_node(node))
        return True

    def _drain_node(self, node: Node):
        # Phase 1: soft drain — wait grace_period for in-flight jobs to finish
        try:
            yield self.env.timeout(self.grace_period)
        except simpy.Interrupt:
            pass  # forced early termination (not used currently)

        # Phase 2: hard migration — forcibly migrate any remaining jobs
        self._hard_migrate(node)
        node.state = NodeState.TERMINATED
        node.drain_process = None
        node.active_jobs.clear()

    def _hard_migrate(self, node: Node) -> None:
        jobs_to_migrate = list(node.active_jobs)
        for job in jobs_to_migrate:
            if job.service_process is not None and job.service_process.is_alive:
                # Compute remaining service time before interrupting
                elapsed = self.env.now - job.service_start_time
                remaining = max(0.0, job.service_time - elapsed)
                job.service_time = remaining
                job.service_start_time = None
                job.assigned_node_id = None
                job.is_migrated = True
                job.service_process.interrupt("migrate")
                job.service_process = None
                # Return resources
                node.cpu_used = max(0.0, node.cpu_used - job.cpu_req)
                node.mem_used = max(0.0, node.mem_used - job.mem_req)
                self.step_migrations += 1
                # Re-enqueue with migration tag (dispatched first)
                self.queue.appendleft(job)
        node.active_jobs = [j for j in node.active_jobs if not j.is_migrated]

    # ------------------------------------------------------------------ #
    # Arrival process
    # ------------------------------------------------------------------ #

    def _arrivals(self):
        while True:
            rate = self._current_rate()
            inter_arrival = self.rng.exponential(1.0 / max(rate, 1e-6))
            yield self.env.timeout(inter_arrival)
            job = self._generate_job()
            self.queue.append(job)
            self._try_dispatch()

    def _current_rate(self) -> float:
        if self.workload_fn is not None:
            mult = self.workload_fn(self.env.now)
        else:
            mult = 1.0
        return self.base_arrival_rate * max(mult, 0.01)

    def _generate_job(self) -> Job:
        service_time = float(self.rng.lognormal(self.lognormal_mu, self.lognormal_sigma))
        service_time = max(1.0, min(service_time, 3600.0))  # clamp 1s–1hr
        cpu_req = float(self.rng.choice([0.5, 1.0, 2.0], p=[0.5, 0.4, 0.1]))
        mem_req = float(self.rng.choice([0.25, 0.5, 1.0], p=[0.5, 0.4, 0.1]))
        priority = int(self.rng.integers(0, 5))
        job = Job(
            job_id=self._next_job_id,
            arrival_time=self.env.now,
            service_time=service_time,
            cpu_req=cpu_req,
            mem_req=mem_req,
            priority=priority,
        )
        self._next_job_id += 1
        return job

    # ------------------------------------------------------------------ #
    # Dispatcher
    # ------------------------------------------------------------------ #

    def _try_dispatch(self) -> None:
        """Dispatch queued jobs to available nodes using SJF with MIGRATED priority."""
        if not self.queue:
            return

        # Sort: migrated jobs first, then by priority desc, then by service_time asc
        sorted_q = sorted(
            self.queue,
            key=lambda j: (not j.is_migrated, -j.priority, j.service_time),
        )

        dispatched = set()
        for job in sorted_q:
            node = self._find_node_for_job(job)
            if node is None:
                break  # no capacity available
            dispatched.add(job.job_id)
            self._start_job(job, node)

        self.queue = deque(j for j in self.queue if j.job_id not in dispatched)

    def _find_node_for_job(self, job: Job) -> Optional[Node]:
        """Find ACTIVE node with enough CPU+mem, prefer least-loaded."""
        candidates = [
            n for n in self.nodes
            if n.state == NodeState.ACTIVE
            and n.available_cpu() >= job.cpu_req
            and n.available_mem() >= job.mem_req
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda n: n.cpu_util)

    def _start_job(self, job: Job, node: Node) -> None:
        job.service_start_time = self.env.now
        job.assigned_node_id = node.node_id
        node.cpu_used += job.cpu_req
        node.mem_used += job.mem_req
        node.active_jobs.append(job)
        job.service_process = self.env.process(self._serve_job(job, node))
        self.step_wait_times.append(job.wait_time)

    def _serve_job(self, job: Job, node: Node):
        try:
            yield self.env.timeout(job.service_time)
            # Job completed normally
            job.completion_time = self.env.now
            sojourn = job.completion_time - job.arrival_time
            self.step_sojourn_times.append(sojourn)
            self._recent_sojourns.append(sojourn)
            self.step_completions += 1
        except simpy.Interrupt as e:
            if str(e.cause) == "migrate":
                return  # migrated; job re-queued by _hard_migrate
            return
        finally:
            # Release resources (guard against double-release on migrate)
            if job.assigned_node_id == node.node_id:
                node.cpu_used = max(0.0, node.cpu_used - job.cpu_req)
                node.mem_used = max(0.0, node.mem_used - job.mem_req)
                if job in node.active_jobs:
                    node.active_jobs.remove(job)
                job.service_process = None
        # Try dispatching jobs that were waiting for capacity
        self._try_dispatch()

    # ------------------------------------------------------------------ #
    # Apply scheduling action (re-order queue by agent-assigned priorities)
    # ------------------------------------------------------------------ #

    def apply_scheduling_action(self, priority_overrides: Dict[int, int]) -> None:
        """
        priority_overrides: {job_id: new_priority_bucket (0..4)}
        Applies the scheduling agent's priority assignments to queued jobs.
        """
        for job in self.queue:
            if job.job_id in priority_overrides:
                job.priority = priority_overrides[job.job_id]
        self._try_dispatch()

    # ------------------------------------------------------------------ #
    # Cost accrual
    # ------------------------------------------------------------------ #

    def _accrue_cost(self) -> None:
        dt_hr = self.step_duration / 3600.0
        for node in self.nodes:
            if node.state in (NodeState.ACTIVE, NodeState.DRAINING, NodeState.BOOTING):
                self.total_cost += node.node_type.cost_per_hr * dt_hr

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #

    def get_metrics(self) -> dict:
        active_nodes = [n for n in self.nodes if n.state == NodeState.ACTIVE]
        booting_nodes = [n for n in self.nodes if n.state == NodeState.BOOTING]
        draining_nodes = [n for n in self.nodes if n.state == NodeState.DRAINING]

        n_active = len(active_nodes)
        n_booting = len(booting_nodes)
        n_draining = len(draining_nodes)

        cpu_utils = [n.cpu_util for n in active_nodes]
        mem_utils = [n.mem_util for n in active_nodes]
        mean_cpu = float(np.mean(cpu_utils)) if cpu_utils else 0.0
        mean_mem = float(np.mean(mem_utils)) if mem_utils else 0.0

        queue_len = len(self.queue)

        # P95 sojourn (latency proxy)
        if len(self._recent_sojourns) >= 5:
            p95 = float(np.percentile(list(self._recent_sojourns), 95))
        else:
            p95 = 0.0

        mean_wait = float(np.mean(self.step_wait_times)) if self.step_wait_times else 0.0

        step_cost = sum(
            n.node_type.cost_per_hr * (self.step_duration / 3600.0)
            for n in self.nodes
            if n.state in (NodeState.ACTIVE, NodeState.DRAINING, NodeState.BOOTING)
        )

        return {
            "n_active": n_active,
            "n_booting": n_booting,
            "n_draining": n_draining,
            "mean_cpu_util": mean_cpu,
            "mean_mem_util": mean_mem,
            "queue_len": queue_len,
            "p95_latency": p95,
            "mean_wait": mean_wait,
            "step_cost": step_cost,
            "total_cost": self.total_cost,
            "step_completions": self.step_completions,
            "step_migrations": self.step_migrations,
        }

    def get_node_features(self, current_time: float, n_max: int) -> np.ndarray:
        """
        Returns (n_max, 6) array of per-node features, zero-padded.
        Features per node: [cpu_util, mem_util, age_norm, is_booting, is_active, is_draining]
        """
        features = np.zeros((n_max, 6), dtype=np.float32)
        live_nodes = [n for n in self.nodes if n.state != NodeState.TERMINATED]
        for i, node in enumerate(live_nodes[:n_max]):
            age = current_time - node.boot_start_time
            age_norm = min(age / 3600.0, 1.0)
            features[i, 0] = node.cpu_util
            features[i, 1] = node.mem_util
            features[i, 2] = age_norm
            features[i, 3] = float(node.state == NodeState.BOOTING)
            features[i, 4] = float(node.state == NodeState.ACTIVE)
            features[i, 5] = float(node.state == NodeState.DRAINING)
        return features

    def get_job_features(self, k: int, current_time: float) -> np.ndarray:
        """
        Returns (k, 4) array of top-k queued job features, zero-padded.
        Features per job: [size_norm, wait_norm, priority_norm, deadline_urgency]
        """
        features = np.zeros((k, 4), dtype=np.float32)
        # Sort queue same way as dispatcher for consistency
        sorted_q = sorted(
            self.queue,
            key=lambda j: (not j.is_migrated, -j.priority, j.service_time),
        )
        for i, job in enumerate(sorted_q[:k]):
            wait = current_time - job.arrival_time
            features[i, 0] = job.estimated_size_norm
            features[i, 1] = min(wait / 300.0, 1.0)   # normalize to 5min
            features[i, 2] = job.priority_norm
            features[i, 3] = job.deadline_urgency
        return features

    def get_top_k_job_ids(self, k: int) -> List[int]:
        sorted_q = sorted(
            self.queue,
            key=lambda j: (not j.is_migrated, -j.priority, j.service_time),
        )
        return [j.job_id for j in sorted_q[:k]]

    def get_active_node_mask(self, n_max: int) -> np.ndarray:
        """Binary mask (n_max,): 1 if slot i has a live (non-terminated) node."""
        mask = np.zeros(n_max, dtype=np.float32)
        live_nodes = [n for n in self.nodes if n.state != NodeState.TERMINATED]
        for i in range(min(len(live_nodes), n_max)):
            mask[i] = 1.0
        return mask

    def get_drainable_node_ids(self, n_max: int) -> List[int]:
        """Return node IDs for live nodes in slot order (same order as node features)."""
        live_nodes = [n for n in self.nodes if n.state != NodeState.TERMINATED]
        return [n.node_id for n in live_nodes[:n_max]]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _active_count(self) -> int:
        return sum(1 for n in self.nodes if n.state != NodeState.TERMINATED)

    def _get_node(self, node_id: int) -> Optional[Node]:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None

    def _active_nodes(self) -> List[Node]:
        return [n for n in self.nodes if n.state == NodeState.ACTIVE]

    def node_type_for_demand(self, predicted_demand_norm: float) -> str:
        """Greedy node type selection: smallest type meeting predicted demand."""
        if predicted_demand_norm > 0.75:
            return "xlarge"
        elif predicted_demand_norm > 0.5:
            return "large"
        elif predicted_demand_norm > 0.25:
            return "medium"
        return "small"
