"""
Tests for SafetyCoordinator — all 4 filter cases, N_min boundary, uncertainty threshold.
"""
import pytest
import numpy as np

from autocloud.coordinator.safety import SafetyCoordinator
from autocloud.simulator.node import Node, NodeState, NODE_TYPES


# ------------------------------------------------------------------ #
# Fixtures / helpers
# ------------------------------------------------------------------ #

def make_node(node_id, state=NodeState.ACTIVE, cpu_util=0.5, boot_start_time=0.0):
    n = Node(
        node_id=node_id,
        node_type=NODE_TYPES["small"],   # cpu=2
        state=state,
        boot_start_time=boot_start_time,
    )
    # Set cpu_used so that cpu_util property returns correct value
    # cpu_util = cpu_used / node_type.cpu  (only when ACTIVE)
    if state == NodeState.ACTIVE:
        n.cpu_used = cpu_util * n.node_type.cpu
    return n


def make_coordinator(**kwargs):
    defaults = dict(n_min=3, warmup_period=60.0, sigma_high=0.3)
    defaults.update(kwargs)
    return SafetyCoordinator(**defaults)


def _prop_vec(slots):
    """Create consolidation vector with 1s at given slot indices."""
    v = np.zeros(20, dtype=np.float32)
    for s in slots:
        v[s] = 1.0
    return v


# ------------------------------------------------------------------ #
# Basic pass-through
# ------------------------------------------------------------------ #

class TestPassThrough:
    def test_no_drain_proposals_unchanged(self):
        coord = make_coordinator()
        nodes = [make_node(i) for i in range(5)]
        con_vec = np.zeros(20, dtype=np.float32)
        a_so, con_out, a_sch = coord.resolve(
            a_scaleout=0, consolidation_vec=con_vec,
            a_scheduling=2, nodes=nodes,
            current_time=300.0, sigma_t5=0.0
        )
        assert a_so == 0
        assert np.all(con_out == 0)
        assert a_sch == 2

    def test_scaleout_unchanged(self):
        coord = make_coordinator()
        nodes = [make_node(i) for i in range(5)]
        a_so, _, _ = coord.resolve(
            a_scaleout=2, consolidation_vec=np.zeros(20),
            a_scheduling=0, nodes=nodes,
            current_time=300.0, sigma_t5=0.0
        )
        assert a_so == 2

    def test_scheduling_unchanged(self):
        coord = make_coordinator()
        nodes = [make_node(i) for i in range(5)]
        _, _, a_sch = coord.resolve(
            a_scaleout=0, consolidation_vec=np.zeros(20),
            a_scheduling=4, nodes=nodes,
            current_time=300.0, sigma_t5=0.0
        )
        assert a_sch == 4


# ------------------------------------------------------------------ #
# Filter 1 — Boot-protection
# ------------------------------------------------------------------ #

class TestFilter1BootProtection:
    def test_young_node_protected(self):
        """Node booted at t=250 with warmup=60 → protected until t=310. At t=300 it's still protected."""
        coord = make_coordinator(warmup_period=60.0)
        nodes = [make_node(i, boot_start_time=250.0) for i in range(4)]
        # Try to drain slot 0
        con_vec = _prop_vec([0])
        _, con_out, _ = coord.resolve(
            a_scaleout=0, consolidation_vec=con_vec,
            a_scheduling=0, nodes=nodes,
            current_time=300.0, sigma_t5=0.0
        )
        assert con_out[0] == 0.0, "Young node should be boot-protected"

    def test_old_node_not_protected(self):
        """Node booted at t=0 with warmup=60 → not protected at t=300."""
        coord = make_coordinator(warmup_period=60.0, n_min=1)
        nodes = [make_node(i, boot_start_time=0.0) for i in range(4)]
        con_vec = _prop_vec([0])
        _, con_out, _ = coord.resolve(
            a_scaleout=0, consolidation_vec=con_vec,
            a_scheduling=0, nodes=nodes,
            current_time=300.0, sigma_t5=0.0
        )
        assert con_out[0] == 1.0, "Old node should be drainable"

    def test_boot_protection_boundary(self):
        """Node booted at t=240 with warmup=60, small node boot_time=30.
        Protected until age < (30+60)=90, i.e. until t=240+90=330.
        At t=329 still protected; at t=331 free."""
        coord = make_coordinator(warmup_period=60.0, n_min=1)
        # small node_type.boot_time = 30s  →  protection window = 30+60 = 90s
        nodes = [make_node(i, boot_start_time=240.0) for i in range(4)]
        con_vec = _prop_vec([0])

        # At t=329, age=89 < 90 → still protected
        _, con_at_329, _ = coord.resolve(
            a_scaleout=0, consolidation_vec=con_vec,
            a_scheduling=0, nodes=nodes, current_time=329.0, sigma_t5=0.0
        )
        assert con_at_329[0] == 0.0

        # At t=331, age=91 ≥ 90 → free to drain
        _, con_at_331, _ = coord.resolve(
            a_scaleout=0, consolidation_vec=con_vec,
            a_scheduling=0, nodes=nodes, current_time=331.0, sigma_t5=0.0
        )
        assert con_at_331[0] == 1.0


# ------------------------------------------------------------------ #
# Filter 2 — N_min floor
# ------------------------------------------------------------------ #

class TestFilter2NMin:
    def test_nmin_floor_respected(self):
        """With n_min=3 and 4 active nodes, only 1 can be drained."""
        coord = make_coordinator(n_min=3)
        nodes = [make_node(i, boot_start_time=0.0, cpu_util=0.5) for i in range(4)]
        # Try to drain all 4
        con_vec = _prop_vec([0, 1, 2, 3])
        _, con_out, _ = coord.resolve(
            a_scaleout=0, consolidation_vec=con_vec,
            a_scheduling=0, nodes=nodes, current_time=300.0, sigma_t5=0.0
        )
        assert con_out.sum() == 1.0, f"Only 1 drain allowed, got {con_out.sum()}"

    def test_nmin_floor_exact(self):
        """With n_min=3 and exactly 3 active nodes, no drain allowed."""
        coord = make_coordinator(n_min=3)
        nodes = [make_node(i, boot_start_time=0.0) for i in range(3)]
        con_vec = _prop_vec([0])
        _, con_out, _ = coord.resolve(
            a_scaleout=0, consolidation_vec=con_vec,
            a_scheduling=0, nodes=nodes, current_time=300.0, sigma_t5=0.0
        )
        assert con_out.sum() == 0.0, "Cannot drain when already at N_min"

    def test_nmin_drain_keeps_highest_cpu(self):
        """When trimming drain_set to meet N_min, highest-CPU nodes are kept active (lowest-CPU drained)."""
        coord = make_coordinator(n_min=3)
        # 4 nodes: cpu_utils = [0.9, 0.1, 0.8, 0.2] at slots 0-3
        cpu_utils = [0.9, 0.1, 0.8, 0.2]
        nodes = [make_node(i, boot_start_time=0.0, cpu_util=cpu_utils[i]) for i in range(4)]
        # Propose drain slots 0 and 1
        con_vec = _prop_vec([0, 1])
        _, con_out, _ = coord.resolve(
            a_scaleout=0, consolidation_vec=con_vec,
            a_scheduling=0, nodes=nodes, current_time=300.0, sigma_t5=0.0
        )
        # Only 1 drain allowed. Slot 1 (cpu=0.1) should be drained (lowest CPU in drain_set)
        # Slot 0 (cpu=0.9) should be kept active
        assert con_out.sum() == 1.0
        assert con_out[1] == 1.0, f"Lowest-CPU node (slot 1) should be drained, got {con_out}"
        assert con_out[0] == 0.0, f"Highest-CPU node (slot 0) should remain, got {con_out}"

    def test_nmin_with_terminated_nodes(self):
        """Terminated nodes don't count towards active count."""
        coord = make_coordinator(n_min=3)
        nodes = [
            make_node(0, state=NodeState.TERMINATED, boot_start_time=0.0),
            make_node(1, boot_start_time=0.0),
            make_node(2, boot_start_time=0.0),
            make_node(3, boot_start_time=0.0),
            make_node(4, boot_start_time=0.0),
        ]
        # 4 active nodes (node 0 is terminated), propose drain slot index 1
        con_vec = _prop_vec([1])  # slot 1 in live_nodes = node_id 1
        _, con_out, _ = coord.resolve(
            a_scaleout=0, consolidation_vec=con_vec,
            a_scheduling=0, nodes=nodes, current_time=300.0, sigma_t5=0.0
        )
        assert con_out.sum() == 1.0, "Should allow 1 drain (4 active → 3 ≥ n_min)"


# ------------------------------------------------------------------ #
# Filter 3 — Uncertainty suppression
# ------------------------------------------------------------------ #

class TestFilter3Uncertainty:
    def test_high_uncertainty_suppresses_drain(self):
        coord = make_coordinator(sigma_high=0.3, n_min=1)
        nodes = [make_node(i, boot_start_time=0.0) for i in range(5)]
        con_vec = _prop_vec([0])
        _, con_out, _ = coord.resolve(
            a_scaleout=0, consolidation_vec=con_vec,
            a_scheduling=0, nodes=nodes, current_time=300.0, sigma_t5=0.35
        )
        assert con_out.sum() == 0.0, "High uncertainty should suppress all drains"

    def test_low_uncertainty_allows_drain(self):
        coord = make_coordinator(sigma_high=0.3, n_min=1)
        nodes = [make_node(i, boot_start_time=0.0) for i in range(5)]
        con_vec = _prop_vec([0])
        _, con_out, _ = coord.resolve(
            a_scaleout=0, consolidation_vec=con_vec,
            a_scheduling=0, nodes=nodes, current_time=300.0, sigma_t5=0.2
        )
        assert con_out.sum() == 1.0, "Low uncertainty should allow drain"

    def test_uncertainty_exact_threshold(self):
        """At exactly sigma_high, drain is NOT suppressed (strict >)."""
        coord = make_coordinator(sigma_high=0.3, n_min=1)
        nodes = [make_node(i, boot_start_time=0.0) for i in range(5)]
        con_vec = _prop_vec([0])
        _, con_out, _ = coord.resolve(
            a_scaleout=0, consolidation_vec=con_vec,
            a_scheduling=0, nodes=nodes, current_time=300.0, sigma_t5=0.3
        )
        assert con_out.sum() == 1.0, "sigma == sigma_high should NOT suppress (strict >)"


# ------------------------------------------------------------------ #
# Filter 4 — Simultaneous scaleout suppression
# ------------------------------------------------------------------ #

class TestFilter4ScaleoutLock:
    def test_scaleout_suppresses_drain(self):
        coord = make_coordinator(n_min=1)
        nodes = [make_node(i, boot_start_time=0.0) for i in range(5)]
        con_vec = _prop_vec([0])
        _, con_out, _ = coord.resolve(
            a_scaleout=1, consolidation_vec=con_vec,
            a_scheduling=0, nodes=nodes, current_time=300.0, sigma_t5=0.0
        )
        assert con_out.sum() == 0.0, "Scaleout should lock out consolidation"

    def test_no_scaleout_allows_drain(self):
        coord = make_coordinator(n_min=1)
        nodes = [make_node(i, boot_start_time=0.0) for i in range(5)]
        con_vec = _prop_vec([0])
        _, con_out, _ = coord.resolve(
            a_scaleout=0, consolidation_vec=con_vec,
            a_scheduling=0, nodes=nodes, current_time=300.0, sigma_t5=0.0
        )
        assert con_out.sum() == 1.0

    def test_scaleout_2_also_suppresses(self):
        coord = make_coordinator(n_min=1)
        nodes = [make_node(i, boot_start_time=0.0) for i in range(5)]
        con_vec = _prop_vec([0])
        _, con_out, _ = coord.resolve(
            a_scaleout=2, consolidation_vec=con_vec,
            a_scheduling=0, nodes=nodes, current_time=300.0, sigma_t5=0.0
        )
        assert con_out.sum() == 0.0, "scaleout=2 should also suppress drain"


# ------------------------------------------------------------------ #
# Filter interaction / priority
# ------------------------------------------------------------------ #

class TestFilterInteraction:
    def test_filter3_overrides_filter2(self):
        """High uncertainty suppresses even if N_min would allow drains."""
        coord = make_coordinator(sigma_high=0.3, n_min=1)
        nodes = [make_node(i, boot_start_time=0.0) for i in range(10)]
        con_vec = _prop_vec([0, 1])
        _, con_out, _ = coord.resolve(
            a_scaleout=0, consolidation_vec=con_vec,
            a_scheduling=0, nodes=nodes, current_time=300.0, sigma_t5=0.5
        )
        assert con_out.sum() == 0.0

    def test_filter4_overrides_filter2(self):
        """Scaleout suppresses consolidation even if N_min would allow."""
        coord = make_coordinator(n_min=1)
        nodes = [make_node(i, boot_start_time=0.0) for i in range(10)]
        con_vec = _prop_vec([0, 1])
        _, con_out, _ = coord.resolve(
            a_scaleout=1, consolidation_vec=con_vec,
            a_scheduling=0, nodes=nodes, current_time=300.0, sigma_t5=0.0
        )
        assert con_out.sum() == 0.0

    def test_draining_state_nodes_ignored(self):
        """Nodes already in DRAINING state cannot be re-drained."""
        coord = make_coordinator(n_min=1)
        nodes = [
            make_node(0, state=NodeState.DRAINING, boot_start_time=0.0),
            make_node(1, boot_start_time=0.0),
            make_node(2, boot_start_time=0.0),
        ]
        # Propose draining slot 0 (already DRAINING)
        con_vec = _prop_vec([0])
        _, con_out, _ = coord.resolve(
            a_scaleout=0, consolidation_vec=con_vec,
            a_scheduling=0, nodes=nodes, current_time=300.0, sigma_t5=0.0
        )
        assert con_out[0] == 0.0, "Already-draining node should not be re-added to drain_set"


# ------------------------------------------------------------------ #
# Output shape
# ------------------------------------------------------------------ #

class TestOutputShape:
    def test_consolidation_output_shape(self):
        coord = make_coordinator()
        nodes = [make_node(i, boot_start_time=0.0) for i in range(8)]
        _, con_out, _ = coord.resolve(
            a_scaleout=0, consolidation_vec=np.zeros(20),
            a_scheduling=0, nodes=nodes, current_time=300.0, sigma_t5=0.0
        )
        assert con_out.shape == (20,)
        assert con_out.dtype == np.float32

    def test_consolidation_binary(self):
        coord = make_coordinator(n_min=1)
        nodes = [make_node(i, boot_start_time=0.0) for i in range(5)]
        con_vec = _prop_vec([0, 1])
        _, con_out, _ = coord.resolve(
            a_scaleout=0, consolidation_vec=con_vec,
            a_scheduling=0, nodes=nodes, current_time=300.0, sigma_t5=0.0
        )
        assert set(np.unique(con_out)).issubset({0.0, 1.0})


# ------------------------------------------------------------------ #
# Filter report
# ------------------------------------------------------------------ #

class TestFilterReport:
    def test_report_structure(self):
        coord = make_coordinator()
        nodes = [make_node(i, boot_start_time=0.0) for i in range(5)]
        report = coord.get_filter_report(
            a_scaleout=0,
            consolidation_vec=_prop_vec([0]),
            nodes=nodes,
            current_time=300.0,
            sigma_t5=0.0,
        )
        assert "initial_drain_count"      in report
        assert "f1_boot_protect_removed"  in report
        assert "f2_nmin_removed"          in report
        assert "f3_uncertainty_fired"     in report
        assert "f4_scaleout_fired"        in report
        assert "final_drain_count"        in report

    def test_report_f4_fires(self):
        coord = make_coordinator(n_min=1)
        nodes = [make_node(i, boot_start_time=0.0) for i in range(5)]
        report = coord.get_filter_report(
            a_scaleout=1,
            consolidation_vec=_prop_vec([0]),
            nodes=nodes,
            current_time=300.0,
            sigma_t5=0.0,
        )
        assert report["f4_scaleout_fired"] is True
        assert report["final_drain_count"] == 0

    def test_report_f3_fires(self):
        coord = make_coordinator(sigma_high=0.3, n_min=1)
        nodes = [make_node(i, boot_start_time=0.0) for i in range(5)]
        report = coord.get_filter_report(
            a_scaleout=0,
            consolidation_vec=_prop_vec([0]),
            nodes=nodes,
            current_time=300.0,
            sigma_t5=0.5,
        )
        assert report["f3_uncertainty_fired"] is True
        assert report["final_drain_count"] == 0
