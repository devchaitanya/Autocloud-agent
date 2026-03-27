"""
Hierarchical Safety Coordinator.

Because Scale-Out only adds and Consolidation only drains, there are zero
capacity-direction conflicts by design. The coordinator's role is purely
safety filtering — it never overrides intent, only enforces constraints.

Five filters applied in sequence:

  Filter 1 — Boot-protection:
    Nodes younger than boot_time + warmup_period are protected from draining.
    Prevents Consolidation from cannibalizing freshly-provisioned nodes that
    have 0% utilization during their boot window (prime but wrong drain targets).

  Filter 2 — N_min = 3 floor:
    If draining would leave < N_min active nodes, remove drain candidates
    until the floor is met. Tie-break: keep highest-CPU nodes (they're serving
    real load); drain lowest-CPU ones first.

  Filter 3 — Uncertainty-aware drain suppression:
    If forecast uncertainty σ_{t+5} > sigma_high, suppress the entire drain
    set. High uncertainty means the forecaster isn't confident about near-future
    demand — conservative action is safest.

  Filter 4 — Simultaneous scale-out suppression:
    If Scale-Out just fired (a_scaleout > 0), suppress Consolidation for this
    step. Don't drain while booting new nodes — the new capacity hasn't arrived
    yet and draining simultaneously wastes the provisioning cost.

  Filter 5 — Uncertainty-driven proactive scale-out (REACTIVE SAFETY):
    If σ_{t+5} > sigma_high AND CPU is rising, force at least 1 new node even
    if the ScaleOut agent chose 0. This catches out-of-distribution early spikes
    that the Transformer forecaster hasn't seen during training — exactly the
    "Early Shock" failure mode from Days 5 & 6 of the Alibaba trace.
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Dict

from environment.node import Node, NodeState


class SafetyCoordinator:
    def __init__(
        self,
        n_min: int = 3,
        warmup_period: float = 60.0,
        sigma_high: float = 0.3,
    ):
        self.n_min = n_min
        self.warmup_period = warmup_period
        self.sigma_high = sigma_high

    def resolve(
        self,
        a_scaleout: int,
        consolidation_vec: np.ndarray,      # (N_max,) binary — agent's drain proposals
        a_scheduling: np.ndarray,           # (K,) priority buckets — passed through unchanged
        nodes: List[Node],                  # current live node list (same slot order as obs)
        current_time: float,
        sigma_t5: float,                    # forecast uncertainty at t+5
        cpu_rising: bool = False,           # True when current CPU > previous step CPU
        cpu_delta: float = 0.0,             # absolute change in CPU util this step
        n_max: int = 20,
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Apply safety filters and return the final action triple.

        Returns:
            a_scaleout_final     — unchanged (we never block provisioning)
            consolidation_final  — (N_max,) binary after filtering
            a_scheduling_final   — unchanged (scheduling is always safe)
        """
        live_nodes = [n for n in nodes if n.state != NodeState.TERMINATED]

        # Build initial drain set from agent's proposal
        drain_set: List[int] = []   # node_ids
        for slot_idx, flag in enumerate(consolidation_vec[:len(live_nodes)]):
            if flag > 0.5 and live_nodes[slot_idx].state == NodeState.ACTIVE:
                drain_set.append(live_nodes[slot_idx].node_id)

        # ── Filter 1: Boot-protection ──────────────────────────────────
        protected = set()
        for node_id in drain_set:
            node = self._get_node(nodes, node_id)
            if node is not None and node.is_protected(current_time, self.warmup_period):
                protected.add(node_id)
        drain_set = [nid for nid in drain_set if nid not in protected]

        # ── Filter 2: N_min floor ──────────────────────────────────────
        n_active = sum(1 for n in live_nodes if n.state == NodeState.ACTIVE)
        would_remain = n_active - len(drain_set)
        if would_remain < self.n_min:
            excess = self.n_min - would_remain   # how many to remove from drain_set
            # Sort drain_set ascending: keep highest-CPU nodes active, drain lowest-CPU first
            drain_set_nodes = [
                (nid, self._get_node(nodes, nid))
                for nid in drain_set
            ]
            # Sort ascending by CPU util → remove lowest-util first (safest to drain,
            # but we must keep them to meet N_min → remove from drain_set)
            drain_set_nodes.sort(key=lambda x: x[1].cpu_util if x[1] else 0.0)
            # Drain the lowest-CPU nodes first (they're idle); protect highest-CPU ones
            # by removing them from drain_set so they stay active.
            to_remove = set(nid for nid, _ in drain_set_nodes[-excess:])
            drain_set = [nid for nid in drain_set if nid not in to_remove]

        # ── Filter 3: Uncertainty suppression ─────────────────────────
        if sigma_t5 > self.sigma_high:
            drain_set = []

        # ── Filter 4: Simultaneous scale-out suppression ──────────────
        if a_scaleout > 0:
            drain_set = []

        # ── Filter 5: Uncertainty-driven proactive scale-out ──────────
        # Fires when EITHER:
        #   (a) forecaster uncertainty is high AND cpu is rising
        #       → out-of-distribution spike the Transformer didn't predict
        #   (b) cpu spiked sharply (>6% in one 30s step) even with no forecaster
        #       → pure reactive fallback for early-shock workloads
        cpu_spike = cpu_delta > 0.06
        if (sigma_t5 > self.sigma_high or cpu_spike) and cpu_rising:
            a_scaleout = max(a_scaleout, 1)
            drain_set = []   # don't drain while forcing scale-out

        # Rebuild binary vector from filtered drain_set
        drain_ids_set = set(drain_set)
        consolidation_final = np.zeros(n_max, dtype=np.float32)
        for slot_idx, node in enumerate(live_nodes[:n_max]):
            if node.node_id in drain_ids_set:
                consolidation_final[slot_idx] = 1.0

        return a_scaleout, consolidation_final, a_scheduling

    def get_filter_report(
        self,
        a_scaleout: int,
        consolidation_vec: np.ndarray,
        nodes: List[Node],
        current_time: float,
        sigma_t5: float,
        n_max: int = 20,
    ) -> Dict:
        """Diagnostic: returns which filters fired and how many drains each removed."""
        live_nodes = [n for n in nodes if n.state != NodeState.TERMINATED]
        initial_drain = [
            live_nodes[i].node_id
            for i, flag in enumerate(consolidation_vec[:len(live_nodes)])
            if flag > 0.5 and live_nodes[i].state == NodeState.ACTIVE
        ]
        n_initial = len(initial_drain)

        protected = sum(
            1 for nid in initial_drain
            if (node := self._get_node(nodes, nid)) and node.is_protected(current_time, self.warmup_period)
        )
        after_f1 = n_initial - protected

        n_active = sum(1 for n in live_nodes if n.state == NodeState.ACTIVE)
        would_remain = n_active - after_f1
        f2_removed = max(0, self.n_min - would_remain) if would_remain < self.n_min else 0
        after_f2 = after_f1 - f2_removed

        f3_fired = sigma_t5 > self.sigma_high
        f4_fired = a_scaleout > 0

        return {
            "initial_drain_count": n_initial,
            "f1_boot_protect_removed": protected,
            "f2_nmin_removed": f2_removed,
            "f3_uncertainty_fired": f3_fired,
            "f4_scaleout_fired": f4_fired,
            "final_drain_count": 0 if (f3_fired or f4_fired) else after_f2,
        }

    @staticmethod
    def _get_node(nodes: List[Node], node_id: int) -> Optional[Node]:
        for n in nodes:
            if n.node_id == node_id:
                return n
        return None
