from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    import simpy


class NodeState(IntEnum):
    BOOTING = 0
    ACTIVE = 1
    DRAINING = 2
    TERMINATED = 3


@dataclass
class NodeType:
    name: str
    cpu: int
    memory: float       # GB
    cost_per_hr: float
    boot_time: float    # seconds


NODE_TYPES = {
    "small":  NodeType("small",  cpu=2,  memory=4.0,  cost_per_hr=0.05, boot_time=30.0),
    "medium": NodeType("medium", cpu=4,  memory=8.0,  cost_per_hr=0.10, boot_time=45.0),
    "large":  NodeType("large",  cpu=8,  memory=16.0, cost_per_hr=0.20, boot_time=60.0),
    "xlarge": NodeType("xlarge", cpu=16, memory=32.0, cost_per_hr=0.40, boot_time=90.0),
}


@dataclass
class Node:
    node_id: int
    node_type: NodeType
    state: NodeState = NodeState.BOOTING
    boot_start_time: float = 0.0
    drain_start_time: Optional[float] = None

    # Runtime metrics (updated each step)
    cpu_used: float = 0.0           # cores currently in use
    mem_used: float = 0.0           # GB currently in use

    # SimPy process handles (set by simulator)
    boot_process: Optional[object] = field(default=None, repr=False)
    drain_process: Optional[object] = field(default=None, repr=False)

    # Jobs currently running on this node
    active_jobs: List[object] = field(default_factory=list, repr=False)

    @property
    def cpu_util(self) -> float:
        if self.state != NodeState.ACTIVE:
            return 0.0
        cap = self.node_type.cpu
        return min(self.cpu_used / cap, 1.0) if cap > 0 else 0.0

    @property
    def mem_util(self) -> float:
        cap = self.node_type.memory
        return min(self.mem_used / cap, 1.0) if cap > 0 else 0.0

    @property
    def age(self) -> float:
        """Filled in by simulator using current sim time."""
        return 0.0  # placeholder; simulator computes this dynamically

    def is_protected(self, current_time: float, warmup: float) -> bool:
        """True if node should not be drained (too young)."""
        age = current_time - self.boot_start_time
        return age < (self.node_type.boot_time + warmup)

    def available_cpu(self) -> float:
        if self.state != NodeState.ACTIVE:
            return 0.0
        return max(0.0, self.node_type.cpu - self.cpu_used)

    def available_mem(self) -> float:
        if self.state != NodeState.ACTIVE:
            return 0.0
        return max(0.0, self.node_type.memory - self.mem_used)
