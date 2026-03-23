from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Job:
    job_id: int
    arrival_time: float
    service_time: float         # total service duration (seconds)
    cpu_req: float = 1.0        # CPU cores required
    mem_req: float = 0.5        # GB memory required
    priority: int = 0           # 0=low … 4=critical
    deadline: Optional[float] = None  # absolute sim time deadline (None = no deadline)
    is_migrated: bool = False

    # Set when job starts service on a node
    service_start_time: Optional[float] = None
    assigned_node_id: Optional[int] = None
    service_process: Optional[object] = field(default=None, repr=False)

    # Filled on completion
    completion_time: Optional[float] = None

    @property
    def wait_time(self) -> float:
        if self.service_start_time is None:
            return 0.0
        return self.service_start_time - self.arrival_time

    @property
    def sojourn_time(self) -> Optional[float]:
        if self.completion_time is None or self.service_start_time is None:
            return None
        return self.completion_time - self.arrival_time

    @property
    def remaining_service_time(self) -> float:
        """Used during migration: how much service is left."""
        if self.service_start_time is None:
            return self.service_time
        # This is approximate; simulator tracks elapsed time precisely
        return self.service_time

    @property
    def deadline_urgency(self) -> float:
        """Normalized urgency in [0,1]; 1 = deadline already passed."""
        if self.deadline is None or self.arrival_time is None:
            return 0.0
        slack = self.deadline - self.arrival_time
        if slack <= 0:
            return 1.0
        return min(1.0, self.service_time / slack)

    @property
    def estimated_size_norm(self) -> float:
        """Normalized service time (capped at 600s → 1.0)."""
        return min(self.service_time / 600.0, 1.0)

    @property
    def priority_norm(self) -> float:
        return self.priority / 4.0
