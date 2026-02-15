"""
DAG data models for task dependency management.

Provides serializable data structures for representing task graphs
with dependency relationships, status tracking, and wavefront execution state.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class DAGTaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DAGTask:
    """Single node in the task DAG."""
    task_id: str
    title: str
    description: str
    tools: List[str] = field(default_factory=list)
    priority: str = "medium"
    depends_on: List[str] = field(default_factory=list)
    status: DAGTaskStatus = DAGTaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "tools": self.tools,
            "priority": self.priority,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DAGTask":
        return cls(
            task_id=data["task_id"],
            title=data.get("title", ""),
            description=data.get("description", ""),
            tools=data.get("tools", []),
            priority=data.get("priority", "medium"),
            depends_on=data.get("depends_on", []),
            status=DAGTaskStatus(data.get("status", "pending")),
            result=data.get("result"),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )

    def to_task_dict(self) -> Dict[str, Any]:
        """Convert to the flat task dict format used by _execute_single_task."""
        return {
            "id": self.task_id,
            "title": self.title,
            "description": self.description,
            "tools": self.tools,
            "priority": self.priority,
            **self.metadata,
        }


@dataclass
class DAGState:
    """Complete DAG execution state, serializable to/from dict for AgentState."""
    tasks: Dict[str, DAGTask] = field(default_factory=dict)
    execution_order: List[List[str]] = field(default_factory=list)
    current_wavefront: int = 0
    completed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    _dependents_cache: Optional[Dict[str, List[str]]] = field(default=None, repr=False)

    def get_dependents(self, task_id: str) -> List[str]:
        """Return task_ids that directly depend on the given task_id.

        Builds a reverse index on first access (O(n) once), then returns
        cached results in O(1) for subsequent calls.
        """
        if self._dependents_cache is None:
            # Build reverse index: task_id -> list of tasks that depend on it
            self._dependents_cache = {tid: [] for tid in self.tasks}
            for tid, task in self.tasks.items():
                for dep_id in task.depends_on:
                    if dep_id in self._dependents_cache:
                        self._dependents_cache[dep_id].append(tid)
        return self._dependents_cache.get(task_id, [])

    def invalidate_dependents_cache(self) -> None:
        """Invalidate the dependents cache (call when task graph changes)."""
        self._dependents_cache = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tasks": {tid: t.to_dict() for tid, t in self.tasks.items()},
            "execution_order": self.execution_order,
            "current_wavefront": self.current_wavefront,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "skipped_count": self.skipped_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DAGState":
        tasks = {
            tid: DAGTask.from_dict(tdata)
            for tid, tdata in data.get("tasks", {}).items()
        }
        return cls(
            tasks=tasks,
            execution_order=data.get("execution_order", []),
            current_wavefront=data.get("current_wavefront", 0),
            completed_count=data.get("completed_count", 0),
            failed_count=data.get("failed_count", 0),
            skipped_count=data.get("skipped_count", 0),
        )
