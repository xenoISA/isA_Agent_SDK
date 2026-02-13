"""
Task DAG (Directed Acyclic Graph) system for dependency-aware task execution.
"""

__all__ = ["DAGTask", "DAGTaskStatus", "DAGState", "DAGScheduler"]


def __getattr__(name):
    if name in ("DAGTask", "DAGTaskStatus", "DAGState"):
        from .models import DAGTask, DAGTaskStatus, DAGState
        mapping = {
            "DAGTask": DAGTask,
            "DAGTaskStatus": DAGTaskStatus,
            "DAGState": DAGState,
        }
        return mapping[name]
    if name == "DAGScheduler":
        from .scheduler import DAGScheduler
        return DAGScheduler
    raise AttributeError(f"module 'isa_agent_sdk.dag' has no attribute '{name}'")
