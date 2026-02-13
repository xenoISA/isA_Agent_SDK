"""
DAG scheduler - validates task graphs and computes wavefront execution order.

Uses Kahn's algorithm for topological sorting and cycle detection.
Wavefronts group independent tasks that can execute in parallel.
"""
from collections import deque
from typing import Dict, List, Set

from .models import DAGTask, DAGTaskStatus, DAGState


class DAGScheduler:
    """Validates DAG structure and computes wavefront execution order."""

    @staticmethod
    def build_dag(task_list: List[Dict]) -> DAGState:
        """Convert a flat task list (with optional depends_on) into a DAGState.

        Each task dict should have at least an 'id' (or 'task_id') and 'title'.
        """
        dag = DAGState()
        for i, task_data in enumerate(task_list):
            if not isinstance(task_data, dict):
                continue
            task_id = task_data.get("task_id") or task_data.get("id") or f"task_{i + 1}"

            # Separate known fields from metadata passthrough
            known_keys = {
                "task_id", "id", "title", "description", "tools",
                "priority", "depends_on", "status", "result", "error", "metadata",
            }
            metadata = {
                k: v for k, v in task_data.items() if k not in known_keys
            }
            # Merge explicit metadata if present
            metadata.update(task_data.get("metadata", {}))

            dag_task = DAGTask(
                task_id=task_id,
                title=task_data.get("title", f"Task {i + 1}"),
                description=task_data.get("description", task_data.get("title", "")),
                tools=task_data.get("tools", []),
                priority=task_data.get("priority", "medium"),
                depends_on=task_data.get("depends_on", []),
                status=DAGTaskStatus.PENDING,
                metadata=metadata,
            )
            dag.tasks[task_id] = dag_task
        return dag

    @staticmethod
    def validate(dag: DAGState) -> List[str]:
        """Return list of validation errors (empty means valid).

        Checks:
        - Self-dependencies
        - References to non-existent task IDs
        - Cycles (via Kahn's algorithm)
        """
        errors: List[str] = []
        all_ids = set(dag.tasks.keys())

        for tid, task in dag.tasks.items():
            # Self-dependency
            if tid in task.depends_on:
                errors.append(f"Task '{tid}' depends on itself")

            # Missing references
            for dep_id in task.depends_on:
                if dep_id not in all_ids:
                    errors.append(
                        f"Task '{tid}' depends on non-existent task '{dep_id}'"
                    )

        # Cycle detection via Kahn's algorithm
        if not errors:
            in_degree: Dict[str, int] = {tid: 0 for tid in all_ids}
            for task in dag.tasks.values():
                for dep_id in task.depends_on:
                    if dep_id in in_degree:
                        # dep_id -> task (task depends on dep_id)
                        # We track in-degree of each task
                        pass
            # Recompute: in-degree = number of dependencies for each task
            in_degree = {tid: len(dag.tasks[tid].depends_on) for tid in all_ids}

            queue = deque(tid for tid, deg in in_degree.items() if deg == 0)
            processed = 0

            # Build adjacency: task -> list of tasks that depend on it
            dependents: Dict[str, List[str]] = {tid: [] for tid in all_ids}
            for tid, task in dag.tasks.items():
                for dep_id in task.depends_on:
                    if dep_id in dependents:
                        dependents[dep_id].append(tid)

            while queue:
                node = queue.popleft()
                processed += 1
                for child in dependents[node]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)

            if processed != len(all_ids):
                errors.append(
                    f"Cycle detected: only {processed}/{len(all_ids)} tasks are acyclic"
                )

        return errors

    @staticmethod
    def compute_wavefronts(dag: DAGState) -> List[List[str]]:
        """Topological sort into wavefronts using Kahn's algorithm.

        Each wavefront is a list of task_ids whose dependencies are all
        in earlier wavefronts. Wavefronts execute sequentially; tasks
        within a wavefront can run in parallel.

        Raises ValueError if the DAG contains cycles.
        """
        all_ids = set(dag.tasks.keys())
        in_degree = {tid: len(dag.tasks[tid].depends_on) for tid in all_ids}

        # Build adjacency: parent -> children that depend on it
        dependents: Dict[str, List[str]] = {tid: [] for tid in all_ids}
        for tid, task in dag.tasks.items():
            for dep_id in task.depends_on:
                if dep_id in dependents:
                    dependents[dep_id].append(tid)

        # Seed with zero-indegree tasks
        current_wave = sorted(
            tid for tid, deg in in_degree.items() if deg == 0
        )
        wavefronts: List[List[str]] = []
        processed = 0

        while current_wave:
            wavefronts.append(current_wave)
            processed += len(current_wave)
            next_wave_set: Set[str] = set()

            for node in current_wave:
                for child in dependents[node]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        next_wave_set.add(child)

            current_wave = sorted(next_wave_set)

        if processed != len(all_ids):
            raise ValueError(
                f"Cycle detected: only {processed}/{len(all_ids)} tasks are acyclic"
            )

        return wavefronts

    @staticmethod
    def get_ready_tasks(dag: DAGState) -> List[str]:
        """Return task_ids that are PENDING and have all dependencies COMPLETED."""
        ready = []
        for tid, task in dag.tasks.items():
            if task.status not in (DAGTaskStatus.PENDING, DAGTaskStatus.READY):
                continue
            # Check all dependencies are completed
            deps_met = all(
                dag.tasks[dep_id].status == DAGTaskStatus.COMPLETED
                for dep_id in task.depends_on
                if dep_id in dag.tasks
            )
            if deps_met:
                ready.append(tid)
        return sorted(ready)

    @staticmethod
    def mark_task_completed(dag: DAGState, task_id: str, result: str) -> DAGState:
        """Mark a task as completed and update counters."""
        task = dag.tasks.get(task_id)
        if task:
            task.status = DAGTaskStatus.COMPLETED
            task.result = result
            dag.completed_count += 1
        return dag

    @staticmethod
    def mark_task_failed(dag: DAGState, task_id: str, error: str) -> DAGState:
        """Mark a task as failed and cascade SKIPPED to all transitive dependents."""
        task = dag.tasks.get(task_id)
        if not task:
            return dag

        task.status = DAGTaskStatus.FAILED
        task.error = error
        dag.failed_count += 1

        # Cascade: skip all transitive dependents
        to_skip = deque(_direct_dependents(dag, task_id))
        visited: Set[str] = set()

        while to_skip:
            dep_tid = to_skip.popleft()
            if dep_tid in visited:
                continue
            visited.add(dep_tid)

            dep_task = dag.tasks.get(dep_tid)
            if dep_task and dep_task.status in (
                DAGTaskStatus.PENDING,
                DAGTaskStatus.READY,
            ):
                dep_task.status = DAGTaskStatus.SKIPPED
                dep_task.error = f"Skipped: upstream task '{task_id}' failed"
                dag.skipped_count += 1
                # Continue cascading
                to_skip.extend(_direct_dependents(dag, dep_tid))

        return dag

    @staticmethod
    def is_dag_complete(dag: DAGState) -> bool:
        """True if no tasks are PENDING, READY, or RUNNING."""
        return all(
            task.status
            in (DAGTaskStatus.COMPLETED, DAGTaskStatus.FAILED, DAGTaskStatus.SKIPPED)
            for task in dag.tasks.values()
        )


def _direct_dependents(dag: DAGState, task_id: str) -> List[str]:
    """Return task_ids that directly depend on the given task_id."""
    return [
        tid
        for tid, task in dag.tasks.items()
        if task_id in task.depends_on
    ]
