#!/usr/bin/env python3
"""
Background Jobs Module - NATS + Redis based task queue

Complete background task processing system:
- NATS JetStream for task queue (Celery replacement)
- Redis for state management and pub/sub
- Worker pool for task execution
- Progress tracking and streaming updates

Usage:
    # Enqueue a task
    from isa_agent_sdk.services.background_jobs import enqueue_task, TaskDefinition, ToolCallInfo

    task = TaskDefinition(
        job_id="job_123",
        session_id="sess_456",
        tools=[
            ToolCallInfo(
                tool_name="web_crawl",
                tool_args={"url": "https://example.com"},
                tool_call_id="call_1"
            )
        ]
    )

    sequence = await enqueue_task(task)

    # Check task status
    from isa_agent_sdk.services.background_jobs import get_task_status

    progress = await get_task_status("job_123")
    print(f"Status: {progress.status}, Progress: {progress.progress_percent}%")

    # Get task result
    from isa_agent_sdk.services.background_jobs import get_task_result

    result = await get_task_result("job_123")
    print(f"Completed: {result.successful_tools}/{result.total_tools} tools")
"""

from .task_models import (
    TaskStatus,
    TaskDefinition,
    TaskProgress,
    TaskResult,
    ToolCallInfo,
    ToolExecutionResult,
    ProgressEvent
)

from .nats_task_queue import NATSTaskQueue, get_task_queue
from .redis_state_manager import RedisStateManager, get_state_manager
from .task_worker import TaskWorker

# ============================================
# High-level API Functions (All Async)
# ============================================


async def enqueue_task(task: TaskDefinition) -> int:
    """
    Enqueue a background task

    Args:
        task: Task definition

    Returns:
        NATS sequence number

    Example:
        >>> task = TaskDefinition(
        ...     job_id="job_123",
        ...     session_id="sess_456",
        ...     tools=[ToolCallInfo(...)]
        ... )
        >>> sequence = await enqueue_task(task)
    """
    queue = await get_task_queue()
    sequence = await queue.enqueue_task(task)

    # Increment counter
    state_mgr = await get_state_manager()
    await state_mgr.increment_task_counter("tasks_queued")

    # Store initial status
    progress = TaskProgress(
        job_id=task.job_id,
        status=TaskStatus.QUEUED,
        total_tools=len(task.tools),
        completed_tools=0,
        failed_tools=0,
        progress_percent=0.0
    )
    await state_mgr.store_task_status(task.job_id, progress)

    return sequence


async def get_task_status(job_id: str) -> TaskProgress:
    """
    Get task execution status

    Args:
        job_id: Job ID

    Returns:
        Task progress object

    Example:
        >>> progress = await get_task_status("job_123")
        >>> print(f"{progress.status}: {progress.progress_percent}%")
    """
    state_mgr = await get_state_manager()
    return await state_mgr.get_task_status(job_id)


async def get_task_result(job_id: str) -> TaskResult:
    """
    Get final task result

    Args:
        job_id: Job ID

    Returns:
        Task result object

    Example:
        >>> result = await get_task_result("job_123")
        >>> print(f"Success: {result.successful_tools}/{result.total_tools}")
    """
    state_mgr = await get_state_manager()
    return await state_mgr.get_task_result(job_id)


async def cancel_task(job_id: str):
    """
    Cancel a queued or running task

    Note: This marks the task as cancelled in Redis.
    Running tasks will complete their current tool but won't start new ones.

    Args:
        job_id: Job ID
    """
    state_mgr = await get_state_manager()
    progress = await state_mgr.get_task_status(job_id)

    if progress:
        progress.status = TaskStatus.CANCELLED
        await state_mgr.store_task_status(job_id, progress)

        # Publish cancellation event
        event = ProgressEvent(
            type="task_error",
            job_id=job_id,
            data={"reason": "cancelled_by_user"}
        )
        await state_mgr.publish_progress_event(event)


async def get_queue_statistics():
    """
    Get queue statistics

    Returns:
        Dictionary with queue stats

    Example:
        >>> stats = await get_queue_statistics()
        >>> print(f"Queued: {stats['tasks_queued']}")
        >>> print(f"Completed: {stats['tasks_completed']}")
    """
    state_mgr = await get_state_manager()
    return await state_mgr.get_task_statistics()


async def submit_tool_execution_task(
    task_data: dict,
    priority: str = "normal",
    max_retries: int = 2
) -> dict:
    """
    Submit a tool execution task to the background queue (used by ToolNode)

    Args:
        task_data: Dictionary with:
            - job_id: Unique job identifier
            - session_id: User session ID
            - user_id: User ID
            - tools: List of tool execution dicts
            - config: Serialized runtime config
        priority: Task priority (high/normal/low)
        max_retries: Maximum retry attempts

    Returns:
        dict with task_id and queue info

    Example (from ToolNode):
        >>> result = await submit_tool_execution_task(
        ...     task_data={
        ...         "job_id": "job_abc123",
        ...         "session_id": "sess_456",
        ...         "user_id": "user_789",
        ...         "tools": [
        ...             {
        ...                 "tool_name": "web_crawl",
        ...                 "tool_args": {"url": "https://example.com"},
        ...                 "tool_call_id": "call_1"
        ...             }
        ...         ],
        ...         "config": {"user_id": "user_789", "session_id": "sess_456"}
        ...     },
        ...     priority="high"
        ... )
        >>> print(result["task_id"])
    """
    import uuid

    # Generate task ID
    task_id = f"task_{uuid.uuid4().hex[:16]}"

    # Convert to TaskDefinition
    tool_calls = [
        ToolCallInfo(
            tool_name=tool["tool_name"],
            tool_args=tool["tool_args"],
            tool_call_id=tool["tool_call_id"]
        )
        for tool in task_data.get("tools", [])
    ]

    task = TaskDefinition(
        job_id=task_data["job_id"],
        session_id=task_data["session_id"],
        user_id=task_data.get("user_id", "unknown"),
        tools=tool_calls,
        priority=priority,
        max_retries=max_retries,
        metadata={
            "config": task_data.get("config", {}),
            "task_type": "tool_execution",
            "submitted_from": "tool_node"
        }
    )

    # Enqueue to NATS
    sequence = await enqueue_task(task)

    return {
        "task_id": task_id,
        "job_id": task_data["job_id"],
        "nats_sequence": sequence,
        "priority": priority,
        "queue": "nats_jetstream"
    }


# Export all public APIs
__all__ = [
    # Models
    "TaskStatus",
    "TaskDefinition",
    "TaskProgress",
    "TaskResult",
    "ToolCallInfo",
    "ToolExecutionResult",
    "ProgressEvent",

    # Core classes
    "NATSTaskQueue",
    "RedisStateManager",
    "TaskWorker",

    # Singleton getters
    "get_task_queue",
    "get_state_manager",

    # High-level API (all async)
    "enqueue_task",
    "get_task_status",
    "get_task_result",
    "cancel_task",
    "get_queue_statistics",
    "submit_tool_execution_task",  # For ToolNode integration
]
