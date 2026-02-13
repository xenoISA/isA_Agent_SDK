# Long-Running Tasks

This guide covers running agents that may take minutes to hours, and how to keep them reliable.

## 1) Use Durable Execution + Checkpointing

For long tasks, enable collaborative execution with checkpoints so work can resume after interruptions.

```python
from isa_agent_sdk import query, ISAAgentOptions, ExecutionMode

options = ISAAgentOptions(
    execution_mode=ExecutionMode.COLLABORATIVE,
    checkpoint_frequency=5,  # save every 5 steps
    session_id="long_task_001",
    user_id="user_123",
)

async for msg in query("Run a long workflow...", options=options):
    if msg.is_checkpoint:
        # Optionally show progress or persist state
        print(f"checkpoint: {msg.session_id}")
    elif msg.is_text:
        print(msg.content, end="")
```

To resume:

```python
from isa_agent_sdk import resume

async for msg in resume(session_id="long_task_001", resume_value={"continue": True}):
    print(msg.content, end="" if msg.is_text else "\n")
```

## 2) Use Background Jobs for Hours-Long Tasks

The SDK includes a background job system based on NATS JetStream + Redis.

Location:
- `isa_agent_sdk/services/background_jobs/`

Key components:
- `nats_task_queue.py` (queue)
- `redis_state_manager.py` (state + progress)
- `task_worker.py` (worker execution)

### Prerequisites

- NATS JetStream running (default port 4222)
- Redis running (default port 6379)
- `REDIS_PASSWORD` set if your Redis requires auth
- Worker process started

### Start a Worker

```bash
python -m isa_agent_sdk.services.background_jobs.task_worker --name worker-1
```

If your Redis requires auth:

```bash
export REDIS_PASSWORD=staging_redis_2024
python -m isa_agent_sdk.services.background_jobs.task_worker --name worker-1
```

If you want the worker to only process newly enqueued tasks:

```bash
python -m isa_agent_sdk.services.background_jobs.task_worker --name worker-1 --delivery-policy new
```

If you need a shared task namespace across enqueuers and workers:

```bash
export BACKGROUND_JOBS_USER_ID=agent-service
```

### Enqueue a Task

```python
from isa_agent_sdk.services.background_jobs import enqueue_task, TaskDefinition, ToolCallInfo

task = TaskDefinition(
    job_id="job_123",
    session_id="sess_456",
    user_id="user_789",
    tools=[
        ToolCallInfo(
            tool_name="web_crawl",
            tool_args={"url": "https://example.com"},
            tool_call_id="call_1",
        )
    ],
    priority="high",
)

sequence = await enqueue_task(task)
print(sequence)
```

### Poll Status / Result

```python
from isa_agent_sdk.services.background_jobs import get_task_status, get_task_result

status = await get_task_status("job_123")
print(status.status, status.progress_percent)

result = await get_task_result("job_123")
print(result.successful_tools, result.total_tools)
```

## 3) Environment Overrides (Single Source of Truth)

Models are configured in `ModelConfig.from_env()` and then reused across agent settings.

Important env vars:
- `AI_MODEL` / `DEFAULT_LLM`
- `AI_PROVIDER` / `DEFAULT_LLM_PROVIDER`
- `REASON_MODEL` / `REASON_LLM`
- `REASON_MODEL_PROVIDER` / `REASON_LLM_PROVIDER`
- `RESPONSE_MODEL` / `RESPONSE_LLM`
- `RESPONSE_MODEL_PROVIDER` / `RESPONSE_LLM_PROVIDER`
- `BACKGROUND_JOBS_USER_ID` (shared Redis/NATS namespace for background jobs)

ReasonNode uses `settings.reason_model`, so `REASON_MODEL` is the critical override for reasoning.

## 4) Recommended Practices

- Use explicit `session_id` for long tasks
- Use checkpointing for resumability
- Use background jobs for hours-long execution
- Use polling or SSE to surface progress

## 5) Troubleshooting

- If jobs never run: confirm NATS + Redis are healthy and the worker is running
- If tasks stay in QUEUED: check worker logs for NATS/Redis auth or Consul discovery errors
- If tasks stay in QUEUED and worker is running: ensure `BACKGROUND_JOBS_USER_ID` matches between enqueuer and worker
- If ReasonNode uses wrong model: check `REASON_MODEL` env var
- If responses are slow: check `RESPONSE_MODEL` env var
