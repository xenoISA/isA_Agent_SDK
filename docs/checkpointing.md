# Checkpointing & Durable Execution

Checkpointing enables agents to persist their state and resume execution after interruptions, process restarts, or failures.

## Overview

Durable execution provides:
- **State persistence** - Save execution state at checkpoints
- **Resume capability** - Continue from last checkpoint
- **Failure recovery** - Survive process crashes and restarts
- **Long-running tasks** - Handle tasks that span hours or days

## Checkpointer Backends

| Backend | Use Case | Persistence | Performance |
|---------|----------|-------------|-------------|
| `session_service` | Production (recommended) | Durable | High |
| `postgres` | Production (traditional) | Durable | High |
| `memory` | Development/testing | None | Fastest |

## Basic Usage

### Enable Durable Execution

```python
from isa_agent_sdk import query, ISAAgentOptions, ExecutionMode

options = ISAAgentOptions(
    execution_mode=ExecutionMode.COLLABORATIVE,
    checkpoint_frequency=5  # Checkpoint every 5 steps
)

async for msg in query("Long running task...", options=options):
    print(msg.content, end="" if msg.is_text else "\n")
```

### Session Management

```python
from isa_agent_sdk import query, ISAAgentOptions

# Start a new session with explicit ID
options = ISAAgentOptions(
    session_id="my-task-123",
    user_id="user-456"
)

async for msg in query("Start processing files...", options=options):
    if msg.is_checkpoint:
        print(f"Checkpoint saved: {msg.session_id}")
    elif msg.is_text:
        print(msg.content, end="")
```

### Resume Execution

```python
from isa_agent_sdk import resume

# Resume from a previous session
async for msg in resume(
    session_id="my-task-123",
    resume_value={"continue": True}  # Optional: provide input
):
    print(msg.content, end="" if msg.is_text else "\n")
```

### Synchronous Resume

```python
from isa_agent_sdk import resume_sync

result = resume_sync(
    session_id="my-task-123",
    resume_value={"authorized": True}
)
print(result.content)
```

## Checkpointer Configuration

### Using Session Service (Recommended)

```python
# Set via environment variable
# CHECKPOINTER_BACKEND=session_service

# Or configure in options
options = ISAAgentOptions(
    execution_mode=ExecutionMode.COLLABORATIVE
)
```

### Using PostgreSQL Directly

```python
# Set via environment variable
# CHECKPOINTER_BACKEND=postgres
# DATABASE_URL=postgresql://user:pass@host:5432/db
```

### Using Memory (Development)

```python
# Set via environment variable
# CHECKPOINTER_BACKEND=memory
```

## Programmatic Access

### Get the Checkpointer

```python
from isa_agent_sdk.services.persistence import (
    get_checkpointer,
    get_durable_service
)

# Get the durable service
durable = get_durable_service()

# Get the checkpointer instance
checkpointer = durable.get_checkpointer()
```

### Initialize Async Pool (FastAPI)

```python
from isa_agent_sdk.services.persistence import initialize_async_pool
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    # Initialize the connection pool at startup
    await initialize_async_pool()
    yield
    # Cleanup handled automatically
```

## Session Service Checkpointer

The `SessionServiceCheckpointer` integrates with the isA platform's session microservice:

```python
from isa_agent_sdk.services.persistence import SessionServiceCheckpointer

# Automatically used when backend is "session_service"
# Requires session_service_url in config
```

Features:
- Thread-safe async operations
- Automatic serialization
- Distributed session management
- Built-in retry logic

## Execution Manager

For advanced control over execution state:

```python
from isa_agent_sdk.services.persistence import ExecutionManager

manager = ExecutionManager()

# Check execution status
status = await manager.get_status("my-task-123")
print(f"Status: {status}")

# List pending executions
pending = await manager.list_pending(user_id="user-456")
for execution in pending:
    print(f"- {execution.session_id}: {execution.status}")
```

## Checkpoint Events in Streaming

```python
async for msg in query("Process all files", options=options):
    if msg.is_checkpoint:
        # Execution paused at checkpoint
        checkpoint_data = msg.metadata
        print(f"Checkpoint: {checkpoint_data}")

        # Respond to continue
        await msg.respond({"continue": True})

    elif msg.type == "progress":
        print(f"Progress: {msg.progress_percent}%")

    elif msg.is_text:
        print(msg.content, end="")
```

## Failure Recovery

### Automatic Retry

The SDK automatically handles transient failures:

```python
options = ISAAgentOptions(
    execution_mode=ExecutionMode.COLLABORATIVE,
    max_iterations=30  # Prevents infinite loops
)
```

### Manual Recovery

```python
from isa_agent_sdk import get_session_state, resume

# Check session state
state = await get_session_state("my-task-123")

if state and state.get("status") == "interrupted":
    # Resume from last checkpoint
    async for msg in resume("my-task-123"):
        print(msg.content)
```

## Integration with HIL

Checkpointing works seamlessly with Human-in-the-Loop:

```python
from isa_agent_sdk import query, request_tool_permission, ExecutionMode

options = ISAAgentOptions(
    execution_mode=ExecutionMode.COLLABORATIVE
)

async for msg in query("Delete old files", options=options):
    if msg.is_hil_request:
        # Execution paused, waiting for human input
        # State is saved - can resume even after restart

        tool = msg.metadata.get("tool_name")
        print(f"Permission requested for: {tool}")

        # User approves (could be hours later)
        await msg.respond({"authorized": True})

    elif msg.is_text:
        print(msg.content, end="")
```

## Best Practices

1. **Use explicit session IDs** for tasks you might need to resume
2. **Set appropriate checkpoint frequency** based on task duration
3. **Handle checkpoint events** in streaming for visibility
4. **Use COLLABORATIVE mode** for long-running tasks
5. **Configure production backend** (session_service or postgres)
6. **Implement cleanup** for completed sessions

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CHECKPOINTER_BACKEND` | Backend type | `session_service` |
| `DATABASE_URL` | PostgreSQL connection | - |
| `SESSION_SERVICE_URL` | Session service URL | From config |
| `ENVIRONMENT` | Environment name | `dev` |

## Next Steps

- [Human-in-the-Loop](./human-in-the-loop.md) - Combine with HIL
- [Options](./options.md) - Configure execution modes
- [Streaming](./streaming.md) - Handle checkpoint events
