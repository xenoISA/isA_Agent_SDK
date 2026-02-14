# Quickstart

Get started with isA Agent SDK in minutes.

## Installation

```bash
pip install isa-agent-sdk

# With FastAPI server support
pip install isa-agent-sdk[server]
```

## Basic Query

The simplest way to use the SDK is with the `query()` function:

```python
from isa_agent_sdk import query

# Simple usage
async for msg in query("Hello, world!"):
    if msg.is_text:
        print(msg.content, end="")
```

## With Options

Configure agent behavior with `ISAAgentOptions`:

```python
from isa_agent_sdk import query, ISAAgentOptions

async for msg in query(
    "Fix the bug in auth.py",
    options=ISAAgentOptions(
        allowed_tools=["Read", "Edit", "Bash"],
        execution_mode="collaborative",
        max_iterations=20,
    )
):
    if msg.is_text:
        print(msg.content, end="")
    elif msg.is_tool_use:
        print(f"\n[Tool: {msg.tool_name}]")
```

## Streaming Messages

Handle different message types during streaming:

```python
from isa_agent_sdk import query

async for msg in query("Analyze the codebase"):
    if msg.is_text:
        print(msg.content, end="")
    elif msg.is_tool_use:
        print(f"\n[Using {msg.tool_name}: {msg.tool_input}]")
    elif msg.is_tool_result:
        print(f"[Result: {msg.content[:100]}...]")
    elif msg.is_error:
        print(f"Error: {msg.content}")
```

## Non-Streaming (Sync)

For synchronous contexts:

```python
from isa_agent_sdk import query_sync

# Returns a generator
for msg in query_sync("What's in the README?"):
    if msg.is_text:
        print(msg.content, end="")
```

**Note**: `query_sync()` cannot be called from async contexts. Use `query()` instead.

## Human-in-the-Loop

Request permission before dangerous operations:

```python
from isa_agent_sdk import request_tool_permission

authorized = await request_tool_permission(
    "delete_file",
    {"path": "important_data.txt"}
)
if authorized:
    # proceed with deletion
    pass
```

Create checkpoints for durable execution:

```python
from isa_agent_sdk import checkpoint

await checkpoint("about_to_deploy", {
    "version": "1.0.0",
    "environment": "production"
})
```

## Multi-Agent Swarm

For dynamic multi-agent workflows:

```python
from isa_agent_sdk import Agent, ISAAgentOptions
from isa_agent_sdk.agents import SwarmOrchestrator, SwarmAgent

researcher = SwarmAgent(
    agent=Agent("researcher", ISAAgentOptions(
        system_prompt="You gather facts. Always hand off to writer when done.",
    )),
    description="Research and information gathering",
)

writer = SwarmAgent(
    agent=Agent("writer", ISAAgentOptions(
        system_prompt="You write polished summaries from research.",
    )),
    description="Writing and documentation",
)

swarm = SwarmOrchestrator(
    agents=[researcher, writer],
    entry_agent="researcher",
    max_handoffs=5,
)

result = await swarm.run("Research Python and write a summary")
print(result.text)
print(f"Final agent: {result.final_agent}")
print(f"Handoffs: {result.handoff_trace}")
```

## DAG Task Execution

For dependency-ordered workflows:

```python
from isa_agent_sdk.agents import SwarmOrchestrator

result = await swarm.run_dag([
    {
        "id": "research",
        "title": "Research Python",
        "description": "Gather key facts about Python.",
        "agent": "researcher",
    },
    {
        "id": "write",
        "title": "Write Summary",
        "description": "Write a summary based on research.",
        "agent": "writer",
        "depends_on": ["research"],  # Runs after research completes
    },
])

# Access per-task results
print(result.agent_outputs["researcher:research"].text)
print(result.agent_outputs["writer:write"].text)
```

## HTTP Client (for deployed apps)

Connect to a deployed agent service:

```python
from isa_agent_sdk import ISAAgent

client = ISAAgent(base_url="http://localhost:8000")

# Non-streaming
response = client.chat.create(
    message="Explain quantum computing",
    user_id="user123"
)
print(response.content)

# Streaming
for event in client.chat.stream(
    message="Write a Python function",
    user_id="user123"
):
    if event.is_content:
        print(event.content, end="")
```

## Next Steps

- [Core Concepts](./concepts.md) - Understand the architecture
- [Configuration](./configuration.md) - Configure agent behavior
- [Multi-Agent](./multi-agent.md) - Build multi-agent systems
- [Swarm Orchestration](./swarm.md) - Dynamic agent handoffs and DAG execution
- [Examples](./examples.md) - More code examples
- [API Reference](./api-reference.md) - Complete API documentation
