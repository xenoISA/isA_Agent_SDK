# isA Agent SDK

Build intelligent AI agents with advanced features including streaming, tools, human-in-the-loop, and durable execution.

## Installation

```bash
pip install isa-agent-sdk
```

## Quick Start

```python
from isa_agent_sdk import query

async for msg in query("Hello, world!"):
    print(msg.content, end="" if msg.is_text else "\n")
```

## Core Concepts

### Streaming vs Single Mode

The SDK supports both streaming and single-turn interactions:

```python
# Streaming - get messages as they arrive
async for msg in query("Explain AI"):
    if msg.is_text:
        print(msg.content, end="")
    elif msg.is_tool_use:
        print(f"[Using: {msg.tool_name}]")

# Single mode - get complete response
from isa_agent_sdk import ask

result = await ask("What is 2+2?")
print(result.content)
```

[Learn more about streaming](./streaming.md)

### Configuration Options

Control agent behavior with `ISAAgentOptions`:

```python
from isa_agent_sdk import query, ISAAgentOptions

options = ISAAgentOptions(
    model="gpt-4.1-nano",
    allowed_tools=["web_search", "read_file"],
    max_iterations=30,
    skills=["code-review", "debug"]
)

async for msg in query("Review this code", options=options):
    print(msg.content, end="" if msg.is_text else "\n")
```

[Full options reference](./options.md)

### Tools & MCP Integration

Access a wide range of tools via the Model Context Protocol:

```python
from isa_agent_sdk import execute_tool, get_available_tools

# Discover tools
tools = await get_available_tools(user_query="search the web")

# Execute directly
result = await execute_tool("web_search", {"query": "Python tutorials"})
```

[Tools documentation](./tools.md)

### Human-in-the-Loop

Enable agents to pause and request human approval:

```python
from isa_agent_sdk import request_tool_permission

authorized = await request_tool_permission(
    tool_name="delete_file",
    tool_args={"path": "/data/file.txt"},
    reason="Cleaning up old data"
)

if authorized:
    # proceed
```

[HIL documentation](./human-in-the-loop.md)

### Skills System

Activate specialized behaviors via prompt injection:

```python
options = ISAAgentOptions(
    skills=["code-review", "debug", "refactor"]
)
```

Built-in skills:
- `code-review` - Expert code reviewer
- `debug` - Systematic debugger
- `refactor` - Refactoring specialist
- `test-writer` - Test coverage expert
- `documentation` - Technical writer

[Skills documentation](./skills.md)

### Event Triggers

Enable proactive agent activation:

```python
from isa_agent_sdk import register_trigger, TriggerType

trigger_id = await register_trigger(
    user_id="user-123",
    trigger_type=TriggerType.THRESHOLD,
    description="Alert on price drop",
    conditions={"threshold_value": 5.0, "direction": "down"},
    action_config={"prompt": "Analyze the drop"}
)
```

[Triggers documentation](./triggers.md)

### Structured Outputs

Get validated JSON matching your schema:

```python
from pydantic import BaseModel
from isa_agent_sdk import query, ISAAgentOptions, OutputFormat

class Recipe(BaseModel):
    name: str
    ingredients: list[str]
    prep_time_minutes: int

async for msg in query(
    "Find a chocolate chip cookie recipe",
    options=ISAAgentOptions(
        output_format=OutputFormat.from_pydantic(Recipe)
    )
):
    if msg.has_structured_output:
        recipe = msg.parse(Recipe)
        print(f"Recipe: {recipe.name}")
```

[Structured outputs documentation](./structured-outputs.md)

### Durable Execution

Checkpoint and resume long-running tasks:

```python
from isa_agent_sdk import query, resume, ExecutionMode

options = ISAAgentOptions(
    execution_mode=ExecutionMode.COLLABORATIVE,
    session_id="my-task-123"
)

# Later, resume from checkpoint
async for msg in resume("my-task-123"):
    print(msg.content)
```

[Checkpointing documentation](./checkpointing.md)

### Deploy to isA Cloud

Deploy your custom agents to production:

```bash
# 1. Create deployment
curl -X POST http://localhost:8095/api/v1/deployments \
  -H "X-User-ID: user123" \
  -d '{"app_name": "my-agent"}'

# 2. Upload code (auto-builds)
tar -czf my-agent.tar.gz -C my-agent .
curl -X POST http://localhost:8095/api/v1/deployments/{id}/upload \
  -F "file=@my-agent.tar.gz"

# 3. Deploy version
curl -X POST http://localhost:8095/api/v1/deployments/{id}/deploy \
  -d '{"version": "v1.0.0", "replicas": 2}'

# 4. Query your deployed agent
curl -X POST http://localhost:8095/api/v1/apps/{id}/query \
  -d '{"prompt": "Hello from the cloud!"}'
```

[Deployment Guide](./deployment-guide.md)

## Documentation

### Getting Started
- [Quick Start](./quickstart.md) - First steps with the SDK
- [Streaming](./streaming.md) - Streaming vs single mode

### Configuration
- [Options](./options.md) - Full configuration reference
- [Skills](./skills.md) - Skill system guide

### Features
- [Tools](./tools.md) - Tools & MCP integration
- [Human-in-the-Loop](./human-in-the-loop.md) - Approval workflows
- [Structured Outputs](./structured-outputs.md) - JSON schema & Pydantic integration
- [Triggers](./triggers.md) - Event-based activation
- [Checkpointing](./checkpointing.md) - Durable execution
- [Messages](./messages.md) - Message types reference

### Deployment
- [Deployment Guide](./deployment-guide.md) - Deploy agents to isA Cloud

### Testing
- [Testing & Verification](./testing.md) - Verified E2E test results

## API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `query(prompt, options)` | Stream agent responses |
| `ask(prompt, options)` | Get single response |
| `resume(session_id)` | Resume from checkpoint |
| `execute_tool(name, args)` | Execute tool directly |
| `get_available_tools()` | List available tools |

### Message Types

| Type | Description |
|------|-------------|
| `text` | Text content |
| `thinking` | Chain-of-thought |
| `tool_use` | Tool being called |
| `tool_result` | Tool result |
| `checkpoint` | Requires input |
| `error` | Error message |

### Execution Modes

| Mode | Description |
|------|-------------|
| `REACTIVE` | Standard request-response |
| `COLLABORATIVE` | Durable with checkpoints |
| `PROACTIVE` | Event-driven autonomous |

## HTTP Client

For deployed applications:

```python
from isa_agent_sdk import ISAAgent

client = ISAAgent(base_url="http://localhost:8000")

# Simple chat
response = client.chat.create(message="Hello!", user_id="user123")

# Streaming
for event in client.chat.stream(message="Explain AI"):
    if event.is_content:
        print(event.content, end="")
```

## Examples

### Code Review Agent

```python
from isa_agent_sdk import query, ISAAgentOptions

options = ISAAgentOptions(
    skills=["code-review"],
    allowed_tools=["read_file"]
)

async for msg in query("Review main.py for issues", options=options):
    print(msg.content, end="" if msg.is_text else "\n")
```

### Research Agent

```python
options = ISAAgentOptions(
    allowed_tools=["web_search", "fetch_url"],
    max_iterations=50
)

async for msg in query("Research latest AI developments", options=options):
    if msg.is_tool_use:
        print(f"[Searching: {msg.tool_args}]")
    elif msg.is_text:
        print(msg.content, end="")
```

### File Processing Agent

```python
options = ISAAgentOptions(
    execution_mode=ExecutionMode.COLLABORATIVE,
    allowed_tools=["read_file", "write_file", "bash"]
)

async for msg in query("Process all CSV files in /data", options=options):
    if msg.is_checkpoint:
        await msg.respond({"continue": True})
    elif msg.is_text:
        print(msg.content, end="")
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ISA_API_KEY` | API key for isA platform |
| `ISA_MODEL_URL` | Model service URL |
| `ISA_MCP_URL` | MCP service URL |
| `CHECKPOINTER_BACKEND` | Checkpointer backend |

## Support

- Report issues at the project repository
- Check documentation for troubleshooting guides
