# isA Agent SDK

A complete AI Agent SDK for building intelligent agents with advanced features. Compatible with Claude Agent SDK patterns, with additional capabilities.

## Features

- **Claude Agent SDK Compatible** - Familiar API patterns
- **Streaming Messages** - Real-time response streaming
- **Built-in Tools** - Read, Write, Edit, Bash, WebSearch, etc.
- **MCP Integration** - Model Context Protocol support
- **Human-in-the-Loop** - Durable execution with checkpoints
- **Skills System** - Prompt injection and customization
- **Event Triggers** - Proactive agent activation
- **Multiple Execution Modes** - Reactive, Collaborative, Proactive

## Installation

```bash
pip install isa-agent-sdk

# With FastAPI server support
pip install isa-agent-sdk[server]
```

## Quick Start

### Basic Query

```python
from isa_agent_sdk import query, ISAAgentOptions

# Simple usage
async for msg in query("Hello, world!"):
    print(msg.content)

# With options
async for msg in query(
    "Fix the bug in auth.py",
    options=ISAAgentOptions(
        allowed_tools=["Read", "Edit", "Bash"],
        execution_mode="collaborative"
    )
):
    if msg.is_text:
        print(msg.content, end="")
    elif msg.is_tool_use:
        print(f"[Tool: {msg.tool_name}]")
```

### Human-in-the-Loop

```python
from isa_agent_sdk import request_tool_permission, checkpoint

# Request permission before dangerous operations
authorized = await request_tool_permission(
    "delete_file",
    {"path": "important_data.txt"}
)
if authorized:
    # proceed with deletion
    pass

# Create checkpoints for durable execution
await checkpoint("about_to_deploy", {
    "version": "1.0.0",
    "environment": "production"
})
```

### HTTP Client (for deployed apps)

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

### Building a FastAPI Agent Service

```python
from fastapi import FastAPI
from isa_agent_sdk import query, ISAAgentOptions

app = FastAPI()

@app.post("/query")
async def agent_query(prompt: str):
    responses = []
    async for msg in query(prompt):
        if msg.is_text:
            responses.append(msg.content)
    return {"response": "".join(responses)}
```

## Execution Modes

- **Reactive** - Responds to explicit requests
- **Collaborative** - Checkpoints for human approval
- **Proactive** - Anticipates needs, suggests actions

## License

MIT
