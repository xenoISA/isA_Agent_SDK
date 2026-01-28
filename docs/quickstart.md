# Quick Start

Get started with the isA Agent SDK in minutes.

## Installation

```bash
pip install isa-agent-sdk
```

## Basic Usage

### Simple Query

The simplest way to use the SDK:

```python
from isa_agent_sdk import ask

# Get a direct answer
response = await ask("What is the capital of France?")
print(response)  # "Paris"
```

### Streaming Response

For real-time streaming responses:

```python
from isa_agent_sdk import query, ISAAgentOptions

options = ISAAgentOptions(
    model="gpt-4.1-nano",
    allowed_tools=["web_search", "read_file"]
)

async for msg in query("Explain quantum computing", options=options):
    if msg.is_text:
        print(msg.content, end="", flush=True)
    elif msg.is_tool_use:
        print(f"\n[Using tool: {msg.tool_name}]")
    elif msg.is_complete:
        print("\n[Done]")
```

### Synchronous Usage

For non-async contexts:

```python
from isa_agent_sdk import query_sync, ask_sync

# Simple sync query
answer = ask_sync("What is 2+2?")
print(answer)  # "4"

# Streaming sync query
for msg in query_sync("List 3 programming languages"):
    if msg.is_text:
        print(msg.content, end="")
```

## Configuration

### Using Options

```python
from isa_agent_sdk import query, ISAAgentOptions, ExecutionMode

options = ISAAgentOptions(
    # Model settings
    model="gpt-4.1-nano",

    # Tool access
    allowed_tools=["web_search", "read_file", "write_file"],

    # Execution mode
    execution_mode=ExecutionMode.COLLABORATIVE,

    # Session management
    session_id="my-session-123",
    user_id="user-456",

    # Safety settings
    guardrails_enabled=True,
    max_iterations=30
)

async for msg in query("Help me debug this code", options=options):
    print(msg.content, end="" if msg.is_text else "\n")
```

### Loading from Config File

```python
from isa_agent_sdk import ISAAgentOptions

# Load from YAML config
options = ISAAgentOptions.from_file("agent_config.yaml")
```

Example `agent_config.yaml`:

```yaml
model: gpt-4.1-nano
allowed_tools:
  - web_search
  - read_file
  - write_file
execution_mode: collaborative
guardrails_enabled: true
max_iterations: 30
```

## Tool Execution

### Execute a Single Tool

```python
from isa_agent_sdk import execute_tool

result = await execute_tool(
    tool_name="web_search",
    tool_args={"query": "Python best practices 2024"},
    session_id="my-session"
)

print(result.content)
```

### Get Available Tools

```python
from isa_agent_sdk import get_available_tools

# List all tools
tools = await get_available_tools()
for tool in tools[:5]:
    print(f"- {tool['name']}: {tool['description']}")

# Semantic search for relevant tools
relevant_tools = await get_available_tools(
    user_query="I need to search the web",
    max_results=5
)
```

## Session Management

### Get Session State

```python
from isa_agent_sdk import get_session_state

state = await get_session_state("my-session-123")
print(f"Messages: {state['messages_count']}")
print(f"Summary: {state['summary']}")
print(f"Next action: {state['next_action']}")
```

### Resume a Session

```python
from isa_agent_sdk import resume

# Resume from a checkpoint (e.g., after HIL approval)
async for msg in resume(
    session_id="my-session-123",
    resume_value={"authorized": True}
):
    print(msg.content, end="" if msg.is_text else "\n")
```

## Error Handling

```python
from isa_agent_sdk import query, ISAAgentOptions

try:
    async for msg in query("Do something risky"):
        if msg.is_error:
            print(f"Error: {msg.content}")
            break
        print(msg.content, end="" if msg.is_text else "\n")
except Exception as e:
    print(f"Query failed: {e}")
```

## Next Steps

- [Streaming vs Single Mode](./streaming.md) - Understanding response modes
- [Configuration Options](./options.md) - Full options reference
- [Human-in-the-Loop](./human-in-the-loop.md) - Approval workflows
- [Tools & MCP](./tools.md) - Tool integration guide
