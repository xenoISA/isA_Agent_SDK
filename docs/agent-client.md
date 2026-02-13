# Agent Client

The SDK provides two client interfaces for agent interactions:

1. **`ISAAgentClient`** - Bidirectional client for interactive conversations (Claude SDK compatible)
2. **`ISAAgent`** - HTTP client for calling deployed agents

## ISAAgentClient (Recommended)

The bidirectional client provides a Claude SDK-compatible interface with `query()` and `receive()` methods.

### Import

```python
from isa_agent_sdk import ISAAgentClient, ISAAgentClientSync, ISAAgentOptions
```

### Basic Usage

```python
async with ISAAgentClient() as client:
    await client.query("What files are in this directory?")
    async for msg in client.receive():
        if msg.is_text:
            print(msg.content, end="")
```

### Multi-turn Conversation

```python
async with ISAAgentClient() as client:
    # First turn
    await client.query("Remember this number: 42")
    async for msg in client.receive():
        print(msg.content)

    # Continue same session
    await client.query("What number did I ask you to remember?")
    async for msg in client.receive():
        print(msg.content)  # Will mention 42
```

### Convenience Method: ask()

```python
async with ISAAgentClient() as client:
    response = await client.ask("What is 2 + 2?")
    print(response)  # "4"
```

### With Options

```python
options = ISAAgentOptions(
    model="deepseek-reasoner",
    allowed_tools=["Read", "Edit", "Bash"],
    max_iterations=20
)

async with ISAAgentClient(options=options) as client:
    await client.query("Fix the bug in main.py")
    async for msg in client.receive():
        if msg.is_tool_use:
            print(f"[Using: {msg.tool_name}]")
        elif msg.is_text:
            print(msg.content, end="")
```

### Session Forking

Fork a session to explore alternative approaches:

```python
async with ISAAgentClient() as client:
    await client.query("Analyze this code")
    async for msg in client.receive():
        print(msg.content)

    # Fork to try different approach
    forked = client.fork()
    async with forked:
        await forked.query("Try a different refactoring approach")
        async for msg in forked.receive():
            print(msg.content)

    # Original client still has original conversation
```

### Remote API Mode

Connect to a deployed agent service:

```python
async with ISAAgentClient(
    base_url="http://api.example.com",
    api_key="your-api-key"
) as client:
    await client.query("Hello!")
    async for msg in client.receive():
        print(msg.content)
```

### Session Info

```python
async with ISAAgentClient() as client:
    info = client.get_session_info()
    print(f"Session: {info['session_id']}")
    print(f"Messages: {info['message_count']}")
    print(f"Mode: {info['mode']}")  # 'local' or 'remote'
```

### Synchronous Usage

For non-async contexts:

```python
from isa_agent_sdk import ISAAgentClientSync

with ISAAgentClientSync() as client:
    client.query("Hello!")
    for msg in client.receive():
        print(msg.content)
```

---

## ISAAgent (HTTP Client)

Lightweight HTTP client for calling deployed agents. Use when your agent runs as a service.

### Import

```python
from isa_agent_sdk import ISAAgent, ISAAgentSync
```

### Non-Streaming Request

```python
client = ISAAgent(base_url="http://localhost:8000")

response = client.chat.create(
    message="Summarize the meeting notes",
    user_id="user-123"
)

print(response.content)
```

### Streaming Request

```python
client = ISAAgent(base_url="http://localhost:8000")

for event in client.chat.stream(
    message="Draft a product launch plan",
    user_id="user-123"
):
    if event.is_content:
        print(event.content, end="")
```

### Session Continuation

```python
session_id = "creative-session-001"

response = client.chat.create(
    message="Create a 30s storyboard",
    user_id="user-123",
    session_id=session_id,
)

followup = client.chat.create(
    message="Now generate voiceover script",
    user_id="user-123",
    session_id=session_id,
)
```

---

## When to Use Which

| Use Case | Client |
|----------|--------|
| Interactive development | `ISAAgentClient` |
| Multi-turn conversations | `ISAAgentClient` |
| Claude SDK compatibility | `ISAAgentClient` |
| Deployed agent service | `ISAAgent` |
| Simple HTTP integration | `ISAAgent` |

---

## Error Handling

Both clients raise typed errors:

```python
from isa_agent_sdk import (
    ISASDKError,
    ConnectionError,
    SessionError,
    ToolExecutionError
)

try:
    async with ISAAgentClient() as client:
        await client.query("Do something")
        async for msg in client.receive():
            ...
except ConnectionError as e:
    print(f"Connection failed: {e.service} at {e.url}")
except SessionError as e:
    print(f"Session error: {e.session_id}")
except ISASDKError as e:
    print(f"SDK error: {e}")
```

See [Error Classes](#error-classes) for the full hierarchy.

---

## Error Classes

The SDK provides a complete error hierarchy:

```python
from isa_agent_sdk import (
    # Base
    ISASDKError,

    # Connection & Infrastructure
    ConnectionError,
    TimeoutError,
    CircuitBreakerError,
    RateLimitError,

    # Execution
    ExecutionError,
    ToolExecutionError,
    ModelError,
    MaxIterationsError,

    # Session & State
    SessionError,
    SessionNotFoundError,
    SessionExpiredError,
    CheckpointError,

    # Validation
    ValidationError,
    SchemaError,
    ConfigurationError,

    # Permission
    PermissionError,
    ToolPermissionError,
    HILDeniedError,

    # MCP
    MCPError,
    MCPConnectionError,
    MCPToolNotFoundError,
)
```

All errors include rich details:

```python
try:
    ...
except ToolExecutionError as e:
    print(e.message)      # Error message
    print(e.tool_name)    # Which tool failed
    print(e.tool_args)    # Arguments passed
    print(e.session_id)   # Session context
    print(e.details)      # Full details dict
```
