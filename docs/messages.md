# Message Types

The `AgentMessage` class is the primary interface for receiving agent responses in streaming mode.

## Overview

Messages represent all types of events that occur during agent execution:
- Text content and thinking
- Tool calls and results
- Human-in-the-loop requests
- Progress updates
- Errors and system events

## AgentMessage

### Basic Properties

```python
from isa_agent_sdk import query

async for msg in query("Explain async programming"):
    print(f"Type: {msg.type}")
    print(f"Content: {msg.content}")
    print(f"Timestamp: {msg.timestamp}")
    print(f"Session: {msg.session_id}")
    print(f"Metadata: {msg.metadata}")
```

### Type Checking Properties

```python
async for msg in query("Write a function"):
    if msg.is_text:
        # Text content from the agent
        print(msg.content, end="")

    elif msg.is_thinking:
        # Internal reasoning (chain-of-thought)
        print(f"[Thinking: {msg.content}]")

    elif msg.is_tool_use:
        # Agent is calling a tool
        print(f"[Tool: {msg.tool_name}]")

    elif msg.is_tool_result:
        # Tool execution completed
        print(f"[Result: {msg.tool_result_value}]")

    elif msg.is_checkpoint:
        # Requires human input
        await msg.respond({"continue": True})

    elif msg.is_error:
        # Error occurred
        print(f"Error: {msg.content}")
```

## Message Types Reference

| Type | Property | Description |
|------|----------|-------------|
| `text` | `is_text` | Text content from agent |
| `thinking` | `is_thinking` | Chain-of-thought reasoning |
| `tool_use` | `is_tool_use` | Tool being called |
| `tool_result` | `is_tool_result` | Tool execution result |
| `checkpoint` | `is_checkpoint` | Requires human input |
| `hil_request` | `is_hil_request` | Human-in-the-loop request |
| `error` | `is_error` | Error message |
| `result` | `is_complete` | Final result |
| `progress` | - | Progress update |
| `session_start` | - | Session started |
| `session_end` | - | Session ended |

## Tool Information

### Tool Use Messages

```python
if msg.is_tool_use:
    print(f"Tool: {msg.tool_name}")
    print(f"Arguments: {msg.tool_args}")
    print(f"Tool ID: {msg.metadata.get('tool_use_id')}")
```

### Tool Result Messages

```python
if msg.is_tool_result:
    print(f"Tool: {msg.tool_name}")
    print(f"Result: {msg.tool_result_value}")

    if msg.tool_error:
        print(f"Error: {msg.tool_error}")
```

## Progress Information

```python
if msg.type == "progress":
    print(f"Progress: {msg.progress_percent}%")
    print(f"Step: {msg.progress_step}")
```

## Checkpoint/HIL Response

```python
if msg.is_checkpoint or msg.is_hil_request:
    question = msg.metadata.get("question")
    options = msg.metadata.get("options")
    hil_type = msg.metadata.get("hil_type")

    print(f"Question: {question}")

    # Respond to continue execution
    await msg.respond({
        "authorized": True,
        "input": "user provided data"
    })
```

## Creating Messages (Factory Methods)

### Text Message

```python
from isa_agent_sdk import AgentMessage

msg = AgentMessage.text("Hello, world!", session_id="session-123")
```

### Thinking Message

```python
msg = AgentMessage.thinking("Let me analyze this...", session_id="session-123")
```

### Tool Use Message

```python
msg = AgentMessage.tool_use(
    tool_name="web_search",
    args={"query": "Python tutorials"},
    session_id="session-123",
    tool_use_id="tool-456"
)
```

### Tool Result Message

```python
msg = AgentMessage.tool_result(
    tool_name="web_search",
    result={"urls": ["https://..."]},
    error=None,
    session_id="session-123",
    tool_use_id="tool-456"
)
```

### Error Message

```python
msg = AgentMessage.error(
    "Connection timeout",
    error_type="network",
    session_id="session-123"
)
```

### Progress Message

```python
msg = AgentMessage.progress(
    step="Processing file 3 of 10",
    percent=30.0,
    session_id="session-123"
)
```

### HIL Request Message

```python
msg = AgentMessage.hil_request(
    question="Delete these files?",
    request_type="authorization",
    options=["Yes", "No"],
    session_id="session-123"
)
```

## ConversationHistory

Collect and manage multiple messages:

```python
from isa_agent_sdk import ConversationHistory

history = ConversationHistory(session_id="session-123")

# Add messages
history.add_user_message("Write a function")
history.add_assistant_message("Here's a function...")

# Access messages
print(f"Total messages: {len(history.messages)}")
print(f"Last message: {history.last_message}")

# Get specific content
print(f"All text: {history.get_text_content()}")
print(f"Tool calls: {history.get_tool_calls()}")
print(f"Thinking: {history.get_thinking()}")

# Check completion
print(f"Is complete: {history.is_complete}")
```

## Converting from LangChain Messages

```python
from langchain_core.messages import AIMessage, ToolMessage
from isa_agent_sdk import AgentMessage

# From AIMessage
ai_msg = AIMessage(content="Hello!")
agent_msg = AgentMessage.from_langchain_message(ai_msg)

# From ToolMessage
tool_msg = ToolMessage(content="Result", name="search")
agent_msg = AgentMessage.from_langchain_message(tool_msg)
```

## Converting to isA Event Data

```python
# Get underlying isA EventData
event_data = msg.to_event_data()
print(f"Event type: {event_data.type}")

# Get isA EventType enum
event_type = msg.event_type
print(f"Event type enum: {event_type}")
```

## Claude SDK Compatible Types

For compatibility with Claude Agent SDK patterns:

### TextBlock

```python
from isa_agent_sdk._messages import TextBlock

block = TextBlock(text="Hello, world!")
```

### ToolUseBlock

```python
from isa_agent_sdk._messages import ToolUseBlock

block = ToolUseBlock(
    id="tool-123",
    name="web_search",
    input={"query": "test"}
)
```

### ToolResultBlock

```python
from isa_agent_sdk._messages import ToolResultBlock

block = ToolResultBlock(
    tool_use_id="tool-123",
    content="Search results...",
    is_error=False
)
```

### AssistantMessage

```python
from isa_agent_sdk._messages import AssistantMessage

# Create from AgentMessage
assistant_msg = AssistantMessage.from_agent_message(agent_msg)
print(f"Content blocks: {assistant_msg.content}")
```

### ResultMessage

```python
from isa_agent_sdk._messages import ResultMessage

result = ResultMessage(
    subtype="success",
    duration_ms=1500,
    num_turns=3,
    total_cost_usd=0.02,
    result="Task completed"
)
```

## Streaming Helper

Check if a message is part of streaming content:

```python
if msg.is_streaming:
    # Streaming content (text or thinking)
    print(msg.content, end="", flush=True)
else:
    # Complete message
    print(msg.content)
```

## Node Information

For debugging graph execution:

```python
if msg.type in ("node_enter", "node_exit"):
    print(f"Node: {msg.node_name}")
```

## Skill Information

When skills are activated:

```python
if msg.skill_name:
    print(f"Skill: {msg.skill_name}")
```

## Intent Classification

From the Sense node:

```python
if msg.intent:
    print(f"Intent: {msg.intent}")

if msg.is_simple_intent:
    print("Simple query - no complex processing needed")
```

## Best Practices

1. **Use type checking properties** (`is_text`, `is_tool_use`) for clarity
2. **Handle all message types** in streaming for complete visibility
3. **Respond to checkpoints** promptly for durable execution
4. **Check tool_error** in tool_result messages
5. **Use ConversationHistory** for multi-turn conversations

## Next Steps

- [Streaming](./streaming.md) - Handle messages in streams
- [Human-in-the-Loop](./human-in-the-loop.md) - Checkpoint responses
- [Tools](./tools.md) - Tool call messages
