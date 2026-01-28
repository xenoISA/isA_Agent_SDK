# Streaming vs Single Mode

The isA Agent SDK supports both streaming and single-response modes for different use cases.

## Streaming Mode

Streaming mode provides real-time updates as the agent processes your request. Use this for:
- Interactive UIs with live feedback
- Long-running tasks with progress updates
- Tool execution visibility
- Human-in-the-loop interactions

### Basic Streaming

```python
from isa_agent_sdk import query

async for msg in query("Analyze this codebase and suggest improvements"):
    if msg.is_text:
        # Stream text tokens as they arrive
        print(msg.content, end="", flush=True)
    elif msg.is_thinking:
        # Agent's reasoning process
        print(f"\n[Thinking: {msg.content}]")
    elif msg.is_tool_use:
        # Tool being called
        print(f"\n[Calling: {msg.tool_name}({msg.tool_args})]")
    elif msg.is_tool_result:
        # Tool result received
        print(f"\n[Result: {msg.tool_result_value}]")
    elif msg.is_complete:
        # Final response
        print(f"\n\n=== Complete ===")
```

### Message Types

| Type | Property | Description |
|------|----------|-------------|
| `text` | `msg.is_text` | Streaming text content |
| `thinking` | `msg.is_thinking` | Agent reasoning |
| `tool_use` | `msg.is_tool_use` | Tool invocation |
| `tool_result` | `msg.is_tool_result` | Tool response |
| `result` | `msg.is_complete` | Final complete response |
| `error` | `msg.is_error` | Error occurred |
| `checkpoint` | `msg.is_checkpoint` | HIL checkpoint |
| `progress` | `msg.type == "progress"` | Progress update |
| `session_start` | `msg.type == "session_start"` | Session began |
| `session_end` | `msg.type == "session_end"` | Session ended |

### Progress Tracking

```python
async for msg in query("Process all files in /src"):
    if msg.type == "progress":
        print(f"Progress: {msg.progress_percent:.1f}% - {msg.progress_step}")
    elif msg.is_text:
        print(msg.content, end="")
```

## Single Mode (Non-Streaming)

Single mode waits for the complete response. Use this for:
- Simple Q&A interactions
- Backend processing
- When streaming isn't needed

### Using `ask()`

```python
from isa_agent_sdk import ask

# Returns the final text response directly
answer = await ask("What is the capital of Japan?")
print(answer)  # "Tokyo"
```

### Sync Version

```python
from isa_agent_sdk import ask_sync

answer = ask_sync("Explain recursion in one sentence")
print(answer)
```

### Collecting Full Response from Stream

```python
from isa_agent_sdk import query

parts = []
async for msg in query("Write a haiku about coding"):
    if msg.is_text or msg.is_complete:
        parts.append(msg.content or "")

full_response = "".join(parts)
print(full_response)
```

## Handling Different Event Types

### Complete Event Handler

```python
from isa_agent_sdk import query, AgentMessage

async def handle_message(msg: AgentMessage):
    match msg.type:
        case "text":
            return {"action": "append", "content": msg.content}
        case "thinking":
            return {"action": "thinking", "content": msg.content}
        case "tool_use":
            return {
                "action": "tool_start",
                "tool": msg.tool_name,
                "args": msg.tool_args
            }
        case "tool_result":
            return {
                "action": "tool_end",
                "tool": msg.tool_name,
                "result": msg.tool_result_value,
                "error": msg.tool_error
            }
        case "checkpoint" | "hil_request":
            return {
                "action": "pause",
                "question": msg.metadata.get("question"),
                "options": msg.metadata.get("options")
            }
        case "error":
            return {"action": "error", "message": msg.content}
        case "result":
            return {"action": "complete", "content": msg.content}
        case _:
            return {"action": "ignore"}

async for msg in query("Help me refactor this code"):
    result = await handle_message(msg)
    print(result)
```

### Tool Execution Visibility

```python
from isa_agent_sdk import query

tools_used = []

async for msg in query("Search for Python tutorials and summarize"):
    if msg.is_tool_use:
        print(f"Starting: {msg.tool_name}")
        tools_used.append({
            "name": msg.tool_name,
            "args": msg.tool_args,
            "id": msg.metadata.get("tool_use_id")
        })
    elif msg.is_tool_result:
        print(f"Completed: {msg.tool_name}")
        # Find matching tool and add result
        for tool in tools_used:
            if tool["id"] == msg.metadata.get("tool_use_id"):
                tool["result"] = msg.tool_result_value
                tool["error"] = msg.tool_error
    elif msg.is_text:
        print(msg.content, end="")

print(f"\n\nTools used: {len(tools_used)}")
for tool in tools_used:
    print(f"  - {tool['name']}: {'success' if not tool.get('error') else 'failed'}")
```

## Server-Sent Events (SSE) for Web

### FastAPI SSE Endpoint

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from isa_agent_sdk import query
import json

app = FastAPI()

@app.post("/chat/stream")
async def stream_chat(prompt: str):
    async def generate():
        async for msg in query(prompt):
            data = {
                "type": msg.type,
                "content": msg.content,
                "metadata": msg.metadata
            }
            yield f"data: {json.dumps(data)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

### JavaScript Client

```javascript
const eventSource = new EventSource('/chat/stream?prompt=Hello');

eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);

    switch (data.type) {
        case 'text':
            appendToChat(data.content);
            break;
        case 'tool_use':
            showToolIndicator(data.metadata.tool_name);
            break;
        case 'result':
            completeMessage();
            eventSource.close();
            break;
        case 'error':
            showError(data.content);
            eventSource.close();
            break;
    }
};
```

## Choosing Between Modes

| Use Case | Recommended Mode | Function |
|----------|------------------|----------|
| Chat UI with live typing | Streaming | `query()` |
| Simple Q&A bot | Single | `ask()` |
| Background processing | Single | `ask_sync()` |
| Progress indicators | Streaming | `query()` |
| Tool execution feedback | Streaming | `query()` |
| HIL approval workflows | Streaming | `query()` |
| API endpoint | Either | Depends on client |

## Next Steps

- [Configuration Options](./options.md) - Customize streaming behavior
- [Human-in-the-Loop](./human-in-the-loop.md) - Handle checkpoints
- [Messages Reference](./messages.md) - Full message type reference
