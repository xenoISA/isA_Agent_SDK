# Human-in-the-Loop (HIL)

The Human-in-the-Loop system enables agents to pause execution and request human approval, input, or review before proceeding with sensitive operations.

## Overview

HIL provides:
- **Durable execution** - Survives process restarts via LangGraph checkpointing
- **Async responses** - Can wait hours or days for human input
- **Multiple interaction types** - Authorization, input collection, review
- **MCP tool integration** - `ask_human` and `request_authorization` tools
- **Security levels** - LOW, MEDIUM, HIGH, CRITICAL risk classification
- **Schema validation** - Validate user input with JSON schemas and retry support

## Comparison with Claude SDK

| Feature | Claude SDK | isA Agent SDK | Advantage |
|---------|-----------|---------------|-----------|
| Basic Input | `human_input()` | `collect_input()` | Equivalent |
| Selection | N/A | `collect_selection()` | More powerful |
| Authorization | Hooks | `request_authorization()` | More powerful |
| Review/Edit | N/A | `request_review()` | More powerful |
| Combined Input+Auth | N/A | `request_input_with_authorization()` | More powerful |
| Durable Execution | No | Yes (LangGraph checkpoints) | More powerful |
| MCP Integration | No | Yes (`ask_human` tool) | More powerful |
| Schema Validation | No | Yes (with retry) | More powerful |
| Security Levels | No | Yes (LOW→CRITICAL) | More powerful |

The isA HIL system is a superset of Claude SDK's `human_input()` functionality, providing additional methods for structured input, authorization workflows, and durable execution.

## Basic Authorization

### Request Tool Permission

Pause execution until human approves a tool use:

```python
from isa_agent_sdk import request_tool_permission

# In your agent logic
authorized = await request_tool_permission(
    tool_name="delete_file",
    tool_args={"path": "/important/data.txt"},
    reason="Need to clean up old data files",
    security_level="high"
)

if authorized:
    # Proceed with deletion
    await delete_file("/important/data.txt")
else:
    # Handle rejection
    print("User declined the operation")
```

### Generic Authorization

For any action requiring approval:

```python
from isa_agent_sdk import request_authorization

authorized = await request_authorization(
    action="Deploy to production",
    reason="All tests passed, ready for release",
    context={
        "version": "2.1.0",
        "changes": 15,
        "tests_passed": 142
    },
    risk_level="high"
)
```

## Input Collection

### Collect Text Input

```python
from isa_agent_sdk import collect_input

# Simple text input
api_key = await collect_input(
    prompt="Please enter your API key",
    description="Required for authentication",
    input_type="text"
)

# Number input
count = await collect_input(
    prompt="How many items to process?",
    input_type="number"
)

# Boolean input
confirm = await collect_input(
    prompt="Enable debug mode?",
    input_type="boolean"
)
```

### Collect with Schema Validation

```python
config = await collect_input(
    prompt="Enter configuration",
    input_type="object",
    schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer", "minimum": 1},
            "enabled": {"type": "boolean"}
        },
        "required": ["name", "count"]
    }
)
```

### Collect Credentials

Secure credential collection:

```python
from isa_agent_sdk import collect_credentials

# API key
api_key = await collect_credentials(
    prompt="Enter your OpenAI API key",
    credential_type="api_key",
    description="Required for LLM access"
)

# Password
password = await collect_credentials(
    prompt="Enter database password",
    credential_type="password"
)

# Token
token = await collect_credentials(
    prompt="Enter GitHub token",
    credential_type="token"
)
```

### Collect Selection

Choose from predefined options:

```python
from isa_agent_sdk import collect_selection

choice = await collect_selection(
    prompt="Select deployment environment",
    options=["development", "staging", "production"],
    description="Choose where to deploy"
)

print(f"Deploying to: {choice}")
```

## Review and Approval

### Request Content Review

Ask human to review generated content:

```python
from isa_agent_sdk import request_review

# Review code
result = await request_review(
    content=generated_code,
    content_type="code",
    instructions="Please review this code for security issues",
    editable=True  # Allow user to edit
)

if result["approved"]:
    final_code = result.get("edited_content", generated_code)
    await save_code(final_code)
else:
    action = result.get("action", "rejected")
    print(f"Review result: {action}")
```

**Content Types:**
- `text` - General text content
- `code` - Source code
- `plan` - Execution plan
- `config` - Configuration data

### Request Plan Approval

Specialized for execution plans:

```python
from isa_agent_sdk import request_plan_approval

plan = """
## Deployment Plan

1. Run database migrations
2. Deploy backend services
3. Deploy frontend
4. Run smoke tests
5. Switch traffic to new version
"""

result = await request_plan_approval(
    plan=plan,
    plan_title="Production Deployment v2.1.0"
)

if result["approved"]:
    await execute_deployment()
```

### Request Execution Choice

Let user choose between options:

```python
from isa_agent_sdk import request_execution_choice

choice = await request_execution_choice(
    prompt="How should I handle the failing tests?",
    options=[
        {"id": "fix", "label": "Fix the tests", "description": "I'll update the test code"},
        {"id": "skip", "label": "Skip for now", "description": "Continue without these tests"},
        {"id": "abort", "label": "Abort", "description": "Stop the current operation"}
    ],
    recommendation="fix"  # Highlight recommended option
)

match choice:
    case "fix":
        await fix_tests()
    case "skip":
        await continue_without_tests()
    case "abort":
        raise AbortException("User aborted")
```

## Handling HIL in Streaming

### Detecting HIL Requests

```python
from isa_agent_sdk import query

async for msg in query("Delete old log files"):
    if msg.is_hil_request or msg.is_checkpoint:
        # HIL request detected
        question = msg.metadata.get("question")
        options = msg.metadata.get("options")
        hil_type = msg.metadata.get("hil_type")

        print(f"Approval needed: {question}")

        # Respond to the checkpoint
        await msg.respond({"authorized": True})

    elif msg.is_text:
        print(msg.content, end="")
```

### Resume After HIL

If the process restarts during HIL wait:

```python
from isa_agent_sdk import resume

# Resume with user's response
async for msg in resume(
    session_id="my-session-123",
    resume_value={"authorized": True, "input": "user provided data"}
):
    print(msg.content, end="" if msg.is_text else "\n")
```

## HIL Statistics

### Get Interrupt Stats

```python
from isa_agent_sdk import get_hil_stats

stats = get_hil_stats()
print(f"Total interrupts: {stats.total}")
print(f"By type: {stats.by_type}")
print(f"By node: {stats.by_node}")
print(f"Latest: {stats.latest}")
```

**Note:** `get_hil_stats()` returns an `InterruptStats` dataclass, not a dictionary.

### Clear HIL History

```python
from isa_agent_sdk import clear_hil_history

clear_hil_history()  # For testing/cleanup
```

## Integration Example

### Complete HIL Workflow

```python
from isa_agent_sdk import (
    query,
    request_tool_permission,
    request_review,
    collect_input,
    ISAAgentOptions,
    ExecutionMode
)

async def safe_file_operation():
    options = ISAAgentOptions(
        execution_mode=ExecutionMode.COLLABORATIVE,
        allowed_tools=["read_file", "write_file", "delete_file"]
    )

    async for msg in query("Clean up the temp directory", options=options):
        if msg.is_checkpoint:
            hil_type = msg.metadata.get("hil_type")

            if hil_type == "tool_permission":
                # Ask for tool permission
                tool = msg.metadata.get("tool_name")
                args = msg.metadata.get("tool_args")

                print(f"\nPermission requested for: {tool}")
                print(f"Arguments: {args}")

                user_input = input("Approve? (y/n): ")
                await msg.respond({"authorized": user_input.lower() == "y"})

            elif hil_type == "review":
                # Show content for review
                content = msg.metadata.get("content")
                print(f"\nReview requested:\n{content}")

                user_input = input("Approve? (y/n): ")
                await msg.respond({"approved": user_input.lower() == "y"})

        elif msg.is_text:
            print(msg.content, end="")

        elif msg.is_error:
            print(f"\nError: {msg.content}")
```

### Web Application Integration

```python
from fastapi import FastAPI, WebSocket
from isa_agent_sdk import query, resume
import json

app = FastAPI()

# Store pending HIL requests
pending_hil = {}

@app.websocket("/chat/{session_id}")
async def chat(websocket: WebSocket, session_id: str):
    await websocket.accept()

    async for msg in query("User prompt here", options={"session_id": session_id}):
        if msg.is_checkpoint:
            # Store HIL request and notify client
            pending_hil[session_id] = msg
            await websocket.send_json({
                "type": "hil_request",
                "question": msg.metadata.get("question"),
                "options": msg.metadata.get("options"),
                "hil_type": msg.metadata.get("hil_type")
            })
            return  # Wait for response via separate endpoint

        await websocket.send_json({
            "type": msg.type,
            "content": msg.content
        })

@app.post("/hil/{session_id}/respond")
async def respond_hil(session_id: str, response: dict):
    if session_id in pending_hil:
        msg = pending_hil.pop(session_id)
        await msg.respond(response)
        return {"status": "ok"}
    return {"status": "not_found"}
```

## MCP Tool Integration

The HIL system exposes tools via MCP (Model Context Protocol) that agents can call directly:

### ask_human Tool

The `ask_human` MCP tool allows agents to request human input during execution:

```python
# Tool definition in MCP
@mcp.tool()
async def ask_human(question: str, context: str = "", user_id: str = "default") -> str:
    """Ask the human for additional information or clarification"""
    # Returns structured response with:
    # - status: "human_input_requested"
    # - action: "ask_human"
    # - data: { question, context, user_id, instruction }
```

**When agents call this:**
```json
{
  "status": "human_input_requested",
  "action": "ask_human",
  "data": {
    "question": "What deployment environment should I use?",
    "context": "Deploying application v2.1.0",
    "instruction": "This request requires human input. The client should handle the interaction."
  }
}
```

### request_authorization Tool

The `request_authorization` MCP tool handles permission requests for sensitive operations:

```python
# Tool definition in MCP
@mcp.tool()
async def request_authorization(
    tool_name: str,
    reason: str,
    user_id: str = "default",
    tool_args: dict | None = None
) -> str:
    """Request human authorization before executing a tool"""
    # Automatically determines security level based on tool name:
    # - "delete", "forget" → HIGH
    # - "admin" → CRITICAL
    # - Default → MEDIUM
```

**Response includes:**
- `request_id` - Unique authorization request ID
- `security_level` - AUTO-determined: MEDIUM, HIGH, or CRITICAL
- `expires_at` - Authorization expiration timestamp

### Direct HIL Service Access

For advanced use cases, access the HIL service directly:

```python
from isa_agent_sdk.services.human_in_the_loop import get_hil_service

hil_service = get_hil_service()

# Ask with interrupt (pauses graph execution)
response = hil_service.ask_human_with_interrupt(
    question="Should I proceed?",
    context="About to delete 50 files",
    node_source="cleanup_node"
)
```

## Best Practices

1. **Use appropriate risk levels** - Set `security_level` or `risk_level` accurately
2. **Provide clear context** - Include enough information for informed decisions
3. **Handle rejections gracefully** - Always have a fallback path
4. **Use durable execution** - Enable `COLLABORATIVE` mode for long operations
5. **Timeout handling** - Consider what happens if user doesn't respond
6. **Use MCP tools** - Prefer `ask_human` and `request_authorization` for standardized interactions

## Next Steps

- [Checkpointing](./checkpointing.md) - Durable execution details
- [Options](./options.md) - Configure HIL behavior
- [Streaming](./streaming.md) - Handle HIL in streams
