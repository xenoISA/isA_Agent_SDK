# isA Agent SDK vs Claude Agent SDK Comparison

This document compares the isA Agent SDK with Anthropic's Claude Agent SDK, highlighting equivalent features, isA's additional capabilities, and gaps to address.

## Feature Matrix

### Core Features

| Feature | Claude SDK | isA Agent SDK | Status |
|---------|-----------|---------------|--------|
| `query()` function | ✅ Async iterator | ✅ Async iterator | ✅ Equivalent |
| Streaming | ✅ Real-time tokens | ✅ Async generators + SSE | ✅ Equivalent |
| Hooks | ✅ Pre/PostToolUse, Stop, SessionStart/End | ✅ HookMatcher in options.py | ✅ Equivalent |
| Structured Outputs | ✅ JSON schema | ✅ OutputFormat + Pydantic | ✅ Equivalent |
| System Prompts | ✅ preset + append | ✅ SystemPromptConfig + MCP | ✅ Better |
| Sessions | ✅ resume, fork | ✅ DurableService + resume() | ✅ Equivalent |
| Subagents | ✅ AgentDefinition + Task tool | ✅ AgentDefinition + Task tool | ✅ Equivalent |
| MCP Servers | ✅ External subprocess | ✅ Full MCPClient | ✅ Equivalent |
| Permissions | ✅ permissionMode | ✅ HIL + PermissionMode | ✅ Better |
| User Input | ✅ AskUserQuestion | ✅ HIL: collect_input(), etc. | ✅ Better |

### Built-in Tools

| Tool | Claude SDK | isA Agent SDK | Status |
|------|-----------|---------------|--------|
| Read | ✅ | ✅ via MCP | ✅ |
| Write | ✅ | ✅ via MCP | ✅ |
| Edit | ✅ | ✅ via MCP | ✅ |
| Bash | ✅ | ✅ via MCP | ✅ |
| Glob | ✅ | ✅ via MCP | ✅ |
| Grep | ✅ | ✅ via MCP | ✅ |
| WebSearch | ✅ | ✅ via MCP | ✅ |
| WebFetch | ✅ | ✅ via MCP | ✅ |

---

## GAPS - What Claude SDK Has That isA Needs

### 1. ~~`@tool` Decorator & `create_sdk_mcp_server()`~~ ✅ IMPLEMENTED

```python
# isA Agent SDK - Now supported!
from isa_agent_sdk import tool, create_sdk_mcp_server, query, ISAAgentOptions

@tool("greet", "Greet a user by name")
async def greet_user(name: str) -> str:
    return f"Hello, {name}!"

server = create_sdk_mcp_server("mytools", [greet_user])

async for msg in query(
    "Greet Alice",
    options=ISAAgentOptions(
        mcp_servers={"mytools": server},
        allowed_tools=["mcp__mytools__greet"]
    )
):
    print(msg.content)
```

**Implemented features:**
- `@tool` decorator with schema inference from type hints
- Custom JSON schema support
- `SDKMCPServer` class for in-process execution
- `create_sdk_mcp_server()` factory function
- Mixed servers (SDK + external MCP)
- Automatic error handling

---

### 2. Slash Commands (.claude/commands/*.md) 🟡 MEDIUM PRIORITY

Claude SDK supports filesystem-based custom commands:

```
.claude/commands/
├── fix-bug.md       # /fix-bug command
├── review.md        # /review command
└── deploy.md        # /deploy command
```

Each command file contains a prompt template that gets injected when the user types `/fix-bug`.

**Implementation Plan:**
1. Create `_commands.py` module
2. Add `CommandManager` class to discover/load commands
3. Support `{PROJECT_ROOT}/.isa/commands/*.md` pattern
4. Integrate with query() to detect `/command` prefix

---

### 3. ~~Memory (CLAUDE.md / ISA.md)~~ ✅ IMPLEMENTED

Claude SDK reads project context from `CLAUDE.md`, and isA supports the same:

```markdown
# CLAUDE.md
This project is a Python web service using FastAPI.
Always use type hints and follow PEP 8.
The database is PostgreSQL with SQLAlchemy ORM.
```

This gets automatically included in the system prompt.

**Implemented features:**
- `ISA.md`, `CLAUDE.md`, `.isa/CONTEXT.md`, `.claude/CONTEXT.md` discovery
- `project_context="auto"` for automatic lookup
- Direct path or inline content supported
- Injected into system prompt memory slot

---

### 4. Plugins System 🟢 LOW PRIORITY

Claude SDK has programmatic plugins:

```python
options = ClaudeAgentOptions(
    plugins=[
        {"name": "my-plugin", "commands": [...], "agents": [...]}
    ]
)
```

**Implementation Plan:**
1. Define `Plugin` dataclass
2. Add `plugins` option to `ISAAgentOptions`
3. Support plugin discovery and loading
4. Register plugin commands/agents with main system

---

### 5. ~~`ClaudeSDKClient` Bidirectional Client~~ ✅ IMPLEMENTED

Claude SDK has an async context manager for interactive conversations:

```python
async with ClaudeSDKClient(options=options) as client:
    await client.query("Greet Alice")
    async for msg in client.receive_response():
        print(msg)

    # Continue conversation
    await client.query("Now greet Bob")
    async for msg in client.receive_response():
        print(msg)
```

**isA Agent SDK - Now supported!**
```python
from isa_agent_sdk import ISAAgentClient, ISAAgentOptions

# Local execution (LangGraph)
async with ISAAgentClient() as client:
    await client.query("Analyze this codebase")
    async for msg in client.receive():
        print(msg.content)

    # Continue same conversation
    await client.query("Now fix the bug")
    async for msg in client.receive():
        print(msg.content)

# Remote API execution
async with ISAAgentClient(base_url="http://api.example.com") as client:
    await client.query("Hello!")
    async for msg in client.receive():
        print(msg.content)

# Convenience method
response = await client.ask("What is 2 + 2?")

# Fork session for exploration
forked = client.fork()
async with forked:
    await forked.query("Try alternative approach")
```

**Implemented features:**
- Async context manager (`async with`)
- `query()` and `receive()` separation
- `ask()` convenience method
- Session state management
- Local (LangGraph) and Remote (HTTP) modes
- Session forking
- Sync wrapper (`ISAAgentClientSync`)

---

### 6. ~~Specific Error Classes~~ ✅ IMPLEMENTED

Claude SDK has typed errors:

```python
from claude_agent_sdk import (
    ClaudeSDKError,      # Base
    CLINotFoundError,    # CLI not installed
    CLIConnectionError,  # Connection issues
    ProcessError,        # Process failed
    CLIJSONDecodeError,  # JSON parsing
)
```

**isA Agent SDK - Now supported!**
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
    GraphExecutionError,

    # Session & State
    SessionError,
    SessionNotFoundError,
    SessionExpiredError,
    CheckpointError,
    ResumeError,

    # Validation
    ValidationError,
    SchemaError,
    ConfigurationError,

    # Permission & Authorization
    PermissionError,
    ToolPermissionError,
    HILDeniedError,
    HILTimeoutError,

    # MCP
    MCPError,
    MCPConnectionError,
    MCPToolNotFoundError,
)
```

**Implemented features:**
- Full error hierarchy with base `ISASDKError`
- Rich error details (service, url, session_id, tool_name, etc.)
- All errors exported from `isa_agent_sdk`

---

## isA ADVANTAGES - What isA Has That Claude SDK Doesn't

### 1. Multi-Model Support ⭐

isA works with any LLM, not just Claude:

```python
# DeepSeek
options = ISAAgentOptions(model="deepseek-reasoner")

# GPT-4
options = ISAAgentOptions(model="gpt-4-turbo")

# Local models
options = ISAAgentOptions(model="llama-3.1-70b")
```

Claude SDK only works with Anthropic models.

---

### 2. Desktop Execution ⭐

Route tools to user's local machine:

```python
options = ISAAgentOptions(
    env=ExecutionEnv.DESKTOP,
    user_id="xenodennis",
    allowed_tools=["read_file", "bash_execute"]
)
```

Claude SDK doesn't have this concept.

---

### 3. Steward Tools (管家) ⭐

Personal assistant capabilities:

| Category | Tools |
|----------|-------|
| Task Management | 9 tools |
| Calendar | 9 tools |
| Event Triggers | 6 tools |

```python
# Via conversation
"Create a todo for 'Review PR' with high priority"
"Schedule a meeting with John tomorrow at 2pm"
"Alert me when Bitcoin drops 5%"
```

---

### 4. Event Triggers (Proactive Activation) ⭐

```python
from isa_agent_sdk import register_price_trigger, register_schedule_trigger

# Price alert
await register_price_trigger(
    user_id="user123",
    product="Bitcoin",
    threshold=5.0,
    direction="down",
    action_prompt="Analyze the market situation"
)

# Scheduled task
await register_schedule_trigger(
    user_id="user123",
    cron="0 9 * * *",
    action_prompt="Summarize today's tasks"
)
```

---

### 5. Advanced HIL System ⭐

7+ methods vs Claude's single `human_input()`:

| Method | Description |
|--------|-------------|
| `collect_input()` | Text/number/boolean input |
| `collect_selection()` | Choose from options |
| `collect_credentials()` | Secure credential input |
| `request_authorization()` | Approve/reject actions |
| `request_review()` | Review and edit content |
| `request_plan_approval()` | Approve execution plans |
| `request_execution_choice()` | Choose between options |

Plus:
- Durable execution (survives restarts)
- Security levels (LOW → CRITICAL)
- Schema validation with retry
- MCP tool integration (`ask_human`, `request_authorization`)

---

### 6. Execution Modes ⭐

```python
# Reactive - respond to user prompts
options = ISAAgentOptions(execution_mode=ExecutionMode.REACTIVE)

# Collaborative - HIL checkpoints
options = ISAAgentOptions(execution_mode=ExecutionMode.COLLABORATIVE)

# Proactive - trigger-based activation
options = ISAAgentOptions(execution_mode=ExecutionMode.PROACTIVE)
```

---

### 7. Multi-Environment Support ⭐

```python
# Cloud shared MCP (default)
options = ISAAgentOptions(env=ExecutionEnv.CLOUD_SHARED)

# User's desktop
options = ISAAgentOptions(env=ExecutionEnv.DESKTOP)

# Isolated cloud VM (coming soon)
options = ISAAgentOptions(env=ExecutionEnv.CLOUD_POOL)
```

---

### 8. Background Jobs (NATS + Redis) ⭐

Run hours-long tool execution off the request thread:

```python
from isa_agent_sdk.services.background_jobs import submit_tool_execution_task

result = await submit_tool_execution_task(
    task_data={
        "job_id": "job_123",
        "session_id": "sess_1",
        "user_id": "user_1",
        "tools": [...],
    }
)
```

Claude SDK doesn't have a built-in background job system.

---

## Implementation Roadmap

### Phase 1: Parity (High Priority) ✅ COMPLETE

| Feature | Effort | Priority | Status |
|---------|--------|----------|--------|
| `@tool` decorator + SDK MCP server | 3-4 days | 🔴 HIGH | ✅ DONE |
| `ISAAgentClient` bidirectional | 2-3 days | 🟡 MEDIUM | ✅ DONE |
| Memory file support (ISA.md) | 1-2 days | 🟡 MEDIUM | ✅ DONE |
| Error class hierarchy | 1 day | 🟢 LOW | ✅ DONE |

### Phase 2: Feature Complete (Medium Priority)

| Feature | Effort | Priority | Status |
|---------|--------|----------|--------|
| Slash commands | 2-3 days | 🟡 MEDIUM | Skipped (not needed) |
| Plugins system | 3-4 days | 🟢 LOW | Skipped (have equivalents) |

### Phase 3: Polish

- Documentation parity with Claude SDK
- Migration guide for Claude SDK users
- Example agents (email assistant, research agent)

---

## Current Status Summary

```
Feature Coverage:  100% of Claude SDK core features ✅
Advantages:        Multi-model, Desktop, Steward, Triggers, Advanced HIL, Background Jobs
Remaining Gaps:    None (slash commands & plugins skipped by design)
```

**isA Agent SDK is now a complete superset of Claude Agent SDK** with:
- All core features implemented
- `@tool` decorator for custom tools
- `ISAAgentClient` bidirectional client
- Full error class hierarchy
- Memory file support (ISA.md/CLAUDE.md)
- Plus significant additional capabilities (multi-model, desktop execution, steward tools, event triggers, advanced HIL, background jobs)
