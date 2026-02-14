# Core Concepts

## Architecture Overview

isA Agent SDK is built on LangGraph and provides a complete agent execution framework with nodes, state management, and streaming support.

### Agent Execution Flow

```
User Query
    ↓
Entry Node (Session validation, memory retrieval, tool discovery)
    ↓
Reason Node (Analyze request, plan approach)
    ↓
Model Node (LLM interaction)
    ↓
Router Node (Determine next action)
    ↓
Tool Node (Execute tools) OR Response Node (Format response)
    ↓
Result
```

## Core Components

### 1. Agent State

The `AgentState` is a TypedDict that flows through the graph:

```python
class AgentState(TypedDict):
    messages: List[BaseMessage]           # Conversation history
    session_id: str                       # Session identifier
    shared_state: Dict[str, Any]          # Mutable shared data
    task_list: List[Dict]                 # Flat task list
    task_dag: Optional[DAGState]          # DAG task structure
    current_task_index: int               # Current task in flat list
    confidence: float                     # Response confidence score
    # ... and more
```

State fields use annotated reducers for merging:
- `preserve_latest` - Keep the most recent value
- `add_messages` - Append to message list
- `sum_numeric` - Add numeric values
- `merge_dicts` - Deep merge dictionaries

### 2. Nodes

Nodes are the building blocks of agent execution:

#### Entry Node
- Validates session existence or creates new sessions
- Retrieves conversation memory
- Discovers available tools and skills
- Prepares enhanced system prompt

#### Reason Node
- Analyzes the current request
- Plans the approach
- Determines if tools are needed
- Manages reasoning iterations

#### Model Node
- Handles LLM (Large Language Model) interactions
- Manages streaming responses
- Integrates billing and usage tracking
- Detects sensitive requests for human-in-the-loop

#### Router Node
- Analyzes AI responses to determine next action
- Detects autonomous planning requests
- Routes to appropriate next node
- Manages workflow direction based on tool call patterns

#### Tool Node
- Executes tool calls (Read, Write, Edit, Bash, etc.)
- Manages MCP tool integration
- Handles autonomous task planning
- Detects and builds DAG structures from task lists

#### Response Node
- Formats final responses
- Applies output format schemas
- Handles structured outputs
- Validates response structure

#### Guardrail Node
- Applies safety guardrails
- Checks for PII and sensitive data
- Enforces compliance rules (HIPAA, etc.)
- Validates response safety

#### Revise Node
- Revises responses based on guardrail feedback
- Improves response quality
- Handles confidence-based revision
- Manages revision iterations

#### Failsafe Node
- Handles errors gracefully
- Categorizes error types (UNCERTAINTY, INSUFFICIENT_INFO, etc.)
- Provides context-aware fallback responses
- Ensures transparent failure communication

### 3. Execution Modes

The SDK supports three execution modes:

#### Reactive (Default)
- Responds to explicit requests only
- No proactive suggestions
- Straightforward request-response pattern

#### Collaborative
- Creates checkpoints for human approval
- Requests permission for sensitive operations
- Interactive workflow with user involvement
- Durable execution with state persistence

#### Proactive
- Anticipates user needs
- Suggests next actions
- More autonomous behavior
- Still respects tool permissions

### 4. Streaming

All agent operations support streaming:

```python
async for msg in query("Hello"):
    if msg.is_text:
        print(msg.content, end="")
    elif msg.is_tool_use:
        print(f"\n[Using {msg.tool_name}]")
    elif msg.is_tool_result:
        print(f"[Result: {msg.content[:50]}...]")
```

Message types:
- `is_text` - Text content
- `is_tool_use` - Tool invocation
- `is_tool_result` - Tool execution result
- `is_error` - Error message
- `is_result` - Final result (for structured outputs)

### 5. Multi-Agent Systems

#### MultiAgentOrchestrator
Fixed routing with explicit router function:

```python
def router(state, last_result):
    outputs = state.get("outputs", {})
    if "planner" not in outputs:
        return "planner"
    if "renderer" not in outputs:
        return "renderer"
    return None
```

Best for: Known routing patterns, sequential workflows.

#### SwarmOrchestrator
Dynamic routing with LLM-directed handoffs:

```python
swarm = SwarmOrchestrator(
    agents=[researcher, writer],
    entry_agent="researcher",
    max_handoffs=5,
)
```

Agents decide when to hand off via `[HANDOFF: agent_name]` directives.

Best for: Dynamic workflows, specialist agents, collaborative tasks.

### 6. DAG Task Execution

Tasks with dependencies execute in wavefronts:

```python
tasks = [
    {"id": "a", "title": "Task A"},
    {"id": "b", "title": "Task B", "depends_on": ["a"]},
    {"id": "c", "title": "Task C", "depends_on": ["a"]},
    {"id": "d", "title": "Task D", "depends_on": ["b", "c"]},
]
```

Execution order:
```
Wavefront 0: [a]         # No dependencies
Wavefront 1: [b, c]      # Depend on a (parallel)
Wavefront 2: [d]         # Depends on b and c
```

Features:
- **Cycle detection** - Kahn's algorithm validates DAG structure
- **Failure cascade** - Failed tasks mark dependents as SKIPPED
- **Multi-agent** - Different agents execute different tasks in parallel
- **Status tracking** - PENDING → READY → RUNNING → COMPLETED/FAILED/SKIPPED

### 7. Options Configuration

`ISAAgentOptions` controls agent behavior:

```python
options = ISAAgentOptions(
    allowed_tools=["Read", "Edit", "Bash"],
    execution_mode="collaborative",
    max_iterations=20,
    confidence_threshold=0.7,
    system_prompt="You are a helpful assistant.",
    skills=["documentation", "code-review"],
    output_format=OutputFormat.from_pydantic(MySchema),
)
```

Key options:
- `allowed_tools` - Tool whitelist
- `execution_mode` - reactive/collaborative/proactive
- `max_iterations` - Maximum graph iterations (must be > 0)
- `confidence_threshold` - Minimum confidence for responses (0.0-1.0)
- `system_prompt` - System prompt or SystemPromptConfig
- `skills` - Skills to load
- `output_format` - Structured output schema

### 8. Human-in-the-Loop

Request permission before dangerous operations:

```python
authorized = await request_tool_permission(
    "delete_file",
    {"path": "important_data.txt"}
)
```

Create checkpoints for durable execution:

```python
await checkpoint("before_deployment", {
    "version": "2.0.0",
    "environment": "production"
})
```

### 9. Sessions and Memory

Sessions maintain conversation context:

```python
# Create session
session = await create_session(user_id="user123")

# Query with session
async for msg in query(
    "What did we discuss earlier?",
    options=ISAAgentOptions(session_id=session.id)
):
    print(msg.content)
```

Sessions store:
- Conversation history
- Shared state
- Checkpoints
- Task progress

### 10. Tools

Built-in tools:
- **Read** - Read files
- **Write** - Create/overwrite files
- **Edit** - Edit specific file sections
- **Bash** - Execute shell commands
- **WebSearch** - Search the web
- **WebFetch** - Fetch web content
- **Glob** - Find files by pattern
- **Grep** - Search file contents

Tools integrate with MCP (Model Context Protocol) for extensibility.

### 11. Skills

Skills are reusable agent capabilities defined in markdown:

```
.isa/skills/my-skill/SKILL.md
```

Skills are loaded:
1. First from `.isa/skills/` (local, project-specific)
2. Then from MCP servers (shared, global)

This allows project-specific overrides of global skills.

## Data Flow

### Query Execution
```
query(prompt, options)
    → SmartAgentGraphBuilder.build()
    → Graph execution with streaming
    → AgentMessage stream
    → User receives responses
```

### Swarm Execution
```
swarm.run(prompt)
    → Entry agent runs with handoff prompt injection
    → Parse response for [HANDOFF:] or [COMPLETE]
    → If handoff: switch agent, inject context
    → If complete: return SwarmRunResult
    → Max handoffs safety cap
```

### DAG Execution
```
swarm.run_dag(tasks)
    → DAGScheduler.build_dag(tasks)
    → DAGScheduler.validate(dag)
    → DAGScheduler.compute_wavefronts(dag)
    → For each wavefront:
        → Run all tasks in parallel (per agent)
        → Aggregate results
        → Pass to dependent tasks
    → Return SwarmRunResult with agent_outputs
```

## Best Practices

1. **Set appropriate max_iterations** - Default is 3, but complex tasks need 15-20 for proper reasoning cycles
2. **Use collaborative mode for risky operations** - Get human approval before destructive actions
3. **Leverage DAG for complex workflows** - Dependencies ensure correct execution order
4. **Use Swarm for specialist agents** - Each agent focuses on one skill
5. **Configure tools carefully** - Only allow tools needed for the task
6. **Use structured outputs for data** - Type-safe with Pydantic schemas
7. **Handle errors gracefully** - Check `msg.is_error` in streaming
8. **Test with comprehensive coverage** - Follow patterns in test_*.py files

## Next Steps

- [Configuration](./configuration.md) - Detailed configuration guide
- [Multi-Agent](./multi-agent.md) - Multi-agent orchestration patterns
- [Swarm](./swarm.md) - Swarm orchestration and DAG execution
- [Examples](./examples.md) - Practical code examples
- [API Reference](./api-reference.md) - Complete API documentation
