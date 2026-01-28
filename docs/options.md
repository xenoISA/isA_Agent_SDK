# Configuration Options

The `ISAAgentOptions` class controls all aspects of agent behavior.

## Basic Usage

```python
from isa_agent_sdk import query, ISAAgentOptions

options = ISAAgentOptions(
    model="gpt-4.1-nano",
    allowed_tools=["web_search", "read_file"],
    max_iterations=30
)

async for msg in query("Your prompt", options=options):
    print(msg.content, end="" if msg.is_text else "\n")
```

## Core Options

### Model Configuration

```python
options = ISAAgentOptions(
    # LLM model to use
    model="gpt-4.1-nano",  # Default model
)
```

**Available Models:**
- `gpt-4.1-nano` - Fast, cost-effective (default)
- `gpt-4o` - High capability
- `deepseek-reasoner` - Reasoning model (chain-of-thought)
- `llama-3.3-70b` - Open source alternative

### System Prompts (Claude SDK Compatible)

Customize agent behavior with system prompts. Supports three modes:

```python
from isa_agent_sdk import ISAAgentOptions, SystemPromptConfig, SystemPromptPreset

# 1. Simple string - appends to default prompt (backwards compatible)
options = ISAAgentOptions(
    system_prompt="Always respond in formal English. Be concise."
)

# 2. Preset with custom additions (recommended)
options = ISAAgentOptions(
    system_prompt=SystemPromptConfig(
        preset=SystemPromptPreset.REASON,  # Use MCP template
        append="Focus on security implications in your analysis."
    )
)

# 3. Full replacement (loses built-in capabilities)
options = ISAAgentOptions(
    system_prompt=SystemPromptConfig(
        replace="You are a specialized security auditor. Analyze code for vulnerabilities."
    )
)
```

**Available Presets:**
| Preset | MCP Template | Description |
|--------|--------------|-------------|
| `REASON` | `default_reason_prompt` | For reasoning/planning phase |
| `RESPONSE` | `default_response_prompt` | For final response generation |
| `RAG_REASON` | `rag_reason_prompt` | Reasoning with user's uploaded files |
| `REVIEW` | `default_review_prompt` | For evaluating execution results |
| `MINIMAL` | `minimal_prompt` | Minimal base, mostly custom instructions |

**How it works:**
- MCP stores the base prompt templates with variables (`{{memory}}`, `{{tools}}`, etc.)
- Your `append` text is injected as `{{user_instructions}}` into the template
- Use `replace` only when you need complete control (not recommended)

### Tool Configuration

```python
options = ISAAgentOptions(
    # Explicit tool allowlist
    allowed_tools=["web_search", "read_file", "write_file", "bash"],

    # Tool discovery mode
    tool_discovery=ToolDiscoveryMode.HYBRID,  # explicit, semantic, hybrid
)
```

**Tool Discovery Modes:**
| Mode | Description |
|------|-------------|
| `EXPLICIT` | Only use tools in `allowed_tools` list |
| `SEMANTIC` | Discover tools based on query context |
| `HYBRID` | Combine explicit list with semantic discovery |

### Permission Modes

```python
from isa_agent_sdk import PermissionMode

options = ISAAgentOptions(
    permission_mode=PermissionMode.DEFAULT
)
```

| Mode | Description |
|------|-------------|
| `DEFAULT` | Ask for permission on sensitive operations |
| `ACCEPT_EDITS` | Auto-approve file edits, ask for others |
| `BYPASS_PERMISSIONS` | No permission checks (use carefully) |

## Execution Modes

### Reactive Mode (Default)

Standard request-response interaction:

```python
from isa_agent_sdk import ExecutionMode

options = ISAAgentOptions(
    execution_mode=ExecutionMode.REACTIVE
)
```

### Collaborative Mode

For long-running tasks with checkpoints:

```python
options = ISAAgentOptions(
    execution_mode=ExecutionMode.COLLABORATIVE,
    checkpoint_frequency=5,  # Checkpoint every 5 tasks
)
```

Features:
- Durable execution (survives restarts)
- Periodic checkpoints
- Resume capability

### Proactive Mode

For event-driven autonomous operation:

```python
options = ISAAgentOptions(
    execution_mode=ExecutionMode.PROACTIVE,
    proactive_suggestions=True,  # Enable suggestions
)
```

Features:
- Event triggers
- Autonomous task execution
- Proactive suggestions

## Safety & Guardrails

```python
from isa_agent_sdk import GuardrailMode

options = ISAAgentOptions(
    # Enable guardrails
    guardrails_enabled=True,

    # Guardrail strictness
    guardrail_mode=GuardrailMode.MODERATE,  # permissive, moderate, strict

    # Confidence-based failsafe
    failsafe_enabled=True,
    failsafe_confidence_threshold=0.7,  # 0.0-1.0

    # Iteration limit
    max_iterations=30,
)
```

**Guardrail Modes:**
| Mode | Description |
|------|-------------|
| `PERMISSIVE` | Minimal restrictions |
| `MODERATE` | Balanced safety (default) |
| `STRICT` | Maximum safety checks |

## Session Management

```python
options = ISAAgentOptions(
    # Session identification
    session_id="my-session-123",  # Auto-generated if not provided
    user_id="user-456",

    # Session lifetime
    session_ttl=3600,  # 1 hour (seconds)

    # Resume previous session
    resume="previous-session-id",

    # Additional metadata
    metadata={
        "project": "my-project",
        "environment": "production"
    }
)
```

## Execution Environment

```python
from isa_agent_sdk import ExecutionEnv

options = ISAAgentOptions(
    # Where to run the agent
    execution_env=ExecutionEnv.CLOUD_POOL,  # cloud_pool, cloud_shared, desktop

    # Pool configuration (for cloud_pool)
    pool_config=PoolConfig(
        pool_type="standard",
        ttl=600,
        memory=512,
        cpu=1
    ),

    # Desktop agent URL (for desktop)
    desktop_agent_url="ws://localhost:9000"
)
```

**Execution Environments:**
| Environment | Description |
|-------------|-------------|
| `CLOUD_POOL` | Isolated VM per session |
| `CLOUD_SHARED` | Shared cloud resources |
| `DESKTOP` | Local desktop agent |

## MCP Server Configuration

### Connect to External MCP Servers

```python
from isa_agent_sdk import MCPServerConfig

options = ISAAgentOptions(
    mcp_servers={
        "github": MCPServerConfig(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_TOKEN": "your-token"}
        ),
        "filesystem": MCPServerConfig(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"]
        ),
        "custom": MCPServerConfig(
            url="http://localhost:8081/mcp"
        )
    }
)
```

## Graph Type Selection

```python
options = ISAAgentOptions(
    # Agent architecture
    graph_type="smart_agent",  # smart_agent, research, coding, conversation
)
```

**Graph Types:**
| Type | Description |
|------|-------------|
| `smart_agent` | General-purpose with reasoning |
| `research` | Optimized for research tasks |
| `coding` | Optimized for code tasks |
| `conversation` | Simple chat without tools |

## Summarization

```python
options = ISAAgentOptions(
    # Auto-summarize long conversations
    summarization_enabled=True,
)
```

## Structured Outputs

Get validated JSON matching a specific schema:

```python
from isa_agent_sdk import OutputFormat

# From Pydantic model (recommended)
from pydantic import BaseModel

class Recipe(BaseModel):
    name: str
    ingredients: list[str]
    prep_time_minutes: int

options = ISAAgentOptions(
    output_format=OutputFormat.from_pydantic(Recipe)
)

# From JSON schema
options = ISAAgentOptions(
    output_format=OutputFormat.json_schema({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "items": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["name"]
    })
)

# Simple JSON mode (no schema validation)
options = ISAAgentOptions(
    output_format=OutputFormat.json_object()
)
```

**OutputFormat Types:**
| Type | Description |
|------|-------------|
| `text` | Default free-form text |
| `json_object` | Valid JSON (no schema) |
| `json_schema` | Schema-validated JSON |

See [Structured Outputs](./structured-outputs.md) for complete documentation.

## Skills Configuration

```python
options = ISAAgentOptions(
    # Activate specialized skills
    skills=["code-review", "debug", "refactor"]
)
```

**Built-in Skills:**
- `code-review` - Expert code reviewer
- `debug` - Systematic debugger
- `refactor` - Refactoring specialist
- `test-writer` - Test coverage expert
- `documentation` - Technical writer

## Loading from File

### YAML Configuration

```yaml
# agent_config.yaml
model: gpt-4.1-nano
allowed_tools:
  - web_search
  - read_file
  - write_file
  - bash
execution_mode: collaborative
guardrails_enabled: true
guardrail_mode: moderate
max_iterations: 30
skills:
  - code-review
  - debug
mcp_servers:
  github:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
```

```python
options = ISAAgentOptions.from_file("agent_config.yaml")
```

## Runtime Configuration

Convert options to runtime configs:

```python
# For SmartAgentGraph
graph_config = options.to_graph_config()

# For LangGraph runtime
runtime_config = options.to_runtime_config()
```

## Full Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model` | str | "gpt-4.1-nano" | LLM model |
| `system_prompt` | str \| SystemPromptConfig | None | Custom system prompt (preset/append/replace) |
| `allowed_tools` | List[str] | None | Tool allowlist |
| `permission_mode` | PermissionMode | DEFAULT | Permission handling |
| `execution_mode` | ExecutionMode | REACTIVE | Execution mode |
| `execution_env` | ExecutionEnv | CLOUD_SHARED | Where to run |
| `graph_type` | str | "smart_agent" | Agent architecture |
| `guardrails_enabled` | bool | True | Enable guardrails |
| `guardrail_mode` | GuardrailMode | MODERATE | Guardrail strictness |
| `failsafe_enabled` | bool | True | Confidence failsafe |
| `failsafe_confidence_threshold` | float | 0.7 | Failsafe threshold |
| `max_iterations` | int | 30 | Max graph iterations |
| `tool_discovery` | ToolDiscoveryMode | HYBRID | Tool discovery mode |
| `summarization_enabled` | bool | True | Auto-summarize |
| `checkpoint_frequency` | int | 5 | Checkpoint frequency |
| `proactive_suggestions` | bool | False | Enable suggestions |
| `session_id` | str | Auto | Session identifier |
| `user_id` | str | None | User identifier |
| `session_ttl` | int | 3600 | Session TTL (seconds) |
| `resume` | str | None | Session to resume |
| `skills` | List[str] | None | Skills to activate |
| `output_format` | OutputFormat | None | Structured output format |
| `mcp_servers` | Dict | {} | MCP server configs |
| `pool_config` | PoolConfig | None | Pool configuration |
| `metadata` | Dict | None | Additional metadata |

## Next Steps

- [Streaming](./streaming.md) - Response handling
- [Structured Outputs](./structured-outputs.md) - JSON schema & Pydantic
- [Human-in-the-Loop](./human-in-the-loop.md) - Approval workflows
- [Skills](./skills.md) - Skill system guide
