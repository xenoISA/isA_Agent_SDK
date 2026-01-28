# System Prompts Guide

Customize agent behavior with flexible system prompts using a Claude SDK-compatible architecture.

## Overview

The isA Agent SDK supports three ways to customize system prompts:

| Mode | Use Case | MCP Template |
|------|----------|--------------|
| **Simple String** | Quick customization | Uses default, appends your text |
| **Preset + Append** | Recommended approach | Selects template, appends your text |
| **Full Replacement** | Complete control | Bypasses MCP entirely |

## Quick Start

```python
from isa_agent_sdk import query, ISAAgentOptions, SystemPromptConfig, SystemPromptPreset

# Simple string (backwards compatible)
options = ISAAgentOptions(
    system_prompt="Always respond in formal English."
)

# Preset with custom additions (recommended)
options = ISAAgentOptions(
    system_prompt=SystemPromptConfig(
        preset=SystemPromptPreset.REASON,
        append="Focus on security implications."
    )
)

# Full replacement (advanced)
options = ISAAgentOptions(
    system_prompt=SystemPromptConfig(
        replace="You are a pirate. Always say Ahoy!"
    )
)
```

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    System Prompt Flow                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ISAAgentOptions.system_prompt                                  │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────┐                                           │
│  │ SystemPromptConfig │                                        │
│  │  - preset       │                                           │
│  │  - append       │                                           │
│  │  - replace      │                                           │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────┐                   │
│  │            ReasonNode / ResponseNode     │                   │
│  │                                          │                   │
│  │  if replace:                            │                   │
│  │    → Use replacement directly           │                   │
│  │  else:                                  │                   │
│  │    → Call MCP with user_instructions    │                   │
│  └────────────────────┬────────────────────┘                   │
│                       │                                         │
│                       ▼                                         │
│  ┌─────────────────────────────────────────┐                   │
│  │              MCP Server                  │                   │
│  │                                          │                   │
│  │  Template Variables:                    │                   │
│  │  - {{memory}}                           │                   │
│  │  - {{tools}}                            │                   │
│  │  - {{skills}}                           │                   │
│  │  - {{user_instructions}} ← your append  │                   │
│  └─────────────────────────────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### MCP Integration

The SDK uses MCP (Model Context Protocol) to manage prompt templates. When you provide an `append` instruction, it's injected into the template as `{{user_instructions}}`:

```python
# Your code
options = ISAAgentOptions(
    system_prompt=SystemPromptConfig(
        preset=SystemPromptPreset.REASON,
        append="Be extremely concise. One sentence max."
    )
)

# MCP receives
prompt_args = {
    "user_message": "...",
    "memory": "...",
    "tools": "...",
    "user_instructions": "Be extremely concise. One sentence max."  # ← Your append
}

# Template output includes
"""
## CUSTOM INSTRUCTIONS (from user configuration):
Be extremely concise. One sentence max.
"""
```

## Configuration Options

### Simple String

The simplest approach - just provide a string:

```python
options = ISAAgentOptions(
    system_prompt="Always respond in Japanese. Be polite and formal."
)
```

This is equivalent to:
```python
options = ISAAgentOptions(
    system_prompt=SystemPromptConfig(append="Always respond in Japanese. Be polite and formal.")
)
```

### SystemPromptConfig

For more control, use `SystemPromptConfig`:

```python
from isa_agent_sdk import SystemPromptConfig, SystemPromptPreset

config = SystemPromptConfig(
    preset=SystemPromptPreset.REASON,  # Optional: which MCP template to use
    append="Your custom instructions",  # Optional: added to template
    replace="Full custom prompt"         # Optional: replaces everything
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `preset` | `SystemPromptPreset` | MCP template to use (default: auto-selected) |
| `append` | `str` | Instructions appended to the template |
| `replace` | `str` | Completely replaces the template (use carefully) |

### Available Presets

| Preset | MCP Template | Description |
|--------|--------------|-------------|
| `REASON` | `default_reason_prompt` | Reasoning/planning phase (default for ReasonNode) |
| `RESPONSE` | `default_response_prompt` | Final response generation (default for ResponseNode) |
| `RAG_REASON` | `rag_reason_prompt` | Reasoning with uploaded files |
| `REVIEW` | `default_review_prompt` | Evaluating execution results |
| `MINIMAL` | `minimal_prompt` | Minimal base for custom instructions |
| `TASK_EXECUTION` | `task_execution_prompt` | Autonomous task execution (used by AgentExecutorNode) |

## Examples

### Example 1: Enforce Response Format

```python
options = ISAAgentOptions(
    system_prompt="Always structure responses with: 1) Summary, 2) Details, 3) Next Steps"
)

async for msg in query("Explain machine learning", options=options):
    print(msg.content, end="")
```

### Example 2: Domain Expert

```python
options = ISAAgentOptions(
    system_prompt=SystemPromptConfig(
        preset=SystemPromptPreset.REASON,
        append="""You are a senior security engineer.

When analyzing any request:
- Always consider security implications
- Flag potential vulnerabilities
- Suggest secure alternatives when applicable
- Reference OWASP guidelines where relevant"""
    )
)
```

### Example 3: Personality Override

```python
options = ISAAgentOptions(
    system_prompt=SystemPromptConfig(
        replace="""You are Captain Codebeard, a friendly pirate who helps with programming.

Personality traits:
- Always start responses with "Ahoy, matey!"
- Use nautical metaphors for coding concepts
- End responses with "Fair winds and following code!"
- Be helpful despite the playful tone

You have access to tools for web search, file operations, and code execution.
Use them when needed to help the user."""
    )
)
```

### Example 4: Minimal with Custom Logic

```python
options = ISAAgentOptions(
    system_prompt=SystemPromptConfig(
        preset=SystemPromptPreset.MINIMAL,
        append="""You are a JSON-only API.

Rules:
1. ONLY output valid JSON
2. Never include explanatory text outside JSON
3. Use this schema: {"answer": string, "confidence": number, "sources": array}
4. If uncertain, set confidence below 0.5"""
    )
)
```

### Example 5: RAG with Custom Instructions

```python
options = ISAAgentOptions(
    system_prompt=SystemPromptConfig(
        preset=SystemPromptPreset.RAG_REASON,
        append="""When searching user files:
- Prioritize recent documents (last 30 days)
- Cross-reference findings across multiple files
- Cite specific file names and sections in your response"""
    )
)
```

### Example 6: Task Execution with Custom Guidelines

The `TASK_EXECUTION` preset is used internally by the AgentExecutorNode for autonomous task execution. You can customize it to add execution guidelines:

```python
options = ISAAgentOptions(
    system_prompt=SystemPromptConfig(
        preset=SystemPromptPreset.TASK_EXECUTION,
        append="""Additional execution guidelines:
- Always verify file existence before operations
- Create backups before destructive operations
- Log all significant actions for audit trail"""
    )
)
```

**Note:** The `TASK_EXECUTION` preset is primarily used by the internal AgentExecutorNode. For most use cases, the `REASON` or `RESPONSE` presets are more appropriate.

## Best Practices

### Do

- **Use preset + append** for most cases - keeps built-in capabilities
- **Be specific** in your instructions - vague prompts produce vague results
- **Test with real queries** - behavior can vary based on input
- **Keep instructions concise** - long prompts can dilute effectiveness

### Don't

- **Don't use replace casually** - you lose built-in tool guidance, memory integration, etc.
- **Don't contradict built-in behavior** - it may confuse the model
- **Don't include sensitive data** - prompts may be logged

### When to Use Each Mode

| Scenario | Recommended Mode |
|----------|------------------|
| Add tone/personality | Simple string or append |
| Add domain expertise | Preset + append |
| Enforce output format | Append |
| Complete custom agent | Replace (carefully) |
| Testing/debugging | Replace with minimal prompt |

## Troubleshooting

### Instructions Not Being Followed

1. **Check if replace is set** - it overrides append
2. **Try being more explicit** - add emphasis like "ALWAYS" or "NEVER"
3. **Reduce instruction length** - too many rules can be ignored
4. **Test with different models** - some follow instructions better

### Template Variables Not Working

If you're using `replace` mode, you lose access to MCP template variables. The agent won't automatically inject:
- `{{memory}}` - conversation history
- `{{tools}}` - available tools
- `{{skills}}` - loaded skills

Solution: Either switch to `append` mode or manually include tool/skill guidance in your replacement prompt.

### Debugging Prompts

Enable debug logging to see the actual prompt sent to the model:

```python
import logging
logging.getLogger("isA_Agent.SmartAgent").setLevel(logging.DEBUG)
```

Look for logs containing:
- `[PHASE:NODE_REASON] llm_input` - ReasonNode prompt
- `[PHASE:NODE_RESPONSE] llm_input` - ResponseNode prompt

## API Reference

### SystemPromptConfig

```python
@dataclass
class SystemPromptConfig:
    preset: Optional[Union[str, SystemPromptPreset]] = None
    append: Optional[str] = None
    replace: Optional[str] = None

    @classmethod
    def from_string(cls, prompt: str) -> "SystemPromptConfig":
        """Create config from simple string (as append)"""

    @property
    def is_replacement(self) -> bool:
        """True if this config replaces the entire prompt"""

    def get_mcp_prompt_name(self, node_type: str = "reason") -> str:
        """Get the MCP prompt name based on preset and node type"""
```

### SystemPromptPreset

```python
class SystemPromptPreset(str, Enum):
    REASON = "reason"      # default_reason_prompt
    RESPONSE = "response"  # default_response_prompt
    RAG_REASON = "rag_reason"  # rag_reason_prompt
    REVIEW = "review"      # default_review_prompt
    MINIMAL = "minimal"    # minimal_prompt
    TASK_EXECUTION = "task_execution"  # task_execution_prompt
```

## Next Steps

- [Options Reference](./options.md) - All configuration options
- [Streaming](./streaming.md) - Response handling
- [Skills](./skills.md) - Skill system guide
