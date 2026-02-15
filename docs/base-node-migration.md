# BaseNode Migration Guide

## Overview

As of version 1.x.x, `BaseNode` has been deprecated in favor of specialized base classes that follow the Interface Segregation Principle (ISP). This improves performance, reduces memory overhead, and provides clearer interfaces for different node types.

## Why the Change?

The original `BaseNode` forced all nodes to inherit 7 mixins, even if they only used 2-3:

- ConfigExtractorMixin
- MCPSearchMixin
- MCPSecurityMixin
- MCPToolExecutionMixin
- MCPAssetsMixin
- StreamingMixin
- ModelCallingMixin

This violated the Interface Segregation Principle, leading to:
- Unnecessary memory overhead (unused capabilities loaded)
- IDE confusion (autocomplete showing irrelevant methods)
- Unclear contracts (what does this node actually do?)

## New Base Classes

### MinimalBaseNode

Core functionality only - use for custom nodes that don't need MCP or model calling.

**Provides:**
- `execute()` entry point with error handling
- `_execute_logic()` abstract method
- `state`/`config` properties (concurrency-safe)
- Basic logging

**Does NOT provide:**
- MCP operations
- Model calling
- Streaming
- Config extraction helpers

**Example:**
```python
from isa_agent_sdk.nodes import MinimalBaseNode
from isa_agent_sdk.agent_types.agent_state import AgentState
from langchain_core.runnables import RunnableConfig

class CustomValidatorNode(MinimalBaseNode):
    def __init__(self):
        super().__init__("CustomValidator")

    async def _execute_logic(self, state: AgentState, config: RunnableConfig) -> AgentState:
        # Custom validation logic
        messages = state.get("messages", [])
        if not messages:
            raise ValueError("No messages to validate")
        return state
```

### ToolExecutionNode

For nodes that execute MCP tools - use for tool execution nodes.

**Provides:**
- All MinimalBaseNode functionality
- ConfigExtractorMixin (get_runtime_context, get_user_id, etc.)
- MCPToolExecutionMixin (mcp_call_tool, etc.)
- MCPSecurityMixin (mcp_get_tool_security_levels, etc.)
- StreamingMixin (stream_tool, stream_custom)

**Example:**
```python
from isa_agent_sdk.nodes import ToolExecutionNode

class CustomToolNode(ToolExecutionNode):
    def __init__(self):
        super().__init__("CustomToolNode")

    async def _execute_logic(self, state: AgentState, config: RunnableConfig) -> AgentState:
        # Execute tool with streaming
        result = await self.mcp_call_tool(
            tool_name="my_tool",
            args={"param": "value"},
            config=config
        )
        self.stream_tool("my_tool", f"Completed: {result}")
        return {"messages": [result]}
```

### ReasoningNode

For nodes that call LLMs - use for reasoning/response nodes.

**Provides:**
- All MinimalBaseNode functionality
- ConfigExtractorMixin (get_runtime_context, get_user_id, etc.)
- ModelCallingMixin (call_model, _messages_to_prompt)
- StreamingMixin (stream_custom)
- MCPAssetsMixin (mcp_get_prompt, mcp_get_resource)

**Example:**
```python
from isa_agent_sdk.nodes import ReasoningNode
from langchain_core.messages import SystemMessage

class CustomReasonNode(ReasoningNode):
    def __init__(self):
        super().__init__("CustomReasonNode")

    async def _execute_logic(self, state: AgentState, config: RunnableConfig) -> AgentState:
        # Get system prompt from MCP
        system_prompt = await self.mcp_get_prompt(
            "my_custom_prompt",
            {"context": "value"},
            config
        )

        # Call model with streaming
        response = await self.call_model(
            messages=[SystemMessage(content=system_prompt)] + state.get("messages", []),
            tools=[],
            stream_tokens=True
        )

        return {"messages": [response]}
```

## Migration Steps

### 1. Identify Your Node Type

Determine which specialized base class fits your node:

| Current Node | Uses Tools? | Calls LLM? | New Base Class |
|-------------|-------------|------------|----------------|
| ToolNode | ✅ Yes | ❌ No | ToolExecutionNode |
| ReasonNode | ❌ No | ✅ Yes | ReasoningNode |
| ResponseNode | ❌ No | ✅ Yes | ReasoningNode |
| Custom validator | ❌ No | ❌ No | MinimalBaseNode |

### 2. Update Import Statement

```python
# Before
from isa_agent_sdk.nodes import BaseNode

# After - for tool nodes
from isa_agent_sdk.nodes import ToolExecutionNode

# After - for reasoning/response nodes
from isa_agent_sdk.nodes import ReasoningNode

# After - for minimal custom nodes
from isa_agent_sdk.nodes import MinimalBaseNode
```

### 3. Update Class Definition

```python
# Before
class MyToolNode(BaseNode):
    def __init__(self):
        super().__init__("MyToolNode")

    async def _execute_logic(self, state, config):
        # ...

# After
class MyToolNode(ToolExecutionNode):
    def __init__(self):
        super().__init__("MyToolNode")

    async def _execute_logic(self, state, config):
        # Same logic - no changes needed!
```

### 4. Test Your Node

Run your tests to ensure everything works:

```bash
pytest tests/test_my_node.py -xvs
```

## Breaking Changes

None! The old `BaseNode` continues to work with a deprecation warning. You can migrate at your own pace.

## Timeline

- **v1.x.x**: BaseNode deprecated with warning
- **v2.0.0**: BaseNode will be removed (planned)

## Benefits After Migration

1. **Better Performance**: Only load mixins you actually use
2. **Clearer Intent**: Base class name reveals node purpose
3. **Better IDE Support**: Autocomplete shows only relevant methods
4. **Reduced Memory**: Smaller object footprint per node

## Need Help?

- Review existing nodes: `ReasonNode`, `ToolNode` use the new base classes
- Check tests: `tests/test_dag.py` shows usage patterns
- Open an issue: https://github.com/yourorg/isa_agent_sdk/issues

## Related Issues

- [#44: BaseNode violates Interface Segregation Principle](https://github.com/yourorg/isa_agent_sdk/issues/44)
