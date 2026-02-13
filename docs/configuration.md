# Configuration

## Configuration

### ISA Client Configuration
- **Default URL**: `http://localhost:8082`
- **Service Type**: `text` for chat interactions
- **Task Type**: `chat` for conversational AI

### Streaming Configuration
- Real-time token delivery enabled by default
- Comprehensive billing tracking
- Tool call detection and processing

## Configuration

### Routing Constants
- **Direct End**: `next_action = "end"`
- **Tool Execution**: `next_action = "call_tool"`
- **Autonomous Mode**: `next_action = "call_tool"` + `is_autonomous = True`

### Strategy Types
- **`"direct"`**: No tool execution needed
- **`"tool_call"`**: Regular tool execution
- **`"autonomous_planning"`**: Autonomous mode activation

## Configuration

### Environment Variables
- No direct environment variable dependencies
- Inherits MCP and session manager configurations

### Initialization Parameters
```python
EntryNode(mcp_manager: MCPManager, session_manager: SessionManager)
```

## References

- [checkpointing.md](./checkpointing.md)
- [deployment-guide.md](./deployment-guide.md)
- [desktop-execution.md](./desktop-execution.md)
- [isa_agent_sdk/nodes/docs/entry_node.md](./isa_agent_sdk/nodes/docs/entry_node.md)
- [isa_agent_sdk/nodes/docs/failsafe_node.md](./isa_agent_sdk/nodes/docs/failsafe_node.md)
- [isa_agent_sdk/nodes/docs/guardrail_node.md](./isa_agent_sdk/nodes/docs/guardrail_node.md)
- [isa_agent_sdk/nodes/docs/model_node.md](./isa_agent_sdk/nodes/docs/model_node.md)
- [isa_agent_sdk/nodes/docs/revise_node.md](./isa_agent_sdk/nodes/docs/revise_node.md)
- [isa_agent_sdk/nodes/docs/router_node.md](./isa_agent_sdk/nodes/docs/router_node.md)
- [isa_agent_sdk/nodes/docs/tool_node.md](./isa_agent_sdk/nodes/docs/tool_node.md)
- [isa_agent_sdk/services/auto_detection/DESIGN.md](./isa_agent_sdk/services/auto_detection/DESIGN.md)
- [isa_agent_sdk/services/background_jobs/docs/INTEGRATION_COMPLETE.md](./isa_agent_sdk/services/background_jobs/docs/INTEGRATION_COMPLETE.md)
- [isa_agent_sdk/services/feedback/README.md](./isa_agent_sdk/services/feedback/README.md)
- [memory.md](./memory.md)
- [options.md](./options.md)
- [product/agent_creation_status.md](./product/agent_creation_status.md)
- [skills.md](./skills.md)
- [steward.md](./steward.md)
- [structured-outputs.md](./structured-outputs.md)
- [swarm.md](./swarm.md)
- [system-prompts.md](./system-prompts.md)
- [tools.md](./tools.md)
- [triggers.md](./triggers.md)
