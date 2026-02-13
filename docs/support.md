# Support

# With FastAPI server support
pip install isa-agent-sdk[server]
```

## Project Setup

Initialize a `.isa` directory for project-specific configuration:

```bash
mkdir -p .isa/skills
```

```
my-project/
├── .isa/
│   ├── config.json          # Project configuration
│   ├── settings.local.json  # MCP servers, permissions
│   └── skills/              # Local skills (loaded first)
│       └── my-skill/
│           └── SKILL.md
└── ...
```

Skills in `.isa/skills/` are loaded before MCP, allowing project-specific overrides.

## Quick Start

### Basic Query

```python
from isa_agent_sdk import query, ISAAgentOptions

### Supported Patterns
- `plan_autonomous_task` - Basic autonomous planning
- `create_plan_autonomous_task_executor` - Extended planning with execution
- `autonomous_plan_autonomous_task_workflow` - Complex workflow planning
- Any tool name containing `plan_autonomous_task` substring

### Common Issues
1. **Tools Not Detected**: Check tool_calls format and naming
2. **Wrong Routing**: Verify autonomous pattern matching
3. **Missing Updates**: Check stream writer configuration
4. **State Loss**: Ensure proper state preservation

## References

- [README.md](./README.md)
- [deployment-guide.md](./deployment-guide.md)
- [desktop-execution.md](./desktop-execution.md)
- [isa_agent_sdk/nodes/docs/failsafe_node.md](./isa_agent_sdk/nodes/docs/failsafe_node.md)
- [isa_agent_sdk/nodes/docs/guardrail_node.md](./isa_agent_sdk/nodes/docs/guardrail_node.md)
- [isa_agent_sdk/nodes/docs/revise_node.md](./isa_agent_sdk/nodes/docs/revise_node.md)
- [isa_agent_sdk/nodes/docs/router_node.md](./isa_agent_sdk/nodes/docs/router_node.md)
- [isa_agent_sdk/nodes/docs/tool_node.md](./isa_agent_sdk/nodes/docs/tool_node.md)
- [isa_agent_sdk/services/auto_detection/MCP_PROGRESS_INTEGRATION.md](./isa_agent_sdk/services/auto_detection/MCP_PROGRESS_INTEGRATION.md)
- [isa_agent_sdk/services/background_jobs/docs/DOCKER_TEST_RESULTS.md](./isa_agent_sdk/services/background_jobs/docs/DOCKER_TEST_RESULTS.md)
- [isa_agent_sdk/services/background_jobs/docs/INTEGRATION_COMPLETE.md](./isa_agent_sdk/services/background_jobs/docs/INTEGRATION_COMPLETE.md)
- [isa_agent_sdk/services/feedback/README.md](./isa_agent_sdk/services/feedback/README.md)
- [isa_agent_sdk/services/human_in_the_loop/README.md](./isa_agent_sdk/services/human_in_the_loop/README.md)
- [memory.md](./memory.md)
- [messages.md](./messages.md)
- [research/claude_comparison.md](./research/claude_comparison.md)
- [swarm.md](./swarm.md)
- [triggers.md](./triggers.md)
