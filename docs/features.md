# Features

## Core Features

- **Claude Agent SDK Compatible** - Familiar API patterns
- **Streaming Messages** - Real-time response streaming
- **Built-in Tools** - Read, Write, Edit, Bash, WebSearch, etc.
- **MCP Integration** - Model Context Protocol support
- **Human-in-the-Loop** - Durable execution with checkpoints
- **Skills System** - Local-first skill loading with MCP fallback
- **Project Config** - `.isa` directory for project-specific settings
- **Event Triggers** - Proactive agent activation
- **Multiple Execution Modes** - Reactive, Collaborative, Proactive
- **A2A Ready** - Agent Card + JSON-RPC client/server adapters

## Multi-Agent Features

### Swarm Orchestration
- **Dynamic Handoffs** - Agents decide when to hand off control via `[HANDOFF: agent_name]` directives
- **Streaming Events** - Track agent transitions with `swarm_agent_start` and `swarm_handoff` events
- **Handoff Trace** - Complete audit trail of agent handoffs
- **Max Handoffs Safety** - Configurable limit prevents infinite loops
- **Shared State** - State accumulates across agent handoffs
- See [Swarm Orchestration](./swarm.md) for details

### DAG Task Execution
- **Dependency Ordering** - Tasks specify `depends_on` for execution order
- **Wavefront Parallelism** - Independent tasks in same wavefront run concurrently
- **Cycle Detection** - Kahn's algorithm validates DAG structure
- **Failure Cascade** - Failed tasks automatically mark dependents as SKIPPED
- **Multi-Agent DAGs** - Different agents execute different tasks in parallel
- **Task Status Tracking** - PENDING, READY, RUNNING, COMPLETED, FAILED, SKIPPED
- See [DAG Scheduler](../isa_agent_sdk/dag/scheduler.py) for implementation

### MultiAgent Orchestrator
- **Fixed Routing** - Explicit router function controls agent transitions
- **Shared State** - Mutable state passed across agents
- **Per-Agent Config** - Each agent has its own skills, tools, and options
- See [Multi-Agent](./multi-agent.md) for details

## Advanced Features

### MCP Resource Management
- **Dynamic Rule Loading**: Patterns loaded from MCP resources at runtime
- **Resource Caching**: Efficient pattern compilation and caching
- **Fallback Handling**: Default patterns when MCP resources unavailable
- **Multi-Resource Support**: PII, medical, and policy resources

### Pattern Extensibility
- **MCP-Driven Patterns**: Patterns defined in MCP resources
- **New PII Types**: Additional violation types via resource updates
- **Severity Levels**: Support for different violation severities
- **Custom Actions**: Extensible enforcement action system

### Medical Compliance
- **HIPAA Integration**: Healthcare data protection rules
- **Medical Keyword Detection**: Healthcare-specific terminology
- **Compliance Recommendations**: Actionable compliance guidance
- **Risk-Based Assessment**: Severity-weighted risk scoring

### Integration Extensions
- **MCP Resource System**: Dynamic policy management via MCP
- **External Compliance**: Integration with external compliance systems
- **Audit Logging**: Comprehensive compliance event logging
- **Reporting**: Violation pattern analysis and reporting
- **Policy Management**: Dynamic policy updates via MCP resources

## Core Features

### Confidence Assessment
- **Rule-based evaluation**: Detects uncertainty indicators, partial response identifiers, error keywords
- **AI-enhanced assessment**: Uses AI models for secondary evaluation in complex cases
- **Multi-dimensional scoring**: Considers response length, language certainty, completeness
- **Dynamic thresholds**: Configurable confidence threshold (default: 0.7)

### Smart Error Categorization
Automatically classifies problematic responses into 6 categories:

| Category | Description | Example Trigger Keywords |
|----------|-------------|-------------------------|
| `UNCERTAINTY` | High uncertainty | "not sure", "don't know", "maybe", "possibly" |
| `INSUFFICIENT_INFO` | Insufficient information | "not enough information", "need more details" |
| `AMBIGUOUS_QUERY` | Ambiguous query | "ambiguous", "unclear", "multiple interpretations" |
| `TOOL_FAILURE` | Tool execution failure | "tool failed", "execution failed", "error occurred" |
| `TIMEOUT` | Request timeout | "timeout", "timed out", "request expired" |
| `TECHNICAL_LIMITATION` | Technical limitation | "technical limitation", "cannot process", "not capable" |

### Graceful Failure Handling
- **Context-aware responses**: Generate appropriate alternative answers based on issue type
- **User-friendly feedback**: Provide constructive guidance and suggestions
- **Transparency**: Honestly acknowledge limitations without providing potentially inaccurate information
- **Actionable advice**: Offer specific next steps for users

## References

- [README.md](./README.md)
- [isa_agent_sdk/nodes/docs/failsafe_node.md](./isa_agent_sdk/nodes/docs/failsafe_node.md)
- [isa_agent_sdk/nodes/docs/guardrail_node.md](./isa_agent_sdk/nodes/docs/guardrail_node.md)
- [isa_agent_sdk/services/auto_detection/DESIGN.md](./isa_agent_sdk/services/auto_detection/DESIGN.md)
- [isa_agent_sdk/services/background_jobs/docs/INTEGRATION_COMPLETE.md](./isa_agent_sdk/services/background_jobs/docs/INTEGRATION_COMPLETE.md)
- [isa_agent_sdk/services/feedback/MIGRATION_SUMMARY.md](./isa_agent_sdk/services/feedback/MIGRATION_SUMMARY.md)
- [isa_agent_sdk/services/feedback/README.md](./isa_agent_sdk/services/feedback/README.md)
- [options.md](./options.md)
- [product/agent_creation_status.md](./product/agent_creation_status.md)
- [research/claude_comparison.md](./research/claude_comparison.md)
