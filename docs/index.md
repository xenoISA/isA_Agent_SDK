# isA Agent SDK Documentation

A complete AI Agent SDK for building intelligent agents with advanced features. Compatible with Claude Agent SDK patterns, with additional capabilities including multi-agent orchestration, DAG task execution, and swarm dynamics.

## Getting Started

New to isA Agent SDK? Start here:

- **[Installation](./installation.md)** - Install the SDK and set up your environment
- **[Quickstart](./quickstart.md)** - Get started in 5 minutes with basic examples
- **[Examples](./examples.md)** - Practical code examples for common use cases

## Core Concepts

Understand the fundamentals:

- **[Core Concepts](./concepts.md)** - Agent architecture, nodes, and execution flow
- **[Configuration](./configuration.md)** - Configure agent behavior and options
- **[Options](./options.md)** - Complete reference for ISAAgentOptions
- **[Messages](./messages.md)** - Message types and streaming
- **[System Prompts](./system-prompts.md)** - System prompt configuration

## Key Features

Deep dives into major capabilities:

### Multi-Agent Systems
- **[Multi-Agent](./multi-agent.md)** - Fixed-routing multi-agent orchestration
- **[Swarm Orchestration](./swarm.md)** - Dynamic agent handoffs and DAG execution
- **[DAG System](../isa_agent_sdk/dag/)** - Dependency-ordered task execution with wavefronts

### Tools & Execution
- **[Tools](./tools.md)** - Built-in tools and custom tool integration
- **[Skills](./skills.md)** - Local-first skill system with MCP fallback
- **[Streaming](./streaming.md)** - Real-time message streaming
- **[Structured Outputs](./structured-outputs.md)** - Type-safe outputs with Pydantic

### Human-in-the-Loop
- **[Human-in-the-Loop](./human-in-the-loop.md)** - Request permissions and create checkpoints
- **[Checkpointing](./checkpointing.md)** - Durable execution with state persistence
- **[Long-Running Tasks](./long-running-tasks.md)** - Background task execution

### Advanced Features
- **[A2A Integration](./a2a.md)** - Agent-to-Agent protocol support
- **[Triggers](./triggers.md)** - Proactive agent activation
- **[Memory](./memory.md)** - Conversation memory and context
- **[Steward](./steward.md)** - Agent stewardship and monitoring

## Development

Build and deploy production agents:

- **[Testing](./testing.md)** - Test your agent code (234 tests in SDK)
- **[Deployment](./deployment.md)** - Deploy agents to production
- **[Deployment Guide](./deployment-guide.md)** - Detailed deployment instructions
- **[Environment](./environment.md)** - Environment variables and configuration
- **[Desktop Execution](./desktop-execution.md)** - Run agents on desktop

## API Reference

Complete API documentation:

- **[API Reference](./api-reference.md)** - Full API documentation
- **[Agent Client](./agent-client.md)** - HTTP client for deployed agents

## Research & Comparisons

Background research and design decisions:

- **[Claude Comparison](./research/claude_comparison.md)** - Comparison with Claude Agent SDK
- **[Agent Teams Analysis](./comparison-claude-agent-teams-vs-isa-sdk.md)** - Multi-agent architecture analysis
- **[Product Status](./product/agent_creation_status.md)** - Development status

## Node Documentation

Internal node implementations:

- [Entry Node](../isa_agent_sdk/nodes/docs/entry_node.md) - Session validation and context preparation
- [Model Node](../isa_agent_sdk/nodes/docs/model_node.md) - LLM interaction and streaming
- [Tool Node](../isa_agent_sdk/nodes/docs/tool_node.md) - Tool execution
- [Router Node](../isa_agent_sdk/nodes/docs/router_node.md) - Routing decisions
- [Response Node](../isa_agent_sdk/nodes/docs/response_node.md) - Response formatting
- [Guardrail Node](../isa_agent_sdk/nodes/docs/guardrail_node.md) - Safety guardrails
- [Revise Node](../isa_agent_sdk/nodes/docs/revise_node.md) - Response revision
- [Failsafe Node](../isa_agent_sdk/nodes/docs/failsafe_node.md) - Error handling

## Service Documentation

Background services and utilities:

- [Auto Detection](../isa_agent_sdk/services/auto_detection/README.md) - Automatic service detection
- [Background Jobs](../isa_agent_sdk/services/background_jobs/README.md) - Async task execution
- [Feedback System](../isa_agent_sdk/services/feedback/README.md) - User feedback integration
- [Human-in-the-Loop Service](../isa_agent_sdk/services/human_in_the_loop/README.md) - HIL implementation
- [Trace Migrations](../isa_agent_sdk/services/trace/migrations/README.md) - Database migrations

## Key Highlights

- ✅ **234 passing tests** - Comprehensive coverage including security audit validation
- ✅ **Multi-agent ready** - Swarm orchestration with dynamic handoffs
- ✅ **DAG execution** - Dependency-ordered task pipelines with parallel wavefronts
- ✅ **Audit complete** - All security findings (Issues #13-#26) resolved
- ✅ **Production ready** - Battle-tested with real-world deployments

## Support

Need help?

- **[Support](./support.md)** - Get help and report issues
- **GitHub Issues** - https://github.com/xenoISA/isA_Agent_SDK/issues
- **Documentation Bugs** - Report at the repository above

---

**Quick Links:**
[Installation](./installation.md) |
[Quickstart](./quickstart.md) |
[Examples](./examples.md) |
[API Reference](./api-reference.md) |
[Swarm](./swarm.md) |
[Testing](./testing.md)
