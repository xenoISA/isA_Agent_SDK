# Core Concepts

## Overview

The `ModelNode` is responsible for LLM (Large Language Model) interactions within the agent workflow. It handles model calls, streaming responses, billing integration, and sensitive request detection for human-in-the-loop scenarios.

## Overview

The `RouterNode` is responsible for intelligent routing decisions within the agent workflow. It analyzes AI responses to determine the next appropriate action, detects autonomous planning requests, and manages workflow direction based on tool call patterns.

## Overview

The `EntryNode` is the first node in the agent workflow, responsible for preparing the agent execution context. It handles session validation, memory retrieval, tool discovery, and enhanced prompt construction.

## References

- [checkpointing.md](./checkpointing.md)
- [deployment-guide.md](./deployment-guide.md)
- [desktop-execution.md](./desktop-execution.md)
- [human-in-the-loop.md](./human-in-the-loop.md)
- [isa_agent_sdk/nodes/docs/entry_node.md](./isa_agent_sdk/nodes/docs/entry_node.md)
- [isa_agent_sdk/nodes/docs/failsafe_node.md](./isa_agent_sdk/nodes/docs/failsafe_node.md)
- [isa_agent_sdk/nodes/docs/guardrail_node.md](./isa_agent_sdk/nodes/docs/guardrail_node.md)
- [isa_agent_sdk/nodes/docs/model_node.md](./isa_agent_sdk/nodes/docs/model_node.md)
- [isa_agent_sdk/nodes/docs/revise_node.md](./isa_agent_sdk/nodes/docs/revise_node.md)
- [isa_agent_sdk/nodes/docs/router_node.md](./isa_agent_sdk/nodes/docs/router_node.md)
- [isa_agent_sdk/nodes/docs/tool_node.md](./isa_agent_sdk/nodes/docs/tool_node.md)
- [isa_agent_sdk/services/auto_detection/DESIGN.md](./isa_agent_sdk/services/auto_detection/DESIGN.md)
- [isa_agent_sdk/services/auto_detection/MCP_PROGRESS_INTEGRATION.md](./isa_agent_sdk/services/auto_detection/MCP_PROGRESS_INTEGRATION.md)
- [isa_agent_sdk/services/background_jobs/docs/INTEGRATION_COMPLETE.md](./isa_agent_sdk/services/background_jobs/docs/INTEGRATION_COMPLETE.md)
- [isa_agent_sdk/services/feedback/INDUSTRY_ANALYSIS.md](./isa_agent_sdk/services/feedback/INDUSTRY_ANALYSIS.md)
- [isa_agent_sdk/services/feedback/MIGRATION_SUMMARY.md](./isa_agent_sdk/services/feedback/MIGRATION_SUMMARY.md)
- [isa_agent_sdk/services/feedback/README.md](./isa_agent_sdk/services/feedback/README.md)
- [isa_agent_sdk/services/human_in_the_loop/README.md](./isa_agent_sdk/services/human_in_the_loop/README.md)
- [isa_agent_sdk/services/human_in_the_loop/old/FEATURE_COMPARISON.md](./isa_agent_sdk/services/human_in_the_loop/old/FEATURE_COMPARISON.md)
- [isa_agent_sdk/services/trace/migrations/README.md](./isa_agent_sdk/services/trace/migrations/README.md)
- [memory.md](./memory.md)
- [messages.md](./messages.md)
- [options.md](./options.md)
- [skills.md](./skills.md)
- [steward.md](./steward.md)
- [structured-outputs.md](./structured-outputs.md)
- [system-prompts.md](./system-prompts.md)
- [tools.md](./tools.md)
- [triggers.md](./triggers.md)
