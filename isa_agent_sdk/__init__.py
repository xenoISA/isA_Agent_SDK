#!/usr/bin/env python3
"""
isA Agent SDK
=============

A complete AI Agent SDK for building intelligent agents with advanced features.
Compatible with Claude Agent SDK patterns, with additional capabilities.

Quick Start:
    from isa_agent_sdk import query, ISAAgentOptions

    # Simple usage
    async for msg in query("Hello, world!"):
        print(msg.content)

    # With options
    async for msg in query(
        "Fix the bug",
        options=ISAAgentOptions(
            allowed_tools=["Read", "Edit", "Bash"],
            execution_mode="collaborative"
        )
    ):
        print(msg)

    # Human-in-the-loop
    from isa_agent_sdk import request_tool_permission, checkpoint

    authorized = await request_tool_permission("delete_file", {"path": "data.txt"})
    if authorized:
        # proceed...

    # HTTP Client (for deployed apps)
    from isa_agent_sdk import ISAAgent

    client = ISAAgent(base_url="http://localhost:8000")
    response = client.chat.create(message="Hello!", user_id="user123")

Features:
    - Claude Agent SDK-compatible API
    - Streaming message interface
    - Built-in tools (Read, Write, Edit, Bash, etc.)
    - MCP server integration
    - Skills system (prompt injection)
    - Human-in-the-Loop (durable execution, survives restarts)
    - Multiple execution modes (reactive, collaborative, proactive)
    - Multiple environments (cloud_pool, cloud_shared, desktop)
    - Event triggers (proactive agent activation)
    - Structured outputs (JSON schema, Pydantic integration)

Structured Outputs Example:
    from pydantic import BaseModel
    from isa_agent_sdk import query, ISAAgentOptions, OutputFormat

    class Recipe(BaseModel):
        name: str
        ingredients: list[str]
        prep_time_minutes: int

    async for msg in query(
        "Find a chocolate chip cookie recipe",
        options=ISAAgentOptions(
            output_format=OutputFormat.from_pydantic(Recipe)
        )
    ):
        if msg.has_structured_output:
            recipe = msg.parse(Recipe)
            print(f"Recipe: {recipe.name}")
"""

# Version
__version__ = "0.1.0"

# Error classes (import early, no dependencies)
from .errors import (
    ISASDKError,
    ConnectionError,
    TimeoutError,
    CircuitBreakerError,
    RateLimitError,
    ExecutionError,
    ToolExecutionError,
    ModelError,
    MaxIterationsError,
    GraphExecutionError,
    SessionError,
    SessionNotFoundError,
    SessionExpiredError,
    CheckpointError,
    ResumeError,
    ValidationError,
    SchemaError,
    ConfigurationError,
    PermissionError,
    ToolPermissionError,
    HILDeniedError,
    HILTimeoutError,
    MCPError,
    MCPConnectionError,
    MCPToolNotFoundError,
)

# Configuration options (no circular imports)
from .options import (
    ISAAgentOptions,
    Options,  # Alias
    ExecutionEnv,
    ExecutionMode,
    ToolDiscoveryMode,
    PermissionMode,
    GuardrailMode,
    OutputFormat,
    OutputFormatType,
    SystemPromptConfig,
    SystemPromptPreset,
    MCPServerConfig,
    AgentDefinition,
    HookMatcher,
    PoolConfig,
    TriggerConfig,
)

# HTTP Client (standalone, no circular imports)
from .agent_client import (
    ISAAgent,
    ISAAgentSync,
    AgentEvent,
    AgentResponse,
    SessionInfo,
    EventType,
)


# Cache for lazily imported modules
_lazy_modules = {}

# Lazy imports for modules with complex dependencies
def __getattr__(name):
    """Lazy import for modules with circular dependency potential"""
    import importlib

    # Core query functions
    if name in ("query", "query_sync", "ask", "ask_sync", "execute_tool",
                "get_available_tools", "get_session_state", "resume",
                "resume_sync", "QueryExecutor"):
        if "_query" not in _lazy_modules:
            _lazy_modules["_query"] = importlib.import_module("._query", "isa_agent_sdk")
        return getattr(_lazy_modules["_query"], name)

    # Message types
    if name in ("AgentMessage", "ConversationHistory", "ISAEventType",
                "EventData", "EventEmitter", "EventCategory",
                "MESSAGE_TYPE_MAP", "REVERSE_TYPE_MAP", "ResultSubtype",
                "ResultMessage"):
        if "_messages" not in _lazy_modules:
            _lazy_modules["_messages"] = importlib.import_module("._messages", "isa_agent_sdk")
        return getattr(_lazy_modules["_messages"], name)

    # Structured output helpers
    if name in ("StructuredOutputResult", "StructuredOutputParser",
                "parse_structured_output", "parse_json_response"):
        if "_structured" not in _lazy_modules:
            _lazy_modules["_structured"] = importlib.import_module("._structured", "isa_agent_sdk")
        return getattr(_lazy_modules["_structured"], name)

    # Human-in-the-Loop
    if name in ("get_hil", "request_tool_permission", "request_authorization",
                "collect_input", "collect_credentials", "collect_selection",
                "request_review", "request_plan_approval", "request_execution_choice",
                "checkpoint", "get_hil_stats", "clear_hil_history"):
        if "_hil" not in _lazy_modules:
            _lazy_modules["_hil"] = importlib.import_module("._hil", "isa_agent_sdk")
        return getattr(_lazy_modules["_hil"], name)

    # Skills system
    if name in ("Skill", "SkillManager", "BUILTIN_SKILLS", "get_skill_manager",
                "load_skill", "load_builtin_skill", "activate_skills",
                "get_skill_injection", "list_builtin_skills"):
        if "_skills" not in _lazy_modules:
            _lazy_modules["_skills"] = importlib.import_module("._skills", "isa_agent_sdk")
        return getattr(_lazy_modules["_skills"], name)

    # Event Triggers
    if name in ("TriggerType", "TriggerCondition", "register_trigger",
                "unregister_trigger", "get_user_triggers", "get_trigger_stats",
                "initialize_triggers", "shutdown_triggers", "get_trigger_manager",
                "register_price_trigger", "register_schedule_trigger",
                "register_event_pattern_trigger"):
        if "_triggers" not in _lazy_modules:
            _lazy_modules["_triggers"] = importlib.import_module("._triggers", "isa_agent_sdk")
        return getattr(_lazy_modules["_triggers"], name)

    # Custom Tools (@tool decorator)
    if name in ("tool", "SDKTool", "SDKMCPServer", "create_sdk_mcp_server",
                "register_sdk_server", "get_sdk_server", "execute_sdk_tool",
                "is_sdk_tool"):
        if "_tools" not in _lazy_modules:
            _lazy_modules["_tools"] = importlib.import_module("._tools", "isa_agent_sdk")
        return getattr(_lazy_modules["_tools"], name)

    # Agent abstractions (single + multi-agent)
    if name in ("Agent", "AgentRunResult", "MultiAgentOrchestrator", "MultiAgentResult",
                "SwarmOrchestrator", "SwarmAgent", "SwarmRunResult", "SwarmState"):
        if "_agents" not in _lazy_modules:
            _lazy_modules["_agents"] = importlib.import_module(".agents", "isa_agent_sdk")
        return getattr(_lazy_modules["_agents"], name)

    # Bidirectional client
    if name in ("ISAAgentClient", "ISAAgentClientSync", "ClientMode"):
        if "_client" not in _lazy_modules:
            _lazy_modules["_client"] = importlib.import_module("._client", "isa_agent_sdk")
        return getattr(_lazy_modules["_client"], name)

    # Project Context (ISA.md/CLAUDE.md support)
    if name in ("load_project_context", "discover_project_context_file",
                "format_project_context_for_prompt"):
        if "_project_context" not in _lazy_modules:
            _lazy_modules["_project_context"] = importlib.import_module(".utils.project_context", "isa_agent_sdk")
        return getattr(_lazy_modules["_project_context"], name)

    # A2A support
    if name in ("A2AAgentCard", "A2AClient", "A2AServerAdapter", "register_a2a_fastapi_routes",
                "build_auth_service_token_validator"):
        if "_a2a" not in _lazy_modules:
            _lazy_modules["_a2a"] = importlib.import_module(".a2a", "isa_agent_sdk")
        return getattr(_lazy_modules["_a2a"], name)

    raise AttributeError(f"module 'isa_agent_sdk' has no attribute '{name}'")

__all__ = [
    # Version
    "__version__",

    # Error classes
    "ISASDKError",
    "ConnectionError",
    "TimeoutError",
    "CircuitBreakerError",
    "RateLimitError",
    "ExecutionError",
    "ToolExecutionError",
    "ModelError",
    "MaxIterationsError",
    "GraphExecutionError",
    "SessionError",
    "SessionNotFoundError",
    "SessionExpiredError",
    "CheckpointError",
    "ResumeError",
    "ValidationError",
    "SchemaError",
    "ConfigurationError",
    "PermissionError",
    "ToolPermissionError",
    "HILDeniedError",
    "HILTimeoutError",
    "MCPError",
    "MCPConnectionError",
    "MCPToolNotFoundError",

    # Core functions
    "query",
    "query_sync",
    "ask",
    "ask_sync",
    "execute_tool",
    "get_available_tools",
    "get_session_state",
    "resume",
    "resume_sync",
    "QueryExecutor",

    # Options
    "ISAAgentOptions",
    "Options",
    "ExecutionEnv",
    "ExecutionMode",
    "ToolDiscoveryMode",
    "PermissionMode",
    "GuardrailMode",
    "OutputFormat",
    "OutputFormatType",
    "SystemPromptConfig",
    "SystemPromptPreset",
    "MCPServerConfig",
    "AgentDefinition",
    "HookMatcher",
    "PoolConfig",
    "TriggerConfig",

    # Messages
    "AgentMessage",
    "ConversationHistory",
    "ResultMessage",
    "ResultSubtype",
    "ISAEventType",
    "EventData",
    "EventEmitter",
    "EventCategory",
    "MESSAGE_TYPE_MAP",
    "REVERSE_TYPE_MAP",

    # Structured Output
    "StructuredOutputResult",
    "StructuredOutputParser",
    "parse_structured_output",
    "parse_json_response",

    # Human-in-the-Loop
    "get_hil",
    "request_tool_permission",
    "request_authorization",
    "collect_input",
    "collect_credentials",
    "collect_selection",
    "request_review",
    "request_plan_approval",
    "request_execution_choice",
    "checkpoint",
    "get_hil_stats",
    "clear_hil_history",

    # Skills
    "Skill",
    "SkillManager",
    "BUILTIN_SKILLS",
    "get_skill_manager",
    "load_skill",
    "load_builtin_skill",
    "activate_skills",
    "get_skill_injection",
    "list_builtin_skills",

    # Event Triggers
    "TriggerType",
    "TriggerCondition",
    "register_trigger",
    "unregister_trigger",
    "get_user_triggers",
    "get_trigger_stats",
    "initialize_triggers",
    "shutdown_triggers",
    "get_trigger_manager",
    "register_price_trigger",
    "register_schedule_trigger",
    "register_event_pattern_trigger",

    # Custom Tools (@tool decorator)
    "tool",
    "SDKTool",
    "SDKMCPServer",
    "create_sdk_mcp_server",
    "register_sdk_server",
    "get_sdk_server",
    "execute_sdk_tool",
    "is_sdk_tool",

    # Agent abstractions
    "Agent",
    "AgentRunResult",
    "MultiAgentOrchestrator",
    "MultiAgentResult",
    "SwarmOrchestrator",
    "SwarmAgent",
    "SwarmRunResult",
    "SwarmState",

    # HTTP Client (legacy)
    "ISAAgent",
    "ISAAgentSync",
    "AgentEvent",
    "AgentResponse",
    "SessionInfo",
    "EventType",

    # Bidirectional Client (Claude SDK compatible)
    "ISAAgentClient",
    "ISAAgentClientSync",
    "ClientMode",

    # Project Context (ISA.md/CLAUDE.md support)
    "load_project_context",
    "discover_project_context_file",
    "format_project_context_for_prompt",

    # A2A
    "A2AAgentCard",
    "A2AClient",
    "A2AServerAdapter",
    "register_a2a_fastapi_routes",
    "build_auth_service_token_validator",
]
