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
"""

# Version
__version__ = "0.1.0"

# Configuration options (no circular imports)
from .options import (
    ISAAgentOptions,
    Options,  # Alias
    ExecutionEnv,
    ExecutionMode,
    ToolDiscoveryMode,
    PermissionMode,
    GuardrailMode,
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
                "MESSAGE_TYPE_MAP", "REVERSE_TYPE_MAP"):
        if "_messages" not in _lazy_modules:
            _lazy_modules["_messages"] = importlib.import_module("._messages", "isa_agent_sdk")
        return getattr(_lazy_modules["_messages"], name)

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

    raise AttributeError(f"module 'isa_agent_sdk' has no attribute '{name}'")

__all__ = [
    # Version
    "__version__",

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
    "MCPServerConfig",
    "AgentDefinition",
    "HookMatcher",
    "PoolConfig",
    "TriggerConfig",

    # Messages
    "AgentMessage",
    "ConversationHistory",
    "ISAEventType",
    "EventData",
    "EventEmitter",
    "EventCategory",
    "MESSAGE_TYPE_MAP",
    "REVERSE_TYPE_MAP",

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

    # HTTP Client
    "ISAAgent",
    "ISAAgentSync",
    "AgentEvent",
    "AgentResponse",
    "SessionInfo",
    "EventType",
]
