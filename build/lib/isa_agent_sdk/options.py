#!/usr/bin/env python3
"""
isA Agent SDK - Options Configuration
=====================================

ISAAgentOptions: Configuration options for agent execution.
Fully compatible with Claude Agent SDK options plus advanced isA features.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Callable
from enum import Enum


class ExecutionEnv(str, Enum):
    """Execution environment options"""
    CLOUD_POOL = "cloud_pool"      # Semi-persistent isolated VMs
    CLOUD_SHARED = "cloud_shared"  # Session-based ephemeral
    DESKTOP = "desktop"            # Local machine via desktop agent


class ExecutionMode(str, Enum):
    """Agent execution mode"""
    REACTIVE = "reactive"          # Responds to explicit requests
    COLLABORATIVE = "collaborative"  # Checkpoints for human approval
    PROACTIVE = "proactive"        # Anticipates needs, suggests actions


class ToolDiscoveryMode(str, Enum):
    """How tools are discovered and selected"""
    EXPLICIT = "explicit"    # Only use explicitly allowed tools
    SEMANTIC = "semantic"    # Find tools by semantic similarity
    HYBRID = "hybrid"        # Explicit core + semantic discovery


class PermissionMode(str, Enum):
    """Permission handling mode (Claude SDK compatible)"""
    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    BYPASS_PERMISSIONS = "bypassPermissions"


class GuardrailMode(str, Enum):
    """Guardrail strictness level"""
    PERMISSIVE = "permissive"
    MODERATE = "moderate"
    STRICT = "strict"


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection"""
    command: Optional[str] = None    # Command to spawn server
    args: Optional[List[str]] = None  # Arguments for command
    url: Optional[str] = None         # URL for existing server
    env: Optional[Dict[str, str]] = None  # Environment variables

    def __post_init__(self):
        if not self.command and not self.url:
            raise ValueError("MCPServerConfig requires either 'command' or 'url'")


@dataclass
class AgentDefinition:
    """Definition for a sub-agent (Claude SDK compatible + extensions)"""
    description: str                     # Agent description
    prompt: str                          # System prompt for agent
    tools: Optional[List[str]] = None    # Allowed tools for this agent

    # isA extensions
    graph_type: Optional[str] = None     # Graph type: "smart_agent", "research", "coding"
    execution_mode: Optional[str] = None  # Mode: "reactive", "collaborative", "proactive"
    max_iterations: Optional[int] = None  # Max iterations for this sub-agent


@dataclass
class HookMatcher:
    """Hook matcher configuration (Claude SDK compatible)"""
    matcher: str                          # Regex pattern to match (e.g., "Edit|Write")
    hooks: List[Callable]                 # Hook callback functions

    def matches(self, target: str) -> bool:
        """Check if the target matches this hook's pattern"""
        import re
        if self.matcher == "*":
            return True
        return bool(re.match(self.matcher, target))


@dataclass
class PoolConfig:
    """Configuration for cloud pool execution"""
    pool_type: str = "vm"         # Pool type: "vm", "container", "sandbox"
    ttl: int = 3600               # Time-to-live in seconds
    snapshot: bool = True         # Enable state snapshots
    max_memory: Optional[int] = None  # Max memory in MB
    max_cpu: Optional[float] = None   # Max CPU cores


@dataclass
class TriggerConfig:
    """
    Configuration for event-driven triggers (proactive agent activation)

    Event triggers enable agents to start working WITHOUT a user chat message,
    triggered by external events like price thresholds, schedules, or IoT events.

    Example:
        trigger_config = TriggerConfig(
            enabled=True,
            event_service_url="http://localhost:8080/events",
            task_service_url="http://localhost:8080/tasks"
        )
    """
    enabled: bool = False
    """Enable event triggers (default: disabled for backward compatibility)"""

    event_service_url: Optional[str] = None
    """URL of the Event Service API for event subscriptions"""

    task_service_url: Optional[str] = None
    """URL of the Task Service API for scheduled tasks"""

    use_nats_bus: bool = True
    """Use NATS JetStream for event delivery (recommended for production)"""

    max_triggers_per_user: int = 50
    """Maximum number of triggers a single user can register"""

    trigger_cooldown_seconds: int = 60
    """Minimum seconds between trigger fires (prevents spam)"""

    event_bus: Optional[Any] = None
    """Optional pre-configured event bus instance (for testing/custom setups)"""

    def to_graph_config(self) -> Dict[str, Any]:
        """Convert to graph configuration dictionary"""
        if not self.enabled:
            return {}

        return {
            "event_service_url": self.event_service_url,
            "task_service_url": self.task_service_url,
            "event_bus": self.event_bus,
            "use_nats_bus": self.use_nats_bus,
            "max_triggers_per_user": self.max_triggers_per_user,
            "trigger_cooldown_seconds": self.trigger_cooldown_seconds
        }


@dataclass
class ISAAgentOptions:
    """
    Configuration options for agent execution.

    Fully compatible with Claude Agent SDK, plus advanced isA features.

    Example (Claude SDK compatible):
        options = ISAAgentOptions(
            allowed_tools=["Read", "Write", "Bash"],
            model="claude-sonnet"
        )

    Example (isA advanced):
        options = ISAAgentOptions(
            allowed_tools=["Read", "Write"],
            tool_discovery="hybrid",
            execution_mode="collaborative",
            execution_env="cloud_pool",
            skills=["code-review", "refactor"]
        )
    """

    # === Claude Agent SDK Compatible ===
    allowed_tools: Optional[List[str]] = None
    """Explicit tool allowlist. None means all available tools."""

    permission_mode: Union[str, PermissionMode] = PermissionMode.DEFAULT
    """Permission handling: "default", "acceptEdits", "bypassPermissions" """

    model: str = "gpt-5-nano"
    """Model to use for reasoning (routes through isA_Model)"""

    system_prompt: Optional[str] = None
    """Custom system prompt (appended to default)"""

    mcp_servers: Optional[Dict[str, Union[MCPServerConfig, Dict]]] = None
    """External MCP servers to connect to"""

    hooks: Optional[Dict[str, List[HookMatcher]]] = None
    """Lifecycle hooks: PreToolUse, PostToolUse, etc."""

    agents: Optional[Dict[str, AgentDefinition]] = None
    """Sub-agent definitions for Task tool"""

    resume: Optional[str] = None
    """Session ID to resume from"""

    cwd: Optional[str] = None
    """Working directory for file operations"""

    setting_sources: Optional[List[str]] = None
    """Config sources: ["project"] for .isa/ style config"""

    # === isA Advanced Features ===
    execution_env: Union[str, ExecutionEnv] = ExecutionEnv.CLOUD_SHARED
    """Execution environment: "cloud_pool", "cloud_shared", "desktop" """

    graph_type: str = "smart_agent"
    """Graph type: "smart_agent", "research", "coding", "conversation" """

    skills: Optional[List[str]] = None
    """Skill names to load (e.g., ["code-review", "debug"])"""

    execution_mode: Union[str, ExecutionMode] = ExecutionMode.REACTIVE
    """Execution mode: "reactive", "collaborative", "proactive" """

    guardrails_enabled: bool = False
    """Enable safety guardrails"""

    guardrail_mode: Union[str, GuardrailMode] = GuardrailMode.MODERATE
    """Guardrail strictness: "permissive", "moderate", "strict" """

    failsafe_enabled: bool = False
    """Enable confidence-based failsafe"""

    failsafe_confidence_threshold: float = 0.7
    """Minimum confidence before triggering failsafe (0.0-1.0)"""

    summarization_enabled: bool = True
    """Auto-summarize long conversations"""

    summarization_threshold: int = 20
    """Number of messages before summarization"""

    preserve_last_n: int = 5
    """Number of recent messages to preserve verbatim"""

    max_iterations: int = 50
    """Maximum graph iterations"""

    tool_discovery: Union[str, ToolDiscoveryMode] = ToolDiscoveryMode.EXPLICIT
    """Tool discovery mode: "explicit", "semantic", "hybrid" """

    # Collaborative mode options
    checkpoint_frequency: int = 3
    """Checkpoint every N tasks (collaborative mode)"""

    # Proactive mode options
    proactive_suggestions: bool = True
    """Enable proactive suggestions (proactive mode)"""

    # Pool configuration
    pool_config: Optional[PoolConfig] = None
    """Configuration for cloud_pool execution"""

    # Desktop agent configuration
    desktop_agent_url: Optional[str] = None
    """WebSocket URL for desktop agent (default: ws://localhost:9000)"""

    # Session configuration
    session_ttl: int = 1800
    """Session time-to-live in seconds (cloud_shared)"""

    # Trigger configuration (for proactive agent activation)
    trigger_config: Optional[TriggerConfig] = None
    """Configuration for event-driven triggers (proactive mode)"""

    # User/context info
    user_id: Optional[str] = None
    """User identifier"""

    session_id: Optional[str] = None
    """Session identifier (auto-generated if not provided)"""

    metadata: Optional[Dict[str, Any]] = None
    """Additional metadata"""

    def __post_init__(self):
        """Validate and normalize options"""
        # Convert string enums to actual enums
        if isinstance(self.permission_mode, str):
            self.permission_mode = PermissionMode(self.permission_mode)
        if isinstance(self.execution_env, str):
            self.execution_env = ExecutionEnv(self.execution_env)
        if isinstance(self.execution_mode, str):
            self.execution_mode = ExecutionMode(self.execution_mode)
        if isinstance(self.tool_discovery, str):
            self.tool_discovery = ToolDiscoveryMode(self.tool_discovery)
        if isinstance(self.guardrail_mode, str):
            self.guardrail_mode = GuardrailMode(self.guardrail_mode)

        # Convert MCP server dicts to MCPServerConfig
        if self.mcp_servers:
            normalized = {}
            for name, config in self.mcp_servers.items():
                if isinstance(config, dict):
                    normalized[name] = MCPServerConfig(**config)
                else:
                    normalized[name] = config
            self.mcp_servers = normalized

        # Convert agent definition dicts to AgentDefinition
        if self.agents:
            normalized = {}
            for name, defn in self.agents.items():
                if isinstance(defn, dict):
                    normalized[name] = AgentDefinition(**defn)
                else:
                    normalized[name] = defn
            self.agents = normalized

    def to_graph_config(self) -> Dict[str, Any]:
        """Convert options to SmartAgentGraph configuration"""
        config = {
            "guardrail_enabled": self.guardrails_enabled,
            "guardrail_mode": self.guardrail_mode.value if isinstance(self.guardrail_mode, GuardrailMode) else self.guardrail_mode,
            "failsafe_enabled": self.failsafe_enabled,
            "confidence_threshold": self.failsafe_confidence_threshold,
            "summarization_enabled": self.summarization_enabled,
            "summarization_threshold": self.summarization_threshold,
            "preserve_last_n": self.preserve_last_n,
            "max_graph_iterations": self.max_iterations,
        }

        # Add trigger config if enabled
        if self.trigger_config and self.trigger_config.enabled:
            config["trigger_config"] = self.trigger_config.to_graph_config()

        return config

    def to_runtime_config(self) -> Dict[str, Any]:
        """Convert options to LangGraph runtime configuration"""
        return {
            "configurable": {
                "thread_id": self.session_id,
                "model": self.model,
                "allowed_tools": self.allowed_tools,
                "tool_discovery": self.tool_discovery.value if isinstance(self.tool_discovery, ToolDiscoveryMode) else self.tool_discovery,
                "execution_mode": self.execution_mode.value if isinstance(self.execution_mode, ExecutionMode) else self.execution_mode,
                "skills": self.skills,
                "cwd": self.cwd,
                "user_id": self.user_id,
            },
            "recursion_limit": self.max_iterations,
        }

    @classmethod
    def from_file(cls, path: str) -> "ISAAgentOptions":
        """
        Load options from a YAML configuration file.

        Args:
            path: Path to .isa/config.yaml or similar

        Returns:
            ISAAgentOptions instance
        """
        import yaml

        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        # Map config file structure to options
        options_dict = {}

        if 'defaults' in config:
            options_dict['model'] = config['defaults'].get('model')
            options_dict['execution_env'] = config['defaults'].get('execution_env')
            options_dict['execution_mode'] = config['defaults'].get('execution_mode')

        if 'tools' in config:
            options_dict['allowed_tools'] = config['tools'].get('allowed')
            options_dict['tool_discovery'] = config['tools'].get('discovery')

        if 'skills' in config:
            options_dict['skills'] = config['skills'].get('enabled')

        if 'guardrails' in config:
            options_dict['guardrails_enabled'] = config['guardrails'].get('enabled', False)
            options_dict['guardrail_mode'] = config['guardrails'].get('mode')

        if 'mcp_servers' in config:
            options_dict['mcp_servers'] = config['mcp_servers']

        # Filter out None values
        options_dict = {k: v for k, v in options_dict.items() if v is not None}

        return cls(**options_dict)


# Convenience aliases
Options = ISAAgentOptions


__all__ = [
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
]
