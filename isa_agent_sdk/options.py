#!/usr/bin/env python3
"""
isA Agent SDK - Options Configuration
=====================================

ISAAgentOptions: Configuration options for agent execution.
Fully compatible with Claude Agent SDK options plus advanced isA features.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Callable, Type, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from pydantic import BaseModel
    from ._tools import SDKMCPServer


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


class OutputFormatType(str, Enum):
    """Output format types for structured outputs"""
    TEXT = "text"              # Default free-form text
    JSON_OBJECT = "json_object"  # Basic JSON mode (model returns valid JSON)
    JSON_SCHEMA = "json_schema"  # Strict schema-validated JSON


@dataclass
class OutputFormat:
    """
    Configuration for structured output format (Claude Agent SDK compatible).

    Use this to get validated JSON responses that match a specific schema.
    The agent will return structured data in the `structured_output` field.

    Example with JSON Schema:
        output_format = OutputFormat(
            type="json_schema",
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name"]
            }
        )

    Example with Pydantic:
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        output_format = OutputFormat.from_pydantic(Person)
    """
    type: Union[str, OutputFormatType] = OutputFormatType.TEXT
    """Output format type: 'text', 'json_object', or 'json_schema'"""

    schema: Optional[Dict[str, Any]] = None
    """JSON Schema for 'json_schema' type. Defines the structure of the output."""

    strict: bool = True
    """If True, enforce strict schema validation (constrained sampling)"""

    # Internal: Pydantic model reference for parsing
    _pydantic_model: Optional[Type["BaseModel"]] = field(default=None, repr=False)

    def __post_init__(self):
        """Validate and normalize format type"""
        if isinstance(self.type, str):
            self.type = OutputFormatType(self.type)

        # Validate schema is provided for json_schema type
        if self.type == OutputFormatType.JSON_SCHEMA and not self.schema:
            raise ValueError("schema is required when type is 'json_schema'")

    @classmethod
    def from_pydantic(cls, model: Type["BaseModel"], strict: bool = True) -> "OutputFormat":
        """
        Create OutputFormat from a Pydantic model.

        Args:
            model: Pydantic BaseModel class
            strict: Whether to enforce strict validation

        Returns:
            OutputFormat configured with the model's JSON schema

        Example:
            from pydantic import BaseModel

            class Task(BaseModel):
                title: str
                priority: int
                completed: bool = False

            output_format = OutputFormat.from_pydantic(Task)
        """
        try:
            # Get JSON schema from Pydantic model
            schema = model.model_json_schema()

            instance = cls(
                type=OutputFormatType.JSON_SCHEMA,
                schema=schema,
                strict=strict
            )
            instance._pydantic_model = model
            return instance

        except AttributeError:
            raise TypeError(
                f"{model} does not appear to be a Pydantic model. "
                "Ensure it inherits from pydantic.BaseModel."
            )

    @classmethod
    def json_object(cls) -> "OutputFormat":
        """Create a simple JSON object format (no schema validation)"""
        return cls(type=OutputFormatType.JSON_OBJECT)

    @classmethod
    def json_schema(cls, schema: Dict[str, Any], strict: bool = True) -> "OutputFormat":
        """Create a JSON schema format with the given schema"""
        return cls(type=OutputFormatType.JSON_SCHEMA, schema=schema, strict=strict)

    def to_model_format(self) -> Dict[str, Any]:
        """
        Convert to format expected by isA_Model service.

        Returns:
            Dict compatible with response_format parameter
        """
        if self.type == OutputFormatType.TEXT:
            return {}
        elif self.type == OutputFormatType.JSON_OBJECT:
            return {"type": "json_object"}
        elif self.type == OutputFormatType.JSON_SCHEMA:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": self.schema,
                    "strict": self.strict
                }
            }
        return {}


class SystemPromptPreset(str, Enum):
    """Available system prompt presets (templates from MCP)"""
    REASON = "reason"          # default_reason_prompt - for reasoning/planning
    RESPONSE = "response"      # default_response_prompt - for final responses
    RAG_REASON = "rag_reason"  # rag_reason_prompt - reasoning with file context
    REVIEW = "review"          # default_review_prompt - for evaluation
    MINIMAL = "minimal"        # Minimal prompt, mostly user instructions
    TASK_EXECUTION = "task_execution"  # task_execution_prompt - for autonomous task execution


@dataclass
class SystemPromptConfig:
    """
    Configuration for system prompts (Claude Agent SDK compatible).

    Supports three modes:
    1. Default preset with optional additions (recommended)
    2. Full replacement with custom prompt
    3. Simple string (backwards compatible, treated as append)

    Example - Use preset with additions:
        system_prompt = SystemPromptConfig(
            preset="reason",
            append="Always respond in formal English. Be concise."
        )

    Example - Full custom replacement:
        system_prompt = SystemPromptConfig(
            replace="You are a specialized code reviewer..."
        )

    Example - Simple string (backwards compatible):
        system_prompt = "Always be helpful and concise."
        # This is equivalent to:
        system_prompt = SystemPromptConfig(append="Always be helpful and concise.")
    """
    preset: Optional[Union[str, SystemPromptPreset]] = None
    """
    Preset template to use from MCP. Options:
    - "reason": For reasoning/planning phase (default_reason_prompt)
    - "response": For final response generation (default_response_prompt)
    - "rag_reason": For reasoning with user's uploaded files
    - "review": For evaluating execution results
    - "minimal": Minimal base, mostly your custom instructions
    If None, uses the node's default preset.
    """

    append: Optional[str] = None
    """Custom instructions to append to the preset template."""

    replace: Optional[str] = None
    """
    Full replacement prompt. If provided, preset and append are ignored.
    Use this for complete customization (loses built-in capabilities).
    """

    def __post_init__(self):
        """Validate configuration"""
        if self.replace and (self.preset or self.append):
            import warnings
            warnings.warn(
                "SystemPromptConfig: 'replace' is set, 'preset' and 'append' will be ignored."
            )

        # Normalize preset to enum
        if isinstance(self.preset, str):
            try:
                self.preset = SystemPromptPreset(self.preset)
            except ValueError:
                # Allow custom preset names for future extensibility
                pass

    @classmethod
    def from_string(cls, prompt: str) -> "SystemPromptConfig":
        """Create config from simple string (treated as append)"""
        return cls(append=prompt)

    @property
    def is_replacement(self) -> bool:
        """Whether this config replaces the entire prompt"""
        return self.replace is not None

    @property
    def is_default(self) -> bool:
        """Whether this uses default behavior (no customization)"""
        return not self.replace and not self.append and not self.preset

    def get_mcp_prompt_name(self, node_type: str = "reason") -> str:
        """
        Get the MCP prompt name for this configuration.

        Args:
            node_type: The type of node requesting the prompt ("reason", "response", "review")

        Returns:
            MCP prompt name to use
        """
        if self.preset:
            preset_map = {
                SystemPromptPreset.REASON: "default_reason_prompt",
                SystemPromptPreset.RESPONSE: "default_response_prompt",
                SystemPromptPreset.RAG_REASON: "rag_reason_prompt",
                SystemPromptPreset.REVIEW: "default_review_prompt",
                SystemPromptPreset.MINIMAL: "minimal_prompt",
                SystemPromptPreset.TASK_EXECUTION: "task_execution_prompt",
            }
            if isinstance(self.preset, SystemPromptPreset):
                return preset_map.get(self.preset, f"default_{node_type}_prompt")
            return self.preset  # Custom preset name

        # Default based on node type
        return f"default_{node_type}_prompt"


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
            env="cloud_pool",
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

    system_prompt: Optional[Union[str, SystemPromptConfig]] = None
    """
    Custom system prompt configuration (Claude Agent SDK compatible).

    Accepts either:
    - str: Simple string to append to the default prompt
    - SystemPromptConfig: Full configuration with preset/append/replace options

    Examples:
        # Simple string (appends to default)
        system_prompt="Always respond in formal English."

        # Use preset with custom additions
        system_prompt=SystemPromptConfig(
            preset="reason",
            append="Focus on security implications."
        )

        # Full replacement (loses built-in capabilities)
        system_prompt=SystemPromptConfig(
            replace="You are a specialized security auditor..."
        )
    """

    mcp_servers: Optional[Dict[str, Union[MCPServerConfig, "SDKMCPServer", Dict]]] = None
    """
    MCP servers to connect to.

    Supports three types:
    1. MCPServerConfig: External MCP servers (subprocess)
    2. SDKMCPServer: In-process SDK servers (from @tool decorator)
    3. Dict: Will be converted to MCPServerConfig

    Example:
        from isa_agent_sdk import tool, create_sdk_mcp_server

        @tool("greet", "Greet someone")
        async def greet(name: str) -> str:
            return f"Hello, {name}!"

        options = ISAAgentOptions(
            mcp_servers={
                "project": create_sdk_mcp_server("project", [greet]),
                "external": {"command": "npx", "args": ["@some/mcp-server"]}
            },
            allowed_tools=["mcp__project__greet"]
        )
    """

    hooks: Optional[Dict[str, List[HookMatcher]]] = None
    """Lifecycle hooks: PreToolUse, PostToolUse, etc."""

    agents: Optional[Dict[str, AgentDefinition]] = None
    """Sub-agent definitions for Task tool"""

    resume: Optional[str] = None
    """Session ID to resume from"""

    cwd: Optional[str] = None
    """Working directory for file operations"""

    output_format: Optional[Union[OutputFormat, Dict[str, Any]]] = None
    """
    Structured output format configuration (Claude Agent SDK compatible).

    Use this to get validated JSON responses matching a specific schema.
    The response will include a `structured_output` field with validated data.

    Example with dict:
        output_format={
            "type": "json_schema",
            "schema": {"type": "object", "properties": {"name": {"type": "string"}}}
        }

    Example with OutputFormat:
        output_format=OutputFormat.from_pydantic(MyModel)
    """

    setting_sources: Optional[List[str]] = None
    """Config sources: ["project"] for .isa/ style config"""

    project_context: Optional[str] = None
    """
    Project context file path or content (Claude SDK CLAUDE.md compatible).

    Supports:
    - File path: "ISA.md", "CLAUDE.md", ".isa/CONTEXT.md"
    - Direct content: Multi-line string with project context
    - "auto": Auto-discover from project root (checks ISA.md, CLAUDE.md, .isa/CONTEXT.md)

    The content is injected into the system prompt as persistent project memory.

    Example:
        # Auto-discover project context file
        options = ISAAgentOptions(project_context="auto")

        # Use specific file
        options = ISAAgentOptions(project_context="./ISA.md")

        # Direct content
        options = ISAAgentOptions(project_context='''
            This project uses FastAPI and PostgreSQL.
            Always use type hints and follow PEP 8.
        ''')
    """

    # === isA Service Configuration ===
    isa_mcp_url: Optional[str] = None
    """URL for isA MCP server (SSE). Default: from ISA_MCP_URL env or http://localhost:8081"""

    isa_model_url: Optional[str] = None
    """URL for isA Model service. Default: from ISA_API_URL env or http://localhost:8082"""

    # === isA Advanced Features ===
    env: Union[str, ExecutionEnv] = ExecutionEnv.CLOUD_SHARED
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
        if isinstance(self.env, str):
            self.env = ExecutionEnv(self.env)
        if isinstance(self.execution_mode, str):
            self.execution_mode = ExecutionMode(self.execution_mode)
        if isinstance(self.tool_discovery, str):
            self.tool_discovery = ToolDiscoveryMode(self.tool_discovery)
        if isinstance(self.guardrail_mode, str):
            self.guardrail_mode = GuardrailMode(self.guardrail_mode)

        # Convert MCP server dicts to MCPServerConfig (preserve SDKMCPServer as-is)
        if self.mcp_servers:
            # Import here to avoid circular imports
            from ._tools import SDKMCPServer
            normalized = {}
            for name, config in self.mcp_servers.items():
                if isinstance(config, SDKMCPServer):
                    # SDK servers pass through unchanged
                    normalized[name] = config
                elif isinstance(config, dict):
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

        # Convert output_format dict to OutputFormat
        if self.output_format and isinstance(self.output_format, dict):
            self.output_format = OutputFormat(**self.output_format)

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
        config = {
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

        # Add output_format if specified
        if self.output_format:
            if isinstance(self.output_format, OutputFormat):
                config["configurable"]["output_format"] = self.output_format.to_model_format()
                config["configurable"]["output_schema"] = self.output_format.schema
            elif isinstance(self.output_format, dict):
                config["configurable"]["output_format"] = self.output_format
                config["configurable"]["output_schema"] = self.output_format.get("schema")

        return config

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
            options_dict['env'] = config['defaults'].get('env', config['defaults'].get('execution_env'))
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
    "OutputFormat",
    "OutputFormatType",
    "MCPServerConfig",
    "AgentDefinition",
    "HookMatcher",
    "PoolConfig",
    "TriggerConfig",
    "SystemPromptConfig",
    "SystemPromptPreset",
]
