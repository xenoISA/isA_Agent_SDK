#!/usr/bin/env python3
"""
Project-based custom tools with @tool decorator.

This module allows users to define custom tools as Python functions
that execute in-process, without needing to add them to the central MCP server.

Example:
    from isa_agent_sdk import tool, create_sdk_mcp_server, query, ISAAgentOptions

    @tool("greet", "Greet a user by name")
    async def greet_user(name: str) -> str:
        return f"Hello, {name}!"

    @tool("calculate", "Perform a calculation", schema={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression"}
        },
        "required": ["expression"]
    })
    async def calculate(expression: str) -> dict:
        result = eval(expression)  # Note: use safe eval in production
        return {"result": result, "expression": expression}

    # Create SDK MCP server with your tools
    server = create_sdk_mcp_server(
        name="my-project-tools",
        tools=[greet_user, calculate]
    )

    # Use in query
    async for msg in query(
        "Greet Alice and calculate 2+2",
        options=ISAAgentOptions(
            mcp_servers={"project": server},
            allowed_tools=["mcp__project__greet", "mcp__project__calculate"]
        )
    ):
        print(msg.content)
"""

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union, get_type_hints
from functools import wraps


@dataclass
class SDKTool:
    """Represents a tool defined with the @tool decorator."""
    name: str
    description: str
    func: Callable
    schema: Optional[Dict[str, Any]] = None
    _inferred_schema: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Infer schema from function signature if not provided."""
        if self.schema is None:
            self._inferred_schema = self._infer_schema()
        else:
            self._inferred_schema = self.schema

    def _infer_schema(self) -> Dict[str, Any]:
        """Infer JSON schema from function signature and type hints."""
        sig = inspect.signature(self.func)
        hints = get_type_hints(self.func) if hasattr(self.func, '__annotations__') else {}

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'cls'):
                continue

            # Get type hint
            param_type = hints.get(param_name, Any)

            # Convert Python type to JSON schema type
            json_type = self._python_type_to_json(param_type)

            properties[param_name] = {
                "type": json_type,
                "description": f"Parameter: {param_name}"
            }

            # Check if required (no default value)
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required
        }

    def _python_type_to_json(self, python_type) -> str:
        """Convert Python type to JSON schema type."""
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
            type(None): "null",
        }

        # Handle Optional, Union, etc.
        origin = getattr(python_type, '__origin__', None)
        if origin is Union:
            args = python_type.__args__
            # Optional[X] is Union[X, None]
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return self._python_type_to_json(non_none[0])

        return type_map.get(python_type, "string")

    def get_tool_definition(self) -> Dict[str, Any]:
        """Get MCP-compatible tool definition."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self._inferred_schema
        }

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given arguments."""
        try:
            # Call the function
            if asyncio.iscoroutinefunction(self.func):
                result = await self.func(**arguments)
            else:
                result = self.func(**arguments)

            # Normalize result to MCP format
            return self._normalize_result(result)

        except Exception as e:
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Error: {str(e)}"}]
            }

    def _normalize_result(self, result: Any) -> Dict[str, Any]:
        """Normalize result to MCP tool result format."""
        # Already in correct format
        if isinstance(result, dict) and "content" in result:
            return result

        # Convert to text content
        if isinstance(result, str):
            text = result
        elif isinstance(result, (dict, list)):
            text = json.dumps(result, indent=2, ensure_ascii=False)
        else:
            text = str(result)

        return {
            "content": [{"type": "text", "text": text}]
        }


def tool(
    name: str,
    description: str,
    schema: Optional[Dict[str, Any]] = None
) -> Callable:
    """
    Decorator to define a custom tool.

    Args:
        name: Tool name (used in allowed_tools as mcp__{server}__{name})
        description: Human-readable description of what the tool does
        schema: Optional JSON schema for input validation. If not provided,
                it will be inferred from function signature and type hints.

    Returns:
        Decorated function with SDKTool metadata attached.

    Example:
        @tool("fetch_weather", "Get current weather for a city")
        async def fetch_weather(city: str, units: str = "celsius") -> str:
            # Implementation
            return f"Weather in {city}: 20°{units[0].upper()}"

        @tool("complex_tool", "A tool with custom schema", schema={
            "type": "object",
            "properties": {
                "data": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["data"]
        })
        async def complex_tool(data: list) -> dict:
            return {"processed": len(data)}
    """
    def decorator(func: Callable) -> Callable:
        # Create SDKTool instance
        sdk_tool = SDKTool(
            name=name,
            description=description,
            func=func,
            schema=schema
        )

        # Attach metadata to function
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)

        wrapper._sdk_tool = sdk_tool
        wrapper._is_sdk_tool = True

        return wrapper

    return decorator


class SDKMCPServer:
    """
    In-process MCP server for project-based tools.

    This executes tools directly in the Python process without
    subprocess communication overhead.
    """

    def __init__(
        self,
        name: str,
        tools: List[Union[Callable, SDKTool]],
        version: str = "1.0.0"
    ):
        """
        Initialize SDK MCP server.

        Args:
            name: Server name (used in tool names as mcp__{name}__{tool})
            tools: List of @tool decorated functions or SDKTool instances
            version: Server version string
        """
        self.name = name
        self.version = version
        self._tools: Dict[str, SDKTool] = {}

        # Register tools
        for t in tools:
            self.register_tool(t)

    def register_tool(self, t: Union[Callable, SDKTool]) -> None:
        """Register a tool with the server."""
        if isinstance(t, SDKTool):
            sdk_tool = t
        elif hasattr(t, '_sdk_tool'):
            sdk_tool = t._sdk_tool
        elif hasattr(t, '_is_sdk_tool'):
            sdk_tool = t._sdk_tool
        else:
            raise ValueError(
                f"Tool {t} is not decorated with @tool. "
                "Use @tool decorator or pass SDKTool instance."
            )

        self._tools[sdk_tool.name] = sdk_tool

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools in MCP format."""
        return [t.get_tool_definition() for t in self._tools.values()]

    def get_tool(self, name: str) -> Optional[SDKTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        """Check if a tool exists."""
        return name in self._tools

    async def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a tool by name.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            MCP-formatted tool result
        """
        tool = self._tools.get(name)
        if not tool:
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Tool '{name}' not found"}]
            }

        return await tool.execute(arguments)

    def get_full_tool_name(self, tool_name: str) -> str:
        """Get the full MCP tool name (mcp__{server}__{tool})."""
        return f"mcp__{self.name}__{tool_name}"

    def __repr__(self) -> str:
        return f"SDKMCPServer(name={self.name!r}, tools={list(self._tools.keys())})"


def create_sdk_mcp_server(
    name: str,
    tools: List[Union[Callable, SDKTool]],
    version: str = "1.0.0"
) -> SDKMCPServer:
    """
    Create an in-process SDK MCP server for custom tools.

    This is the main factory function for creating project-based tool servers.

    Args:
        name: Server name. Tools will be accessible as mcp__{name}__{tool_name}
        tools: List of @tool decorated functions
        version: Server version (default: "1.0.0")

    Returns:
        SDKMCPServer instance

    Example:
        from isa_agent_sdk import tool, create_sdk_mcp_server, ISAAgentOptions

        @tool("greet", "Greet someone")
        async def greet(name: str) -> str:
            return f"Hello, {name}!"

        @tool("add", "Add two numbers")
        def add(a: int, b: int) -> int:
            return a + b

        server = create_sdk_mcp_server("mytools", [greet, add])

        options = ISAAgentOptions(
            mcp_servers={"mytools": server},
            allowed_tools=["mcp__mytools__greet", "mcp__mytools__add"]
        )
    """
    return SDKMCPServer(name=name, tools=tools, version=version)


# Global registry for SDK servers (used by query execution)
_sdk_servers: Dict[str, SDKMCPServer] = {}


def register_sdk_server(name: str, server: SDKMCPServer) -> None:
    """Register an SDK server globally."""
    _sdk_servers[name] = server


def get_sdk_server(name: str) -> Optional[SDKMCPServer]:
    """Get a registered SDK server by name."""
    return _sdk_servers.get(name)


def get_all_sdk_servers() -> Dict[str, SDKMCPServer]:
    """Get all registered SDK servers."""
    return _sdk_servers.copy()


def clear_sdk_servers() -> None:
    """Clear all registered SDK servers."""
    _sdk_servers.clear()


async def execute_sdk_tool(
    full_tool_name: str,
    arguments: Dict[str, Any],
    servers: Optional[Dict[str, SDKMCPServer]] = None
) -> Optional[Dict[str, Any]]:
    """
    Execute a tool if it belongs to an SDK server.

    Args:
        full_tool_name: Full tool name (e.g., "mcp__mytools__greet")
        arguments: Tool arguments
        servers: Optional dict of servers to check. If None, uses global registry.

    Returns:
        Tool result if found, None if not an SDK tool
    """
    # Check if this is an SDK tool (mcp__{server}__{tool})
    if not full_tool_name.startswith("mcp__"):
        return None

    parts = full_tool_name.split("__")
    if len(parts) != 3:
        return None

    _, server_name, tool_name = parts

    # Check provided servers first
    if servers:
        server = servers.get(server_name)
        if server and isinstance(server, SDKMCPServer):
            if server.has_tool(tool_name):
                return await server.call_tool(tool_name, arguments)

    # Fall back to global registry
    server = _sdk_servers.get(server_name)
    if server and server.has_tool(tool_name):
        return await server.call_tool(tool_name, arguments)

    return None


def is_sdk_tool(
    full_tool_name: str,
    servers: Optional[Dict[str, SDKMCPServer]] = None
) -> bool:
    """
    Check if a tool name refers to an SDK tool.

    Args:
        full_tool_name: Full tool name (e.g., "mcp__mytools__greet")
        servers: Optional dict of servers to check

    Returns:
        True if this is an SDK tool, False otherwise
    """
    if not full_tool_name.startswith("mcp__"):
        return False

    parts = full_tool_name.split("__")
    if len(parts) != 3:
        return False

    _, server_name, tool_name = parts

    # Check provided servers
    if servers:
        server = servers.get(server_name)
        if server and isinstance(server, SDKMCPServer):
            return server.has_tool(tool_name)

    # Check global registry
    server = _sdk_servers.get(server_name)
    return server is not None and server.has_tool(tool_name)
