# Tools & MCP Integration

The isA Agent SDK integrates with the Model Context Protocol (MCP) to provide access to a wide range of tools.

## Overview

Tools enable agents to:
- Search the web
- Read and write files
- Execute code
- Interact with APIs
- Access databases
- And much more...

## Tool Discovery

### List Available Tools

```python
from isa_agent_sdk import get_available_tools

# Get all available tools
tools = await get_available_tools()
for tool in tools[:10]:
    print(f"- {tool['name']}: {tool['description']}")
```

### Semantic Tool Search

Find tools relevant to a specific task:

```python
# Find tools for web research
web_tools = await get_available_tools(
    user_query="I need to search the web and fetch URLs",
    max_results=5
)

# Find tools for file operations
file_tools = await get_available_tools(
    user_query="read and write files",
    max_results=5
)
```

## Tool Configuration

### Explicit Tool List

Specify exactly which tools to allow:

```python
from isa_agent_sdk import query, ISAAgentOptions

options = ISAAgentOptions(
    allowed_tools=[
        "web_search",
        "fetch_url",
        "read_file",
        "write_file"
    ]
)

async for msg in query("Research Python best practices", options=options):
    print(msg.content, end="" if msg.is_text else "\n")
```

### Tool Discovery Modes

```python
from isa_agent_sdk import ToolDiscoveryMode

# Explicit only - use only allowed_tools
options = ISAAgentOptions(
    tool_discovery=ToolDiscoveryMode.EXPLICIT,
    allowed_tools=["web_search", "read_file"]
)

# Semantic - auto-discover based on query
options = ISAAgentOptions(
    tool_discovery=ToolDiscoveryMode.SEMANTIC
)

# Hybrid - combine explicit list with semantic discovery
options = ISAAgentOptions(
    tool_discovery=ToolDiscoveryMode.HYBRID,
    allowed_tools=["read_file", "write_file"]  # Always available
)
```

## Direct Tool Execution

### Execute a Single Tool

```python
from isa_agent_sdk import execute_tool

# Web search
result = await execute_tool(
    tool_name="web_search",
    tool_args={"query": "Python async programming"},
    session_id="my-session"
)
print(result.content)

# Read file
result = await execute_tool(
    tool_name="read_file",
    tool_args={"path": "/path/to/file.txt"}
)

# Execute bash command
result = await execute_tool(
    tool_name="bash",
    tool_args={"command": "ls -la"}
)
```

### Handle Tool Results

```python
result = await execute_tool("web_search", {"query": "test"})

if result.is_tool_result:
    if result.tool_error:
        print(f"Tool failed: {result.tool_error}")
    else:
        print(f"Result: {result.tool_result_value}")
```

## Custom Project Tools (@tool decorator)

Define custom tools directly in your Python code using the `@tool` decorator. These tools execute in-process without needing to add them to the central MCP server.

### Basic Usage

```python
from isa_agent_sdk import tool, create_sdk_mcp_server, query, ISAAgentOptions

# Define custom tools with @tool decorator
@tool("greet", "Greet a user by name")
async def greet_user(name: str) -> str:
    return f"Hello, {name}!"

@tool("calculate", "Evaluate a math expression")
def calculate(expression: str) -> dict:
    result = eval(expression)  # Use safe eval in production
    return {"expression": expression, "result": result}

# Create SDK MCP server with your tools
server = create_sdk_mcp_server("mytools", [greet_user, calculate])

# Use in query
async for msg in query(
    "Greet Alice and then calculate 15 * 7",
    options=ISAAgentOptions(
        mcp_servers={"mytools": server},
        allowed_tools=["mcp__mytools__greet", "mcp__mytools__calculate"]
    )
):
    if msg.is_tool_use:
        print(f"[Tool] {msg.tool_name}")
    elif msg.is_text:
        print(msg.content, end="")
```

### @tool Decorator Options

```python
from isa_agent_sdk import tool

# Basic - schema inferred from type hints
@tool("my_tool", "Description of what it does")
async def my_tool(param1: str, param2: int = 10) -> str:
    ...

# With custom JSON schema
@tool("complex_tool", "Handle complex input", schema={
    "type": "object",
    "properties": {
        "items": {"type": "array", "items": {"type": "string"}},
        "config": {"type": "object"}
    },
    "required": ["items"]
})
async def complex_tool(items: list, config: dict = None) -> dict:
    ...
```

### Return Values

Tools can return various types:

```python
@tool("string_tool", "Returns a string")
def string_tool() -> str:
    return "Simple text result"

@tool("dict_tool", "Returns JSON")
def dict_tool() -> dict:
    return {"key": "value", "list": [1, 2, 3]}  # Auto-serialized to JSON

@tool("mcp_format_tool", "Returns MCP format directly")
def mcp_format_tool() -> dict:
    return {"content": [{"type": "text", "text": "Direct MCP format"}]}
```

### Error Handling

Errors are automatically captured and returned in MCP format:

```python
@tool("may_fail", "A tool that might fail")
async def may_fail(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return open(path).read()

# Error is captured as:
# {"isError": True, "content": [{"type": "text", "text": "Error: File not found: /missing"}]}
```

### Mixed Servers (SDK + External)

Combine in-process SDK tools with external MCP servers:

```python
from isa_agent_sdk import tool, create_sdk_mcp_server, ISAAgentOptions, MCPServerConfig

@tool("local_process", "Process data locally")
async def local_process(data: str) -> str:
    return data.upper()

options = ISAAgentOptions(
    mcp_servers={
        # In-process SDK server (fast, no IPC)
        "local": create_sdk_mcp_server("local", [local_process]),

        # External MCP server (subprocess)
        "github": MCPServerConfig(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"]
        )
    },
    allowed_tools=[
        "mcp__local__local_process",  # SDK tool
        "mcp__github__search_code"     # External tool
    ]
)
```

### Benefits of SDK Tools

| Aspect | SDK Tools (@tool) | External MCP |
|--------|------------------|--------------|
| Performance | Fast (in-process) | IPC overhead |
| Debugging | Easy (Python stack) | Harder |
| Deployment | Single process | Multiple processes |
| Type Safety | Python type hints | JSON schema |
| State Sharing | Direct access | Serialization needed |

---

## MCP Server Configuration

### Built-in MCP Server

The SDK connects to isA_MCP by default:

```python
options = ISAAgentOptions(
    # isA_MCP is used automatically
    # URL configured via ISA_MCP_URL environment variable
)
```

### External MCP Servers

Connect to additional MCP servers:

```python
from isa_agent_sdk import ISAAgentOptions, MCPServerConfig

options = ISAAgentOptions(
    mcp_servers={
        # GitHub MCP server
        "github": MCPServerConfig(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_TOKEN": "your-token"}
        ),

        # Filesystem MCP server
        "filesystem": MCPServerConfig(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/allowed/path"]
        ),

        # Custom HTTP MCP server
        "custom": MCPServerConfig(
            url="http://localhost:9000/mcp"
        ),

        # Docker-based MCP server
        "docker-mcp": MCPServerConfig(
            command="docker",
            args=["run", "-i", "my-mcp-server"]
        )
    }
)
```

### MCPServerConfig Options

```python
MCPServerConfig(
    # For stdio-based servers
    command="npx",           # Command to run
    args=["server-name"],    # Arguments
    env={"KEY": "value"},    # Environment variables

    # For HTTP-based servers
    url="http://host:port/mcp"
)
```

## Tool Categories

### Web Tools

```python
# Search
await execute_tool("web_search", {"query": "search terms"})

# Fetch URL content
await execute_tool("fetch_url", {"url": "https://example.com"})

# Crawl website
await execute_tool("web_crawl", {"url": "https://example.com", "depth": 2})
```

### File Tools

```python
# Read file
await execute_tool("read_file", {"path": "/path/to/file"})

# Write file
await execute_tool("write_file", {
    "path": "/path/to/file",
    "content": "file content"
})

# List directory
await execute_tool("list_directory", {"path": "/path/to/dir"})

# Delete file
await execute_tool("delete_file", {"path": "/path/to/file"})
```

### Code Execution Tools

```python
# Bash command
await execute_tool("bash", {"command": "echo hello"})

# Python execution
await execute_tool("python_exec", {"code": "print(2+2)"})
```

### Database Tools

```python
# SQL query
await execute_tool("database_query", {
    "connection": "postgresql://...",
    "query": "SELECT * FROM users LIMIT 10"
})
```

### Time Tools

```python
# Get current time
result = await execute_tool("get_current_time", {})
# Returns: {"iso": "2024-01-15T10:30:00Z", "date": "2024-01-15", ...}

# Get current date
result = await execute_tool("get_current_date", {})
```

## Tool Execution in Streaming

### Monitor Tool Usage

```python
from isa_agent_sdk import query

tools_used = []

async for msg in query("Analyze the project structure"):
    if msg.is_tool_use:
        print(f"[Tool] {msg.tool_name}({msg.tool_args})")
        tools_used.append(msg.tool_name)

    elif msg.is_tool_result:
        if msg.tool_error:
            print(f"[Error] {msg.tool_error}")
        else:
            print(f"[Result] {str(msg.tool_result_value)[:100]}...")

    elif msg.is_text:
        print(msg.content, end="")

print(f"\n\nTools used: {tools_used}")
```

### Tool Execution with Progress

```python
async for msg in query("Search multiple sources and compile results"):
    match msg.type:
        case "tool_use":
            print(f"Starting: {msg.tool_name}")
        case "tool_result":
            status = "failed" if msg.tool_error else "completed"
            print(f"Tool {status}: {msg.tool_name}")
        case "progress":
            print(f"Progress: {msg.progress_percent}%")
        case "text":
            print(msg.content, end="")
```

## Tool Profiling

The SDK tracks tool execution times for optimization:

```python
from isa_agent_sdk.services.auto_detection import get_tool_profiler

profiler = await get_tool_profiler()

# Record custom execution
await profiler.record_execution(
    tool_name="my_tool",
    execution_time_ms=1500,
    tool_args={"param": "value"},
    session_id="session-123",
    success=True
)

# Get execution estimate
estimate = await profiler.estimate_time("web_search", {"query": "test"})
print(f"Estimated time: {estimate}ms")

# Get statistics
stats = await profiler.get_statistics("web_search")
print(f"Average: {stats['avg_time_ms']}ms")
print(f"P90: {stats['p90_time_ms']}ms")
```

## Error Handling

### Handle Tool Errors

```python
from isa_agent_sdk import execute_tool

result = await execute_tool("web_search", {"query": "test"})

if result.tool_error:
    error_type = result.metadata.get("error_type", "unknown")
    match error_type:
        case "timeout":
            print("Tool timed out, retrying...")
        case "rate_limit":
            print("Rate limited, waiting...")
        case "not_found":
            print("Tool not found")
        case _:
            print(f"Error: {result.tool_error}")
```

### Retry Logic

```python
import asyncio

async def execute_with_retry(tool_name, args, max_retries=3):
    for attempt in range(max_retries):
        result = await execute_tool(tool_name, args)
        if not result.tool_error:
            return result
        if "rate_limit" in str(result.tool_error):
            await asyncio.sleep(2 ** attempt)
        else:
            break
    return result
```

## Best Practices

1. **Use explicit tool lists** for production - More predictable behavior
2. **Handle tool errors** - Always check for failures
3. **Monitor tool usage** - Track which tools are being used
4. **Set appropriate timeouts** - Prevent hanging operations
5. **Use semantic discovery** for exploration - Let the agent find relevant tools

## Next Steps

- [Human-in-the-Loop](./human-in-the-loop.md) - Tool permission workflows
- [Options](./options.md) - Configure tool behavior
- [MCP Reference](./mcp-reference.md) - Full MCP protocol details
