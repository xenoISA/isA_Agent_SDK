# Desktop Execution Guide

This guide explains how to route tool execution to a user's local desktop via the Pool Manager and Desktop Agent.

## Overview

The isA Agent SDK supports three execution environments:

| Environment | Description |
|-------------|-------------|
| `desktop` | Execute tools on user's local machine via Desktop Agent |
| `cloud_pool` | Execute tools in isolated cloud VM (coming soon) |
| `cloud_shared` | Execute tools via shared MCP server (default) |

**Desktop Execution Flow:**

```
User Request (env=desktop)
    │
    ▼
isA Agent (FastAPI service)
    │
    ▼
isA Agent SDK (LangGraph)
    │
    ├── sense_node (intent classification)
    │
    ├── reason_node (LLM decides which tools)
    │
    ├── tool_node (executes tools)
    │       │
    │       ▼
    │   BaseNode.mcp_call_tool()
    │       │
    │       ├── if env == "desktop"
    │       │       │
    │       │       ▼
    │       │   DesktopExecutionClient
    │       │       │
    │       │       ▼
    │       │   Pool Manager /desktop/execute
    │       │       │
    │       │       ▼
    │       │   Desktop Agent (WebSocket)
    │       │       │
    │       │       ▼
    │       │   Local Filesystem / Shell
    │       │
    │       └── else → MCP Server (local/cloud)
    │
    ├── format_response (synthesize answer)
    │
    ▼
Response to User
```

## Prerequisites

### 1. Pool Manager (port 8090)

The Pool Manager routes requests to Desktop Agents:

```bash
# Start Pool Manager
cd /path/to/pool_manager
./deployment/local-dev.sh --run
```

### 2. Desktop Agent

The Desktop Agent runs on the user's machine and connects to Pool Manager:

```bash
# Start Desktop Agent
cd /path/to/desktop_os
./deployment/local-dev.sh --run
```

Verify connection:
```bash
curl -s http://localhost:8090/admin/desktops | jq '.desktops[] | {desktop_id, status, owner_id}'
```

### 3. isA Agent (port 8888)

The main agent service using the SDK:

```bash
cd /path/to/isA_Agent
./deployment/local-dev.sh --run
```

## Usage

### Basic Request with Desktop Execution

```bash
curl -X POST "http://localhost:8888/api/v1/agents/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "read the file at /tmp/test.txt",
    "user_id": "your_user_id",
    "session_id": "session_123",
    "env": "desktop"
  }'
```

### Python SDK Usage

```python
from isa_agent_sdk import query, ISAAgentOptions, ExecutionEnv

options = ISAAgentOptions(
    env=ExecutionEnv.DESKTOP,
    user_id="your_user_id",
    allowed_tools=["read_file", "write_file", "bash_execute", "glob_files"]
)

async for msg in query("List all Python files in my home directory", options=options):
    if msg.is_text:
        print(msg.content)
```

## Architecture Deep Dive

### How Tool Routing Works

The routing decision happens in `BaseNode.mcp_call_tool()`:

```python
# isa_agent_sdk/nodes/base_node.py

async def mcp_call_tool(self, tool_name, arguments, config, progress_callback=None):
    # Get execution environment from config
    env = config.get("configurable", {}).get("env", "cloud_shared")
    user_id = config.get("configurable", {}).get("user_id", "unknown")

    # Route based on environment
    if env == "desktop":
        return await self._call_tool_via_desktop(tool_name, arguments, config)

    # Default: MCP execution
    return await mcp_service.call_tool(tool_name, arguments)
```

### Desktop Execution Client

The `DesktopExecutionClient` handles communication with Pool Manager:

```python
# isa_agent_sdk/clients/desktop_execution_client.py

class DesktopExecutionClient:
    async def acquire_desktop(self, user_id: str) -> Optional[str]:
        """Find and acquire user's online desktop"""
        # 1. Get online desktops from Pool Manager
        # 2. Filter by user_id
        # 3. Return first available desktop_id

    async def execute_tool(self, tool_name: str, arguments: dict) -> dict:
        """Execute tool on desktop via Pool Manager"""
        # POST to /desktop/execute with desktop_id
        # Pool Manager forwards to Desktop Agent via WebSocket
        # Returns result from Desktop Agent
```

### Config Flow

The `env` parameter flows through the system:

```
1. Request body: {"env": "desktop", "user_id": "xenodennis"}
       │
       ▼
2. ISAAgentOptions(env=ExecutionEnv.DESKTOP, user_id="xenodennis")
       │
       ▼
3. _query.py builds RunnableConfig:
   {
     "configurable": {
       "thread_id": "session_123",
       "user_id": "xenodennis",
       "env": "desktop",  # <-- Here
       ...
     }
   }
       │
       ▼
4. BaseNode.mcp_call_tool() reads config["configurable"]["env"]
       │
       ▼
5. If "desktop" → _call_tool_via_desktop()
```

## Supported Desktop Tools

The Desktop Agent supports these tools:

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents |
| `write_file` | Write content to file |
| `edit_file` | Edit file with string replacement |
| `bash_execute` | Execute shell commands |
| `glob_files` | Find files by pattern |
| `grep_search` | Search file contents with regex |
| `ls_directory` | List directory contents |
| `system_info` | Get system information |

## Example: Multi-Step Reasoning

Request:
```bash
curl -X POST "http://localhost:8888/api/v1/agents/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Find all Python files in /path/to/project and tell me which ones contain the word API",
    "user_id": "xenodennis",
    "env": "desktop"
  }'
```

The LLM intelligently decides the execution strategy:

```
Step 1: reason_node analyzes request
        → Decides: "I need to find Python files first"
        → Calls: glob_files with pattern "**/*.py"

Step 2: reason_node receives glob results
        → Decides: "Now search for 'API' in those files"
        → Calls: bash_execute with "grep -l 'API' <files>"

Step 3: reason_node receives grep results
        → Decides: "I have all the information"
        → Proceeds to: format_response

Step 4: format_response synthesizes answer
        → Returns comprehensive response to user
```

Each tool call is routed through:
```
SDK → Pool Manager → Desktop Agent → Local Filesystem → Result
```

## Troubleshooting

### Desktop Not Found

```
Error: No online desktop found for user xenodennis
```

**Solution:** Ensure Desktop Agent is running and connected:
```bash
# Check desktop status
curl -s http://localhost:8090/admin/desktops | jq '.desktops[] | select(.owner_id == "xenodennis")'

# Verify WebSocket connection
curl -s http://localhost:8090/desktop/online | jq .
```

### Desktop Not Connected

```
Error: Desktop desktop_xxx is not connected
```

**Solution:** The Desktop Agent's WebSocket connection dropped. Restart it:
```bash
cd /path/to/desktop_os
./deployment/local-dev.sh --stop
./deployment/local-dev.sh --run
```

### Tool Execution Timeout

```
Error: Desktop execution timed out
```

**Solution:** Increase timeout in request or check Desktop Agent logs:
```bash
tail -f /path/to/desktop_os/logs/desktop_agent.log
```

## Security Considerations

1. **User Isolation**: Each desktop is owned by a specific user. Tools only execute on desktops owned by the requesting user.

2. **Visibility Control**: Desktops can be:
   - `private`: Only owner can use
   - `shared`: Anyone can acquire from pool
   - `friends`: Only approved users
   - `organization`: Only org members

3. **Sandbox Mode**: Desktop Agent can run in sandbox mode to restrict file access:
   ```bash
   ./deployment/local-dev.sh --run --sandbox
   ```

4. **Dangerous Command Blocking**: The Desktop Agent blocks dangerous commands like `rm -rf /`.

## Related Documentation

- [Options Reference](./options.md) - All ISAAgentOptions parameters
- [Tools Guide](./tools.md) - Working with MCP tools
- [Streaming Guide](./streaming.md) - Handling streaming responses
- [Deployment Guide](./deployment-guide.md) - Production deployment
