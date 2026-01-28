# isA Agent SDK - Deployment Guide

Build and deploy custom AI agents to the isA Cloud platform.

## Overview

The isA Agent SDK allows developers to build custom FastAPI-based agents that can be deployed to the isA Cloud pool-manager. This guide covers the complete workflow from development to deployment.

### Architecture

```
Developer                     isA Platform                      Runtime
─────────                    ─────────────                     ───────

  Build Agent    ────────►   Code Deployment Service
  (FastAPI app)              ┌─────────────────────┐
                             │ 1. Upload code      │
                             │ 2. Build image      │
                             │ 3. Deploy version   │
                             └──────────┬──────────┘
                                        │
                                        ▼
  User Request   ────────►   Pool Manager
                             ┌─────────────────────┐     ┌────────────────┐
                             │ Route to agent      │────►│ Cloud OS VM    │
                             │ (isolated execution)│     │ (your agent)   │
                             └─────────────────────┘     └────────────────┘
```

## Installation

```bash
pip install isa-agent-sdk
```

This installs all dependencies:
- `isa-common` - Shared utilities (Consul, logging, etc.)
- `isa-mcp` - MCP client for tool execution
- `isa-model` - Model service client for LLM calls
- `langchain-core`, `langgraph` - Agent execution framework

## Quick Start

### 1. Create Your Agent

Create a new directory for your agent project:

```bash
mkdir my-agent
cd my-agent
```

Create `main.py`:

```python
"""
My Custom Agent - Built with isA Agent SDK
"""
import os
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from isa_agent_sdk import query, ISAAgentOptions, ExecutionMode, AgentMessage

app = FastAPI(
    title="My Custom Agent",
    description="A custom agent built with isA Agent SDK",
    version="1.0.0"
)

# ============================================================================
# Models
# ============================================================================

class QueryRequest(BaseModel):
    prompt: str
    context: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None

# ============================================================================
# Agent Configuration
# ============================================================================

SYSTEM_PROMPT = """You are a helpful assistant. Be concise and accurate."""

def get_agent_options(user_id: str, session_id: str) -> ISAAgentOptions:
    """Configure agent options"""
    return ISAAgentOptions(
        execution_mode=ExecutionMode.COLLABORATIVE,
        allowed_tools=["WebSearch", "WebFetch", "Read"],
        system_prompt=SYSTEM_PROMPT,
        session_id=session_id,
        user_id=user_id,
        max_iterations=30,
    )

# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint (required)"""
    return {
        "status": "healthy",
        "service": "my-agent",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """Non-streaming query endpoint"""
    try:
        user_id = os.getenv("ISA_USER_ID", "anonymous")
        session_id = os.getenv("ISA_SESSION_ID", f"session_{datetime.now().timestamp()}")

        options = get_agent_options(user_id, session_id)

        response_parts = []
        async for msg in query(request.prompt, options=options):
            if msg.type == "result" and msg.content:
                response_parts.append(msg.content)

        return QueryResponse(
            success=True,
            response="".join(response_parts)
        )
    except Exception as e:
        return QueryResponse(success=False, error=str(e))

@app.post("/stream")
async def stream_agent(request: QueryRequest):
    """Streaming query endpoint (SSE)"""
    import json

    user_id = os.getenv("ISA_USER_ID", "anonymous")
    session_id = os.getenv("ISA_SESSION_ID", f"session_{datetime.now().timestamp()}")

    async def generate():
        try:
            options = get_agent_options(user_id, session_id)

            async for msg in query(request.prompt, options=options):
                event = {
                    "type": msg.type,
                    "content": msg.content or "",
                    "timestamp": datetime.now().isoformat()
                }
                yield f"data: {json.dumps(event)}\n\n"

            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.get("/")
async def root():
    """Service info"""
    return {
        "service": "my-agent",
        "version": "1.0.0",
        "endpoints": ["/health", "/query", "/stream"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

Create `requirements.txt`:

```
# Additional dependencies (isa-agent-sdk is in base image)
# Add any extra packages your agent needs here
```

### 2. Test Locally

```bash
# Install SDK
pip install isa-agent-sdk fastapi uvicorn

# Run locally
python main.py
```

Test the endpoints:

```bash
# Health check
curl http://localhost:8080/health

# Query
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?"}'

# Stream
curl -X POST http://localhost:8080/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing"}'
```

## Deployment to isA Cloud

### Step 1: Create Deployment

```bash
# Create a new deployment with configuration
curl -X POST http://localhost:8095/api/v1/deployments \
  -H "X-User-ID: user123" \
  -H "Content-Type: application/json" \
  -d '{
    "app_name": "my-agent",
    "description": "My custom AI agent",
    "config": {
      "port": 8080,
      "replicas": 1,
      "resources": {
        "cpu": 2,
        "memory": 2048,
        "disk": 10240,
        "timeout": 300
      },
      "env_vars": {
        "MY_CONFIG": "value"
      },
      "warm_pool": true
    }
  }'
```

Response:
```json
{
  "deployment_id": "dep_abc123",
  "app_name": "my-agent",
  "status": "PENDING",
  "created_at": "2026-01-27T10:30:00Z"
}
```

#### Deployment Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `port` | int | 8080 | Application port |
| `replicas` | int (1-10) | 1 | Number of instances |
| `resources.cpu` | int (1-8) | 2 | CPU cores |
| `resources.memory` | int (512-16384) | 2048 | Memory in MB |
| `resources.disk` | int (1024-102400) | 10240 | Disk in MB |
| `resources.timeout` | int (30-3600) | 300 | Request timeout in seconds |
| `env_vars` | dict | {} | Custom environment variables |
| `warm_pool` | bool | false | Keep instances pre-warmed |

### Step 2: Upload Code

Package and upload your agent:

```bash
# Create tarball (must include main.py at root)
tar -czf my-agent.tar.gz -C my-agent .

# Upload to deployment (triggers build automatically)
curl -X POST http://localhost:8095/api/v1/deployments/dep_abc123/upload \
  -H "X-User-ID: user123" \
  -F "file=@my-agent.tar.gz"
```

Response:
```json
{
  "upload_id": "upload_123",
  "build_id": "build_xyz789",
  "message": "Upload successful, build started"
}
```

#### Code Requirements

Your code must include:
- `main.py` with a FastAPI `app` instance
- `/health` endpoint (required for health checks)
- `/query` and/or `/stream` endpoints for agent interactions

**Forbidden patterns** (security validation):
- `subprocess.call`, `subprocess.run`, `subprocess.Popen`
- `os.system`, `os.popen`
- `eval`, `exec`, `__import__`

### Step 3: Monitor Build

```bash
# Check build status
curl http://localhost:8095/api/v1/deployments/dep_abc123/builds/build_xyz789 \
  -H "X-User-ID: user123"
```

Response:
```json
{
  "build_id": "build_xyz789",
  "status": "SUCCEEDED",
  "image_tag": "localhost:5000/dep_abc123:build_xyz789",
  "started_at": "2026-01-27T10:30:00Z",
  "completed_at": "2026-01-27T10:32:00Z"
}
```

#### Build Status Values

| Status | Description |
|--------|-------------|
| `PENDING` | Build queued |
| `VALIDATING` | Checking code requirements |
| `BUILDING` | Docker image building |
| `PUSHING` | Pushing to registry |
| `SUCCEEDED` | Build complete |
| `FAILED` | Build failed |

#### Stream Build Logs (SSE)

```bash
curl http://localhost:8095/api/v1/deployments/dep_abc123/builds/build_xyz789/logs \
  -H "X-User-ID: user123"
```

### Step 4: Create Version

After a successful build, create a versioned release:

```bash
curl -X POST http://localhost:8095/api/v1/deployments/dep_abc123/versions \
  -H "X-User-ID: user123" \
  -H "Content-Type: application/json" \
  -d '{
    "build_id": "build_xyz789",
    "version": "v1.0.0",
    "release_notes": "Initial release with weather features"
  }'
```

### Step 5: Deploy

Deploy the version to Pool Manager:

```bash
curl -X POST http://localhost:8095/api/v1/deployments/dep_abc123/deploy \
  -H "X-User-ID: user123" \
  -H "Content-Type: application/json" \
  -d '{
    "version": "v1.0.0",
    "replicas": 2,
    "warm_pool": true
  }'
```

Response:
```json
{
  "deployment_id": "dep_abc123",
  "version": "v1.0.0",
  "status": "ACTIVE",
  "replicas": 2,
  "instances": [
    {"instance_id": "inst_001", "status": "READY"},
    {"instance_id": "inst_002", "status": "READY"}
  ]
}
```

### Step 6: Query Your Deployed Agent

```bash
# Non-streaming query
curl -X POST http://localhost:8095/api/v1/apps/dep_abc123/query \
  -H "X-User-ID: user123" \
  -H "X-Session-ID: session_001" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello from the cloud!"}'

# Streaming query (SSE)
curl -X POST http://localhost:8095/api/v1/apps/dep_abc123/stream \
  -H "X-User-ID: user123" \
  -H "X-Session-ID: session_001" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing", "stream": true}'
```

### Step 7: Release Session (Optional)

Release the instance back to the pool:

```bash
curl -X POST http://localhost:8095/api/v1/apps/dep_abc123/release \
  -H "X-User-ID: user123" \
  -H "X-Session-ID: session_001"
```

## Version Management

### List Versions

```bash
curl http://localhost:8095/api/v1/deployments/dep_abc123/versions \
  -H "X-User-ID: user123"
```

### Get Latest Version

```bash
curl http://localhost:8095/api/v1/deployments/dep_abc123/versions/latest \
  -H "X-User-ID: user123"
```

### Rollback to Previous Version

```bash
curl -X POST http://localhost:8095/api/v1/deployments/dep_abc123/rollback \
  -H "X-User-ID: user123" \
  -H "Content-Type: application/json" \
  -d '{"version": "v0.9.0"}'
```

## Deployment Lifecycle

```
PENDING → BUILDING → READY → DEPLOYING → ACTIVE
                                    ↓
                                  STOPPED ← (manual stop)
                                    ↓
                                  FAILED ← (error)
```

### Check Deployment Health

```bash
curl http://localhost:8095/api/v1/deployments/dep_abc123/health \
  -H "X-User-ID: user123"
```

### Delete Deployment

```bash
curl -X DELETE http://localhost:8095/api/v1/deployments/dep_abc123 \
  -H "X-User-ID: user123"
```

## Real-World Example: Weather Agent

Here's a complete example of a weather information agent:

### `main.py`

```python
"""
Weather Agent - Provides weather information and recommendations
"""
import os
import json
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from isa_agent_sdk import query, ISAAgentOptions, ExecutionMode

app = FastAPI(title="Weather Agent", version="1.0.0")

class QueryRequest(BaseModel):
    prompt: str
    location: Optional[str] = None

class QueryResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None

WEATHER_SYSTEM_PROMPT = """You are a helpful weather assistant. You can:
1. Provide current weather information for any location
2. Give weather forecasts
3. Recommend activities based on weather conditions

When providing weather information:
- Include temperature in both Celsius and Fahrenheit
- Mention humidity, wind, and precipitation chances
- Suggest appropriate clothing
- Recommend indoor/outdoor activities

If you don't have real-time data, provide helpful general guidance."""

def get_options(user_id: str, session_id: str) -> ISAAgentOptions:
    return ISAAgentOptions(
        execution_mode=ExecutionMode.COLLABORATIVE,
        allowed_tools=["WebSearch", "WebFetch"],
        system_prompt=WEATHER_SYSTEM_PROMPT,
        session_id=session_id,
        user_id=user_id,
        max_iterations=20,
    )

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "weather-agent"}

@app.post("/query", response_model=QueryResponse)
async def weather_query(request: QueryRequest):
    try:
        user_id = os.getenv("ISA_USER_ID", "anonymous")
        session_id = f"weather_{datetime.now().timestamp()}"

        # Enhance prompt with location if provided
        prompt = request.prompt
        if request.location:
            prompt = f"{request.prompt} (Location: {request.location})"

        options = get_options(user_id, session_id)

        results = []
        async for msg in query(prompt, options=options):
            if msg.type == "result" and msg.content:
                results.append(msg.content)

        return QueryResponse(success=True, response="".join(results))
    except Exception as e:
        return QueryResponse(success=False, error=str(e))

@app.post("/stream")
async def weather_stream(request: QueryRequest):
    async def generate():
        try:
            user_id = os.getenv("ISA_USER_ID", "anonymous")
            session_id = f"weather_{datetime.now().timestamp()}"

            prompt = request.prompt
            if request.location:
                prompt = f"{request.prompt} (Location: {request.location})"

            options = get_options(user_id, session_id)

            async for msg in query(prompt, options=options):
                yield f"data: {json.dumps({'type': msg.type, 'content': msg.content})}\n\n"

            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/")
async def root():
    return {
        "service": "weather-agent",
        "description": "Weather information and recommendations",
        "endpoints": {
            "/health": "Health check",
            "/query": "POST - Get weather info",
            "/stream": "POST - Stream weather info"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

### Deployment Commands

```bash
# 1. Create deployment
curl -X POST http://localhost:8095/api/v1/deployments \
  -H "Content-Type: application/json" \
  -d '{"app_name": "weather-agent", "user_id": "user123"}'

# 2. Package and upload
tar -czf weather-agent.tar.gz main.py requirements.txt
curl -X POST http://localhost:8095/api/v1/deployments/{deployment_id}/upload \
  -F "file=@weather-agent.tar.gz"

# 3. Build
curl -X POST http://localhost:8095/api/v1/deployments/{deployment_id}/builds

# 4. Deploy (after build succeeds)
curl -X POST http://localhost:8095/api/v1/deployments/{deployment_id}/deploy \
  -d '{"build_id": "{build_id}"}'

# 5. Test
curl -X POST http://localhost:8095/api/v1/apps/{deployment_id}/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the weather like in Tokyo?", "location": "Tokyo, Japan"}'
```

## SDK API Reference

### Core Functions

#### `query(prompt, options)`

Execute an agent query and stream messages.

```python
from isa_agent_sdk import query, ISAAgentOptions

async for msg in query("Hello", options=ISAAgentOptions()):
    print(f"{msg.type}: {msg.content}")
```

**Parameters:**
- `prompt` (str): The user's request
- `options` (ISAAgentOptions): Configuration options

**Yields:** `AgentMessage` objects

### ISAAgentOptions

```python
from isa_agent_sdk import ISAAgentOptions, ExecutionMode

options = ISAAgentOptions(
    # Execution mode
    execution_mode=ExecutionMode.COLLABORATIVE,  # or REACTIVE, PROACTIVE

    # Tools
    allowed_tools=["WebSearch", "Read", "Write"],

    # Model
    model="gpt-5-nano",

    # Prompts
    system_prompt="You are a helpful assistant.",

    # Session
    session_id="my-session",
    user_id="user123",

    # Limits
    max_iterations=50,
)
```

### AgentMessage

Message types returned by `query()`:

| Type | Description |
|------|-------------|
| `session_start` | Session began |
| `session_end` | Session completed |
| `system` | System information |
| `node_enter` | Entering graph node |
| `node_exit` | Exiting graph node |
| `tool_use` | Tool being called |
| `tool_result` | Tool returned result |
| `result` | Final response content |
| `error` | Error occurred |

```python
async for msg in query(prompt, options):
    if msg.type == "result":
        print(f"Response: {msg.content}")
    elif msg.type == "tool_use":
        print(f"Using tool: {msg.metadata.get('tool_name')}")
    elif msg.type == "error":
        print(f"Error: {msg.content}")
```

## Environment Variables

When deployed, these environment variables are available:

| Variable | Description |
|----------|-------------|
| `ISA_USER_ID` | User ID for the request |
| `ISA_SESSION_ID` | Session identifier |
| `ISA_DEPLOYMENT_ID` | Deployment identifier |
| `ISA_VERSION_ID` | Version identifier |
| `ISA_NETWORK_MODE` | Network mode (proxy/direct) |

## Best Practices

### 1. Always Implement Health Check

```python
@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### 2. Handle Errors Gracefully

```python
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        # ... your logic
        return QueryResponse(success=True, response=result)
    except Exception as e:
        return QueryResponse(success=False, error=str(e))
```

### 3. Use Streaming for Long Responses

```python
@app.post("/stream")
async def stream_endpoint(request: QueryRequest):
    async def generate():
        async for msg in query(request.prompt, options):
            yield f"data: {json.dumps(msg.__dict__)}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
```

### 4. Configure Appropriate Tools

Only enable the tools your agent needs:

```python
options = ISAAgentOptions(
    allowed_tools=["WebSearch"],  # Minimal tools
)
```

### 5. Set Reasonable Iteration Limits

```python
options = ISAAgentOptions(
    max_iterations=20,  # Prevent infinite loops
)
```

## Security & Isolation

### VM Isolation Model

Deployed agents run in isolated Firecracker microVMs with:

- **Network isolation**: `network_mode: "none"` - No direct internet access
- **Resource limits**: CPU, memory, and disk enforced per VM
- **Credential injection**: API keys injected via Unix socket proxy
- **Separate processes**: Each deployment runs in its own VM

```
┌─────────────────────────────────────────────────────────┐
│                     Pool Manager                         │
│  ┌─────────────────────────────────────────────────────┐│
│  │              API Proxy Service                       ││
│  │  ┌─────────────────┐  ┌─────────────────┐           ││
│  │  │ Vault Service   │  │ Model Service   │           ││
│  │  │ (credentials)   │  │ (LLM calls)     │           ││
│  │  └────────┬────────┘  └────────┬────────┘           ││
│  │           │                    │                     ││
│  │           └────────┬───────────┘                     ││
│  │                    │                                 ││
│  │    Unix Socket: /var/run/isa-proxy/{instance}.sock  ││
│  └─────────────────────────────────────────────────────┘│
│                        ▲                                 │
│                        │                                 │
│  ┌─────────────────────┴───────────────────────────────┐│
│  │              Firecracker VM (your agent)            ││
│  │  ┌─────────────────────────────────────────────────┐││
│  │  │  FastAPI App (main.py)                          │││
│  │  │  - Network: none                                │││
│  │  │  - Only access: Unix socket proxy               │││
│  │  │  - Env: ISA_PROXY_SOCKET, ISA_USER_ID, etc.    │││
│  │  └─────────────────────────────────────────────────┘││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

### Credential Injection

Your agent never sees raw API keys. All external calls go through the proxy:

```python
# Inside your agent, the SDK automatically uses the proxy socket
# You just call the SDK normally:
async for msg in query("Hello", options=options):
    print(msg.content)

# Behind the scenes:
# 1. SDK sends request to Unix socket
# 2. Proxy looks up user's credentials from Vault
# 3. Proxy injects Authorization header
# 4. Proxy forwards to Model Service
# 5. Response returned to your agent
```

### Session Affinity

Requests with the same `X-Session-ID` header are routed to the same instance:

```python
# First request creates/acquires instance
curl -H "X-Session-ID: session_001" .../query

# Subsequent requests with same session go to same instance
curl -H "X-Session-ID: session_001" .../query  # Same instance

# Different session may get different instance
curl -H "X-Session-ID: session_002" .../query  # May be different instance
```

## Deployment Script

Here's a helper script to automate deployment:

```bash
#!/bin/bash
# deploy.sh - Deploy an isA Agent to the cloud

set -e

# Configuration
DEPLOYMENT_URL="${ISA_DEPLOYMENT_URL:-http://localhost:8095}"
USER_ID="${ISA_USER_ID:-$(whoami)}"
APP_DIR="${1:-.}"
APP_NAME="${2:-my-agent}"

echo "=== isA Agent Deployment ==="
echo "App: $APP_NAME"
echo "Directory: $APP_DIR"
echo "User: $USER_ID"
echo ""

# Validate
if [ ! -f "$APP_DIR/main.py" ]; then
    echo "Error: main.py not found in $APP_DIR"
    exit 1
fi

# Step 1: Create deployment
echo "Creating deployment..."
RESPONSE=$(curl -s -X POST "$DEPLOYMENT_URL/api/v1/deployments" \
  -H "X-User-ID: $USER_ID" \
  -H "Content-Type: application/json" \
  -d "{\"app_name\": \"$APP_NAME\", \"description\": \"Deployed via script\"}")

DEPLOYMENT_ID=$(echo $RESPONSE | jq -r '.deployment_id')
echo "Deployment ID: $DEPLOYMENT_ID"

# Step 2: Package and upload
echo "Packaging and uploading..."
TARBALL="/tmp/${APP_NAME}.tar.gz"
tar -czf "$TARBALL" -C "$APP_DIR" .

UPLOAD_RESPONSE=$(curl -s -X POST "$DEPLOYMENT_URL/api/v1/deployments/$DEPLOYMENT_ID/upload" \
  -H "X-User-ID: $USER_ID" \
  -F "file=@$TARBALL")

BUILD_ID=$(echo $UPLOAD_RESPONSE | jq -r '.build_id')
echo "Build ID: $BUILD_ID"

# Step 3: Wait for build
echo "Waiting for build to complete..."
while true; do
    BUILD_STATUS=$(curl -s "$DEPLOYMENT_URL/api/v1/deployments/$DEPLOYMENT_ID/builds/$BUILD_ID" \
      -H "X-User-ID: $USER_ID" | jq -r '.status')

    echo "  Build status: $BUILD_STATUS"

    if [ "$BUILD_STATUS" = "SUCCEEDED" ]; then
        break
    elif [ "$BUILD_STATUS" = "FAILED" ]; then
        echo "Build failed!"
        exit 1
    fi

    sleep 5
done

# Step 4: Create version
echo "Creating version..."
VERSION="v$(date +%Y%m%d.%H%M%S)"
curl -s -X POST "$DEPLOYMENT_URL/api/v1/deployments/$DEPLOYMENT_ID/versions" \
  -H "X-User-ID: $USER_ID" \
  -H "Content-Type: application/json" \
  -d "{\"build_id\": \"$BUILD_ID\", \"version\": \"$VERSION\"}" > /dev/null

# Step 5: Deploy
echo "Deploying version $VERSION..."
curl -s -X POST "$DEPLOYMENT_URL/api/v1/deployments/$DEPLOYMENT_ID/deploy" \
  -H "X-User-ID: $USER_ID" \
  -H "Content-Type: application/json" \
  -d "{\"version\": \"$VERSION\", \"replicas\": 1, \"warm_pool\": true}" > /dev/null

echo ""
echo "=== Deployment Complete ==="
echo "Deployment ID: $DEPLOYMENT_ID"
echo "Version: $VERSION"
echo ""
echo "Query endpoint:"
echo "  curl -X POST $DEPLOYMENT_URL/api/v1/apps/$DEPLOYMENT_ID/query \\"
echo "    -H 'X-User-ID: $USER_ID' \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"prompt\": \"Hello!\"}'"

# Cleanup
rm -f "$TARBALL"
```

Usage:
```bash
chmod +x deploy.sh
./deploy.sh ./my-agent my-custom-agent
```

## API Reference

### Code Deployment Service Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/deployments` | Create deployment |
| `GET` | `/api/v1/deployments` | List user's deployments |
| `GET` | `/api/v1/deployments/{id}` | Get deployment details |
| `POST` | `/api/v1/deployments/{id}/upload` | Upload code (auto-builds) |
| `POST` | `/api/v1/deployments/{id}/builds` | Trigger manual build |
| `GET` | `/api/v1/deployments/{id}/builds/{bid}` | Get build status |
| `GET` | `/api/v1/deployments/{id}/builds/{bid}/logs` | Stream build logs (SSE) |
| `POST` | `/api/v1/deployments/{id}/versions` | Create version |
| `GET` | `/api/v1/deployments/{id}/versions` | List versions |
| `GET` | `/api/v1/deployments/{id}/versions/latest` | Get latest version |
| `POST` | `/api/v1/deployments/{id}/deploy` | Deploy version |
| `POST` | `/api/v1/deployments/{id}/rollback` | Rollback to version |
| `GET` | `/api/v1/deployments/{id}/health` | Get deployment health |
| `DELETE` | `/api/v1/deployments/{id}` | Delete deployment |

### App Query Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/apps/{app_id}/query` | Execute query |
| `POST` | `/api/v1/apps/{app_id}/stream` | Stream query (SSE) |
| `GET` | `/api/v1/apps/{app_id}/health` | Get app health |
| `POST` | `/api/v1/apps/{app_id}/release` | Release session instance |

### Required Headers

| Header | Description |
|--------|-------------|
| `X-User-ID` | User identifier (required) |
| `X-Session-ID` | Session for affinity (optional) |
| `Content-Type` | `application/json` for JSON bodies |

## Troubleshooting

### SDK Not Available

If `sdk_available: false` in health check:
- Ensure `isa-agent-sdk` is installed
- Check for import errors in container logs

### Model Calls Failing

- Verify `ISA_API_URL` points to model service
- Check model service is running on port 8082

### Tools Not Working

- Verify `MCP_SERVER_URL` points to MCP service
- Check MCP service is running on port 8081

### Build Failed - Validation Error

Common validation errors:
- `main.py not found` - Ensure main.py is at the root of your tarball
- `FastAPI app not found` - Ensure you have `app = FastAPI()` in main.py
- `Forbidden pattern detected` - Remove subprocess/os.system/eval calls

### Connection Refused

In Docker, use `host.docker.internal` to reach host services:
```bash
-e ISA_API_URL=http://host.docker.internal:8082
-e MCP_SERVER_URL=http://host.docker.internal:8081
```

### Instance Not Responding

Check deployment health:
```bash
curl http://localhost:8095/api/v1/deployments/{id}/health \
  -H "X-User-ID: user123"
```

If unhealthy, check build logs and redeploy.

## Support

- GitHub Issues: https://github.com/isa-agent/sdk/issues
- Documentation: https://docs.isa-agent.com/sdk
