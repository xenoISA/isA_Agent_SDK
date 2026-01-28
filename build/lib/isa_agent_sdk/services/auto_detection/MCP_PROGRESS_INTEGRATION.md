# MCP Progress Integration Guide

## Overview

We've enhanced `mcp_service.py` to capture real-time progress updates from MCP tools that use the Context API for progress reporting and logging.

---

## What Was Added

### 1. **Progress-Aware Tool Execution**

```python
# New signature with progress callback
result = await mcp_service.call_tool(
    tool_name="web_search",
    arguments={"query": "AI news"},
    progress_callback=my_progress_handler  # ✅ NEW!
)
```

### 2. **SSE (Server-Sent Events) Parsing**

MCP tools can send progress updates using the Context API:

```python
# Inside MCP tool implementation
await self.report_progress(ctx, current=3, total=10, message="Processing...")
await self.log_info(ctx, "Step completed")
```

These are sent as SSE events:

```
event: progress
data: {"progress": 3, "total": 10, "message": "Processing..."}

event: log
data: {"level": "info", "message": "Step completed"}

event: message
data: {"jsonrpc": "2.0", "result": {...}}
```

### 3. **New Methods in MCPService**

#### `_request_with_progress(method, params, progress_callback)`
- Handles streaming responses from MCP server
- Detects SSE vs JSON responses
- Invokes progress callback for real-time updates

#### `_parse_sse_response(response, progress_callback)`
- Parses SSE stream line by line
- Extracts progress events, log events, and final results
- Calls progress_callback with structured data

---

## Progress Callback Signature

```python
def progress_callback(
    current: int,        # Current step (for progress events)
    total: int,          # Total steps (for progress events)
    message: str,        # Progress/log message
    event_type: str,     # 'progress' or 'log'
    log_level: str = ''  # 'info', 'debug', 'warning', 'error' (for log events)
):
    """
    Called for each progress update or log message from MCP tool
    """
    if event_type == 'progress':
        print(f"[{current}/{total}] {message}")
    elif event_type == 'log':
        print(f"[{log_level.upper()}] {message}")
```

---

## Usage Examples

### Example 1: Simple Progress Tracking

```python
from src.components.mcp_service import MCPService

mcp = MCPService(mcp_url="http://localhost:8081")
await mcp.initialize()

def my_progress_handler(current, total, message, event_type, **kwargs):
    if event_type == 'progress':
        percent = int((current / total) * 100)
        print(f"Progress: {percent}% - {message}")
    elif event_type == 'log':
        print(f"Log: {message}")

# Execute tool with progress tracking
result = await mcp.call_tool(
    tool_name="test_context_progress",
    arguments={"total_steps": 10, "delay_ms": 100},
    progress_callback=my_progress_handler
)

# Output:
# Progress: 10% - Processing step 1/10
# Progress: 20% - Processing step 2/10
# ...
# Progress: 100% - Processing step 10/10
```

### Example 2: Redis Pub/Sub Progress Broadcasting

```python
import redis
import json

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

def broadcast_progress(job_id: str):
    """Create a progress callback that broadcasts to Redis pub/sub"""

    def callback(current, total, message, event_type, **kwargs):
        channel = f"job_progress:{job_id}"

        if event_type == 'progress':
            redis_client.publish(channel, json.dumps({
                "type": "progress",
                "current": current,
                "total": total,
                "message": message,
                "percent": int((current / total) * 100)
            }))
        elif event_type == 'log':
            redis_client.publish(channel, json.dumps({
                "type": "log",
                "level": kwargs.get('log_level', 'info'),
                "message": message
            }))

    return callback

# Use it
job_id = "job_12345"
result = await mcp.call_tool(
    tool_name="long_running_tool",
    arguments={...},
    progress_callback=broadcast_progress(job_id)
)
```

### Example 3: Background Job with Progress

```python
# In background_job_service.py
from src.components.mcp_service import MCPService

async def execute_tool_with_progress(job_id, tool_name, tool_args):
    """Execute tool and publish progress to Redis"""

    mcp = MCPService(mcp_url=settings.mcp_url)
    await mcp.initialize()

    def progress_handler(current, total, message, event_type, **kwargs):
        # Publish to Redis for real-time updates
        publish_job_progress(job_id, {
            "type": event_type,
            "current": current,
            "total": total,
            "message": message,
            "log_level": kwargs.get('log_level', 'info')
        })

    try:
        result = await mcp.call_tool(
            tool_name=tool_name,
            arguments=tool_args,
            progress_callback=progress_handler
        )
        return result
    finally:
        await mcp.close()
```

---

## Integration with Auto-Detection System

### How Progress Helps Auto-Detection

1. **Detect Long-Running Tools**
   - Tools that report progress are likely long-running
   - If a tool reports > 5 progress steps, it's a background candidate

2. **Estimate Execution Time**
   - Track progress rate: `(current - last_current) / (time - last_time)`
   - Estimate remaining time: `(total - current) / progress_rate`

3. **Build Richer Profiles**
   - Store not just execution time, but also:
     - Number of progress steps
     - Average progress rate
     - Logs pattern (info/warning/error counts)

### Proposed Auto-Detection Logic

```python
from src.services.auto_detection.tool_profiler import ToolProfiler

class BackgroundDetector:
    """Decide whether to run tool in background or immediate"""

    def __init__(self):
        self.profiler = ToolProfiler()

    async def should_background(
        self,
        tool_name: str,
        tool_args: dict,
        user_preferences: dict = None
    ) -> bool:
        """
        Intelligent auto-detection based on multiple factors

        Returns:
            True if tool should run in background
        """
        # 1. Check historical execution time
        estimated_time_ms = self.profiler.estimate_time(tool_name, tool_args)

        # 2. Check if tool supports progress (indicator of complexity)
        tool_profile = self.profiler.get_statistics(tool_name)
        supports_progress = False

        if tool_profile:
            # Check if tool has reported progress in past executions
            supports_progress = tool_profile.get('has_progress_history', False)

        # 3. Decision thresholds
        IMMEDIATE_THRESHOLD = 5000  # 5 seconds
        BACKGROUND_THRESHOLD = 8000  # 8 seconds

        if estimated_time_ms > BACKGROUND_THRESHOLD:
            # Definitely background (> 8 seconds)
            return True

        if estimated_time_ms > IMMEDIATE_THRESHOLD and supports_progress:
            # Probably background (> 5 seconds + complex tool)
            return True

        # 4. Check user preferences
        if user_preferences and user_preferences.get('prefer_background', False):
            if estimated_time_ms > IMMEDIATE_THRESHOLD:
                return True

        # Default: run immediately
        return False
```

---

## Next Steps

### 1. **Enhance ToolProfiler**

Add progress tracking to the profiler:

```python
# In tool_profiler.py
def record_execution_with_progress(
    self,
    tool_name: str,
    execution_time_ms: int,
    tool_args: dict,
    session_id: str,
    success: bool,
    progress_steps: int = 0,  # ✅ NEW
    log_count: dict = None     # ✅ NEW: {"info": 5, "warning": 1, "error": 0}
):
    """Record execution with progress metadata"""
    record = {
        "time_ms": execution_time_ms,
        "args_size": len(str(tool_args)),
        "session_id": session_id,
        "timestamp": time.time(),
        "success": success,
        "progress_steps": progress_steps,      # ✅
        "log_count": log_count or {}           # ✅
    }
    # ... store to Redis
```

### 2. **Create BackgroundDetector**

```bash
src/services/auto_detection/
├── tool_profiler.py           # ✅ Already exists
├── background_detector.py     # 🆕 Decision logic
└── progress_tracker.py        # 🆕 Capture progress during execution
```

### 3. **Integrate with BackgroundJobService**

```python
# In background_job_service.py
from src.components.mcp_service import MCPService

@celery_app.task(base=ToolExecutionTask, bind=True)
def queue_tool_execution_job(self, job_data, config):
    job_id = job_data['job_id']
    tools = job_data['tools']

    mcp = MCPService(mcp_url=settings.mcp_url)
    await mcp.initialize()

    for tool_info in tools:
        tool_name = tool_info['tool_name']
        tool_args = tool_info['tool_args']

        # Create progress callback for this tool
        def progress_callback(current, total, message, event_type, **kwargs):
            publish_job_progress(job_id, {
                "type": event_type,
                "tool_name": tool_name,
                "current": current,
                "total": total,
                "message": message
            })

        # Execute with progress tracking
        result = await mcp.call_tool(
            tool_name=tool_name,
            arguments=tool_args,
            progress_callback=progress_callback  # ✅
        )
```

---

## Testing Progress Capture

### Test with Context Test Tools

```bash
# 1. Start MCP server with test tools
cd /Users/xenodennis/Documents/Fun/isA_MCP
docker-compose -f deployment/staging/docker-compose.staging.yml up -d

# 2. Test progress reporting
curl -X POST http://localhost:8081/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "test_context_progress",
      "arguments": {
        "total_steps": 10,
        "delay_ms": 100
      }
    }
  }'

# Expected output:
# event: progress
# data: {"progress": 1, "total": 10, "message": "Processing step 1/10"}
#
# event: progress
# data: {"progress": 2, "total": 10, "message": "Processing step 2/10"}
# ...
# event: message
# data: {"jsonrpc": "2.0", "result": {...}}
```

---

## Benefits

### ✅ Real-Time Feedback
- Users see progress for long-running tools
- Better UX for background jobs

### ✅ Intelligent Auto-Detection
- Detect which tools are truly long-running
- Make smarter background vs immediate decisions

### ✅ Enhanced Profiling
- Build richer tool execution profiles
- Track not just time, but complexity patterns

### ✅ Better Monitoring
- Capture logs from MCP tools
- Debug tool execution in production

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      User Request                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │  BackgroundDetector    │
          │  (Auto-detection)      │
          └────────┬───────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
  [Immediate]          [Background Job]
        │                     │
        │              ┌──────┴─────────┐
        │              │ Celery Worker  │
        │              └──────┬─────────┘
        │                     │
        └──────────┬──────────┘
                   ▼
          ┌────────────────────┐
          │   MCPService       │
          │   .call_tool()     │
          └────────┬───────────┘
                   │
          ┌────────┴───────────┐
          │ _request_with_     │
          │    progress()      │
          └────────┬───────────┘
                   │
          ┌────────┴───────────┐
          │ _parse_sse_        │
          │   response()       │
          └────────┬───────────┘
                   │
         ┏━━━━━━━━━┻━━━━━━━━━┓
         ┃  Progress Events   ┃
         ┃  • report_progress ┃
         ┃  • log_info        ┃
         ┃  • log_debug       ┃
         ┃  • log_warning     ┃
         ┃  • log_error       ┃
         ┗━━━━━━━━━┳━━━━━━━━━┛
                   │
                   ▼
          ┌────────────────────┐
          │ Progress Callback  │
          └────────┬───────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
         ▼                   ▼
   [ToolProfiler]    [Redis Pub/Sub]
   Record metadata   Broadcast updates
         │                   │
         ▼                   ▼
   [Future auto-    [WebSocket/SSE
    detection]       to frontend]
```

---

## Summary

✅ **Updated `mcp_service.py`** to capture progress from MCP tools
✅ **SSE parsing** for real-time progress and logs
✅ **Progress callback system** for flexible integration
🔜 **Next**: Build `BackgroundDetector` using this data
🔜 **Next**: Enhance `ToolProfiler` to store progress patterns
🔜 **Next**: Integrate with background job system

The foundation is now ready to build intelligent auto-detection! 🚀
