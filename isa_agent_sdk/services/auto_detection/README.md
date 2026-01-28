# Auto-Detection Implementation Guide

## What We're Building
Smart auto-detection system that decides: run task now vs queue to background

## Key Detection Factors
1. **Tool execution time** (from historical data in Redis)
2. **Task count** (20 tasks = long, even with short tools)
3. **Task complexity** (tool diversity, dependencies)
4. **System load** (CPU, memory)
5. **Loop detection** (stuck patterns)

## Where to Record Tool Execution Data

**Answer: Record in AGENT (not MCP)**

**Why:**
- Tools execute via MCP server (`/Users/xenodennis/Documents/Fun/isA_MCP/tools`)
- Agent calls tools via `mcp_call_tool()` in `base_node.py:325`
- **Recording happens in agent AFTER tool returns** (lines 1342-1354 in DESIGN.md)
- Agent has access to: execution time, state, session_id, user_id
- MCP is stateless - doesn't know about sessions/users

**Recording Location:** `src/nodes/tool_node.py` (after tool execution)

## Infrastructure Decisions

| Component | Use | Why |
|-----------|-----|-----|
| **Tool history** | Redis | You have it, perfect for time-series |
| **Progress streaming** | NATS | You have it, better than Redis Pub/Sub |
| **Background jobs** | Keep Celery OR use NATS | Keep what works |
| **Loop detection** | In-memory | Fast, ephemeral, no persistence needed |

## Implementation Status

### ✅ Already Exists (Don't Build)
- Parallel execution (`agent_executor_node.py:541-641`)
- PostgreSQL checkpointing (`durable_service.py`)

### ❌ Missing (Need to Build)
- Tool profiler (historical data)
- Task complexity calculator
- Smart detection engine
- Loop monitors (tool + graph level)
- Decision orchestrator

## First Task: Tool Profiler

### What It Does
- Records tool execution time to Redis
- Estimates future execution time (90th percentile)
- Used by smart detector

### Implementation

**File:** `src/services/auto_detection/tool_profiler.py`

```python
import time
import json
from typing import Dict, Any, List
import redis

class ToolProfiler:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.ttl_seconds = 604800  # 7 days

    async def record_execution(
        self,
        tool_name: str,
        execution_time_ms: int,
        tool_args: Dict,
        session_id: str,
        success: bool
    ):
        """Record tool execution to Redis"""
        key = f"tool_profile:{tool_name}"

        record = {
            "time_ms": execution_time_ms,
            "args_size": len(str(tool_args)),
            "session_id": session_id,
            "timestamp": time.time(),
            "success": success
        }

        # Add to sorted set (score = timestamp)
        self.redis.zadd(key, {json.dumps(record): time.time()})

        # Keep only last 100 records
        self.redis.zremrangebyrank(key, 0, -101)

        # Set TTL
        self.redis.expire(key, self.ttl_seconds)

    async def estimate_time(self, tool_name: str, tool_args: Dict) -> int:
        """Estimate execution time in ms (90th percentile)"""
        key = f"tool_profile:{tool_name}"

        # Get recent records
        records_raw = self.redis.zrange(key, -50, -1)

        if not records_raw:
            # Return default estimate
            return self._get_default_estimate(tool_name)

        # Parse records
        records = [json.loads(r) for r in records_raw]

        # Filter by similar argument size
        args_size = len(str(tool_args))
        similar_times = []

        for record in records:
            size_ratio = record["args_size"] / max(args_size, 1)
            if 0.5 < size_ratio < 2.0:  # Similar complexity
                similar_times.append(record["time_ms"])

        if len(similar_times) < 3:
            return self._get_default_estimate(tool_name)

        # Return 90th percentile
        similar_times.sort()
        p90_index = int(len(similar_times) * 0.9)
        return similar_times[p90_index]

    def _get_default_estimate(self, tool_name: str) -> int:
        """Default estimates for cold start (ms)"""
        defaults = {
            "web_search": 3000,
            "web_crawl": 12000,
            "file_read": 500,
            "database_query": 2000,
        }
        return defaults.get(tool_name, 5000)  # 5s default
```

### Integration Point

**File:** `src/nodes/tool_node.py`

```python
# After tool execution (around line 200-300)
async def _execute_logic(self, state, config):
    # ... existing code ...

    # Record execution
    from ..services.auto_detection.tool_profiler import ToolProfiler

    redis_client = get_redis_client()  # Get your Redis client
    profiler = ToolProfiler(redis_client)

    tool_start = time.time()
    result = await self.mcp_call_tool(tool_name, tool_args, config)
    duration_ms = int((time.time() - tool_start) * 1000)

    # Record to Redis
    await profiler.record_execution(
        tool_name=tool_name,
        execution_time_ms=duration_ms,
        tool_args=tool_args,
        session_id=state.get("session_id", ""),
        success=True
    )
```

## Next Steps (In Order)

1. **NOW:** Implement Tool Profiler
   - Create `tool_profiler.py`
   - Add recording in `tool_node.py`
   - Test with real tool execution

2. **NEXT:** Task Complexity Calculator
   - Create `utils.py` with `calculate_task_complexity()`
   - Test with various task lists

3. **THEN:** Smart Detection Engine
   - Create `smart_detector.py`
   - Integrate with Tool Profiler
   - Add task-based logic

4. **LATER:** Loop Monitors & Orchestrator

## Questions to Answer Before Next Step

1. **Do you have Celery workers running?**
   - Yes → Keep Celery
   - No → Consider NATS

2. **Where is Redis client initialized?**
   - Need it for Tool Profiler

3. **Ready to start with Tool Profiler?**
   - I'll create the file and integration code
