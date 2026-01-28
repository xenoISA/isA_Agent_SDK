# Auto-Detection System - Comprehensive Design

## Overview

Intelligent auto-detection system inspired by Claude Code's approach, designed to handle:
1. **Auto-detection of long-running commands**
2. **Smart resource allocation**
3. **Loop detection monitoring patterns**
4. **Resource consumption tracking**
5. **Automatic pause when repetitive patterns detected**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Auto-Detection System                         │
└────────────┬────────────────────────────────────────────────────┘
             │
             ├─── 1. Tool Profiler
             │    ├─ Historical execution time tracking
             │    ├─ Argument complexity analysis
             │    ├─ 90th percentile estimation
             │    └─ Redis-based persistence
             │
             ├─── 2. Smart Detection Engine
             │    ├─ Multi-factor decision making
             │    ├─ User preference integration
             │    ├─ Dynamic threshold adjustment
             │    └─ Parallel execution detection
             │
             ├─── 3. Loop Detection Monitor
             │    ├─ Pattern recognition (same tool repeated)
             │    ├─ Frequency analysis
             │    ├─ Progress tracking
             │    └─ Automatic pause trigger
             │
             ├─── 4. Resource Tracker
             │    ├─ CPU usage monitoring
             │    ├─ Memory consumption tracking
             │    ├─ I/O bottleneck detection
             │    └─ Worker queue depth analysis
             │
             └─── 5. Decision Orchestrator
                  ├─ Aggregates all signals
                  ├─ Applies business rules
                  ├─ Determines execution strategy
                  └─ Triggers appropriate actions
```

---

## Component 1: Tool Profiler

### Purpose
Learn from historical tool executions to predict future execution times.

### Key Features

1. **Execution Time Recording**
   - Store last 100 executions per tool in Redis
   - Track execution time, argument size, success/failure
   - TTL: 7 days for historical data

2. **Time Estimation**
   - Use 90th percentile for pessimistic estimates
   - Weight by argument size similarity
   - Factor in tool complexity (CPU vs I/O bound)

3. **Batch Analysis**
   - Estimate total time for multiple tools
   - Detect if tools can be parallelized
   - Calculate sequential vs parallel time

### Data Schema

**Redis Key**: `tool_profile:{tool_name}`

**Storage**: Sorted Set (score = timestamp)

**Record Format**:
```json
{
  "time_ms": 12500,
  "args_size": 1024,
  "session_id": "session_123",
  "timestamp": 1735689600.0,
  "success": true,
  "cpu_intensive": false,
  "io_intensive": true,
  "memory_mb": 45,
  "tool_complexity": 0.7
}
```

### Default Estimates (Cold Start)

When no historical data exists:

```python
DEFAULT_ESTIMATES = {
    # Web tools
    "web_search": 3000,      # 3s
    "web_crawl": 12000,      # 12s
    "web_scrape": 8000,      # 8s

    # File tools
    "file_read": 500,        # 0.5s
    "file_write": 1000,      # 1s
    "file_process": 5000,    # 5s

    # Database tools
    "database_query": 2000,  # 2s
    "database_insert": 3000, # 3s

    # API tools
    "api_call": 4000,        # 4s
    "api_batch": 15000,      # 15s

    # MCP tools (default)
    "mcp_tool": 5000,        # 5s

    # Heavy computation
    "data_analysis": 20000,  # 20s
    "image_process": 10000,  # 10s
    "video_process": 60000,  # 60s
}
```

### Algorithm: Time Estimation

```python
def estimate_execution_time(tool_name, tool_args):
    """
    Estimate execution time based on historical data + context

    Returns: time_ms (int)
    """
    # Step 1: Get historical records
    records = redis.zrange(f"tool_profile:{tool_name}", -50, -1)

    if not records:
        return get_default_estimate(tool_name)

    # Step 2: Filter by argument size similarity
    args_size = len(str(tool_args))
    similar_records = []

    for record in records:
        size_ratio = record["args_size"] / max(args_size, 1)
        if 0.5 < size_ratio < 2.0:  # Similar complexity
            similar_records.append(record["time_ms"])

    if len(similar_records) < 3:  # Not enough data
        return get_default_estimate(tool_name)

    # Step 3: Use 90th percentile (pessimistic)
    similar_records.sort()
    p90_index = int(len(similar_records) * 0.9)
    estimated_time = similar_records[p90_index]

    # Step 4: Apply complexity factor
    complexity_factor = analyze_argument_complexity(tool_args)
    adjusted_time = estimated_time * complexity_factor

    return int(adjusted_time)
```

---

## Component 2: Smart Detection Engine

### Purpose
Decide whether to execute synchronously, prompt user, or auto-queue to background.

### Multi-Factor Decision Making

**Inputs**:
1. Estimated total execution time (from Tool Profiler)
2. **Task count and complexity** (NEW - critical for sequential processes)
3. **Execution mode overhead** (NEW - sequential vs parallel)
4. User preferences (thresholds, auto-background enabled)
5. Tool types (CPU/IO intensive)
6. Current system load
7. Historical user choices
8. Tool dependency analysis
9. **Checkpoint overhead** (NEW - from durable_service.py)

**Key Insight**: We cannot rely only on tool execution time. A task may need 20 sequential sub-tasks with short tool calls, which still equals a long execution overall.

**Decision Tree**:

```
                    Estimated Time?
                          │
          ┌───────────────┼───────────────┐
          │               │               │
        < 15s         15-45s           > 45s
          │               │               │
       Execute      ┌─────┴─────┐    Auto-Background?
       Sync         │           │         │
                 User Pref?  System Load?  ├─ Yes → Queue
                    │           │          └─ No → Prompt User
              ┌─────┴─────┐     │
              │           │     │
           Default     Auto   High Load?
           Prompt    Enabled     │
                       │      ┌──┴──┐
                       │      │     │
                      Queue  Defer Wait
```

### User Preferences Schema

**Storage**: User profile in database

```python
{
    "user_id": "user_123",
    "background_job_preferences": {
        # Thresholds
        "auto_background_threshold_seconds": 45,   # >45s = auto-queue
        "prompt_threshold_seconds": 15,            # 15-45s = prompt

        # Auto-mode settings
        "auto_background_enabled": true,           # Enable auto-queue
        "remember_choices": true,                  # Learn from user choices

        # Execution preferences
        "parallel_execution_preferred": false,     # Prefer parallel when possible
        "default_choice": "background",            # Default: background/wait/quick

        # Notification preferences
        "notify_on_completion": true,
        "notification_channels": ["email", "push"]
    }
}
```

### Smart Detection Algorithm (UPDATED)

```python
async def detect_long_running_task(tool_info_list, user_id, state, config):
    """
    Smart detection using multiple signals - UPDATED with task-based detection

    Critical: Don't only rely on tool execution time!
    Example: 20 tasks with short tools = long execution

    Returns: {
        "decision": "execute_sync" | "prompt_user" | "auto_background",
        "reason": str,
        "estimated_time_seconds": float,
        "task_count": int,
        "task_complexity_score": float,
        "confidence": float
    }
    """
    # Step 1: Get time estimates from Tool Profiler
    profiler = ToolProfiler(redis_client)
    estimate = await profiler.get_batch_estimate(tool_info_list)

    tool_execution_time = estimate["total_time_seconds"]
    can_parallelize = estimate["can_parallelize"]
    parallelized_time = estimate.get("parallelized_time_seconds", tool_execution_time)

    # Step 2: CRITICAL - Get task list from state (from agent_executor_node.py)
    task_list = state.get("task_list", [])
    task_count = len(task_list)

    # Step 3: Calculate task complexity (NEW)
    task_complexity = calculate_task_complexity(task_list)

    # Step 4: Determine execution mode overhead (NEW)
    execution_mode = state.get("execution_mode", "sequential")
    if execution_mode == "sequential":
        # Sequential adds context-switching overhead between tasks
        context_switching_overhead = task_count * 2.0  # 2s per task
    else:
        # Parallel has less overhead
        max_parallel = config.get("max_parallel_tasks", 3)
        context_switching_overhead = (task_count / max_parallel) * 1.5

    # Step 5: Calculate checkpoint overhead (from durable_service.py)
    # Checkpoints occur every N tasks (see agent_executor_node.py:759)
    checkpoint_frequency = config.get("checkpoint_frequency", 5)
    num_checkpoints = task_count // checkpoint_frequency
    checkpoint_overhead = num_checkpoints * 10.0  # ~10s per checkpoint

    # Step 6: Calculate total estimated time (UPDATED FORMULA)
    total_estimated_time = (
        tool_execution_time +
        context_switching_overhead +
        checkpoint_overhead +
        (task_complexity * 5.0)  # Complexity factor adds 5s per complexity point
    )

    # Step 7: Get user preferences
    user_prefs = await get_user_background_preferences(user_id)
    auto_threshold = user_prefs.get("auto_background_threshold_seconds", 45)
    prompt_threshold = user_prefs.get("prompt_threshold_seconds", 15)
    auto_enabled = user_prefs.get("auto_background_enabled", False)

    # Step 8: Get system load
    system_load = await get_system_load()
    worker_queue_depth = await get_worker_queue_depth()

    # Step 9: Decision logic with confidence scoring
    decision_factors = []

    # Factor 1: Time-based decision (using TOTAL time, not just tool time)
    if total_estimated_time < prompt_threshold:
        decision_factors.append(("time", "execute_sync", 0.9))
    elif total_estimated_time >= auto_threshold:
        if auto_enabled:
            decision_factors.append(("time", "auto_background", 0.85))
        else:
            decision_factors.append(("time", "prompt_user", 0.7))
    else:
        decision_factors.append(("time", "prompt_user", 0.8))

    # Factor 2: Task count (NEW - can override time)
    if task_count > 15:  # Many tasks
        decision_factors.append(("task_count", "prompt_user", 0.85))
        if task_count > 25:  # Very many tasks
            decision_factors.append(("task_count", "auto_background", 0.9))

    # Factor 3: Task complexity (NEW)
    if task_complexity > 0.7:  # High complexity
        decision_factors.append(("task_complexity", "prompt_user", 0.8))

    # Factor 4: Execution mode (NEW)
    if execution_mode == "sequential" and task_count > 10:
        decision_factors.append(("execution_mode", "auto_background", 0.75))

    # Factor 5: System load (adjust if high load)
    if system_load > 0.8:  # 80% CPU usage
        if total_estimated_time > 30:
            decision_factors.append(("system_load", "auto_background", 0.75))
        else:
            decision_factors.append(("system_load", "defer", 0.6))

    # Factor 6: Worker queue depth (don't overload)
    if worker_queue_depth > 10:  # Many jobs queued
        if total_estimated_time < 20:
            decision_factors.append(("queue_depth", "execute_sync", 0.7))

    # Factor 7: Historical user choices (learning)
    if user_prefs.get("remember_choices", False):
        historical_choice = await get_historical_choice(user_id, tool_info_list)
        if historical_choice:
            decision_factors.append(("history", historical_choice, 0.65))

    # Step 10: Aggregate decisions using weighted voting
    decision_scores = {}
    for factor_name, decision, confidence in decision_factors:
        if decision not in decision_scores:
            decision_scores[decision] = []
        decision_scores[decision].append(confidence)

    # Calculate average confidence per decision
    final_scores = {
        decision: sum(confidences) / len(confidences)
        for decision, confidences in decision_scores.items()
    }

    # Select decision with highest confidence
    final_decision = max(final_scores.items(), key=lambda x: x[1])

    return {
        "decision": final_decision[0],
        "confidence": final_decision[1],
        "estimated_time_seconds": total_estimated_time,
        "tool_execution_time_seconds": tool_execution_time,  # Separate
        "task_count": task_count,
        "task_complexity_score": task_complexity,
        "execution_mode": execution_mode,
        "context_switching_overhead": context_switching_overhead,
        "checkpoint_overhead": checkpoint_overhead,
        "parallelized_time_seconds": parallelized_time,
        "can_parallelize": can_parallelize,
        "tool_estimates": estimate["tool_estimates"],
        "decision_factors": decision_factors,
        "reason": generate_reason(decision_factors, final_decision)
    }


def calculate_task_complexity(task_list: List[dict]) -> float:
    """
    Calculate task complexity score (0.0-1.0)

    Factors:
    - Number of tasks
    - Tool diversity (many different tools = complex)
    - Task dependencies
    - Task priority distribution
    - Estimated duration variance

    Returns: float (0.0 = simple, 1.0 = very complex)
    """
    if not task_list:
        return 0.0

    # Factor 1: Task count (normalized to 0-1)
    task_count_score = min(len(task_list) / 30.0, 1.0)  # 30+ tasks = max

    # Factor 2: Tool diversity
    tools_used = set()
    for task in task_list:
        tool = task.get("tool_name", "unknown")
        tools_used.add(tool)
    tool_diversity_score = min(len(tools_used) / 10.0, 1.0)  # 10+ tools = max

    # Factor 3: Task dependencies (if tasks reference each other)
    dependency_score = 0.0
    for task in task_list:
        if task.get("dependencies") or task.get("depends_on"):
            dependency_score += 0.1
    dependency_score = min(dependency_score, 1.0)

    # Factor 4: Priority distribution (varied priorities = complex)
    priorities = [task.get("priority", 0) for task in task_list]
    if priorities:
        priority_variance = (max(priorities) - min(priorities)) / max(max(priorities), 1)
        priority_score = min(priority_variance, 1.0)
    else:
        priority_score = 0.0

    # Factor 5: Estimated duration variance (some long, some short = complex)
    durations = []
    for task in task_list:
        est = task.get("estimated_duration", 0) or task.get("duration", 0)
        if est:
            durations.append(est)

    if len(durations) > 1:
        avg_duration = sum(durations) / len(durations)
        duration_variance = sum(abs(d - avg_duration) for d in durations) / (avg_duration * len(durations))
        duration_score = min(duration_variance, 1.0)
    else:
        duration_score = 0.0

    # Weighted average
    complexity = (
        task_count_score * 0.3 +
        tool_diversity_score * 0.25 +
        dependency_score * 0.2 +
        priority_score * 0.15 +
        duration_score * 0.1
    )

    return round(complexity, 2)
```

---

## Component 3: Loop Detection Monitor

### Purpose
Detect infinite loops and repetitive patterns that indicate stuck execution.

**Critical Update**: Analysis shows we currently have NO loop detection beyond basic recursion limit (max_graph_iterations=50). We need TWO types of loop detection:

1. **Tool-Level Loop Detection** - Same tool called repeatedly
2. **Graph-Level Loop Detection** (NEW) - Graph nodes cycling infinitely

### Detection Strategies

#### A. Tool-Level Loop Detection

1. **Exact Tool Repetition**
   - Same tool called N times consecutively
   - Threshold: 5 consecutive calls

2. **Cyclic Pattern Detection**
   - Tool sequence repeats (e.g., A→B→C→A→B→C)
   - Threshold: 3 cycle repetitions

3. **No Progress Detection**
   - Same tool with identical arguments
   - Threshold: 3 identical calls

4. **Time-based Stall**
   - Tool takes significantly longer than estimated
   - Threshold: 3x estimated time

#### B. Graph-Level Loop Detection (NEW)

**Problem Found**: In `smart_agent_graph.py`, we have potential infinite loops:
- `reason_model` → `call_tool` → `reason_model` (can loop forever)
- Only protection: `max_graph_iterations = 50` (basic counter)

**New Detection Needed**:

1. **Node Transition Pattern**
   - Track node-to-node transitions
   - Detect: reason_model → call_tool → reason_model (repeated 5+ times)

2. **State Convergence Detection**
   - Check if state is changing between iterations
   - If state identical for 3+ iterations → stuck

3. **Progress Indicator Tracking**
   - Monitor: message count, tool results, decisions made
   - If no new information for 5+ iterations → no progress

4. **Time-per-Iteration Monitoring**
   - Track average iteration time
   - If iteration time drops to <100ms → likely stuck in loop

### Data Structure

**In-Memory State** (per session):

```python
{
    "session_id": "session_123",

    # Tool-level tracking
    "tool_execution_history": [
        {"tool": "web_search", "args_hash": "abc123", "timestamp": 1735689600.0},
        {"tool": "web_search", "args_hash": "abc123", "timestamp": 1735689602.0},
        {"tool": "web_crawl", "args_hash": "def456", "timestamp": 1735689615.0},
        # ... last 50 tool calls
    ],

    # Graph-level tracking (NEW)
    "graph_execution_history": [
        {"from_node": "reason_model", "to_node": "call_tool", "timestamp": 1735689600.0, "state_hash": "xyz789"},
        {"from_node": "call_tool", "to_node": "reason_model", "timestamp": 1735689602.0, "state_hash": "xyz789"},
        {"from_node": "reason_model", "to_node": "call_tool", "timestamp": 1735689604.0, "state_hash": "xyz789"},
        # ... last 100 transitions
    ],

    # Progress indicators (NEW)
    "progress_indicators": {
        "last_message_count": 15,
        "last_tool_result_count": 8,
        "last_decision_count": 3,
        "iterations_without_progress": 0
    },

    "loop_warnings": 0,
    "auto_paused": false
}
```

### Loop Detection Algorithm

```python
class LoopDetectionMonitor:
    """Detect and prevent infinite loops"""

    def __init__(self, max_history=50):
        self.max_history = max_history
        self.session_states = {}  # session_id -> state

    async def check_for_loops(self, session_id, tool_name, tool_args):
        """
        Check if current tool execution indicates a loop

        Returns: {
            "loop_detected": bool,
            "loop_type": "exact_repetition" | "cyclic_pattern" | "no_progress" | "time_stall",
            "severity": "warning" | "critical",
            "recommendation": "pause" | "continue" | "abort"
        }
        """
        # Get or create session state
        state = self._get_session_state(session_id)

        # Add current tool to history
        args_hash = hashlib.md5(str(tool_args).encode()).hexdigest()[:8]
        state["tool_execution_history"].append({
            "tool": tool_name,
            "args_hash": args_hash,
            "timestamp": time.time()
        })

        # Keep only recent history
        if len(state["tool_execution_history"]) > self.max_history:
            state["tool_execution_history"] = state["tool_execution_history"][-self.max_history:]

        # Run detection checks
        checks = [
            self._check_exact_repetition(state),
            self._check_cyclic_pattern(state),
            self._check_no_progress(state),
            self._check_time_stall(state)
        ]

        # Return most severe detection
        for check_result in checks:
            if check_result["loop_detected"]:
                return check_result

        return {"loop_detected": False}

    def _check_exact_repetition(self, state):
        """Check if same tool called N times consecutively"""
        history = state["tool_execution_history"]

        if len(history) < 5:
            return {"loop_detected": False}

        # Check last 5 calls
        last_5 = history[-5:]
        tools = [call["tool"] for call in last_5]

        if len(set(tools)) == 1:  # All same tool
            return {
                "loop_detected": True,
                "loop_type": "exact_repetition",
                "severity": "warning",
                "recommendation": "pause",
                "details": f"Tool '{tools[0]}' called 5 times consecutively"
            }

        return {"loop_detected": False}

    def _check_cyclic_pattern(self, state):
        """Check if tool sequence repeats"""
        history = state["tool_execution_history"]

        if len(history) < 9:  # Need at least 3 cycles of 3-tool pattern
            return {"loop_detected": False}

        # Extract tool names
        tools = [call["tool"] for call in history[-12:]]  # Last 12 calls

        # Try to detect 3-tool cycle repeated 3 times
        for cycle_length in [2, 3, 4]:
            if len(tools) >= cycle_length * 3:
                cycle = tools[:cycle_length]

                # Check if pattern repeats 3 times
                is_cycle = True
                for i in range(3):
                    start = i * cycle_length
                    end = start + cycle_length
                    if tools[start:end] != cycle:
                        is_cycle = False
                        break

                if is_cycle:
                    return {
                        "loop_detected": True,
                        "loop_type": "cyclic_pattern",
                        "severity": "critical",
                        "recommendation": "abort",
                        "details": f"Cyclic pattern detected: {' → '.join(cycle)} repeated 3 times"
                    }

        return {"loop_detected": False}

    def _check_no_progress(self, state):
        """Check if same tool with identical arguments"""
        history = state["tool_execution_history"]

        if len(history) < 3:
            return {"loop_detected": False}

        # Check last 3 calls
        last_3 = history[-3:]

        # Check if all have same tool and args_hash
        if (len(set(call["tool"] for call in last_3)) == 1 and
            len(set(call["args_hash"] for call in last_3)) == 1):

            return {
                "loop_detected": True,
                "loop_type": "no_progress",
                "severity": "critical",
                "recommendation": "abort",
                "details": f"Tool '{last_3[0]['tool']}' called 3 times with identical arguments"
            }

        return {"loop_detected": False}

    def _check_time_stall(self, state):
        """Check if tool takes much longer than estimated"""
        history = state["tool_execution_history"]

        if len(history) < 2:
            return {"loop_detected": False}

        current_tool = history[-1]
        previous_tool = history[-2]

        # Check time between last 2 tool calls
        time_diff = current_tool["timestamp"] - previous_tool["timestamp"]

        # Get estimated time for previous tool
        estimated_time = get_estimated_time(previous_tool["tool"])

        # If actual time is 3x estimated, flag as stall
        if time_diff > estimated_time * 3:
            return {
                "loop_detected": True,
                "loop_type": "time_stall",
                "severity": "warning",
                "recommendation": "pause",
                "details": f"Tool '{previous_tool['tool']}' took {time_diff:.1f}s (3x estimated {estimated_time:.1f}s)"
            }

        return {"loop_detected": False}


class GraphLoopMonitor:
    """
    NEW: Detect graph-level loops in smart_agent_graph.py

    Problem: reason_model → call_tool → reason_model can loop infinitely
    Current protection: Only max_graph_iterations counter
    """

    def __init__(self, max_history=100):
        self.max_history = max_history
        self.session_states = {}  # session_id -> state

    async def check_graph_loop(self, session_id, current_node, next_node, state):
        """
        Check if graph is in an infinite loop

        Returns: {
            "loop_detected": bool,
            "loop_type": "node_transition_pattern" | "state_convergence" | "no_progress" | "rapid_iteration",
            "severity": "warning" | "critical",
            "recommendation": "pause" | "abort"
        }
        """
        # Get or create session state
        graph_state = self._get_graph_state(session_id)

        # Calculate state hash (to detect if state is changing)
        state_hash = self._hash_state(state)

        # Add current transition to history
        graph_state["graph_execution_history"].append({
            "from_node": current_node,
            "to_node": next_node,
            "timestamp": time.time(),
            "state_hash": state_hash
        })

        # Keep only recent history
        if len(graph_state["graph_execution_history"]) > self.max_history:
            graph_state["graph_execution_history"] = graph_state["graph_execution_history"][-self.max_history:]

        # Run detection checks
        checks = [
            self._check_node_transition_pattern(graph_state),
            self._check_state_convergence(graph_state),
            self._check_progress_indicators(graph_state, state),
            self._check_rapid_iteration(graph_state)
        ]

        # Return most severe detection
        for check_result in checks:
            if check_result["loop_detected"]:
                return check_result

        return {"loop_detected": False}

    def _check_node_transition_pattern(self, graph_state):
        """
        Detect: reason_model → call_tool → reason_model (repeated 5+ times)
        """
        history = graph_state["graph_execution_history"]

        if len(history) < 10:  # Need at least 5 cycles (2 transitions per cycle)
            return {"loop_detected": False}

        # Check last 10 transitions for pattern
        last_10 = history[-10:]

        # Pattern: A→B→A→B→A→B→A→B→A→B
        pattern_detected = True
        for i in range(0, len(last_10) - 1, 2):
            if i + 1 >= len(last_10):
                break

            # Check if A→B pattern repeats
            if i == 0:
                node_a = last_10[i]["from_node"]
                node_b = last_10[i]["to_node"]
            else:
                if last_10[i]["from_node"] != node_a or last_10[i]["to_node"] != node_b:
                    pattern_detected = False
                    break

        if pattern_detected and len(last_10) >= 10:
            return {
                "loop_detected": True,
                "loop_type": "node_transition_pattern",
                "severity": "critical",
                "recommendation": "abort",
                "details": f"Graph stuck in loop: {node_a} → {node_b} (repeated 5+ times)"
            }

        return {"loop_detected": False}

    def _check_state_convergence(self, graph_state):
        """
        Check if state hash is identical for 3+ iterations (state not changing)
        """
        history = graph_state["graph_execution_history"]

        if len(history) < 6:  # Need 3 iterations (2 transitions per iteration)
            return {"loop_detected": False}

        # Get last 6 state hashes
        last_6_hashes = [h["state_hash"] for h in history[-6:]]

        # Check if all identical
        if len(set(last_6_hashes)) == 1:
            return {
                "loop_detected": True,
                "loop_type": "state_convergence",
                "severity": "critical",
                "recommendation": "abort",
                "details": "State unchanged for 3+ iterations - graph is stuck"
            }

        return {"loop_detected": False}

    def _check_progress_indicators(self, graph_state, current_state):
        """
        Check if graph is making progress (new messages, tools, decisions)
        """
        progress = graph_state["progress_indicators"]

        # Count current progress indicators
        current_message_count = len(current_state.get("messages", []))
        current_tool_count = len(current_state.get("tool_results", []))
        current_decision_count = len(current_state.get("decisions", []))

        # Check if any increased
        progress_made = (
            current_message_count > progress["last_message_count"] or
            current_tool_count > progress["last_tool_result_count"] or
            current_decision_count > progress["last_decision_count"]
        )

        if not progress_made:
            # No progress
            progress["iterations_without_progress"] += 1

            if progress["iterations_without_progress"] >= 5:
                return {
                    "loop_detected": True,
                    "loop_type": "no_progress",
                    "severity": "warning",
                    "recommendation": "pause",
                    "details": "No progress for 5+ iterations - possible stuck state"
                }
        else:
            # Progress made, reset counter
            progress["iterations_without_progress"] = 0
            progress["last_message_count"] = current_message_count
            progress["last_tool_result_count"] = current_tool_count
            progress["last_decision_count"] = current_decision_count

        return {"loop_detected": False}

    def _check_rapid_iteration(self, graph_state):
        """
        Check if iterations are happening too fast (<100ms) - indicates stuck loop
        """
        history = graph_state["graph_execution_history"]

        if len(history) < 5:
            return {"loop_detected": False}

        # Check last 5 transitions
        last_5 = history[-5:]
        times = [t["timestamp"] for t in last_5]

        # Calculate average time between iterations
        time_diffs = [times[i+1] - times[i] for i in range(len(times)-1)]
        avg_time = sum(time_diffs) / len(time_diffs)

        # If average < 100ms, likely stuck in fast loop
        if avg_time < 0.1:  # 100ms
            return {
                "loop_detected": True,
                "loop_type": "rapid_iteration",
                "severity": "critical",
                "recommendation": "abort",
                "details": f"Rapid iterations detected (avg {avg_time*1000:.0f}ms) - graph likely stuck"
            }

        return {"loop_detected": False}

    def _hash_state(self, state):
        """Calculate hash of critical state fields to detect changes"""
        import hashlib
        import json

        # Extract critical fields
        critical_fields = {
            "messages": len(state.get("messages", [])),
            "tool_results": len(state.get("tool_results", [])),
            "task_list": len(state.get("task_list", [])),
            "next_action": state.get("next_action", ""),
        }

        # Hash it
        state_str = json.dumps(critical_fields, sort_keys=True)
        return hashlib.md5(state_str.encode()).hexdigest()[:8]

    def _get_graph_state(self, session_id):
        """Get or initialize graph state for session"""
        if session_id not in self.session_states:
            self.session_states[session_id] = {
                "graph_execution_history": [],
                "progress_indicators": {
                    "last_message_count": 0,
                    "last_tool_result_count": 0,
                    "last_decision_count": 0,
                    "iterations_without_progress": 0
                }
            }
        return self.session_states[session_id]
```

---

## Component 4: Resource Tracker

### Purpose
Monitor system resource consumption to inform execution decisions.

### Metrics Tracked

1. **CPU Usage** (per process + system-wide)
2. **Memory Consumption** (RSS, VMS)
3. **I/O Operations** (read/write bytes, operations)
4. **Network Usage** (for API-heavy tools)
5. **Worker Queue Depth** (Celery queue size)

### Implementation

```python
class ResourceTracker:
    """Track system resource consumption"""

    def __init__(self):
        self.metrics_history = []  # Last 100 measurements
        self.alert_thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "io_wait_percent": 70.0,
            "worker_queue_depth": 15
        }

    async def get_current_metrics(self):
        """
        Get current resource metrics

        Returns: {
            "cpu_percent": 45.2,
            "memory_percent": 62.1,
            "memory_mb": 1024,
            "io_wait_percent": 10.5,
            "network_mbps": 12.3,
            "worker_queue_depth": 3,
            "timestamp": 1735689600.0
        }
        """
        import psutil

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_mb = memory.used / (1024 * 1024)

        # I/O stats
        io_counters = psutil.disk_io_counters()
        io_wait_percent = psutil.cpu_times_percent().iowait if hasattr(psutil.cpu_times_percent(), 'iowait') else 0

        # Network stats
        net_io = psutil.net_io_counters()
        network_mbps = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024)

        # Worker queue depth (Celery)
        worker_queue_depth = await self._get_celery_queue_depth()

        metrics = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "memory_mb": memory_mb,
            "io_wait_percent": io_wait_percent,
            "network_mbps": network_mbps,
            "worker_queue_depth": worker_queue_depth,
            "timestamp": time.time()
        }

        # Store in history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]

        return metrics

    async def _get_celery_queue_depth(self):
        """Get Celery worker queue depth"""
        try:
            from ..background_job_service import celery_app

            # Get queue length
            inspect = celery_app.control.inspect()
            active = inspect.active()
            reserved = inspect.reserved()

            if active is None or reserved is None:
                return 0

            # Count total tasks
            total = 0
            for worker_tasks in active.values():
                total += len(worker_tasks)
            for worker_tasks in reserved.values():
                total += len(worker_tasks)

            return total

        except Exception as e:
            logger.warning(f"Failed to get Celery queue depth: {e}")
            return 0

    def check_resource_alerts(self, metrics):
        """
        Check if any resource exceeds alert thresholds

        Returns: {
            "alerts": [
                {"metric": "cpu_percent", "value": 85.2, "threshold": 80.0, "severity": "warning"},
                ...
            ],
            "recommendation": "defer" | "continue" | "scale_workers"
        }
        """
        alerts = []

        for metric, threshold in self.alert_thresholds.items():
            value = metrics.get(metric, 0)

            if value > threshold:
                # Determine severity
                if value > threshold * 1.2:  # 20% over threshold
                    severity = "critical"
                else:
                    severity = "warning"

                alerts.append({
                    "metric": metric,
                    "value": value,
                    "threshold": threshold,
                    "severity": severity
                })

        # Determine recommendation
        if any(alert["severity"] == "critical" for alert in alerts):
            recommendation = "defer"
        elif len(alerts) >= 2:
            recommendation = "scale_workers"
        else:
            recommendation = "continue"

        return {
            "alerts": alerts,
            "recommendation": recommendation
        }

    def analyze_tool_resource_profile(self, tool_name):
        """
        Analyze historical resource usage for tool

        Returns: {
            "cpu_intensive": bool,
            "memory_intensive": bool,
            "io_intensive": bool,
            "avg_cpu_percent": float,
            "avg_memory_mb": float
        }
        """
        # Get historical executions from profiler
        records = get_tool_profile_records(tool_name)

        if not records:
            return {
                "cpu_intensive": False,
                "memory_intensive": False,
                "io_intensive": False
            }

        # Calculate averages
        avg_cpu = sum(r.get("cpu_percent", 0) for r in records) / len(records)
        avg_memory = sum(r.get("memory_mb", 0) for r in records) / len(records)
        avg_io_wait = sum(r.get("io_wait_percent", 0) for r in records) / len(records)

        return {
            "cpu_intensive": avg_cpu > 60.0,
            "memory_intensive": avg_memory > 500,  # >500MB
            "io_intensive": avg_io_wait > 20.0,
            "avg_cpu_percent": avg_cpu,
            "avg_memory_mb": avg_memory,
            "avg_io_wait_percent": avg_io_wait
        }
```

---

## Component 5: Decision Orchestrator

### Purpose
Aggregate all signals and make final execution decision.

### Orchestration Flow

```python
class DecisionOrchestrator:
    """Orchestrate all auto-detection components"""

    def __init__(self, redis_client, mcp_service):
        self.tool_profiler = ToolProfiler(redis_client)
        self.smart_detector = SmartDetectionEngine(redis_client)
        self.loop_monitor = LoopDetectionMonitor()  # Tool-level loops
        self.graph_loop_monitor = GraphLoopMonitor()  # Graph-level loops (NEW)
        self.resource_tracker = ResourceTracker()

    async def make_execution_decision(
        self,
        tool_info_list: List[tuple],
        user_id: str,
        session_id: str,
        context: dict
    ) -> dict:
        """
        Aggregate all signals and make final decision

        Returns: {
            "decision": "execute_sync" | "prompt_user" | "auto_background" | "pause" | "abort",
            "reason": str,
            "estimated_time_seconds": float,
            "confidence": float,
            "signals": {
                "smart_detection": {...},
                "loop_detection": {...},
                "resource_check": {...}
            },
            "actions": [...]
        }
        """
        signals = {}

        # Signal 1: Smart detection (time + task-based, UPDATED)
        detection_result = await self.smart_detector.detect_long_running_task(
            tool_info_list, user_id, context.get("state"), context.get("config")
        )
        signals["smart_detection"] = detection_result

        # Signal 2: Tool-level loop detection
        for tool_name, tool_args, _ in tool_info_list:
            loop_result = await self.loop_monitor.check_for_loops(
                session_id, tool_name, tool_args
            )
            if loop_result["loop_detected"]:
                signals["tool_loop_detection"] = loop_result
                break

        # Signal 3: Graph-level loop detection (NEW)
        current_node = context.get("current_node")
        next_node = context.get("next_node")
        if current_node and next_node:
            graph_loop_result = await self.graph_loop_monitor.check_graph_loop(
                session_id, current_node, next_node, context.get("state")
            )
            if graph_loop_result["loop_detected"]:
                signals["graph_loop_detection"] = graph_loop_result

        # Signal 4: Resource check
        current_metrics = await self.resource_tracker.get_current_metrics()
        resource_alert = self.resource_tracker.check_resource_alerts(current_metrics)
        signals["resource_check"] = {
            "metrics": current_metrics,
            "alerts": resource_alert
        }

        # Aggregate decisions with priority
        final_decision = self._aggregate_decisions(signals)

        return final_decision

    def _aggregate_decisions(self, signals):
        """
        Aggregate signals using priority rules (UPDATED)

        Priority:
        1. Graph loop detection (highest) - graph-level stuck states
        2. Tool loop detection (high) - tool-level loops
        3. Resource alerts (medium-high) - defer execution
        4. Smart detection (normal) - time + task-based decision
        """
        # Rule 1: Graph loop detection takes precedence (NEW)
        if "graph_loop_detection" in signals and signals["graph_loop_detection"]["loop_detected"]:
            graph_loop_info = signals["graph_loop_detection"]

            if graph_loop_info["recommendation"] == "abort":
                return {
                    "decision": "abort",
                    "reason": f"Graph loop detected: {graph_loop_info['details']}",
                    "confidence": 0.98,
                    "signals": signals,
                    "actions": ["abort_execution", "notify_user", "log_graph_state"]
                }
            elif graph_loop_info["recommendation"] == "pause":
                return {
                    "decision": "pause",
                    "reason": f"Potential graph loop: {graph_loop_info['details']}",
                    "confidence": 0.90,
                    "signals": signals,
                    "actions": ["pause_execution", "request_human_intervention"]
                }

        # Rule 2: Tool loop detection
        if "tool_loop_detection" in signals and signals["tool_loop_detection"]["loop_detected"]:
            tool_loop_info = signals["tool_loop_detection"]

            if tool_loop_info["recommendation"] == "abort":
                return {
                    "decision": "abort",
                    "reason": f"Tool loop detected: {tool_loop_info['details']}",
                    "confidence": 0.95,
                    "signals": signals,
                    "actions": ["abort_execution", "notify_user"]
                }
            elif tool_loop_info["recommendation"] == "pause":
                return {
                    "decision": "pause",
                    "reason": f"Potential tool loop: {tool_loop_info['details']}",
                    "confidence": 0.85,
                    "signals": signals,
                    "actions": ["pause_execution", "request_human_intervention"]
                }

        # Rule 3: Resource alerts
        resource_check = signals.get("resource_check", {})
        if resource_check.get("alerts", {}).get("recommendation") == "defer":
            return {
                "decision": "defer",
                "reason": "System resources critically high",
                "confidence": 0.8,
                "signals": signals,
                "actions": ["queue_for_later", "notify_when_resources_available"]
            }

        # Rule 4: Use smart detection result (time + task-based)
        smart_detection = signals.get("smart_detection", {})
        return {
            "decision": smart_detection.get("decision", "execute_sync"),
            "reason": smart_detection.get("reason", "Default execution"),
            "confidence": smart_detection.get("confidence", 0.7),
            "estimated_time_seconds": smart_detection.get("estimated_time_seconds", 0),
            "signals": signals,
            "actions": self._determine_actions(smart_detection.get("decision"))
        }

    def _determine_actions(self, decision):
        """Determine actions based on decision"""
        action_map = {
            "execute_sync": ["execute_immediately", "track_execution"],
            "prompt_user": ["show_execution_choice_dialog", "wait_for_user_input"],
            "auto_background": ["queue_to_celery", "return_job_id", "notify_on_completion"],
            "pause": ["pause_execution", "request_human_intervention"],
            "abort": ["abort_execution", "notify_user"],
            "defer": ["queue_for_later", "retry_when_resources_available"]
        }
        return action_map.get(decision, ["execute_immediately"])
```

---

## Integration Points

### 1. Integration with `tool_node.py`

**Location**: `src/nodes/tool_node.py:_execute_logic`

**Before Tool Execution**:

```python
# In ToolNode._execute_logic, before executing tools

from ..services.auto_detection import DecisionOrchestrator

# Initialize orchestrator
orchestrator = DecisionOrchestrator(redis_client, mcp_service)

# Make execution decision
decision = await orchestrator.make_execution_decision(
    tool_info_list=tool_info_list,
    user_id=self.get_user_id(config),
    session_id=state.get("session_id"),
    context=self.get_runtime_context(config)
)

# Act on decision
if decision["decision"] == "abort":
    # Abort execution
    return {
        "messages": [SystemMessage(content=f"Execution aborted: {decision['reason']}")],
        "next_action": "end"
    }

elif decision["decision"] == "pause":
    # Pause and request human intervention
    human_response = hil_service.ask_human_with_interrupt(
        question=f"Execution paused: {decision['reason']}\n\nContinue?",
        context=json.dumps(decision["signals"], indent=2),
        node_source="tool_node"
    )
    # ... handle response

elif decision["decision"] == "auto_background":
    # Queue to background
    job_result = await self._queue_background_job(tool_info_list, state, config)
    # ... return job_id

# ... continue with normal execution
```

**After Tool Execution** (Recording):

```python
# After each tool execution
tool_start = time.time()
result = await self.mcp_call_tool(tool_name, tool_args, config)
duration_ms = int((time.time() - tool_start) * 1000)

# Record execution for profiling
await orchestrator.tool_profiler.record_execution(
    tool_name=tool_name,
    execution_time_ms=duration_ms,
    tool_args=tool_args,
    session_id=session_id,
    success=True,
    resource_metrics=await orchestrator.resource_tracker.get_current_metrics()
)
```

---

## File Structure

```
src/services/auto_detection/
├── __init__.py                  # Package initialization
├── DESIGN.md                    # This file
├── tool_profiler.py             # Component 1: Historical profiling
├── smart_detector.py            # Component 2: Smart detection engine
├── loop_monitor.py              # Component 3: Loop detection
├── resource_tracker.py          # Component 4: Resource tracking
├── orchestrator.py              # Component 5: Decision orchestrator
├── schemas.py                   # Data schemas and types
├── utils.py                     # Helper functions
└── tests/
    ├── test_tool_profiler.py
    ├── test_smart_detector.py
    ├── test_loop_monitor.py
    ├── test_resource_tracker.py
    └── test_orchestrator.py
```

---

## Testing Strategy

### Unit Tests

1. **Tool Profiler**
   - Test recording with various argument sizes
   - Test estimation accuracy (synthetic data)
   - Test default estimates
   - Test Redis persistence

2. **Smart Detector**
   - Test decision logic for various time ranges
   - Test user preference integration
   - Test confidence scoring
   - Test learning from history

3. **Loop Monitor**
   - Test exact repetition detection
   - Test cyclic pattern detection
   - Test no-progress detection
   - Test time stall detection

4. **Resource Tracker**
   - Test metric collection
   - Test alert thresholds
   - Test Celery queue depth
   - Test resource profiling

5. **Orchestrator**
   - Test signal aggregation
   - Test priority rules
   - Test action determination
   - End-to-end decision flow

### Integration Tests

1. Simulate long-running task scenario
2. Simulate infinite loop scenario
3. Simulate high system load scenario
4. Test with real MCP tools

---

## Monitoring & Observability

### Metrics to Track

```python
METRICS = {
    # Detection accuracy
    "detection_accuracy": "% of correct long-running predictions",
    "false_positive_rate": "% of false positive detections",
    "false_negative_rate": "% of missed long-running tasks",

    # Loop detection
    "loops_detected": "Count of loops detected",
    "loops_prevented": "Count of executions aborted due to loops",

    # Resource tracking
    "avg_cpu_usage": "Average CPU usage during executions",
    "avg_memory_usage": "Average memory usage",
    "peak_resource_usage": "Peak resource consumption",

    # User behavior
    "user_override_rate": "% of times user overrides recommendation",
    "background_job_acceptance_rate": "% of auto-background accepted",

    # System performance
    "avg_estimation_error": "Average % error in time estimation",
    "profiler_cache_hit_rate": "% of tools with historical data"
}
```

### Logging

```python
# Structured logging format
logger.info(
    "auto_detection_decision | "
    f"session_id={session_id} | "
    f"decision={decision['decision']} | "
    f"confidence={decision['confidence']:.2f} | "
    f"estimated_time={decision.get('estimated_time_seconds', 0):.1f}s | "
    f"loop_detected={bool(decision['signals'].get('loop_detection'))} | "
    f"resource_alerts={len(decision['signals'].get('resource_check', {}).get('alerts', []))} | "
    f"tool_count={len(tool_info_list)}"
)
```

---

---

## CRITICAL UPDATES (Based on Code Analysis)

### Update 1: Task-Based Detection (NOT Just Tool Time)

**Problem Identified**: Original design only considered tool execution time.

**User Insight**: "We can't only rely on the tools execution time right, some task may need sequential process, for instance it may need 20 tasks (long task) but short tools call"

**Solution Implemented**:
- Added `task_count` factor (>15 tasks = prompt, >25 tasks = auto-background)
- Added `task_complexity_score` calculation (0.0-1.0 based on task count, tool diversity, dependencies, priorities, duration variance)
- Added `execution_mode_overhead` (sequential vs parallel)
- Added `checkpoint_overhead` (from durable_service.py checkpointing)
- **New Formula**: `total_time = tool_time + context_switching + checkpoints + complexity_factor`

### Update 2: Graph-Level Loop Detection (Currently Missing)

**Problem Identified**: Analysis of `smart_agent_graph.py` shows:
- Potential infinite loop: `reason_model` → `call_tool` → `reason_model`
- Only protection: `max_graph_iterations = 50` (basic counter)
- No actual loop detection logic

**Solution Implemented**:
- Created new `GraphLoopMonitor` class (separate from tool-level loops)
- Detects 4 types of graph loops:
  1. **Node transition patterns** (A→B→A→B repeated 5+ times)
  2. **State convergence** (state hash identical for 3+ iterations)
  3. **No progress** (no new messages/tools/decisions for 5+ iterations)
  4. **Rapid iteration** (iterations <100ms = stuck loop)
- Integration point: smart_agent_graph.py node transitions

### Update 3: Parallel Execution Confirmed Working

**Analysis Result**: Confirmed that `agent_executor_node.py:541-641` already implements parallel execution using `asyncio.gather()`:
- Max 3 concurrent tasks: `self.max_parallel_tasks = 3`
- 300s timeout per batch: `self.task_timeout = 300`
- No need to implement - already exists ✓

### Integration Points Updated

1. **tool_node.py** (before tool execution):
   - Pass `state` and `config` to smart detector (not just tool_info_list)
   - Add graph loop check with `current_node` and `next_node`

2. **smart_agent_graph.py** (between node transitions):
   - Add graph loop monitor at each node transition
   - Track node-to-node transitions
   - Monitor state hash changes

3. **agent_executor_node.py** (task execution):
   - Pass task_list to smart detector
   - Include execution_mode in detection context

---

## File Structure (UPDATED)

```
src/services/auto_detection/
├── __init__.py                  # Package initialization
├── DESIGN.md                    # This file (UPDATED)
├── ANALYSIS.md                  # Code analysis findings
├── tool_profiler.py             # Component 1: Historical profiling
├── smart_detector.py            # Component 2: Smart detection (UPDATED - task-based)
├── loop_monitor.py              # Component 3a: Tool-level loop detection
├── graph_loop_monitor.py        # Component 3b: Graph-level loop detection (NEW)
├── resource_tracker.py          # Component 4: Resource tracking
├── orchestrator.py              # Component 5: Decision orchestrator (UPDATED)
├── schemas.py                   # Data schemas and types
├── utils.py                     # Helper functions (includes calculate_task_complexity)
└── tests/
    ├── test_tool_profiler.py
    ├── test_smart_detector.py
    ├── test_loop_monitor.py
    ├── test_graph_loop_monitor.py  # NEW
    ├── test_resource_tracker.py
    └── test_orchestrator.py
```

---

## Next Steps (UPDATED)

### Phase 1: Foundation (Week 1)
1. **Implement Tool Profiler** (unchanged - foundation)
2. **Implement `calculate_task_complexity()` utility** (NEW)
3. **Implement Smart Detection Engine** (UPDATED - with task-based logic)

### Phase 2: Loop Detection (Week 2)
4. **Implement Tool Loop Monitor** (unchanged)
5. **Implement Graph Loop Monitor** (NEW - critical missing piece)
6. **Add integration hooks in smart_agent_graph.py** (NEW)

### Phase 3: Resource & Orchestration (Week 3)
7. **Implement Resource Tracker**
8. **Build Decision Orchestrator** (UPDATED - with both loop monitors)
9. **Write comprehensive tests** (UPDATED - include graph loop tests)

### Phase 4: Integration (Week 4)
10. **Integration with tool_node.py**
11. **Integration with agent_executor_node.py** (task-based detection)
12. **Integration with smart_agent_graph.py** (graph loop detection)

### Phase 5: Production (Week 5)
13. **Monitor and tune in production**
14. **Collect metrics on detection accuracy**
15. **Tune thresholds based on real data**

---

## Summary of Key Changes

| Original Design | Updated Design | Reason |
|----------------|----------------|---------|
| Only tool execution time | Tool time + task count + complexity + overheads | User insight: 20 tasks with short tools = long execution |
| Single loop monitor | Tool loop monitor + Graph loop monitor | Code analysis: No graph loop detection exists |
| No parallel execution tracking | Confirmed parallel execution exists | Analysis: agent_executor_node.py:541-641 |
| Generic context parameter | Explicit state, config, current_node, next_node | Need state access for task_list and graph transitions |
| Simple time thresholds | Multi-factor scoring (7 factors) | More accurate detection |

Ready to start implementing?

🎯 **Recommended Order**:
1. Tool Profiler (foundation)
2. Task Complexity Calculator (utility function)
3. Smart Detection Engine (UPDATED with task-based logic)
4. Graph Loop Monitor (NEW - critical gap identified)
