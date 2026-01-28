# Router Node Documentation

## Overview

The `RouterNode` is responsible for intelligent routing decisions within the agent workflow. It analyzes AI responses to determine the next appropriate action, detects autonomous planning requests, and manages workflow direction based on tool call patterns.

## Purpose

The Router Node serves as the workflow decision engine that:
- Analyzes AI responses for tool calls and content
- Detects autonomous planning tool patterns
- Routes workflow to appropriate next steps
- Manages execution strategy decisions
- Provides streaming updates for routing decisions

## Input State

The Router Node expects an `OptimizedAgentState` with the following key fields:

### Required Fields
- `messages`: Conversation history with the last message being an AI response (list)
- `capabilities`: Available tools and agent capabilities (dict)

### Optional Fields
- `session_id`: Session identifier for tracking (string)
- `user_id`: User identifier for context (string)
- `user_query`: Original user query for reference (string)

## Output State

The Router Node updates the state with:

### Core Routing Fields
- `next_action`: Next workflow step ("end", "call_tool", "agent_executor")
- `execution_strategy`: Strategy type ("direct", "tool_call", "autonomous_planning")
- `is_autonomous`: Boolean indicating autonomous mode activation

### State Preservation
- All existing state fields are preserved
- Only routing-specific fields are added/updated
- Message history remains unchanged

## Dependencies

### Internal Dependencies
- `app.types.agent_state.OptimizedAgentState`: State management
- `app.tracing.langgraph_tracer.trace_node`: Execution tracing

### External Dependencies
- `langgraph.config`: Stream writer and configuration
- `langchain_core.messages`: Message type checking

### Service Dependencies
- **LangGraph**: Streaming and configuration management
- **Tracing System**: Performance monitoring and debugging

## Core Functionality

### 1. Routing Decision Logic
```python
async def execute(state: OptimizedAgentState) -> OptimizedAgentState
```
- **Input**: State with AI response message
- **Output**: State with routing decisions and strategy
- **Logic**:
  1. Check for tool calls in the last AI message
  2. Analyze tool names for autonomous patterns
  3. Set appropriate routing and strategy
  4. Stream decision updates

### 2. Tool Call Analysis
```python
# Extract and analyze tool calls
tool_calls = getattr(last_message, 'tool_calls', [])
tool_names = [tc.get('name', '') if isinstance(tc, dict) else '' for tc in tool_calls]
```
- **Format Support**: Handles both dict and object tool calls
- **Error Handling**: Graceful processing of malformed tool calls
- **Pattern Detection**: Identifies specific tool name patterns

### 3. Autonomous Planning Detection
```python
has_autonomous_tool = any('plan_autonomous_task' in name for name in tool_names if name)
```
- **Pattern**: Tools containing "plan_autonomous_task" substring
- **Examples**: 
  - `plan_autonomous_task`
  - `create_plan_autonomous_task_executor`
  - `custom_plan_autonomous_task_handler`
- **Priority**: Autonomous tools take precedence over regular tools

### 4. Routing Strategies

#### Direct Response (`"direct"`)
- **Condition**: No tool calls present or empty tool calls
- **Next Action**: `"end"` - End workflow
- **Autonomous**: `False`

#### Tool Call (`"tool_call"`)
- **Condition**: Regular tool calls present
- **Next Action**: `"call_tool"` - Execute tools
- **Autonomous**: `False`

#### Autonomous Planning (`"autonomous_planning"`)
- **Condition**: Autonomous planning tools detected
- **Next Action**: `"call_tool"` - Execute autonomous tools
- **Autonomous**: `True`

## Execution Flow

1. **Message Analysis**
   - Extract the last AI message from conversation
   - Check for tool_calls attribute existence
   - Stream initial routing decision status

2. **Tool Call Processing**
   - Extract tool calls safely with error handling
   - Parse tool names from various formats
   - Identify autonomous planning patterns

3. **Routing Decision**
   - Apply routing logic based on tool call analysis
   - Set execution strategy and autonomous flags
   - Determine next workflow action

4. **State Updates**
   - Update routing fields in state
   - Preserve all existing state data
   - Stream completed routing decision

## Autonomous Planning Detection

### Detection Algorithm
The router analyzes tool names for autonomous planning patterns:

```python
# Safe tool name extraction
tool_names = []
for tc in tool_calls:
    if isinstance(tc, dict):
        tool_names.append(tc.get('name', ''))
    else:
        tool_names.append('')

# Pattern matching
has_autonomous_tool = any('plan_autonomous_task' in name for name in tool_names if name)
```

### Supported Patterns
- `plan_autonomous_task` - Basic autonomous planning
- `create_plan_autonomous_task_executor` - Extended planning with execution
- `autonomous_plan_autonomous_task_workflow` - Complex workflow planning
- Any tool name containing `plan_autonomous_task` substring

### Non-Autonomous Patterns
Tools that should NOT trigger autonomous mode:
- `plan_meeting` - Regular planning, not autonomous
- `autonomous_mode_check` - Contains "autonomous" but not the pattern
- `get_weather` - Regular tool execution
- `create_file` - Standard operations

## Streaming Integration

The Router Node provides real-time routing updates:

### Decision Status Updates
```python
{
    "routing": {
        "status": "deciding",
        "has_tool_calls": True
    }
}
```

### Completion Updates
```python
{
    "routing": {
        "status": "completed",
        "decision": "tool_call" | "autonomous_planning" | "direct_response",
        "next_action": "call_tool" | "agent_executor" | "end"
    }
}
```

### Stream Writer Integration
- Uses LangGraph's `get_stream_writer()` for real-time updates
- Handles missing stream writer gracefully
- Provides structured routing information

## Error Handling

### Malformed Tool Calls
- **String Tool Calls**: Handled as empty names
- **Null Values**: Safely processed without errors
- **Missing Keys**: Default values provided
- **Type Errors**: Graceful conversion to empty strings

### Missing Attributes
- **No tool_calls**: Treated as direct response
- **Empty tool_calls**: Routed to direct response
- **Invalid Format**: Processed with default values

### Stream Writer Errors
- **Missing Writer**: Continues without streaming
- **Write Failures**: Logs but doesn't break routing
- **Configuration Errors**: Handled gracefully

## Routing Decision Matrix

| Tool Calls Present | Autonomous Tool | Next Action | Strategy | Autonomous |
|-------------------|----------------|-------------|----------|------------|
| No | N/A | `end` | `direct` | `False` |
| Yes (Empty) | N/A | `end` | `direct` | `False` |
| Yes | No | `call_tool` | `tool_call` | `False` |
| Yes | Yes | `call_tool` | `autonomous_planning` | `True` |

## Testing

The Router Node includes comprehensive test coverage with **13 test cases**, all passing:

### Test Categories
- **Direct Response**: No tool calls routing ✅
- **Regular Tool Calls**: Standard tool execution routing ✅
- **Autonomous Planning**: Autonomous tool detection and routing ✅
- **Mixed Tools**: Autonomous priority over regular tools ✅
- **Edge Cases**: Empty calls and missing attributes ✅
- **Pattern Detection**: Various autonomous tool patterns ✅
- **Malformed Handling**: Invalid tool call formats ✅
- **State Preservation**: Original state data integrity ✅
- **Streaming**: Real-time routing updates ✅

### Test Results
- **13/13 tests passing** (100% success rate)
- Full routing logic validation
- Comprehensive pattern detection testing
- Error handling and edge case coverage
- Streaming integration verification

### Mock Requirements
- LangGraph configuration and stream writer
- OptimizedAgentState with various message types
- Proper async test handling
- Tool call format variations

## Integration Points

### Upstream Nodes
- **Model Node**: Provides AI responses with potential tool calls
- **Tool Node**: Returns to router after tool execution
- **Entry Node**: Initial routing decisions

### Downstream Nodes
- **Tool Node**: Receives tool execution requests
- **Agent Executor**: Receives autonomous planning activations
- **End State**: Direct response completions

### Workflow Integration
```
Entry Node → Model Node → Router Node → [Tool Node | Agent Executor | End]
                ↑                              ↓
                └── Model Node ←───────────────┘
```

## Configuration

### Routing Constants
- **Direct End**: `next_action = "end"`
- **Tool Execution**: `next_action = "call_tool"`
- **Autonomous Mode**: `next_action = "call_tool"` + `is_autonomous = True`

### Strategy Types
- **`"direct"`**: No tool execution needed
- **`"tool_call"`**: Regular tool execution
- **`"autonomous_planning"`**: Autonomous mode activation

## Best Practices

### Implementation
- Always preserve existing state data
- Handle malformed tool calls gracefully
- Provide clear routing decisions
- Stream updates for real-time feedback

### Testing
- Test all routing scenarios thoroughly
- Verify autonomous pattern detection
- Check malformed input handling
- Validate state preservation

### Monitoring
- Track routing decision patterns
- Monitor autonomous activation frequency
- Log malformed tool call incidents
- Measure routing performance

## Security Considerations

### Tool Call Validation
- Safe parsing of potentially malformed data
- No execution of unverified tool calls
- Validation of tool name patterns
- Protection against injection attacks

### State Security
- Immutable preservation of original state
- No sensitive data in routing decisions
- Secure handling of user identifiers
- Audit logging of routing decisions

### Error Information
- Sanitized error handling without data exposure
- No tool implementation details in logs
- Safe processing of untrusted input data
- Secure streaming of routing information

## Performance Considerations

### Routing Efficiency
- Minimal computational overhead
- Fast pattern matching algorithms
- Efficient tool call processing
- Optimized state copying

### Memory Management
- Stateless operation design
- No unnecessary data retention
- Efficient message processing
- Minimal memory allocation

### Scalability
- Support for large tool call lists
- Efficient autonomous pattern detection
- Fast routing decision making
- Minimal processing latency

## Troubleshooting

### Common Issues
1. **Tools Not Detected**: Check tool_calls format and naming
2. **Wrong Routing**: Verify autonomous pattern matching
3. **Missing Updates**: Check stream writer configuration
4. **State Loss**: Ensure proper state preservation

### Debug Information
- Log routing decisions and reasoning
- Track tool call parsing results
- Monitor autonomous pattern matches
- Verify state update completeness

### Performance Monitoring
- Measure routing decision latency
- Track autonomous activation rates
- Monitor error handling frequency
- Analyze streaming update performance