# Tool Node Documentation

## Overview

The `ToolNode` is responsible for executing MCP (Model Context Protocol) tool calls within the agent workflow. It handles tool execution, human-in-the-loop interactions, billing processing, and autonomous planning activation.

## Purpose

The Tool Node serves as the execution engine for tool calls that:
- Executes MCP tools through the MCP manager
- Handles human-in-the-loop tools requiring approval
- Processes billing information from tool executions
- Activates autonomous planning mode when appropriate
- Manages streaming updates for real-time feedback

## Input State

The Tool Node expects an `OptimizedAgentState` with the following key fields:

### Required Fields
- `messages`: Conversation history with the last message containing tool calls (list)
- `capabilities`: Available tools and agent capabilities (dict)

### Optional Fields
- `cost`: Current accumulated cost (float, default: 0.0)
- `credits_used`: Current accumulated credits (float, default: 0.0)
- `session_id`: Session identifier for tracking (string)
- `user_id`: User identifier for authorization (string)

## Output State

The Tool Node updates the state with:

### Core Updates
- `messages`: Appends ToolMessage responses for each executed tool
- `cost`: Accumulated USD cost from tool executions
- `credits_used`: Accumulated credits consumed
- `next_action`: Routing directive for the next workflow step

### Autonomous Planning Updates
When autonomous planning tools are executed successfully:
- `is_autonomous`: Set to True
- `execution_strategy`: Set to "autonomous_planning"
- `autonomous_tasks`: List of planned tasks
- `execution_plan`: Detailed execution plan data
- `next_action`: Set to "agent_executor"

## Dependencies

### Internal Dependencies
- `app.types.agent_state.OptimizedAgentState`: State management
- `app.tracing.langgraph_tracer.trace_node`: Execution tracing
- `app.services.human_in_loop_service.human_in_loop_service`: Human approvals

### External Dependencies
- `langchain_core.messages.ToolMessage`: Tool response format
- `langgraph.config`: Stream writer and configuration
- `json`: Tool response parsing

### Service Dependencies
- **MCP Manager**: For tool discovery and execution
  - `call_tool(tool_name, tool_args)`: Executes individual tools
  - Returns tool responses with optional billing information
- **Human-in-the-Loop Service**: For approval workflows
  - `request_approval()`: Handles authorization requests
  - Supports content review and task approval types

## Core Functionality

### 1. Tool Call Processing
```python
async def execute(state: OptimizedAgentState) -> OptimizedAgentState
```
- **Input**: State with tool calls in the last AI message
- **Output**: Updated state with tool responses and routing information
- **Features**:
  - Handles both dict and object-style tool calls
  - Graceful error handling for malformed calls
  - Comprehensive streaming updates

### 2. Human-in-the-Loop Tool Handling
```python
def _is_human_loop_tool(tool_name: str) -> bool
async def _handle_human_loop_tool(tool_name, tool_args, state) -> str
```
- **Supported Tools**: `ask_human`, `request_authorization`, `approve_authorization`, `get_authorization_requests`
- **Integration**: Uses human-in-loop service for approval workflows
- **Response Format**: JSON with action type, data, and timestamps

### 3. Tool Call Format Support
The node handles multiple tool call formats:
- **Dictionary Format**: `{"name": "tool", "args": {...}, "id": "call_123"}`
- **Object Format**: Objects with `.name`, `.args`, and `.id` attributes
- **Malformed Calls**: Graceful handling with default values

### 4. Billing Integration
- **Cost Extraction**: Parses JSON responses for billing information
- **Accumulation**: Updates state cost and credits_used fields
- **Format**: `{"billing": {"total_cost_usd": 0.05}}`

### 5. Autonomous Planning Detection
- **Trigger**: Tool name `create_execution_plan` with successful response
- **Response Format**: 
```json
{
  "status": "success",
  "data": {
    "tasks": [...],
    "plan_id": "plan_123"
  }
}
```
- **State Updates**: Activates autonomous mode and sets execution data

## Execution Flow

1. **Tool Call Extraction**
   - Extract tool calls from the last AI message
   - Handle missing or empty tool_calls gracefully
   - Stream initial execution update

2. **Tool Processing Loop**
   - Process each tool call individually
   - Handle human-in-loop tools specially
   - Execute regular tools through MCP manager
   - Parse responses and extract billing information

3. **Response Creation**
   - Create ToolMessage for each executed tool
   - Include tool_call_id for proper correlation
   - Handle errors with descriptive error messages

4. **State Updates**
   - Accumulate billing costs and credits
   - Detect autonomous planning activation
   - Set appropriate next_action routing

## Human-in-the-Loop Integration

### Supported Tool Types

#### ask_human
- **Purpose**: Request input or clarification from human operator
- **Arguments**: `question`, `context`, `user_id`
- **Decision Type**: `content_review`
- **Response**: Human-provided answer in structured format

#### request_authorization
- **Purpose**: Request approval for sensitive operations
- **Arguments**: `tool_name`, `reason`, `user_id`
- **Decision Type**: `task_approval`
- **Response**: Authorization decision with security metadata

### Authorization Flow
```python
approval_result = human_in_loop_service.request_approval(
    decision_type="task_approval",
    context={
        "tool_name": tool_to_authorize,
        "reason": reason,
        "user_id": user_id,
        "risk_level": "high"
    },
    node_source="call_tool",
    urgency="high"
)
```

## Streaming Integration

The Tool Node provides comprehensive streaming capabilities:

### Tool Execution Status
```python
{
    "tool_execution": {
        "status": "starting" | "executing" | "validating" | "completed",
        "tool_name": "get_weather",
        "tool_index": 1,
        "total_tools": 3,
        "tool_count": 3
    }
}
```

### Execution Progress
- **Starting**: Tool execution beginning
- **Executing**: Individual tool being processed
- **Validating**: Tool arguments validation
- **Completed**: Tool execution finished with preview

## Billing Integration

### MCP Tool Billing
Tools can return billing information in their responses:
```python
{
    "content": "Tool result",
    "billing": {
        "total_cost_usd": 0.05
    }
}
```

### Cost Accumulation
- Automatically extracts and accumulates costs
- Updates both `cost` and `credits_used` fields
- Handles JSON parsing errors gracefully

## Error Handling

### Tool Execution Errors
- Catches all exceptions during tool execution
- Creates descriptive error messages for users
- Continues processing remaining tools
- Maintains conversation flow

### Malformed Tool Calls
- Handles non-dict tool call formats
- Provides default values for missing attributes
- Creates valid tool messages for all calls
- Logs errors without breaking execution

### Human-in-Loop Errors
- Graceful handling of approval service failures
- Meaningful error responses for authorization issues
- Continues workflow with appropriate error states

## Sensitive Tool Detection

### Detection Patterns
```python
def _is_sensitive_tool(tool_name: str) -> bool
```
- **Patterns**: `delete`, `remove`, `plan_autonomous_task`, `destroy`, `drop`
- **Purpose**: Identifies tools requiring special handling
- **Note**: Authorization handled by human-in-loop service

## Configuration

### Tool Categories
- **Regular Tools**: Standard MCP tools executed directly
- **Human-Loop Tools**: Require human approval or input
- **Autonomous Tools**: Trigger autonomous planning mode
- **Sensitive Tools**: Require authorization (handled by service)

### Routing Logic
- **Regular Execution**: `next_action = "call_model"`
- **Autonomous Planning**: `next_action = "agent_executor"`
- **Error Cases**: Continue to model for error handling

## Testing

The Tool Node includes comprehensive test coverage with **13 test cases**, all passing:

### Test Categories
- **Successful Execution**: Multiple tool calls with responses ✅
- **Billing Processing**: Cost extraction and accumulation ✅
- **Autonomous Planning**: Execution plan activation ✅
- **Human-in-Loop**: Ask human and authorization flows ✅
- **Error Handling**: Tool failures and malformed calls ✅
- **Tool Detection**: Human-loop and sensitive tool identification ✅
- **Streaming**: Execution progress updates ✅
- **Edge Cases**: No tool calls and mixed tool types ✅

### Test Results
- **13/13 tests passing** (100% success rate)
- Full tool execution pipeline validation
- Comprehensive error handling verification
- Human-in-loop integration testing
- Autonomous planning activation tests

### Mock Requirements
- MCP Manager with async tool execution
- Human-in-loop service with approval workflows
- LangGraph configuration and stream writer
- Proper async handling for tool calls

## Integration Points

### Upstream Nodes
- **Model Node**: Provides AI messages with tool calls
- **Router Node**: Directs traffic to tool execution

### Downstream Nodes
- **Model Node**: Receives tool responses for continuation
- **Agent Executor**: Receives autonomous planning activation

### External Services
- **MCP Manager**: Core tool execution capability
- **Human-in-Loop Service**: Authorization and approval workflows
- **Billing System**: Cost tracking and credit management

## Best Practices

### Implementation
- Always handle tool execution errors gracefully
- Process billing information consistently
- Maintain proper tool_call_id correlation
- Validate tool call formats before execution

### Testing
- Mock all external dependencies completely
- Test both success and failure scenarios
- Verify human-in-loop integrations thoroughly
- Validate billing calculations and state updates

### Monitoring
- Track tool execution times and success rates
- Monitor human-in-loop approval patterns
- Log autonomous planning activations
- Measure billing accuracy and cost accumulation

## Security Considerations

### Tool Authorization
- Human-in-loop integration for sensitive operations
- Authorization request tracking with unique IDs
- Risk level assessment for tool executions
- Comprehensive audit logging

### Data Protection
- Secure handling of tool arguments and responses
- User ID validation for authorization requests
- Session context isolation
- No sensitive data in error messages

### Error Information
- Sanitized error messages for users
- Detailed logging for administrators
- No tool implementation details in responses
- Secure billing information handling

## Performance Considerations

### Tool Execution
- Sequential tool processing for reliability
- Efficient JSON parsing and validation
- Minimal state copying and mutation
- Optimized streaming update frequency

### Memory Management
- Stateless operation design
- Efficient message handling and storage
- Proper cleanup of temporary data
- Controlled response size limits

### Scalability
- Support for multiple concurrent tool calls
- Efficient billing calculation and storage
- Streamlined human-in-loop processing
- Minimal overhead for simple tool executions