# Model Node Documentation

## Overview

The `ModelNode` is responsible for LLM (Large Language Model) interactions within the agent workflow. It handles model calls, streaming responses, billing integration, and sensitive request detection for human-in-the-loop scenarios.

## Purpose

The Model Node serves as the intelligent LLM interface that:
- Executes LLM calls using the ISA Model Client
- Processes streaming responses with real-time token delivery
- Handles billing calculation and credit management
- Detects sensitive requests requiring human approval
- Integrates with tracing and monitoring systems

## Input State

The Model Node expects an `OptimizedAgentState` with the following key fields:

### Required Fields
- `messages`: Conversation history with system and user messages (list)
- `capabilities`: Available tools and agent capabilities (dict)
- `user_query`: The user's input query (string)

### Optional Fields
- `enhanced_prompt`: Context-aware system prompt (string)
- `session_memory`: Session memory information (dict)
- `formatted_tools`: Available tools for the model (list)

## Output State

The Model Node updates the state with:

### Core Updates
- `messages`: Appends AI response to conversation history
- `credits_used`: Accumulated credits consumed
- `cost`: Total USD cost accumulated
- `billing_records`: Detailed billing information records
- `enhanced_billing`: Latest billing information

### Response Processing
- Streams individual tokens for real-time experience
- Processes final AI response with tool calls if applicable
- Handles billing information from ISA client

## Dependencies

### Internal Dependencies
- `app.nodes.base_node.BaseNode`: Base class with billing and streaming
- `app.types.agent_state.OptimizedAgentState`: State management
- `app.tracing.langgraph_tracer.trace_node`: Execution tracing

### External Dependencies
- `isa_model.ISAModelClient`: LLM interaction client
- `langchain_core.messages.AIMessage`: AI response format
- `langgraph.config`: Stream writer and configuration
- `langgraph.types`: Interrupt handling

### Service Dependencies
- **ISA Model Client**: For LLM calls and streaming
  - `invoke()`: Makes LLM calls with streaming support
  - Handles billing calculation automatically
  - Supports tool calling and context management

## Core Functionality

### 1. LLM Client Management
```python
def _get_isa_client() -> ISAModelClient
```
- **Purpose**: Singleton pattern for ISA client instantiation
- **Configuration**: Connects to localhost:8082 by default
- **Caching**: Reuses client instance across calls

### 2. Sensitive Request Detection
```python
def _is_sensitive_request(formatted_tools: List[Dict]) -> bool
```
- **Input**: List of available tools
- **Output**: Boolean indicating if human review required
- **Logic**: Detects autonomous planning tools (`plan_autonomous_task`)
- **Purpose**: Triggers human-in-the-loop for high-risk operations

### 3. Streaming Response Processing
```python
async def _execute_logic(state: OptimizedAgentState) -> OptimizedAgentState
```
- **Input**: Current agent state with messages and capabilities
- **Output**: Updated state with AI response and billing
- **Features**:
  - Real-time token streaming
  - Tool call detection and handling
  - Comprehensive error handling
  - Billing integration

### 4. ISA Client Integration
- **Request Format**: `{"input_data": messages, "task": "chat", "service_type": "text"}`
- **Tool Support**: Passes `formatted_tools` when available
- **Response Handling**: Processes async stream with tokens and metadata

## Execution Flow

1. **Preparation**
   - Extract messages and capabilities from state
   - Detect sensitive requests and trigger interrupts if needed
   - Initialize streaming updates

2. **LLM Call Setup**
   - Get or create ISA client instance
   - Prepare call arguments with messages and tools
   - Configure streaming parameters

3. **Response Processing**
   - Stream individual tokens for real-time experience
   - Collect full response content
   - Extract billing information from final chunk
   - Handle tool calls if present

4. **State Updates**
   - Process billing through BaseNode
   - Append AI response to messages
   - Update streaming status
   - Return enriched state

## Streaming Integration

The Model Node provides comprehensive streaming capabilities:

### Token-Level Streaming
```python
{
    "llm_token": {
        "status": "streaming",
        "token": "individual_token",
        "token_index": 1
    }
}
```

### Completion Events
```python
{
    "llm_token": {
        "status": "completed",
        "total_tokens": 150,
        "full_content": "complete_response"
    }
}
```

### Model Call Status
```python
{
    "model_call": {
        "status": "starting" | "completed",
        "tools_count": 3,
        "has_enhanced_prompt": true,
        "has_tool_calls": false
    }
}
```

## Billing Integration

### ISA Client Billing
The ISA client returns billing information in the final stream chunk:
```python
{
    "result": AIMessage(...),
    "billing": {
        "cost_usd": 0.002,
        "total_tokens": 100,
        "input_tokens": 60,
        "output_tokens": 40,
        "model": "gpt-4",
        "provider": "openai"
    }
}
```

### BaseNode Processing
- Automatically calculates credits from USD cost
- Creates `EnhancedBillingInfo` records
- Updates state with accumulated costs
- Provides detailed billing breakdown

## Error Handling

### Connection Failures
- Graceful handling of ISA client connection issues
- Fallback to error messages for user feedback
- Maintains conversation flow despite service issues

### Streaming Errors
- Handles incomplete streams gracefully
- Provides meaningful error responses
- Logs detailed error information for debugging

### Tool Call Errors
- Validates tool call structure
- Handles malformed tool requests
- Continues execution with partial responses

## Human-in-the-Loop Integration

### Sensitive Request Detection
The Model Node automatically detects requests requiring human approval:
- Autonomous task planning tools
- High-risk operations
- Administrative functions

### Interrupt Mechanism
```python
interrupt({
    "type": "llm_call_review",
    "message": "Review LLM request before execution",
    "tools_count": len(formatted_tools),
    "user_query": state.get("user_query", "")
})
```

## Configuration

### ISA Client Configuration
- **Default URL**: `http://localhost:8082`
- **Service Type**: `text` for chat interactions
- **Task Type**: `chat` for conversational AI

### Streaming Configuration
- Real-time token delivery enabled by default
- Comprehensive billing tracking
- Tool call detection and processing

## Performance Considerations

### Client Reuse
- Singleton pattern for ISA client
- Reduces connection overhead
- Maintains session consistency

### Streaming Efficiency
- Token-by-token delivery for responsiveness
- Minimal buffering for real-time experience
- Efficient billing processing

### Memory Management
- Stateless operation design
- Minimal state retention between calls
- Efficient message handling

## Testing

The Model Node includes comprehensive test coverage with **14 test cases**, all passing:

### Test Categories
- **Successful Flow**: Normal LLM execution with streaming ✅
- **Tool Integration**: Tool passing and call detection ✅
- **Error Handling**: Connection failures and malformed responses ✅
- **Streaming**: Token delivery and completion events ✅
- **Billing**: Cost calculation and record keeping ✅
- **Sensitive Detection**: Human-in-the-loop triggers ✅
- **Client Management**: ISA client singleton pattern ✅
- **Message Preservation**: Conversation history management ✅

### Test Results
- **14/14 tests passing** (100% success rate)
- Full streaming pipeline validation
- Comprehensive billing integration testing
- Robust error handling scenarios
- Tool call processing verification

### Mock Requirements
- ISA Model Client with async streaming
- LangGraph configuration and stream writer
- BaseNode billing functionality
- Proper async generator mocking for streaming

## Integration Points

### Upstream Nodes
- **Entry Node**: Provides enhanced prompts and session context
- **Router Node**: Directs traffic to model for LLM interactions

### Downstream Nodes
- **Tool Node**: Receives tool calls for execution
- **Revise Node**: Gets model responses for refinement

### External Services
- **ISA Model Service**: Core LLM functionality
- **Billing System**: Credit calculation and tracking
- **Tracing System**: Performance monitoring

## Best Practices

### Implementation
- Always handle streaming errors gracefully
- Process billing information consistently
- Detect sensitive requests before execution
- Maintain conversation context properly

### Testing
- Mock all external dependencies completely
- Test both success and failure scenarios
- Verify streaming behavior thoroughly
- Validate billing calculations

### Monitoring
- Track LLM response times and costs
- Monitor streaming performance
- Log sensitive request patterns
- Measure billing accuracy

## Security Considerations

### Sensitive Request Handling
- Automatic detection of high-risk operations
- Human review for autonomous tasks
- Comprehensive audit logging

### Data Protection
- Secure message handling
- Billing information protection
- Session context isolation

### Error Information
- Sanitized error messages for users
- Detailed logging for administrators
- No sensitive data in error responses