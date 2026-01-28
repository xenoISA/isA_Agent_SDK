# Revise Node Documentation

## Overview

The `ReviseNode` is responsible for intelligent memory management and conversation storage within the agent workflow. It combines guaranteed session storage with AI-powered analysis to determine when specific memory types should be stored, using ReactAgent for intelligent decision-making.

## Purpose

The Revise Node serves as the intelligent memory management engine that:
- **Always stores session conversations** via MCP `store_session_message` tool
- **Uses ReactAgent for intelligent analysis** to determine specific memory storage needs
- **Supports 5 memory types**: concept, episode, fact, procedure, and working memory
- **Provides AI-driven memory decisions** based on conversation content analysis
- **Streams real-time updates** for memory operations and analysis results

## Input State

The Revise Node expects an `OptimizedAgentState` with the following key fields:

### Required Fields
- `messages`: Conversation history to be stored (list)
- `session_id`: Session identifier for memory persistence (string)

### Optional Fields
- `user_id`: User identifier for personalized memory (string, defaults to "anonymous")
- `user_query`: Current user query for context (string)
- `capabilities`: Available tools and agent capabilities (dict)

## Output State

The Revise Node updates the state with:

### Memory Management Fields
- `memory_update_triggered`: Boolean indicating if AI-driven memory tools were used
- `memory_tools_used`: List of memory tools that were executed by ReactAgent
- `memory_analysis`: AI analysis result from ReactAgent about memory decisions
- `memory_update_error`: Error message if memory operations failed (string)

### State Preservation
- All existing state fields are preserved
- Only memory-specific fields are added/updated
- Message history remains unchanged

## Dependencies

### Internal Dependencies
- `app.types.agent_state.OptimizedAgentState`: State management
- `app.tracing.langgraph_tracer.trace_node`: Execution tracing

### External Dependencies
- `langgraph.config`: Stream writer and configuration
- `app.react_agent`: ReactAgent for intelligent memory analysis

### Service Dependencies
- **MCP Manager**: Memory persistence and tool execution
- **ReactAgent**: AI-powered memory analysis and tool execution
- **Session Manager**: Session state management
- **LangGraph**: Streaming and configuration management
- **Tracing System**: Performance monitoring and debugging

## Core Functionality

### 1. Memory Revision Execution
```python
async def execute(state: OptimizedAgentState) -> OptimizedAgentState
```
- **Input**: State with conversation messages
- **Output**: State with memory update status and analysis results
- **Logic**:
  1. **Fixed**: Always save session conversation via MCP `store_session_message`
  2. **Intelligent**: Use ReactAgent to analyze conversation for specific memory needs
  3. **Dynamic**: AI determines which memory tools to call based on content
  4. **Streaming**: Real-time updates for both session storage and AI analysis

### 2. Conversation Storage
```python
async def _save_conversation(state: OptimizedAgentState)
```
- **Purpose**: Persist conversation messages via MCP tools
- **Tool Used**: `store_session_message`
- **Data Stored**:
  - Session ID and User ID
  - Message type and role classification
  - Message content and importance score
- **Error Handling**: Fails silently to not block main workflow

### 3. Intelligent Memory Analysis
```python
async def _intelligent_memory_update(state: OptimizedAgentState, session_id: str, user_id: str) -> dict
```
- **AI Analysis**: ReactAgent analyzes conversation content for memory storage needs
- **Memory Tools**: 5 specific memory types available for AI to use
  - `store_concept`: Conceptual knowledge and definitions
  - `store_episode`: Specific events and experiences
  - `store_fact`: Factual information and data points
  - `store_procedure`: Step-by-step procedures and workflows
  - `store_working_memory`: Temporary information for conversation context
- **Decision Process**: AI determines which tools to call based on conversation content
- **Dynamic Execution**: Only relevant memory tools are called when needed

### 4. ReactAgent Integration
```python
react_agent = create_react_agent(memory_tools, self.mcp_manager)
analysis_result = await react_agent.execute(task_prompt)
```
- **Tool Discovery**: Dynamically loads memory tools from MCP
- **Task Prompt**: Constructs detailed analysis prompt with conversation context
- **AI Analysis**: ReactAgent analyzes conversation for memory storage needs
- **Tool Execution**: AI calls appropriate memory tools based on analysis
- **Result Parsing**: Extracts which tools were used from ReactAgent output

### 5. Message Role Classification
```python
def _get_message_role(message) -> str
```
- **HumanMessage**: Classified as `"user"`
- **AIMessage**: Classified as `"assistant"`
- **ToolMessage**: Classified as `"tool"`
- **Unknown Types**: Default to `"system"`
- **Algorithm**: Class name substring matching

## Execution Flow

1. **Initialization**
   - Extract session and user identifiers
   - Initialize streaming updates
   - Prepare memory operation context

2. **Fixed Session Storage**
   - Always store last message via MCP `store_session_message`
   - Include message metadata and importance scoring
   - Handle storage errors gracefully

3. **Dynamic Memory Tool Discovery**
   - Load available memory tools from MCP
   - Filter for 5 specific memory types
   - Create ReactAgent with discovered tools

4. **AI-Powered Memory Analysis**
   - Format conversation history for AI analysis
   - Construct detailed task prompt with context
   - Execute ReactAgent analysis of conversation content
   - AI determines which memory tools to call

5. **Memory Tool Execution**
   - ReactAgent calls appropriate memory tools based on analysis
   - Tools execute with proper session and user context
   - Results captured for state updates

6. **State Updates**
   - Set `memory_update_triggered` flag based on tools used
   - Add `memory_tools_used` list with executed tools
   - Include `memory_analysis` with AI reasoning
   - Stream completion status with tool usage

## Memory Update Triggers

### Update Threshold Logic
The revise node uses a simple but effective threshold system:

```python
# Memory update conditions
should_update = (
    messages_count > 0 and          # Has messages
    (messages_count % 5 == 0) and   # Every 5 messages
    self.mcp_manager is not None    # MCP available
)
```

### Trigger Examples
- **5 messages**: ✅ Triggers memory update
- **10 messages**: ✅ Triggers memory update
- **15 messages**: ✅ Triggers memory update
- **4 messages**: ❌ No update (below threshold)
- **7 messages**: ❌ No update (not divisible by 5)

### Update Strategy
- **Non-forced Updates**: `force_update: False` for gradual memory building
- **Session-based**: Updates tied to specific session contexts
- **User-aware**: Includes user ID for personalized memory
- **Incremental**: Builds memory over time rather than replacing

## MCP Integration

### Tool: `store_session_message`
```python
await self.mcp_manager.call_tool("store_session_message", {
    "session_id": session_id,
    "user_id": user_id,
    "message_type": type(last_message).__name__.lower(),
    "role": self._get_message_role(last_message),
    "content": str(getattr(last_message, 'content', '')),
    "importance_score": 0.7
})
```

**Parameters:**
- `session_id`: Session identifier for grouping
- `user_id`: User context for personalization
- `message_type`: Technical message class name
- `role`: Functional role (user/assistant/tool/system)
- `content`: Message content as string
- `importance_score`: Fixed at 0.7 for consistent weighting

### Tool: `summarize_session`
```python
await self.mcp_manager.call_tool("summarize_session", {
    "session_id": session_id,
    "user_id": user_id,
    "force_update": False
})
```

**Parameters:**
- `session_id`: Session to summarize
- `user_id`: User context for summary
- `force_update`: False for incremental updates

## Streaming Integration

The Revise Node provides real-time memory operation updates:

### Starting Status Updates
```python
{
    "memory_revision": {
        "status": "starting",
        "session_id": session_id
    }
}
```

### Completion Updates
```python
{
    "memory_revision": {
        "status": "completed",
        "memory_updated": should_update
    }
}
```

### Stream Writer Integration
- Uses LangGraph's `get_stream_writer()` for real-time updates
- Handles missing stream writer gracefully
- Provides structured memory operation information
- No streaming failures block memory operations

## Error Handling

### MCP Connection Errors
- **Storage Failures**: Logged but don't block workflow
- **Tool Unavailability**: Graceful handling with no MCP manager
- **Network Issues**: Silent failure for resilience
- **Authentication Problems**: Handled by MCP layer

### State Validation
- **Missing Session ID**: Operations skipped safely
- **Empty Messages**: No storage attempts made
- **Invalid Message Types**: Default handling applied
- **Malformed Content**: String conversion applied

### Memory Operation Errors
- **Summarization Failures**: Captured in `memory_update_error`
- **Threshold Calculation**: Safe integer operations
- **Role Classification**: Default to "system" for unknown types
- **Content Extraction**: Safe attribute access with defaults

## Testing

The Revise Node includes comprehensive test coverage with **16 test cases**, all passing:

### Test Categories
- **Basic Session Storage**: Fixed session message storage ✅
- **ReactAgent Integration**: AI-powered memory analysis ✅
- **Memory Tool Detection**: Dynamic tool discovery and execution ✅
- **Multiple Memory Types**: Concept, episode, fact, procedure, working memory ✅
- **MCP Integration**: Tool calling and error handling ✅
- **Message Types**: Different message format handling ✅
- **Edge Cases**: Empty states and missing dependencies ✅
- **Error Scenarios**: ReactAgent failures and graceful degradation ✅
- **Role Classification**: Message type to role mapping ✅
- **State Preservation**: Original data integrity ✅
- **Streaming**: Real-time update integration ✅

### Test Results
- **16/16 tests passing** (100% success rate)
- Full ReactAgent integration validation
- Comprehensive memory tool testing
- AI analysis and decision-making verification
- Error handling and edge case coverage
- Streaming integration verification

### Mock Requirements
- AsyncMock for MCP manager operations
- LangGraph configuration and stream writer mocking
- OptimizedAgentState with various message configurations
- Proper async test handling with pytest-asyncio

## Integration Points

### Upstream Nodes
- **Model Node**: Provides AI responses to be stored
- **Tool Node**: Generates tool messages for storage
- **Guardrail Node**: Provides sanitized content for memory

### Downstream Nodes
- **End State**: Final memory operations before completion
- **Router Node**: May receive updated state after memory operations

### Workflow Integration
```
... → Model Node → Guardrail Node → Revise Node → End
```

### Memory Persistence Flow
```
Conversation → MCP Storage → Threshold Check → Summarization → Updated Memory
```

## Configuration

### Memory Constants
- **Update Threshold**: 5 messages
- **Importance Score**: 0.7 (fixed weighting)
- **Force Update**: False (incremental strategy)

### Role Mapping
- **HumanMessage**: `"user"`
- **AIMessage**: `"assistant"`
- **ToolMessage**: `"tool"`
- **Other Types**: `"system"`

### MCP Tools
- **Storage**: `store_session_message`
- **Summarization**: `summarize_session`

## Best Practices

### Implementation
- Always handle MCP manager absence gracefully
- Preserve existing state data completely
- Use consistent importance scoring
- Provide clear error information in state

### Memory Management
- Follow threshold-based update strategy
- Store messages immediately for reliability
- Use incremental summarization approach
- Maintain session-based organization

### Testing
- Mock MCP operations thoroughly
- Test all message type scenarios
- Verify error handling robustness
- Check state preservation integrity

### Monitoring
- Track memory update frequency
- Monitor MCP operation success rates
- Log storage and summarization errors
- Measure memory operation performance

## Security Considerations

### Data Privacy
- Secure handling of conversation content
- User ID protection in memory operations
- Session isolation and access control
- No sensitive data exposure in errors

### MCP Security
- Validation of MCP tool responses
- Safe handling of tool execution failures
- Protection against memory injection attacks
- Secure session identifier management

### Error Information
- Sanitized error messages without data exposure
- No tool implementation details in logs
- Safe processing of conversation content
- Secure streaming of memory information

## Performance Considerations

### Memory Efficiency
- Minimal memory footprint during operations
- Efficient message processing
- Stateless operation design
- Optimized threshold calculations

### Operation Speed
- Fast role classification algorithms
- Efficient content extraction
- Minimal MCP call overhead
- Quick threshold evaluations

### Scalability
- Support for large conversation histories
- Efficient threshold-based updates
- Minimal processing latency
- Optimal memory update strategies

## Troubleshooting

### Common Issues
1. **Memory Not Updating**: Check message count and threshold logic
2. **MCP Failures**: Verify MCP manager availability and tools
3. **Missing Storage**: Check session ID presence and validity
4. **Role Misclassification**: Verify message type patterns

### Debug Information
- Log memory operation decisions and results
- Track MCP tool call outcomes
- Monitor threshold calculation logic
- Verify state update completeness

### Performance Monitoring
- Measure memory operation latency
- Track update trigger frequency
- Monitor MCP operation success rates
- Analyze conversation storage patterns

## Memory Architecture

### Storage Strategy
- **Immediate Storage**: Last message stored on every execution
- **Batch Summarization**: Every 5 messages trigger summary
- **Incremental Updates**: Build memory gradually over time
- **Session Isolation**: Memory tied to specific sessions

### Data Flow
```
Conversation Messages → Role Classification → MCP Storage → Threshold Check → Summarization
```

### Persistence Model
- **Message Level**: Individual message storage with metadata
- **Session Level**: Periodic summarization for context
- **User Level**: Personalized memory across sessions
- **Importance Weighting**: Consistent scoring for message relevance