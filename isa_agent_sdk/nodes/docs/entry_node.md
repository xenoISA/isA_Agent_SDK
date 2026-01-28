# Entry Node Documentation

## Overview

The `EntryNode` is the first node in the agent workflow, responsible for preparing the agent execution context. It handles session validation, memory retrieval, tool discovery, and enhanced prompt construction.

## Purpose

The Entry Node serves as the intelligent gateway that:
- Validates user sessions and handles new session setup
- Retrieves and processes session memory
- Discovers available MCP tools and capabilities
- Constructs enhanced prompts with context
- Manages streaming updates to the frontend

## Input State

The Entry Node expects an `OptimizedAgentState` with the following key fields:

### Required Fields
- `session_id`: Session identifier (string)
- `user_id`: User identifier (string) 
- `user_query`: The user's input query (string)

### Optional Fields
- `messages`: Existing conversation messages (list)
- `capabilities`: Pre-existing agent capabilities (dict)
- `session_memory`: Previous session memory (dict)

## Output State

The Entry Node updates the state with:

### Core Updates
- `session_validated`: Boolean indicating session validation status
- `session_memory`: Retrieved or default session memory structure
- `enhanced_prompt`: Context-aware system prompt
- `capabilities`: Available tools and agent capabilities
- `timestamp`: Current execution timestamp

### Message Updates
- `messages`: Preserves existing messages or creates new system/user messages
  - System message with enhanced prompt
  - User message with the query

### Error Handling
- `preparation_error`: Error message if preparation fails
- `validation_error`: Error message if session validation fails

## Dependencies

### Internal Dependencies
- `app.types.agent_state.OptimizedAgentState`: State management
- `app.tracing.langgraph_tracer.trace_node`: Execution tracing

### External Dependencies
- `langchain_core.messages`: Message types
- `langgraph.config`: Stream writer and configuration
- `langgraph.types`: Interrupt handling

### Service Dependencies
- **MCP Manager**: For tool discovery and resource access
  - `get_capabilities()`: Retrieves available tools
  - `get_resource()`: Accesses session memory
  - `initialize()`: Initializes MCP session
- **Session Manager**: For session operations (injected dependency)

## Core Functionality

### 1. Session Memory Retrieval
```python
async def _get_session_memory(session_id: str) -> Dict[str, Any]
```
- **Input**: Session ID string
- **Output**: Session memory dictionary or default structure
- **Process**: 
  - Queries MCP resource `memory://session/{session_id}`
  - Parses JSON content if available
  - Returns default structure on failure: `{"conversation_summary": "", "user_preferences": {}, "ongoing_tasks": []}`

### 2. MCP Resource Discovery
```python
async def _search_mcp_resources(user_query: str) -> Dict[str, Any]
```
- **Input**: User query for context-aware tool search
- **Output**: Capabilities dictionary with formatted tools
- **Process**:
  - Initializes MCP session if needed
  - Retrieves available capabilities
  - Returns tools in standardized format

### 3. Enhanced Prompt Building
```python
def _build_enhanced_prompt(user_input: str, session_memory: Dict, session_id: str, user_id: str, available_tools: List = None) -> str
```
- **Input**: User input, session memory, session/user IDs, optional tools
- **Output**: Context-rich system prompt
- **Features**:
  - Incorporates session context and memory
  - Lists available tools and their descriptions
  - Provides user preferences and conversation history
  - Includes execution planning guidance

### 4. Session Validation
```python
def _is_new_session(state: OptimizedAgentState) -> bool
```
- **Input**: Current agent state
- **Output**: Boolean indicating if session needs validation
- **Logic**: Returns `True` if session_id is "unknown", empty, or None

### 5. Capability Merging
```python
def _merge_capabilities(base_capabilities: Dict, mcp_resources: Dict) -> Dict
```
- **Input**: Base capabilities and MCP resources
- **Output**: Merged capabilities dictionary
- **Process**: Combines existing capabilities with discovered MCP tools

## Execution Flow

1. **Initialization**
   - Extract current state information
   - Set up streaming writer for real-time updates

2. **Session Validation**
   - Check if session is new/unknown
   - Trigger session validation interrupt if needed
   - Return early for new sessions

3. **Memory Retrieval**
   - Attempt to retrieve session memory from MCP
   - Fall back to default structure on failure
   - Log any retrieval errors

4. **Tool Discovery**
   - Search for available MCP tools and capabilities
   - Handle uninitialized MCP sessions
   - Merge with existing capabilities

5. **Context Preparation**
   - Build enhanced prompt with all available context
   - Preserve existing messages or create new ones
   - Update state with all prepared information

6. **Streaming Updates**
   - Send preparation status updates
   - Include session information and available tools
   - Mark completion with final status

## Error Handling

### MCP Failures
- Gracefully handles MCP resource access failures
- Provides default values when services are unavailable
- Logs errors for debugging while continuing execution

### Session Validation
- Interrupts execution for unknown sessions
- Provides structured validation requests
- Returns appropriate error states

### Memory Access
- Falls back to default memory structure
- Continues execution even if memory retrieval fails
- Preserves user experience during service issues

## Streaming Integration

The Entry Node provides real-time updates through LangGraph's streaming system:

### Update Types
- `entry_preparation.status`: "starting" | "retrieving_memory" | "discovering_tools" | "completed"
- `session_info`: Current session and user information
- `available_tools`: List of discovered tools and capabilities

### Update Content
```python
{
    "entry_preparation": {
        "status": "completed",
        "session_id": "session_123",
        "user_id": "user_456",
        "memory_retrieved": True,
        "tools_discovered": 5,
        "available_tools": [...]
    }
}
```

## Configuration

### Environment Variables
- No direct environment variable dependencies
- Inherits MCP and session manager configurations

### Initialization Parameters
```python
EntryNode(mcp_manager: MCPManager, session_manager: SessionManager)
```

## Performance Considerations

### Async Operations
- All MCP operations are asynchronous
- Concurrent memory and tool discovery where possible
- Non-blocking error handling

### Memory Usage
- Minimal state duplication
- Efficient JSON parsing for session memory
- Streaming updates for real-time feedback

### Error Recovery
- Fast fallback to default values
- Continues execution despite service failures
- Graceful degradation of functionality

## Testing

The Entry Node includes comprehensive test coverage with **17 test cases**, all passing:

### Test Categories
- **Successful Flow**: Normal execution with all services available ✅
- **Error Handling**: MCP failures, memory access issues ✅
- **Session Validation**: New session detection and validation logic ✅
- **Message Preservation**: Existing message handling ✅
- **Streaming Updates**: Real-time update delivery ✅
- **Prompt Building**: Enhanced prompt construction with tools and memory ✅
- **Capability Merging**: MCP resource integration ✅
- **Memory Retrieval**: Session memory access patterns ✅

### Test Results
- **17/17 tests passing** (100% success rate)
- Comprehensive error handling validation
- Robust mocking of external dependencies
- Real-world scenario coverage

### Mock Requirements
- MCP Manager with `get_resource`, `get_capabilities`, `initialize`
- Session Manager for dependency injection
- LangGraph configuration and stream writer
- Proper context configuration for LangGraph components

## Integration Points

### Upstream Nodes
- **None**: Entry node is the starting point of the workflow

### Downstream Nodes
- **Router Node**: Receives prepared state for decision making
- **Model Node**: Uses enhanced prompt for LLM interactions
- **Tool Node**: Accesses discovered capabilities

### External Services
- **MCP Server**: Tool discovery and memory access
- **Session Store**: Session validation and management
- **Tracing System**: Execution monitoring and debugging

## Best Practices

### Implementation
- Always handle MCP service failures gracefully
- Preserve existing state when possible
- Provide meaningful error messages for debugging

### Testing
- Mock all external dependencies
- Test both success and failure scenarios
- Verify streaming update delivery

### Monitoring
- Track MCP response times and failures
- Monitor session validation patterns
- Log preparation errors for debugging