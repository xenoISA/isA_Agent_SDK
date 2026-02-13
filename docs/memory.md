# Memory System

The isA Agent SDK provides a comprehensive multi-tier memory system that enables agents to remember context, learn from interactions, and maintain persistent user knowledge across sessions.

## Overview

The memory system has three complementary tiers:

1. **Static Project Context** - File-based memory (ISA.md/CLAUDE.md) loaded at startup
2. **Session Memory** - Short-term conversation tracking within a session (TTL-based)
3. **Long-term Memory** - Persistent user memories stored in PostgreSQL + Qdrant

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    isA Agent SDK Memory                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Static Context  │  │ Session Memory  │  │ Long-term       │ │
│  │ (ISA.md)        │  │ (Working+       │  │ Memory          │ │
│  │                 │  │  Session)       │  │ (User-scoped)   │ │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘ │
│           │                    │                     │          │
│           └──────────┬─────────┴─────────────────────┘          │
│                      ▼                                          │
│           ┌─────────────────────┐                               │
│           │  Memory Aggregator  │                               │
│           │  (Context Builder)  │                               │
│           └─────────────────────┘                               │
│                      │                                          │
│                      ▼                                          │
│           ┌─────────────────────┐                               │
│           │   Agent Context     │                               │
│           │   (System Prompt)   │                               │
│           └─────────────────────┘                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Static Project Context

### ISA.md / CLAUDE.md Support

Similar to Claude Code's CLAUDE.md, the SDK supports static project context files that are automatically discovered and injected into the system prompt.

**Supported files (in priority order):**
- `ISA.md` - isA Agent SDK native format
- `CLAUDE.md` - Claude Code compatible
- `.isa/CONTEXT.md` - Alternative location
- `.claude/CONTEXT.md` - Claude Code alternative location

### Usage

```python
from isa_agent_sdk import query, ISAAgentOptions

# Auto-discover from project root
options = ISAAgentOptions(
    project_context="auto"
)

# Specific file path
options = ISAAgentOptions(
    project_context="./ISA.md"
)

# Direct inline content
options = ISAAgentOptions(
    project_context="""
    This project uses FastAPI and PostgreSQL.
    Always use type hints and follow PEP 8.
    """
)

async for msg in query("Help me fix this bug", options=options):
    print(msg.content)
```

### File Discovery

When `project_context="auto"`, the SDK:

1. Starts from the current working directory (or `options.cwd` if set)
2. Searches for context files in the current directory
3. Walks up to 3 parent directories if not found
4. Returns empty string if no file found

### Example ISA.md

```markdown
# Project Context

## Tech Stack
- FastAPI 0.100+
- PostgreSQL 15
- SQLAlchemy 2.0

## Coding Standards
- Use type hints everywhere
- Follow PEP 8
- Write docstrings for public functions

## Important Notes
- Never commit .env files
- Run migrations before starting: `alembic upgrade head`
```

### Programmatic Access

```python
from isa_agent_sdk import (
    load_project_context,
    discover_project_context_file,
    format_project_context_for_prompt
)

# Discover file
path = discover_project_context_file("/path/to/project")

# Load content
content = load_project_context("auto", start_dir="/path/to/project")

# Format for prompt injection
formatted = format_project_context_for_prompt(content)
```

---

## Session Memory (Short-term)

Session memory tracks the current conversation and active tasks within a session.

### Components

| Type | Purpose | TTL | Example |
|------|---------|-----|---------|
| **Session Messages** | Conversation history | Session TTL (default 30min) | "User asked about auth" |
| **Working Memory** | Active task context | Configurable (default 24h) | "Currently fixing login bug" |

### Session Configuration

```python
from isa_agent_sdk import query, ISAAgentOptions

options = ISAAgentOptions(
    session_id="my-session-123",      # Unique session identifier
    user_id="user-456",               # User for memory scoping
    session_ttl=1800,                 # Session TTL in seconds (30 min)
)

async for msg in query("Help me with this task", options=options):
    print(msg.content)
```

### Session Memory Tools

```python
from isa_agent_sdk import execute_tool

# Store a session message
result = await execute_tool(
    tool_name="store_session_message",
    tool_args={
        "user_id": "user-123",
        "session_id": "session-456",
        "message_content": "User asked about authentication",
        "message_type": "conversation",
        "role": "user",
        "importance_score": 0.7
    }
)

# Get session context
result = await execute_tool(
    tool_name="get_session_context",
    tool_args={
        "user_id": "user-123",
        "session_id": "session-456",
        "include_summaries": "true",
        "max_recent_messages": 5
    }
)

# Store working memory (active task)
result = await execute_tool(
    tool_name="store_working_memory",
    tool_args={
        "user_id": "user-123",
        "dialog_content": "Working on fixing the login authentication bug",
        "ttl_seconds": 3600,  # 1 hour TTL
        "importance_score": 0.8
    }
)

# Get active working memories
result = await execute_tool(
    tool_name="get_active_working_memories",
    tool_args={"user_id": "user-123"}
)
```

---

## Long-term Memory (Persistent)

Long-term memory persists user information across sessions using PostgreSQL for structured data and Qdrant for vector embeddings.

### Memory Types

| Type | Purpose | AI Extraction | Example |
|------|---------|---------------|---------|
| **Factual** | Facts about entities | Yes - extracts subject/predicate/object | "Alice works at Google" |
| **Procedural** | How to do things | Yes - extracts steps/prerequisites | "Deploy: run ./deploy.sh" |
| **Episodic** | Events and experiences | Yes - extracts events with dates | "Met with team on 2024-01-15" |
| **Semantic** | Concepts and relationships | Yes - extracts concepts/categories | "FastAPI is a Python framework" |

### Storage Flow

```
Dialog Content
     │
     ▼
┌────────────────────┐
│  AI Extraction     │ ◄── Uses LLM to extract structured facts
│  (ISA Model)       │
└────────────────────┘
     │
     ├──────────────────────────────┐
     ▼                              ▼
┌────────────────────┐    ┌────────────────────┐
│  PostgreSQL        │    │  Qdrant            │
│  (Structured Data) │    │  (Vector Embeddings)│
│  - Subject         │    │  - Semantic Search │
│  - Predicate       │    │  - Similarity      │
│  - Object Value    │    │  - User Filtering  │
│  - Metadata        │    │                    │
└────────────────────┘    └────────────────────┘
```

### Storing Long-term Memories

```python
from isa_agent_sdk import execute_tool

# Store factual memory (AI extracts facts automatically)
result = await execute_tool(
    tool_name="store_factual_memory",
    tool_args={
        "user_id": "user-123",
        "dialog_content": """
        Human: My name is Alice and I work at Google as a software engineer.
        I prefer using Python and TypeScript.
        AI: Nice to meet you Alice!
        """,
        "importance_score": 0.8
    }
)
# Extracts: [Alice, works at, Google], [Alice, prefers, Python], etc.

# Store episodic memory (AI extracts events)
result = await execute_tool(
    tool_name="store_episodic_memory",
    tool_args={
        "user_id": "user-123",
        "dialog_content": "We deployed the new API to production yesterday.",
        "importance_score": 0.7
    }
)

# Store procedural memory (AI extracts how-to steps)
result = await execute_tool(
    tool_name="store_procedural_memory",
    tool_args={
        "user_id": "user-123",
        "dialog_content": "To deploy, first run tests, then build, then push to main.",
        "importance_score": 0.9
    }
)

# Store semantic memory (AI extracts concepts)
result = await execute_tool(
    tool_name="store_semantic_memory",
    tool_args={
        "user_id": "user-123",
        "dialog_content": "FastAPI is a modern Python web framework built on Starlette.",
        "importance_score": 0.6
    }
)
```

### Searching Memories

The memory system supports both text-based and vector similarity search. Vector search uses Qdrant for semantic similarity matching with OpenAI embeddings.

```python
from isa_agent_sdk import execute_tool

# Universal vector search across all memory types
# Uses Qdrant for semantic similarity matching
result = await execute_tool(
    tool_name="search_memories",
    tool_args={
        "user_id": "user-123",
        "query": "Where does Alice work?",
        "memory_types": ["factual", "episodic", "semantic", "procedural", "working", "session"],
        "limit": 10,
        "similarity_threshold": 0.15  # Minimum similarity score (0.0-1.0)
    }
)

# Search facts by subject (text-based)
result = await execute_tool(
    tool_name="search_facts_by_subject",
    tool_args={
        "user_id": "user-123",
        "subject": "Alice",
        "limit": 10
    }
)

# Search episodes by event type (text-based)
result = await execute_tool(
    tool_name="search_episodes_by_event_type",
    tool_args={
        "user_id": "user-123",
        "event_type": "deployment",
        "limit": 5
    }
)

# Search concepts by category (text-based)
result = await execute_tool(
    tool_name="search_concepts_by_category",
    tool_args={
        "user_id": "user-123",
        "category": "technology",
        "limit": 10
    }
)
```

### Vector Search Architecture

All 6 memory types support vector similarity search via Qdrant:

```
Query Text
    │
    ▼
┌────────────────────┐
│  ISA Model         │ ◄── Generates embedding (1536 dimensions)
│  text-embedding-   │
│  3-small           │
└────────────────────┘
    │
    ▼
┌────────────────────┐
│  Qdrant            │ ◄── Cosine similarity search
│  Vector Database   │     with user_id filtering
└────────────────────┘
    │
    ▼
┌────────────────────┐
│  PostgreSQL        │ ◄── Fetch full memory data
│  (Memory Storage)  │     by matched IDs
└────────────────────┘
    │
    ▼
Results with similarity_score (0.0-1.0)
```

**Key Features:**
- **User Isolation**: All searches are filtered by user_id at the Qdrant level
- **Similarity Scores**: Results include `similarity_score` (0.0-1.0) for relevance ranking
- **Score Threshold**: Filter results by minimum similarity (default 0.15)
- **Hybrid Search**: Falls back to text-based search if embedding generation fails

---

## Memory Aggregation

The `MemoryAggregator` combines memories from multiple sources into a unified context for the LLM.

### How It Works

```python
from isa_agent_sdk.graphs.utils.memory_utils import (
    MemoryAggregator,
    get_user_memory_context,
    get_session_summary_only,
    get_working_memory_only
)

# Create aggregator
aggregator = MemoryAggregator(mcp_service, max_context_length=2000)

# Get full aggregated context
context = await aggregator.get_aggregated_memory(
    user_id="user-123",
    session_id="session-456",
    query_context="Python development",  # For semantic search
    include_session=True,    # Session conversation
    include_working=True,    # Active tasks
    include_semantic=True,   # Related concepts
    include_episodic=True,   # Past events
    include_factual=True     # User facts
)

# Convenience functions
session_only = await get_session_summary_only(mcp_service, "user-123", "session-456")
working_only = await get_working_memory_only(mcp_service, "user-123")
```

### Aggregation Output Format

```markdown
# Memory Context

## Session Context
Summary: Discussed fixing authentication bugs
Topics: authentication, login, security

## Current Tasks
Task: Fixing login authentication issue (Priority: high)
Task: Writing unit tests for auth module (Priority: medium)

## Relevant Context
Factual: Alice works at Google (Score: 0.95)
Semantic: OAuth2 is used for authentication (Score: 0.87)

## User Information
Alice: Prefers Python
Alice: Uses VS Code
```

---

## Automatic Memory Storage

After each conversation, memories are automatically stored using intelligent extraction:

```python
# This happens automatically after agent query completes
from isa_agent_sdk.graphs.utils.context_update import update_context_after_chat

result = await update_context_after_chat(
    session_id="session-123",
    user_id="user-456",
    final_state=graph_state,
    mcp_service=mcp,
    conversation_complete=True
)

# Stores to all memory types:
# - Session message (conversation tracking)
# - Factual memory (extracted facts, importance: 0.6)
# - Episodic memory (extracted events, importance: 0.5)
# - Semantic memory (extracted concepts, importance: 0.6)
# - Procedural memory (extracted procedures, importance: 0.7)
# - Working memory (current tasks, TTL: 24h, importance: 0.5)
```

---

## MCP Memory Tools Reference

The memory system provides 15 MCP tools for storing, searching, and managing memories.

### Storage Tools (6)

| Tool | Description | Parameters |
|------|-------------|------------|
| `store_factual_memory` | AI extracts and stores facts | user_id, dialog_content, importance_score |
| `store_episodic_memory` | AI extracts and stores events | user_id, dialog_content, importance_score |
| `store_semantic_memory` | AI extracts and stores concepts | user_id, dialog_content, importance_score |
| `store_procedural_memory` | AI extracts and stores procedures | user_id, dialog_content, importance_score |
| `store_working_memory` | Store temporary task context | user_id, dialog_content, ttl_seconds, importance_score |
| `store_session_message` | Store conversation message | user_id, session_id, message_content, message_type, role, importance_score |

> **Note on AI Extraction Response Format**: The AI extraction tools (`store_factual_memory`, `store_episodic_memory`, `store_semantic_memory`, `store_procedural_memory`) return `memory_id: null` at the top level because they may create multiple memories from a single input. The actual memory IDs are available in `data.memory_ids[]` array.

### Search Tools (4)

| Tool | Description | Parameters |
|------|-------------|------------|
| `search_memories` | Vector similarity search across types | user_id, query, memory_types[], limit, similarity_threshold |
| `search_facts_by_subject` | Text search facts by subject | user_id, subject, limit |
| `search_episodes_by_event_type` | Text search events by type | user_id, event_type, limit |
| `search_concepts_by_category` | Text search concepts by category | user_id, category, limit |

**`search_memories` Parameters:**
- `user_id` (required): User identifier for memory scoping
- `query` (required): Search query text (converted to embedding for vector search)
- `memory_types` (optional): Array of types to search - `["factual", "episodic", "semantic", "procedural", "working", "session"]`. Defaults to all types.
- `limit` (optional): Maximum results per memory type. Default: 10
- `similarity_threshold` (optional): Minimum similarity score (0.0-1.0). Default: 0.15

### Retrieval Tools (3)

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_session_context` | Get session conversation | user_id, session_id, include_summaries, max_recent_messages |
| `get_active_working_memories` | Get current tasks | user_id |
| `summarize_session` | AI-generated session summary | user_id, session_id, force_update, compression_level |

**`summarize_session` Parameters:**
- `user_id` (required): User identifier
- `session_id` (required): Session to summarize
- `force_update` (optional): Force regenerate summary even if one exists. Default: false
- `compression_level` (optional): Summary detail level - `"low"`, `"medium"`, `"high"`. Default: "medium"

### Utility Tools (2)

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_memory_statistics` | User memory stats | user_id |
| `memory_health_check` | Service health status | (none) |

---

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `project_context` | str \| None | None | Static context ("auto", file path, or content) |
| `cwd` | str \| None | None | Working directory for auto-discovery |
| `session_id` | str \| None | Auto-generated | Session identifier |
| `user_id` | str \| None | None | User identifier for memory scoping |
| `session_ttl` | int | 1800 | Session TTL in seconds (30 min) |

---

## Best Practices

### 1. Use User IDs Consistently
```python
# Always set user_id for cross-session memory
options = ISAAgentOptions(
    user_id="user-123",  # Same user across sessions
    session_id="session-new"  # New session each time
)
```

### 2. Set Importance Scores Appropriately
- 0.9-1.0: Critical information (user identity, preferences)
- 0.7-0.8: Important context (current tasks, recent decisions)
- 0.5-0.6: General information (conversations, background)
- 0.1-0.4: Low-priority information (casual mentions)

### 3. Use TTL for Temporary Context
```python
# Short TTL for current task
await execute_tool("store_working_memory", {
    "ttl_seconds": 3600,  # 1 hour for immediate tasks
    ...
})

# Longer TTL for ongoing projects
await execute_tool("store_working_memory", {
    "ttl_seconds": 604800,  # 1 week for project context
    ...
})
```

### 4. Leverage Static Context for Project Rules
```markdown
# ISA.md - Always loaded, never expires

## Project Rules
- Use async/await for all I/O operations
- Never hardcode credentials
- Always validate user input
```

---

## Testing Memory

Run the memory end-to-end tests:

```bash
export ISA_MODEL_URL=http://localhost:8082
export ISA_MCP_URL=http://localhost:8081

python tests/test_memory_e2e.py
```

---

## Next Steps

- [Options](./options.md) - Full configuration reference
- [Tools](./tools.md) - Available tools including memory tools
- [Examples](./examples/) - Complete usage examples
