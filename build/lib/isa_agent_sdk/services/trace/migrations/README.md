# Trace Service Migrations

Database migrations for the trace system (LangSmith-inspired architecture).

## Migration Files

- `001_create_traces_table.sql` - Traces table (Agent Runs / API Requests)
- `002_create_spans_table.sql` - Spans table (Operations within traces)
- `003_create_trace_views.sql` - Utility views (tree structure, statistics)

## Schema Overview

```
agent.traces          -- Agent execution traces (one per API request)
agent.spans           -- Operations within traces (tree structure)
agent.trace_tree      -- View: Recursive execution tree
agent.trace_stats     -- View: Aggregated statistics
```

## Data Model

```
Session (会话 - 多轮对话)
  └── Trace (单次 API 请求 / Agent Run)
       ├── Span: ReasonNode execution
       │    └── Span: Model call
       ├── Span: ToolNode execution
       │    ├── Span: MCP call
       │    └── Span: Tool execution
       └── Span: ResponseNode execution
            └── Span: Model call
```

## Running Migrations

### Quick Method (Recommended)

```bash
cd src/services/trace/migrations
./run_migrations.sh
```

### Manual Method

```bash
# Run each migration
docker exec -i staging-postgres psql -U postgres -d isa_platform < 001_create_traces_table.sql
docker exec -i staging-postgres psql -U postgres -d isa_platform < 002_create_spans_table.sql
docker exec -i staging-postgres psql -U postgres -d isa_platform < 003_create_trace_views.sql
```

## Testing Queries

```sql
-- Get all traces for a session
SELECT * FROM agent.traces WHERE session_id = 'session_123' ORDER BY start_time DESC;

-- Get trace execution tree
SELECT * FROM agent.trace_tree WHERE trace_id = 'trace_456';

-- Get trace statistics
SELECT * FROM agent.trace_stats WHERE session_id = 'session_123';

-- Get model calls for a trace
SELECT span_id, name, model, provider, duration_ms, tokens_used
FROM agent.spans
WHERE trace_id = 'trace_456' AND span_type = 'model_call'
ORDER BY start_time;
```

## Adding New Migrations

1. Create new file: `00X_description.sql`
2. Follow the header format:
   ```sql
   -- Trace Service Migration: Description
   -- Version: 00X
   -- Date: YYYY-MM-DD
   ```
3. Use `agent.` schema prefix for all tables/views
4. Add indexes for common query patterns
5. Add comments for documentation
