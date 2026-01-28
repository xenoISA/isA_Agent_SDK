-- Agent Service Migration: Create spans table
-- Version: 002
-- Date: 2025-10-28

-- Create spans table (Operations within a trace)
CREATE TABLE IF NOT EXISTS agent.spans (
    span_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL REFERENCES agent.traces(trace_id) ON DELETE CASCADE,
    parent_span_id TEXT,

    -- Span Type
    span_type TEXT NOT NULL,
    name TEXT NOT NULL,

    -- Timing
    start_time TIMESTAMPTZ DEFAULT NOW(),
    end_time TIMESTAMPTZ,
    duration_ms INTEGER,

    -- Input/Output
    input JSONB,
    output JSONB,

    -- Model-specific fields
    model TEXT,
    provider TEXT,
    tokens_used JSONB,

    -- Status
    status TEXT DEFAULT 'running',
    error TEXT,

    -- Metadata
    metadata JSONB,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_spans_trace_id ON agent.spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_parent_span_id ON agent.spans(parent_span_id);
CREATE INDEX IF NOT EXISTS idx_spans_span_type ON agent.spans(span_type);
CREATE INDEX IF NOT EXISTS idx_spans_name ON agent.spans(name);
CREATE INDEX IF NOT EXISTS idx_spans_created_at ON agent.spans(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_spans_start_time ON agent.spans(start_time DESC);

-- Comments
COMMENT ON TABLE agent.spans IS 'Spans - operations within a trace (tree structure)';
COMMENT ON COLUMN agent.spans.span_id IS 'Unique span identifier (UUID)';
COMMENT ON COLUMN agent.spans.trace_id IS 'Parent trace ID';
COMMENT ON COLUMN agent.spans.parent_span_id IS 'Parent span ID (NULL for root spans)';
COMMENT ON COLUMN agent.spans.span_type IS 'Span type: node, model_call, mcp_call, tool_call';
COMMENT ON COLUMN agent.spans.name IS 'Span name: ReasonNode, ToolNode, etc.';
COMMENT ON COLUMN agent.spans.input IS 'Span input (JSONB)';
COMMENT ON COLUMN agent.spans.output IS 'Span output (JSONB)';
COMMENT ON COLUMN agent.spans.model IS 'Model name (for model_call spans)';
COMMENT ON COLUMN agent.spans.provider IS 'Model provider (for model_call spans)';
COMMENT ON COLUMN agent.spans.tokens_used IS 'Token usage stats (JSONB)';
COMMENT ON COLUMN agent.spans.status IS 'Span status: running, success, error';
COMMENT ON COLUMN agent.spans.metadata IS 'Additional metadata';
