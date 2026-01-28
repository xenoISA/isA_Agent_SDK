-- Agent Service Migration: Create traces table
-- Version: 001
-- Date: 2025-10-28

-- Create agent schema if not exists
CREATE SCHEMA IF NOT EXISTS agent;

-- Create traces table (Agent Run / API Request)
CREATE TABLE IF NOT EXISTS agent.traces (
    trace_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_id TEXT,

    -- Timing
    start_time TIMESTAMPTZ DEFAULT NOW(),
    end_time TIMESTAMPTZ,
    duration_ms INTEGER,

    -- Input/Output
    input JSONB,
    output JSONB,

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
CREATE INDEX IF NOT EXISTS idx_traces_session_id ON agent.traces(session_id);
CREATE INDEX IF NOT EXISTS idx_traces_user_id ON agent.traces(user_id);
CREATE INDEX IF NOT EXISTS idx_traces_status ON agent.traces(status);
CREATE INDEX IF NOT EXISTS idx_traces_created_at ON agent.traces(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_traces_start_time ON agent.traces(start_time DESC);

-- Comments
COMMENT ON TABLE agent.traces IS 'Agent execution traces - one per API request';
COMMENT ON COLUMN agent.traces.trace_id IS 'Unique trace identifier (UUID)';
COMMENT ON COLUMN agent.traces.session_id IS 'Session ID (multi-turn conversation)';
COMMENT ON COLUMN agent.traces.user_id IS 'User identifier';
COMMENT ON COLUMN agent.traces.input IS 'User request (JSONB)';
COMMENT ON COLUMN agent.traces.output IS 'Final response (JSONB)';
COMMENT ON COLUMN agent.traces.status IS 'Trace status: running, success, error';
COMMENT ON COLUMN agent.traces.metadata IS 'Additional metadata (tags, environment, etc.)';
