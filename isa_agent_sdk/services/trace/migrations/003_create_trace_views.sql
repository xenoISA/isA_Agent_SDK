-- Agent Service Migration: Create utility views
-- Version: 003
-- Date: 2025-10-28

-- View: Trace execution tree (recursive)
CREATE OR REPLACE VIEW agent.trace_tree AS
WITH RECURSIVE span_tree AS (
    -- Root spans
    SELECT
        span_id,
        trace_id,
        parent_span_id,
        span_type,
        name,
        start_time,
        end_time,
        duration_ms,
        status,
        1 as level,
        ARRAY[span_id] as path,
        span_id::TEXT as sort_path
    FROM agent.spans
    WHERE parent_span_id IS NULL

    UNION ALL

    -- Child spans
    SELECT
        s.span_id,
        s.trace_id,
        s.parent_span_id,
        s.span_type,
        s.name,
        s.start_time,
        s.end_time,
        s.duration_ms,
        s.status,
        st.level + 1,
        st.path || s.span_id,
        st.sort_path || '>' || s.span_id::TEXT
    FROM agent.spans s
    INNER JOIN span_tree st ON s.parent_span_id = st.span_id
)
SELECT * FROM span_tree
ORDER BY sort_path;

COMMENT ON VIEW agent.trace_tree IS 'Recursive view showing trace execution tree structure';


-- View: Trace summary statistics
CREATE OR REPLACE VIEW agent.trace_stats AS
SELECT
    t.trace_id,
    t.session_id,
    t.user_id,
    t.start_time,
    t.end_time,
    t.duration_ms as total_duration_ms,
    t.status,
    COUNT(s.span_id) as total_spans,
    COUNT(CASE WHEN s.span_type = 'model_call' THEN 1 END) as model_calls,
    COUNT(CASE WHEN s.span_type = 'mcp_call' THEN 1 END) as mcp_calls,
    COUNT(CASE WHEN s.span_type = 'tool_call' THEN 1 END) as tool_calls,
    SUM((s.tokens_used->>'total')::INTEGER) as total_tokens,
    t.created_at
FROM agent.traces t
LEFT JOIN agent.spans s ON t.trace_id = s.trace_id
GROUP BY t.trace_id, t.session_id, t.user_id, t.start_time, t.end_time,
         t.duration_ms, t.status, t.created_at;

COMMENT ON VIEW agent.trace_stats IS 'Trace summary with aggregated statistics';
