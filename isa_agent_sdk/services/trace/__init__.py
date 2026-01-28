"""
Trace Service - LangSmith-inspired tracing for Agent execution

Phase 2 (Current): Database persistence enabled
- model_callback.py: Capture model calls + write to DB
- mcp_callback.py: Capture MCP operations + write to DB
- trace_writer.py: PostgresClient-based DB writer
- Decorators integrated in base_node.py
- Data logged to console AND persisted to agent.spans

Phase 3 (Next): Query service and frontend viewer
- Query service for trace analysis
- Frontend trace viewer
"""

from .model_callback import trace_model_call
from .mcp_callback import trace_mcp_operation
from .trace_writer import TraceWriter, get_trace_writer

__all__ = [
    'trace_model_call',
    'trace_mcp_operation',
    'TraceWriter',
    'get_trace_writer',
]
