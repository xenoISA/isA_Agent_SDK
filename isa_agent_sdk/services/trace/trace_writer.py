#!/usr/bin/env python3
"""
Trace Writer - Database persistence for trace spans
Uses AsyncPostgresClient from isa_common for async connection management
"""
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from isa_common import AsyncPostgresClient

from isa_agent_sdk.utils.logger import agent_logger

logger = agent_logger


class TraceWriter:
    """
    Write trace spans to agent.spans table using async PostgreSQL client
    """

    def __init__(self, host: str = 'localhost', port: int = 50061):
        """
        Initialize trace writer

        Args:
            host: PostgreSQL proxy host (default: localhost)
            port: PostgreSQL proxy port (default: 50061)
        """
        self.host = host
        self.port = port
        self.user_id = 'agent-service'
        self.db = AsyncPostgresClient(host=host, port=port, user_id=self.user_id)

    async def write_span_start(
        self,
        span_id: str,
        trace_id: str,
        session_id: str,
        span_type: str,
        name: str,
        input_data: Dict[str, Any],
        parent_span_id: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Write span start to database

        Args:
            span_id: Unique span identifier
            trace_id: Trace identifier (same as session_id for now)
            session_id: Session identifier
            span_type: Type of span (model_call, mcp_call, node_execution)
            name: Span name (e.g., "ReasonNode.call_model")
            input_data: Input parameters as dict
            parent_span_id: Optional parent span
            model: Model name (for model_call spans)
            provider: Provider name (for model_call spans)
            metadata: Additional metadata

        Returns:
            bool: Success status
        """
        try:
            # Use session_id as trace_id for simplicity (one trace per API request)
            actual_trace_id = trace_id or session_id

            async with self.db:
                # Ensure trace exists (INSERT ON CONFLICT DO NOTHING)
                query = """
                    INSERT INTO agent.traces (trace_id, session_id, status, start_time)
                    VALUES ($1, $2, $3, NOW())
                    ON CONFLICT (trace_id) DO NOTHING
                """
                await self.db.execute(query, [actual_trace_id, session_id, 'running'])

                # Insert span using direct SQL
                now = datetime.utcnow()
                insert_query = """
                    INSERT INTO agent.spans (
                        span_id, trace_id, parent_span_id, span_type, name, start_time,
                        input, model, provider, metadata, status
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """
                await self.db.execute(insert_query, [
                    span_id, actual_trace_id, parent_span_id, span_type, name, now.isoformat(),
                    json.dumps(input_data) if input_data else None,
                    model, provider,
                    json.dumps(metadata) if metadata else None,
                    'running'
                ])

            logger.info(f"[TraceWriter] Wrote span_start: {span_id[:8]} ({span_type})")
            return True

        except Exception as e:
            logger.error(f"[TraceWriter] Failed to write span_start: {e}", exc_info=True)
            return False

    async def write_span_end(
        self,
        span_id: str,
        duration_ms: int,
        output_data: Optional[Dict[str, Any]] = None,
        status: str = 'success',
        error: Optional[str] = None,
        tokens_used: Optional[Dict] = None
    ) -> bool:
        """
        Update span with completion data

        Args:
            span_id: Span identifier to update
            duration_ms: Duration in milliseconds
            output_data: Output data as dict
            status: Status (success, error)
            error: Error message if status=error
            tokens_used: Token usage stats (for model calls)

        Returns:
            bool: Success status
        """
        try:
            now = datetime.utcnow()
            update_data = {
                'end_time': now.isoformat(),
                'duration_ms': duration_ms,
                'status': status
            }

            if output_data:
                update_data['output'] = output_data

            if error:
                update_data['error'] = error

            if tokens_used:
                update_data['tokens_used'] = tokens_used

            async with self.db:
                # Build UPDATE query
                set_clauses = []
                params = []
                param_idx = 1

                for key, value in update_data.items():
                    set_clauses.append(f"{key} = ${param_idx}")
                    params.append(value)
                    param_idx += 1

                # Add span_id as last parameter
                params.append(span_id)

                query = f"""
                    UPDATE agent.spans
                    SET {', '.join(set_clauses)}
                    WHERE span_id = ${param_idx}
                """

                await self.db.execute(query, params)

            logger.info(f"[TraceWriter] Wrote span_end: {span_id[:8]} ({status}, {duration_ms}ms)")
            return True

        except Exception as e:
            logger.error(f"[TraceWriter] Failed to write span_end: {e}", exc_info=True)
            return False

    async def write_span_complete(
        self,
        span_id: str,
        trace_id: str,
        session_id: str,
        span_type: str,
        name: str,
        duration_ms: int,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        status: str = 'success',
        parent_span_id: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        tokens_used: Optional[Dict] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Write complete span in one operation (for short-lived spans)

        Returns:
            bool: Success status
        """
        try:
            actual_trace_id = trace_id or session_id

            async with self.db:
                # Ensure trace exists (INSERT ON CONFLICT DO NOTHING)
                query = """
                    INSERT INTO agent.traces (trace_id, session_id, status, start_time)
                    VALUES ($1, $2, $3, NOW())
                    ON CONFLICT (trace_id) DO NOTHING
                """
                await self.db.execute(query, [actual_trace_id, session_id, 'running'])

                # Insert span using direct SQL
                now = datetime.utcnow()
                insert_query = """
                    INSERT INTO agent.spans (
                        span_id, trace_id, parent_span_id, span_type, name,
                        start_time, end_time, duration_ms,
                        input, output, model, provider, tokens_used, status, error, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                """
                await self.db.execute(insert_query, [
                    span_id, actual_trace_id, parent_span_id, span_type, name,
                    now.isoformat(), now.isoformat(), duration_ms,
                    json.dumps(input_data) if input_data else None,
                    json.dumps(output_data) if output_data else None,
                    model, provider,
                    json.dumps(tokens_used) if tokens_used else None,
                    status, error,
                    json.dumps(metadata) if metadata else None
                ])

            logger.info(f"[TraceWriter] Wrote complete span: {span_id[:8]} ({span_type}, {duration_ms}ms)")
            return True

        except Exception as e:
            logger.error(f"[TraceWriter] Failed to write complete span: {e}", exc_info=True)
            return False

    async def close(self):
        """Close the database connection"""
        if self.db:
            await self.db.close()


# Global instance
_trace_writer = None

def get_trace_writer() -> TraceWriter:
    """Get global trace writer instance"""
    global _trace_writer
    if _trace_writer is None:
        from isa_agent_sdk.core.config import settings
        _trace_writer = TraceWriter(host=settings.postgres_grpc_host, port=settings.postgres_grpc_port)
    return _trace_writer
