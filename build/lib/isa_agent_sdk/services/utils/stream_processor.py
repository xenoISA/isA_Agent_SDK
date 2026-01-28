#!/usr/bin/env python3
"""
Stream Processor - Common streaming logic for LangGraph execution
Handles streaming events from LangGraph execution without trace dependencies
"""

import json
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, AsyncIterator, cast
from langchain_core.messages import convert_to_messages
from langchain_core.runnables import RunnableConfig

from isa_agent_sdk.utils.logger import api_logger
from isa_agent_sdk.agent_types.event_types import EventType, EventEmitter, EventData, get_event_type


class StreamProcessor:
    """Handles common LangGraph streaming logic for different stream modes"""
    
    def __init__(self, logger=None):
        self.logger = logger or api_logger
        self.thinking_token_count = 0
        self.response_token_count = 0
    
    async def process_stream(
        self,
        graph,
        initial_state: Dict[str, Any],
        config: RunnableConfig,
        session_id: str,
        stream_context: str = "chat"  # "chat", "template", "resume"
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process LangGraph stream with proper modes: ["updates", "messages", "custom"]
        
        Args:
            graph: LangGraph instance
            initial_state: Initial state for graph execution
            config: Runnable config for the graph
            session_id: Session ID for events
            stream_context: Context string for event messages ("chat", "template", "resume")
        
        Yields:
            Streaming event dictionaries
        """
        try:
            # Reset token counts for this stream
            self.thinking_token_count = 0
            self.response_token_count = 0

            graph_start = time.time()
            final_state = None
            event_count = 0

            self.logger.info(
                f"[PHASE:STREAM] graph_execution_start | "
                f"session_id={session_id} | "
                f"context={stream_context} | "
                f"stream_modes=['updates','messages','custom','values']"
            )

            # Use proper stream modes: ["updates", "messages", "custom", "values"]
            # "values" mode gives us the full state after each step (for debugging state.messages)
            async for mode, chunk in graph.astream(
                initial_state,
                config=config,
                stream_mode=["updates", "messages", "custom", "values"]
            ):
                event_count += 1

                # Handle different stream modes
                if mode == "messages":
                    async for event in self._handle_message_mode(chunk, session_id, stream_context):
                        yield event

                elif mode == "custom":
                    async for event in self._handle_custom_mode(chunk, session_id, stream_context):
                        yield event

                elif mode == "values":
                    # NEW: Handle full state values for debugging
                    async for event in self._handle_values_mode(chunk, session_id, stream_context):
                        yield event
                    # Store the final state from values
                    if isinstance(chunk, dict):
                        final_state = chunk

                else:  # mode == "updates"
                    # Handle state updates and check for interrupts
                    interrupt_occurred = False
                    async for event in self._handle_updates_mode(chunk, session_id, stream_context):
                        yield event
                        # Check if this was an interrupt event that should stop the stream
                        if event.get("type") == "paused":
                            interrupt_occurred = True

                    # If interrupt occurred, exit
                    if interrupt_occurred:
                        return

            # Process final state and return billing/completion info
            graph_duration = int((time.time() - graph_start) * 1000)

            self.logger.info(
                f"[PHASE:STREAM] graph_execution_complete | "
                f"session_id={session_id} | "
                f"context={stream_context} | "
                f"total_events={event_count} | "
                f"thinking_tokens={self.thinking_token_count} | "
                f"response_tokens={self.response_token_count} | "
                f"duration_ms={graph_duration}"
            )

            async for event in self._handle_completion(final_state, session_id, stream_context):
                yield event

        except Exception as e:
            graph_duration = int((time.time() - graph_start) * 1000) if 'graph_start' in locals() else 0

            self.logger.error(
                f"graph_execution_error | "
                f"session_id={session_id} | "
                f"context={stream_context} | "
                f"duration_ms={graph_duration} | "
                f"error={type(e).__name__} | "
                f"message={str(e)[:200]}",
                exc_info=True
            )
            
            yield EventEmitter.system_error(
                session_id,
                f"{stream_context.capitalize()} processing error: {str(e)}"
            ).to_dict()
    
    async def _handle_message_mode(
        self,
        chunk,
        session_id: str,
        stream_context: str
    ) -> AsyncIterator[Dict[str, Any]]:
        """Handle message events from LangGraph stream"""
        try:
            messages = convert_to_messages([chunk[0]]) if isinstance(chunk, tuple) and len(chunk) > 0 else convert_to_messages([chunk])
            for message in messages:
                msg_type = type(message).__name__
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    # Tool call message
                    tool_calls = message.tool_calls
                    yield EventEmitter.tool_call(
                        session_id,
                        tool_calls[0].get('name', 'unknown') if tool_calls else 'unknown',
                        tool_calls[0].get('args', {}) if tool_calls else {}
                    ).to_dict()
                elif hasattr(message, 'tool_call_id'):
                    # Tool result message
                    content = str(getattr(message, 'content', ''))
                    preview = content[:100] + "..." if len(content) > 100 else content
                    yield EventEmitter.tool_result(
                        session_id,
                        msg_type,
                        preview
                    ).to_dict()
                else:
                    # Regular message - show content
                    content = str(getattr(message, 'content', ''))
                    if content and 'AI' in msg_type:
                        yield EventEmitter.content_complete(
                            session_id,
                            content,
                            msg_type
                        ).to_dict()
        except Exception as e:
            # Fallback to simple representation
            context_prefix = f"{stream_context.capitalize()} " if stream_context != "chat" else ""
            yield EventEmitter.system_info(
                session_id,
                f"📨 {context_prefix}Message: {str(chunk)[:100]}...",
                error=str(e),
                chunk_preview=str(chunk)[:200]
            ).to_dict()
    
    async def _handle_custom_mode(
        self,
        chunk,
        session_id: str,
        stream_context: str
    ) -> AsyncIterator[Dict[str, Any]]:
        """Handle custom streaming events from LangGraph"""
        # Log all custom events for debugging
        chunk_start = time.time()

        # Task planning events
        if isinstance(chunk, dict) and "task_planning" in chunk:
            task_info = chunk["task_planning"]
            task_count = task_info.get("task_count", 0)
            exec_mode = task_info.get("execution_mode", "sequential")
            status = task_info.get("status", "unknown")

            yield EventEmitter.task_plan(
                session_id,
                {"task_count": task_count, "execution_mode": exec_mode, "status": status}
            ).to_dict()

        # Task status updates (HIL approval/rejection)
        elif isinstance(chunk, dict) and "task_status_update" in chunk:
            status_info = chunk["task_status_update"]
            status = status_info.get("status", "unknown")
            message = status_info.get("message", "")
            task_count = status_info.get("task_count", 0)

            status_icons = {
                "approved": "✅",
                "modified": "📝",
                "rejected": "❌"
            }
            icon = status_icons.get(status, "🔄")

            yield EventEmitter.task_progress(
                session_id,
                f"{icon} {message}",
                status=status,
                task_count=task_count,
                message=message
            ).to_dict()

        # Task step start events
        elif isinstance(chunk, dict) and "task_step_start" in chunk:
            task_info = chunk["task_step_start"]
            task_index = task_info.get("task_index", 0)
            task_title = task_info.get("task_title", "Unknown Task")
            total_tasks = task_info.get("total_tasks", 0)

            yield EventEmitter.task_start(
                session_id,
                f"🚀 [{task_index}/{total_tasks}] {task_title}",
                task_index=task_index,
                task_title=task_title,
                total_tasks=total_tasks
            ).to_dict()

        # Task step complete events
        elif isinstance(chunk, dict) and "task_step_complete" in chunk:
            task_info = chunk["task_step_complete"]
            task_index = task_info.get("task_index", 0)
            task_title = task_info.get("task_title", "Unknown Task")
            total_tasks = task_info.get("total_tasks", 0)
            status = task_info.get("status", "unknown")

            status_icons = {
                "success": "✅",
                "failed": "❌"
            }
            icon = status_icons.get(status, "🔄")

            yield EventEmitter.task_complete(
                session_id,
                f"{icon} [{task_index}/{total_tasks}] {task_title}",
                task_index=task_index,
                task_title=task_title,
                total_tasks=total_tasks,
                status=status
            ).to_dict()

        # Task state updates
        elif isinstance(chunk, dict) and "task_state" in chunk:
            state_info = chunk["task_state"]
            completed = state_info.get("completed_tasks", 0)
            total = state_info.get("total_tasks", 0)
            failed = state_info.get("failed_tasks", 0)
            status = state_info.get("status", "unknown")

            yield EventEmitter.task_progress(
                session_id,
                f"📊 {completed}/{total} completed, {failed} failed",
                completed_tasks=completed,
                total_tasks=total,
                failed_tasks=failed,
                status=status
            ).to_dict()
        
        elif isinstance(chunk, dict) and "tool_execution" in chunk:
            # Tool execution updates with detailed logging (unified format)
            # Supports both standard tool execution and MCP progress messages from SSE stream
            tool_info = chunk["tool_execution"]
            status = tool_info.get("status", "unknown")
            tool_name = tool_info.get("tool_name", "unknown")
            tool_count = tool_info.get("tool_count", "")
            progress = tool_info.get("progress", "")  # MCP progress message from SSE stream

            if status == "starting":
                self.logger.info(
                    f"stream_tool_start | "
                    f"session_id={session_id} | "
                    f"tool_count={tool_count}"
                )
                yield EventEmitter.tool_executing(
                    session_id,
                    "tool_execution",
                    f"Starting tool execution: {tool_count} tool(s)"
                ).to_dict()
            elif status == "executing":
                # MCP progress messages come here (from SSE stream via progress_callback)
                # Use progress message if available (from MCP), otherwise default message
                progress_message = progress if progress else f"Executing tool: {tool_name}"
                self.logger.info(
                    f"stream_tool_executing | "
                    f"session_id={session_id} | "
                    f"tool={tool_name} | "
                    f"progress={progress[:100] if progress else 'N/A'}"
                )
                yield EventEmitter.tool_executing(
                    session_id,
                    tool_name,
                    progress_message
                ).to_dict()
            elif status == "completed":
                duration_ms = tool_info.get("duration_ms", 0)
                result_size = tool_info.get("result_size", 0)
                self.logger.info(
                    f"stream_tool_completed | "
                    f"session_id={session_id} | "
                    f"tool={tool_name} | "
                    f"duration_ms={duration_ms} | "
                    f"result_size={result_size}"
                )
                yield EventEmitter.tool_result(
                    session_id,
                    tool_name,
                    f"Tool completed: {tool_name}"
                ).to_dict()
        
        elif isinstance(chunk, dict) and "reasoning_chunk" in chunk:
            # DeepSeek R1 reasoning process streaming (from show_reasoning=True)
            token = chunk["reasoning_chunk"]
            self.thinking_token_count += 1

            # Log first and periodic tokens
            if self.thinking_token_count == 1 or self.thinking_token_count % 50 == 0:
                chunk_time = int((time.time() - chunk_start) * 1000)
                self.logger.info(
                    f"[PHASE:STREAM] reasoning_token_received | "
                    f"session_id={session_id} | "
                    f"token_num={self.thinking_token_count} | "
                    f"token_length={len(token)} | "
                    f"token_preview='{token[:20]}' | "
                    f"chunk_processing_ms={chunk_time}"
                )

            # Reasoning event - for displaying DeepSeek R1 thinking process
            yield EventEmitter.content_thinking(
                session_id,
                token
            ).to_dict()

        elif isinstance(chunk, dict) and "thinking_chunk" in chunk:
            # ReasonNode thinking process streaming (normal models)
            token = chunk["thinking_chunk"]
            self.thinking_token_count += 1

            # Log first and periodic tokens
            if self.thinking_token_count == 1 or self.thinking_token_count % 50 == 0:
                chunk_time = int((time.time() - chunk_start) * 1000)
                self.logger.info(
                    f"[PHASE:STREAM] thinking_token_received | "
                    f"session_id={session_id} | "
                    f"token_num={self.thinking_token_count} | "
                    f"token_length={len(token)} | "
                    f"token_preview='{token[:20]}' | "
                    f"chunk_processing_ms={chunk_time}"
                )

            # Thinking event - for displaying reasoning process
            yield EventEmitter.content_thinking(
                session_id,
                token
            ).to_dict()

        elif isinstance(chunk, dict) and "thinking_complete" in chunk:
            # ReasonNode complete thinking content (non-streaming)
            content = chunk["thinking_complete"]
            chunk_time = int((time.time() - chunk_start) * 1000)
            
            self.logger.info(
                f"[PHASE:STREAM] thinking_complete_received | "
                f"session_id={session_id} | "
                f"content_length={len(content)} | "
                f"chunk_processing_ms={chunk_time}"
            )
            
            # Send complete thinking as a single event with timestamp
            yield EventEmitter.content_complete(
                session_id,
                content,
                "thinking_complete"
            ).to_dict()

        elif isinstance(chunk, dict) and "tool_call" in chunk:
            # ReasonNode tool call event
            tool_info = chunk["tool_call"]
            tool_name = tool_info.get("name", "unknown")
            tool_args = tool_info.get("args", {})
            tool_id = tool_info.get("id", "")
            
            chunk_time = int((time.time() - chunk_start) * 1000)
            self.logger.info(
                f"[PHASE:STREAM] tool_call_received | "
                f"session_id={session_id} | "
                f"tool_name={tool_name} | "
                f"tool_id={tool_id} | "
                f"chunk_processing_ms={chunk_time}"
            )
            
            # Send tool call event
            yield EventEmitter.tool_call(
                session_id,
                tool_name,
                tool_args
            ).to_dict()

        elif isinstance(chunk, dict) and "response_chunk" in chunk:
            # ResponseNode final response streaming
            token = chunk["response_chunk"]
            self.response_token_count += 1

            # Log first and periodic tokens
            if self.response_token_count == 1 or self.response_token_count % 20 == 0:
                chunk_time = int((time.time() - chunk_start) * 1000)
                self.logger.info(
                    f"[PHASE:STREAM] response_token_received | "
                    f"session_id={session_id} | "
                    f"token_num={self.response_token_count} | "
                    f"token_length={len(token)} | "
                    f"token_preview='{token[:20]}' | "
                    f"chunk_processing_ms={chunk_time}"
                )

            # Token event - for displaying final response
            yield EventEmitter.content_token(
                session_id,
                token
            ).to_dict()

        elif isinstance(chunk, dict) and "custom_llm_chunk" in chunk:
            # Fallback for other nodes (legacy compatibility)
            token = chunk["custom_llm_chunk"]

            # Log timing for first token
            chunk_time = int((time.time() - chunk_start) * 1000)
            self.logger.info(
                f"[TIMING] llm_token_received | "
                f"session_id={session_id} | "
                f"chunk_processing_ms={chunk_time}"
            )

            # Lightweight token event - reduce metadata overhead for better performance
            yield EventEmitter.content_token(
                session_id,
                token
            ).to_dict()

        elif isinstance(chunk, dict) and "llm_token" in chunk:
            # LLM token streaming (alternative format)
            token_info = chunk["llm_token"]
            status = token_info.get("status", "streaming")

            if status == "streaming":
                token = token_info.get("token", "")
                # Lightweight token event - reduce metadata overhead for better performance
                yield EventEmitter.content_token(
                    session_id,
                    token
                ).to_dict()
            elif status == "completed":
                total_tokens = token_info.get("total_tokens", 0)
                full_content = token_info.get("full_content", "")
                context_prefix = f"{stream_context.capitalize()} " if stream_context != "chat" else ""
                yield EventEmitter.content_complete(
                    session_id,
                    f"✅ {context_prefix}LLM completed: {total_tokens} tokens",
                    "llm_completed",
                    total_tokens=total_tokens,
                    full_content=full_content
                ).to_dict()

        elif isinstance(chunk, dict) and chunk.get("type") == "context_progress":
            # Context preparation progress (legacy support)
            stage = chunk.get("stage", "unknown")
            details = chunk.get("details", "")
            yield EventEmitter.context_loading(
                session_id,
                stage=stage,
                details=details
            ).to_dict()

        # Context preparation events
        elif isinstance(chunk, dict) and "context_loading" in chunk:
            yield EventEmitter.context_loading(session_id).to_dict()
            
        elif isinstance(chunk, dict) and "context_tools_ready" in chunk:
            tools_count = chunk["context_tools_ready"].get("count", 0)
            yield EventEmitter.context_tools_ready(
                session_id, 
                tools_count=tools_count
            ).to_dict()
            
        elif isinstance(chunk, dict) and "context_prompts_ready" in chunk:
            prompts_count = chunk["context_prompts_ready"].get("count", 0)
            yield EventEmitter.context_prompts_ready(
                session_id, 
                prompts_count=prompts_count
            ).to_dict()
            
        elif isinstance(chunk, dict) and "context_resources_ready" in chunk:
            resources_count = chunk["context_resources_ready"].get("count", 0)
            yield EventEmitter.context_resources_ready(
                session_id, 
                resources_count=resources_count
            ).to_dict()
            
        elif isinstance(chunk, dict) and "context_memory_ready" in chunk:
            memory_length = chunk["context_memory_ready"].get("length", 0)
            yield EventEmitter.context_memory_ready(
                session_id, 
                memory_length=memory_length
            ).to_dict()
            
        elif isinstance(chunk, dict) and "context_knowledge_ready" in chunk:
            files_count = chunk["context_knowledge_ready"].get("files_count", 0)
            yield EventEmitter.context_knowledge_ready(
                session_id, 
                files_count=files_count
            ).to_dict()
            
        elif isinstance(chunk, dict) and "context_complete" in chunk:
            duration_ms = chunk["context_complete"].get("duration_ms", 0)
            yield EventEmitter.context_complete(
                session_id, 
                duration_ms=duration_ms
            ).to_dict()

        # Memory events
        elif isinstance(chunk, dict) and "memory_storing" in chunk:
            yield EventEmitter.memory_storing(session_id).to_dict()
            
        elif isinstance(chunk, dict) and "memory_stored" in chunk:
            memories_count = chunk["memory_stored"].get("count", 0)
            yield EventEmitter.memory_stored(
                session_id, 
                memories_count=memories_count
            ).to_dict()
            
        elif isinstance(chunk, dict) and "memory_curating" in chunk:
            curation_type = chunk["memory_curating"].get("type", "auto")
            yield EventEmitter.memory_curating(
                session_id, 
                curation_type=curation_type
            ).to_dict()
            
        elif isinstance(chunk, dict) and "memory_curated" in chunk:
            curation_type = chunk["memory_curated"].get("type", "auto")
            yield EventEmitter.memory_curated(
                session_id, 
                curation_type=curation_type
            ).to_dict()

        else:
            # Fallback for truly unknown events (should be rare)
            context_prefix = f"{stream_context.capitalize()} " if stream_context != "chat" else ""
            self.logger.warning(f"Unknown custom event type: {chunk}")
            yield EventEmitter.system_warning(
                session_id,
                f"❓ {context_prefix}Unknown event",
                raw_chunk=str(chunk)[:200]
            ).to_dict()
    
    async def _handle_values_mode(
        self,
        chunk,
        session_id: str,
        stream_context: str
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Handle full state values from LangGraph - for debugging state.messages

        This mode streams the FULL state after each node execution, which is helpful
        for debugging what messages are in the LangGraph state at each step.
        """
        try:
            self.logger.info(
                f"[PHASE:STREAM] values_mode_received | "
                f"session_id={session_id} | "
                f"chunk_type={type(chunk).__name__} | "
                f"is_dict={isinstance(chunk, dict)}"
            )

            if isinstance(chunk, dict):
                # Extract messages from state
                messages = chunk.get("messages", [])

                self.logger.info(
                    f"[PHASE:STREAM] values_mode_processing | "
                    f"session_id={session_id} | "
                    f"messages_count={len(messages)} | "
                    f"state_keys={list(chunk.keys())}"
                )

                # Convert LangChain messages to serializable format
                serializable_messages = []
                for msg in messages:
                    msg_dict = {
                        "type": type(msg).__name__,
                        "content": str(getattr(msg, 'content', '')),
                    }

                    # Add message name (used to identify internal vs final messages)
                    if hasattr(msg, 'name') and msg.name:
                        msg_dict["name"] = msg.name

                    # Add additional_kwargs metadata
                    if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs:
                        msg_dict["additional_kwargs"] = msg.additional_kwargs

                    # Add tool calls if present
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        msg_dict["tool_calls"] = [
                            {
                                "name": tc.get("name", "unknown"),
                                "args": tc.get("args", {}),
                                "id": tc.get("id", "")
                            }
                            for tc in msg.tool_calls
                        ]

                    # Add tool_call_id for ToolMessage
                    if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                        msg_dict["tool_call_id"] = msg.tool_call_id

                    serializable_messages.append(msg_dict)

                # Get other useful state info
                next_action = chunk.get("next_action", "")
                summary = chunk.get("summary", "")

                # Emit state snapshot event for debugging
                self.logger.info(
                    f"[PHASE:STREAM] emitting_state_snapshot | "
                    f"session_id={session_id} | "
                    f"messages_count={len(serializable_messages)} | "
                    f"next_action={next_action}"
                )

                yield EventEmitter.state_snapshot(
                    session_id,
                    messages_count=len(serializable_messages),
                    messages=serializable_messages,
                    next_action=next_action,
                    has_summary=bool(summary),
                    summary_preview=summary[:100] if summary else None
                ).to_dict()

        except Exception as e:
            self.logger.warning(
                f"values_mode_error | "
                f"session_id={session_id} | "
                f"error={type(e).__name__} | "
                f"message={str(e)[:200]}"
            )
            # Don't fail the whole stream on values mode errors
            pass

    async def _handle_updates_mode(
        self,
        chunk,
        session_id: str,
        stream_context: str
    ) -> AsyncIterator[Dict[str, Any]]:
        """Handle state updates and check for interrupts"""
        if isinstance(chunk, dict):
            # Check for LangGraph interrupts first
            if "__interrupt__" in chunk:
                interrupt_data = chunk["__interrupt__"]
                
                # Extract interrupt value and make it JSON serializable
                interrupt_value = self._extract_interrupt_value(interrupt_data)
                
                # Convert interrupt data to JSON-serializable format
                serializable_data = self._serialize_interrupt_data(interrupt_value)
                
                # Send interrupt event to client
                context_prefix = f"{stream_context.capitalize()} " if stream_context != "chat" else ""
                yield EventEmitter.hil_request(
                    session_id,
                    serializable_data.get("type", "unknown"),
                    f"{context_prefix}processing paused - human input required",
                    interrupt_data=serializable_data
                ).to_dict()
                
                # LangGraph interrupt detected - execution is paused
                # Stream should end here as execution is waiting for human input
                interrupt_type = serializable_data.get("type", "unknown")
                self.logger.info(
                    f"graph_interrupted | "
                    f"session_id={session_id} | "
                    f"interrupt_type={interrupt_type}"
                )
                
                # Send a paused status and signal to exit the stream
                yield EventEmitter.session_paused(
                    session_id,
                    f"{context_prefix}execution paused - waiting for human input",
                    status="paused_for_human_input",
                    thread_id=session_id
                ).to_dict()
                
                # Return from this method to signal interrupt
                return
            
            # Show node updates
            for node_name, node_data in chunk.items():
                if node_name == "__interrupt__":
                    continue  # Skip interrupt data we handled above
                    
                if isinstance(node_data, dict):
                    # Don't show per-node credits (tracked separately in billing event)
                    next_action = node_data.get('next_action', '')
                    execution_strategy = node_data.get('execution_strategy', '')
                    messages_count = len(node_data.get('messages', []))

                    # Add context-specific metadata
                    metadata = {
                        "node_name": node_name,
                        "next_action": next_action,
                        "execution_strategy": execution_strategy,
                        "messages_count": messages_count
                    }
                    
                    # Add template-specific metadata if applicable
                    if stream_context == "template":
                        template_used = node_data.get('template_prompt') is not None
                        metadata["template_used"] = template_used
                    
                    context_prefix = f"{stream_context.capitalize()} " if stream_context != "chat" else ""
                    yield EventEmitter.node_exit(
                        session_id,
                        node_name,
                        messages_count=messages_count,
                        next_action=next_action,
                        execution_strategy=execution_strategy
                    ).to_dict()
                else:
                    context_prefix = f"{stream_context.capitalize()} " if stream_context != "chat" else ""
                    yield EventEmitter.node_exit(
                        session_id,
                        node_name,
                        node_data=str(node_data)[:200],
                        context=stream_context
                    ).to_dict()
        else:
            context_prefix = f"{stream_context.capitalize()} " if stream_context != "chat" else ""
            yield EventEmitter.system_info(
                session_id,
                f"📊 {context_prefix}Update: {str(chunk)[:100]}...",
                chunk_preview=str(chunk)[:200],
                context=stream_context
            ).to_dict()
    
    def _extract_interrupt_value(self, interrupt_data):
        """Extract interrupt value from LangGraph interrupt data structure"""
        # Handle tuple format: (Interrupt(...), resumable, ns)
        if isinstance(interrupt_data, (tuple, list)) and len(interrupt_data) > 0:
            # Extract the first element which should be the Interrupt object
            actual_interrupt = interrupt_data[0]
            if hasattr(actual_interrupt, 'value'):
                return actual_interrupt.value
            else:
                return actual_interrupt
        elif hasattr(interrupt_data, 'value'):
            return interrupt_data.value
        else:
            return interrupt_data
    
    def _serialize_interrupt_data(self, interrupt_value):
        """Convert interrupt data to JSON-serializable format"""
        try:
            # Debug: Log the interrupt structure for troubleshooting
            self.logger.info(f"🔍 Interrupt structure: {type(interrupt_value)}")
            self.logger.info(f"🔍 Has value attr: {hasattr(interrupt_value, 'value')}")
            if hasattr(interrupt_value, 'value'):
                self.logger.info(f"🔍 Value type: {type(interrupt_value.value)}")
                self.logger.info(f"🔍 Value content: {interrupt_value.value}")
            
            # Extract data from LangGraph Interrupt object  
            # Priority order: direct dict -> nested value -> string parsing
            if isinstance(interrupt_value, dict):
                # Direct dict access - best case
                return interrupt_value.copy()
            elif hasattr(interrupt_value, 'value') and isinstance(interrupt_value.value, dict):
                # Nested value dict - common case
                return interrupt_value.value.copy()
            else:
                # Fallback to string parsing and object inspection
                serializable_data = {}
                
                # Try object attributes first
                if hasattr(interrupt_value, '__dict__'):
                    for attr in ['type', 'context', 'message', 'timeout_seconds', 'urgency']:
                        if hasattr(interrupt_value, attr):
                            serializable_data[attr] = getattr(interrupt_value, attr)
                
                # If still no type, try string parsing
                if 'type' not in serializable_data:
                    str_repr = str(interrupt_value)
                    if "'type': 'ask_human'" in str_repr:
                        serializable_data['type'] = 'ask_human'
                    elif "'type': 'content_review'" in str_repr:
                        serializable_data['type'] = 'content_review'
                    elif "'type': 'tool_execution_approval'" in str_repr:
                        serializable_data['type'] = 'tool_execution_approval'
                    elif "'type': 'task_approval'" in str_repr:
                        serializable_data['type'] = 'task_approval'
                    else:
                        # Last resort: extract from full string representation
                        type_match = re.search(r"'type':\s*'([^']+)'", str_repr)
                        if type_match:
                            serializable_data['type'] = type_match.group(1)
                        else:
                            serializable_data['type'] = 'unknown'
                            self.logger.warning(f"🔍 Could not extract type from: {str_repr[:200]}...")
                
                # If we still have an empty dict, add the string representation
                if not serializable_data:
                    serializable_data = {"message": str(interrupt_value), "type": "unknown"}
                
                return serializable_data
        except Exception as e:
            # Fallback for any serialization issues
            return {"message": "Human input required", "type": "unknown", "error": str(e)}
    
    async def _handle_completion(
        self,
        final_state: Optional[Dict],
        session_id: str,
        stream_context: str
    ) -> AsyncIterator[Dict[str, Any]]:
        """Handle final state processing and billing information"""
        # Process final state and billing
        credits_used = 0
        if final_state:
            # Get credits from the final state (may be nested in node data)
            for node_name, node_data in final_state.items():
                if isinstance(node_data, dict) and 'credits_used' in node_data:
                    credits_used = max(credits_used, node_data.get('credits_used', 0))
        
        if credits_used == 0:
            credits_used = 0.1  # Default minimal credit usage
        
        # Note: Billing information is handled by ChatService to avoid duplicate events
        # StreamProcessor no longer sends billing events to prevent user confusion
        
        # Send completion event
        context_prefix = f"{stream_context.capitalize()} r" if stream_context != "chat" else "R"
        completion_message = f"🚀 {context_prefix}esponse complete"
        
        # Add context-specific completion details
        if stream_context == "template":
            completion_message = f"🚀 Template response complete"
        elif stream_context == "resume":
            completion_message = "🎉 Resumed execution completed successfully"

        yield EventEmitter.session_end(
            session_id,
            completion_message=completion_message,
            final_state=final_state
        ).to_dict()
    
    async def process_resume_stream(
        self,
        graph,
        command,
        config: RunnableConfig,
        session_id: str
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process LangGraph resume stream - specialized for resume operations
        
        Args:
            graph: LangGraph instance
            command: LangGraph Command for resumption
            config: Runnable config for the graph
            session_id: Session ID for events
        
        Yields:
            Streaming event dictionaries
        """
        try:
            # Resume execution and stream the continued results
            async for mode, chunk in graph.astream(
                command,
                config=config,
                stream_mode=["updates", "messages", "custom"]
            ):
                # Use the same streaming logic but with resume context
                if mode == "messages":
                    # Handle message events - simplified for resume
                    try:
                        from langchain_core.messages import convert_to_messages
                        messages = convert_to_messages([chunk[0]]) if isinstance(chunk, tuple) and len(chunk) > 0 else convert_to_messages([chunk])
                        for message in messages:
                            msg_type = type(message).__name__
                            if hasattr(message, 'content') and message.content:
                                content = str(getattr(message, 'content', ''))
                                if 'AI' in msg_type:
                                    yield EventEmitter.content_complete(
                                        session_id,
                                        content,
                                        msg_type
                                    ).to_dict()
                    except Exception as e:
                        yield EventEmitter.system_error(
                            session_id,
                            f"📨 Resume message: {str(chunk)[:100]}...",
                            error=str(e),
                            chunk_preview=str(chunk)[:200]
                        ).to_dict()
                
                elif mode == "custom":
                    # Handle custom streaming events
                    yield EventEmitter.system_info(
                        session_id,
                        f"🔄 Resume custom: {chunk}",
                        raw_chunk=str(chunk)[:200],
                        event_type="resume_custom"
                    ).to_dict()
                
                else:  # mode == "updates"
                    # Handle state updates - check for more interrupts
                    if isinstance(chunk, dict):
                        # Check for another interrupt
                        if "__interrupt__" in chunk:
                            yield EventEmitter.session_paused(
                                session_id,
                                "Another human input required during resume",
                                context="resume_interrupt"
                            ).to_dict()
                            return
                        
                        # Show node updates
                        for node_name, node_data in chunk.items():
                            if isinstance(node_data, dict):
                                next_action = node_data.get('next_action', '')
                                messages_count = len(node_data.get('messages', []))

                                yield EventEmitter.node_exit(
                                    session_id,
                                    node_name,
                                    messages_count=messages_count,
                                    next_action=next_action,
                                    context="resume"
                                ).to_dict()
                            else:
                                # Handle non-dict node data
                                yield EventEmitter.node_exit(
                                    session_id,
                                    node_name,
                                    node_data=str(node_data)[:200],
                                    context="resume"
                                ).to_dict()
                    else:
                        # Handle non-dict chunks
                        yield EventEmitter.system_info(
                            session_id,
                            f"📊 Resume update: {str(chunk)[:100]}...",
                            chunk_preview=str(chunk)[:200],
                            context="resume"
                        ).to_dict()
            
            # Send completion event
            yield EventEmitter.session_end(
                session_id,
                completion_message="🎉 Resumed execution completed successfully"
            ).to_dict()
            
            self.logger.info(f"🔍 Resume execution completed for session {session_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Resume stream processing error: {e}")
            
            yield EventEmitter.system_error(
                session_id,
                f"Resume execution error: {str(e)}"
            ).to_dict()