#!/usr/bin/env python3
"""
Chat Service - Process chat requests
"""

import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, AsyncIterator, cast, List
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from isa_agent_sdk.graphs.smart_agent_graph import SmartAgentGraphBuilder
from isa_agent_sdk.graphs.graph_registry_with_auth import get_graph_registry
from isa_agent_sdk.graphs.utils.context_init import prepare_runtime_context, create_initial_state
from isa_agent_sdk.graphs.utils.context_update import update_context_after_chat
from isa_agent_sdk.components import SessionService
from isa_agent_sdk.components.billing_service import create_billing_handler, billing_service
from isa_agent_sdk.components.storage_service import get_storage_service
from .base_service import BaseService
from .utils.stream_processor import StreamProcessor
from .hardware_service import get_hardware_service
from isa_agent_sdk.core.config import settings
from isa_agent_sdk.agent_types.event_types import EventEmitter
from isa_agent_sdk.clients.pool_manager_client import get_pool_manager_client


class ChatService(BaseService):
    """Chat service for processing chat requests"""
    
    def __init__(self, session_service: SessionService, config: Optional[Dict] = None):
        super().__init__(config)
        self.session_service = session_service
        self.graph_builder = None
        self.graph = None
    
    async def service_init(self):
        """Initialize chat service components"""
        self.graph_builder = SmartAgentGraphBuilder(config=self.config)
        self.logger.info("ChatService initialized")

    async def _execute_proxy_mode(
        self,
        user_input: str,
        session_id: str,
        user_id: str,
        agent_config: Dict[str, Any],
        auth_token: str = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Execute chat in proxy mode - delegate to pool_manager VM

        Instead of running the agent locally, this:
        1. Acquires an agent VM from pool_manager
        2. Sends the query to the isolated VM
        3. Streams responses back to the client
        4. Releases the VM when done
        """
        pool_client = await get_pool_manager_client()
        instance_id = None

        try:
            # Yield start event
            yield EventEmitter.session_start(
                session_id,
                user_id=user_id,
                execution_mode="proxy"
            ).to_dict()

            # Acquire agent VM
            yield EventEmitter.system_info(
                session_id,
                "Acquiring isolated execution environment...",
                event_type="vm_acquiring"
            ).to_dict()

            acquire_result = await pool_client.acquire_agent_vm(
                agent_config=agent_config,
                session_id=session_id,
                user_id=user_id
            )
            instance_id = acquire_result.get("instance_id")

            if not instance_id:
                raise RuntimeError("Failed to acquire agent VM")

            yield EventEmitter.system_info(
                session_id,
                "Execution environment ready",
                event_type="vm_acquired",
                instance_id=instance_id
            ).to_dict()

            # Stream query to VM
            context = {
                "auth_token": auth_token,
                "agent_config": agent_config
            }

            async for chunk in pool_client.stream_query(
                instance_id=instance_id,
                query=user_input,
                context=context
            ):
                # Transform pool_manager events to our event format
                chunk_type = chunk.get("type", "unknown")

                if chunk_type == "token":
                    yield EventEmitter.content_token(
                        session_id,
                        chunk.get("content", "")
                    ).to_dict()
                elif chunk_type == "tool_call":
                    yield EventEmitter.tool_call(
                        session_id,
                        tool_name=chunk.get("tool_name"),
                        tool_args=chunk.get("tool_args", {}),
                        tool_id=chunk.get("tool_id")
                    ).to_dict()
                elif chunk_type == "tool_result":
                    yield EventEmitter.tool_result(
                        session_id,
                        tool_name=chunk.get("tool_name"),
                        result=chunk.get("result"),
                        tool_id=chunk.get("tool_id")
                    ).to_dict()
                elif chunk_type == "error":
                    yield EventEmitter.system_error(
                        session_id,
                        chunk.get("message", "Unknown error")
                    ).to_dict()
                elif chunk_type == "complete":
                    # Stream complete
                    pass
                else:
                    # Pass through other events
                    yield chunk

            # Send completion
            yield EventEmitter.session_end(
                session_id,
                completion_message="Proxy execution completed"
            ).to_dict()

        except Exception as e:
            self.logger.error(f"Proxy execution error: {e}")
            yield EventEmitter.system_error(
                session_id,
                f"Proxy execution error: {str(e)}",
                error_type=type(e).__name__
            ).to_dict()
        finally:
            # Release VM back to pool
            if instance_id:
                try:
                    await pool_client.release_vm(instance_id)
                    self.logger.info(f"Released VM {instance_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to release VM {instance_id}: {e}")
    
    
    async def execute(
        self,
        user_input: str,
        session_id: str,
        user_id: str = "anonymous",
        prompt_name: str = None,
        prompt_args: Dict[str, Any] = None,
        auth_token: str = None,
        graph_type: Optional[str] = None,  # NEW: explicit graph selection
        auto_select_graph: bool = True,  # NEW: auto-select based on task
        confidence_threshold: float = 0.7,
        proactive_predictions: Optional[Dict[str, Any]] = None,
        device_context: Optional[Dict[str, Any]] = None,
        media_files: Optional[list] = None,
        sensor_data: Optional[Dict[str, Any]] = None,
        trigger_type: str = "user_request",
        output_format: Optional[str] = None,  # "json" for structured output
        file_urls: Optional[List[Dict[str, Any]]] = None  # NEW: uploaded file URLs and metadata
    ) -> AsyncIterator[Dict[str, Any]]:
        """Execute chat processing using StreamProcessor"""
        try:
            execute_start = time.time()
            self.logger.info(
                f"[PHASE:SERVICE] execute_start | "
                f"session_id={session_id} | "
                f"user_id={user_id} | "
                f"user_input='{user_input[:100]}{'...' if len(user_input) > 100 else ''}' | "
                f"graph_type={graph_type} | "
                f"auto_select={auto_select_graph} | "
                f"prompt_name={prompt_name} | "
                f"execution_mode={settings.execution_mode}"
            )

            # Check execution mode - delegate to pool_manager if proxy mode
            if settings.execution_mode == "proxy":
                self.logger.info(
                    f"[PHASE:SERVICE] proxy_mode_detected | "
                    f"session_id={session_id} | "
                    f"delegating to pool_manager"
                )
                # Build agent config for the VM
                agent_config = {
                    "graph_type": graph_type,
                    "auto_select_graph": auto_select_graph,
                    "prompt_name": prompt_name,
                    "prompt_args": prompt_args,
                    "output_format": output_format,
                    "device_context": device_context,
                    "file_urls": file_urls
                }
                async for event in self._execute_proxy_mode(
                    user_input=user_input,
                    session_id=session_id,
                    user_id=user_id,
                    agent_config=agent_config,
                    auth_token=auth_token
                ):
                    yield event
                return

            # Local mode - continue with normal execution
            init_start = time.time()
            await self.ensure_initialized()
            init_duration = int((time.time() - init_start) * 1000)
            self.logger.info(
                f"service_init_complete | "
                f"session_id={session_id} | "
                f"user_id={user_id} | "
                f"duration_ms={init_duration}"
            )

            # Process hardware request if device_context is provided
            hardware_start = time.time()
            hardware_service = get_hardware_service()
            request_data = {
                "device_context": device_context,
                "media_files": media_files,
                "sensor_data": sensor_data,
                "trigger_type": trigger_type
            }
            
            # Check if this is a hardware request and process it
            processed_request = await hardware_service.process_hardware_request(
                message=user_input,
                user_id=user_id,
                request_data=request_data
            )
            
            if processed_request:
                # Update request data with processed hardware context
                request_data.update(processed_request)
                self.logger.info(f"Hardware request processed for device: {device_context.get('device_id', 'unknown') if device_context else 'none'}")

            hardware_duration = int((time.time() - hardware_start) * 1000)
            self.logger.info(
                f"hardware_processing_complete | "
                f"session_id={session_id} | "
                f"user_id={user_id} | "
                f"is_hardware={bool(processed_request)} | "
                f"duration_ms={hardware_duration}"
            )

            # Select and build appropriate graph
            graph_registry = get_graph_registry()
            graph_selection_start = time.time()

            # Determine which graph to use
            selection_method_start = time.time()
            if graph_type:
                # User explicitly specified a graph type
                self.logger.info(
                    f"graph_selection | "
                    f"session_id={session_id} | "
                    f"user_id={user_id} | "
                    f"mode=explicit | "
                    f"graph_type={graph_type}"
                )
                selected_graph = await graph_registry.build_graph(
                    user_id=user_id,
                    graph_type=graph_type
                )
                if not selected_graph:
                    # User doesn't have permission, fall back to default
                    yield EventEmitter.system_warning(
                        session_id,
                        f"Access denied to {graph_type}. Using default graph.",
                        graph_type=graph_type,
                        user_id=user_id
                    ).to_dict()
                    selected_graph = await graph_registry.build_graph(
                        user_id=user_id,
                        graph_type="graph:default"
                    )
            elif auto_select_graph:
                # Auto-select based on task
                context = {
                    "device_context": device_context,
                    "prompt_name": prompt_name
                }
                auto_select_start = time.time()
                selected_type = await graph_registry.select_graph_for_task(
                    user_id=user_id,
                    task_description=user_input,
                    context=context
                )
                auto_select_duration = int((time.time() - auto_select_start) * 1000)
                self.logger.info(
                    f"[PHASE:SERVICE] graph_selection | "
                    f"session_id={session_id} | "
                    f"user_id={user_id} | "
                    f"mode=auto | "
                    f"selected_graph_type={selected_type} | "
                    f"select_duration_ms={auto_select_duration}"
                )
                # DEBUG: Write to file for easier debugging
                with open("/tmp/chat_timing_debug.log", "a") as f:
                    f.write(f"{session_id} | select_graph: {auto_select_duration}ms\n")
                build_start = time.time()
                selected_graph = await graph_registry.build_graph(
                    user_id=user_id,
                    graph_type=selected_type
                )
                build_duration = int((time.time() - build_start) * 1000)
                self.logger.info(
                    f"graph_build_after_select | "
                    f"session_id={session_id} | "
                    f"user_id={user_id} | "
                    f"duration_ms={build_duration}"
                )
                # DEBUG: Write to file
                with open("/tmp/chat_timing_debug.log", "a") as f:
                    f.write(f"{session_id} | build_graph: {build_duration}ms\n")
            else:
                # Use user's active graph or default
                selected_graph = await graph_registry.get_active_graph(
                    user_id=user_id
                )
            
            # Fall back to default builder if registry fails
            if not selected_graph:
                self.logger.warning(
                    f"graph_fallback | "
                    f"session_id={session_id} | "
                    f"user_id={user_id} | "
                    f"reason=registry_failed"
                )
                if not self.graph:
                    self.graph = self.graph_builder.build_graph()
                selected_graph = self.graph

            # Use the selected graph
            self.graph = selected_graph

            # Log graph selection timing
            graph_selection_duration = int((time.time() - graph_selection_start) * 1000)
            self.logger.info(
                f"graph_ready | "
                f"session_id={session_id} | "
                f"user_id={user_id} | "
                f"duration_ms={graph_selection_duration}"
            )
            
            # Create initial state (output_format now passed via config instead)
            state_start = time.time()
            initial_state = create_initial_state(user_input)
            state_duration = int((time.time() - state_start) * 1000)
            self.logger.info(
                f"initial_state_created | "
                f"session_id={session_id} | "
                f"user_id={user_id} | "
                f"duration_ms={state_duration}"
            )

            # Get runtime context with optional template processing
            context_start = time.time()
            runtime_context = await prepare_runtime_context(
                user_id=user_id,
                thread_id=session_id,
                session_service=self.session_service,
                mcp_url=self.config.get("mcp_url", settings.resolved_mcp_server_url),
                user_query=user_input,
                prompt_name=prompt_name,
                prompt_args=prompt_args
            )
            context_duration = int((time.time() - context_start) * 1000)

            # Debug: Check available_tools immediately after prepare_runtime_context
            available_tools_debug = runtime_context.get('available_tools', [])
            self.logger.info(
                f"[DEBUG_CHAT_SERVICE] runtime_context_prepared | "
                f"session_id={session_id} | "
                f"available_tools_count={len(available_tools_debug)} | "
                f"tools={[t.get('name', 'unknown') for t in available_tools_debug[:10]]}"
            )

            self.logger.info(
                f"runtime_context_complete | "
                f"session_id={session_id} | "
                f"user_id={user_id} | "
                f"duration_ms={context_duration}"
            )

            # Create billing handler
            billing_start = time.time()
            billing_handler = create_billing_handler(user_id, session_id, auth_token)
            billing_duration = int((time.time() - billing_start) * 1000)
            self.logger.info(
                f"billing_handler_created | "
                f"session_id={session_id} | "
                f"user_id={user_id} | "
                f"duration_ms={billing_duration}"
            )

            # Configure with recursion_limit for LangGraph loop control and billing
            config_start = time.time()
            config = {
                "configurable": {
                    "thread_id": session_id,
                    "output_format": output_format,  # Pass output_format via config instead of state
                    **runtime_context
                },
                "callbacks": [billing_handler],
                "recursion_limit": self.config.get("max_graph_iterations", 50)  # Official pattern
            }
            config_duration = int((time.time() - config_start) * 1000)
            self.logger.info(
                f"config_prepared | "
                f"session_id={session_id} | "
                f"user_id={user_id} | "
                f"duration_ms={config_duration}"
            )

            # Log total initialization time before streaming starts
            total_init_duration = int((time.time() - execute_start) * 1000)
            self.logger.info(
                f"pre_stream_complete | "
                f"session_id={session_id} | "
                f"user_id={user_id} | "
                f"total_init_ms={total_init_duration} | "
                f"breakdown_ms={{init:{init_duration},hardware:{hardware_duration},graph:{graph_selection_duration},state:{state_duration},context:{context_duration},billing:{billing_duration},config:{config_duration}}}"
            )

            # Send start event
            yield EventEmitter.session_start(
                session_id,
                user_id=user_id,
                graph_type=graph_type or "auto",
                init_duration_ms=total_init_duration
            ).to_dict()

            # Send detailed context preparation events (5 core components)
            tools_count = len(runtime_context.get('available_tools', []))
            prompts_count = len(runtime_context.get('default_prompts', {}))
            resources_count = len(runtime_context.get('default_resources', []))
            memory_length = len(runtime_context.get('memory_context', ''))
            files_count = runtime_context.get('file_count', 0)
            
            # Context loading started
            yield EventEmitter.context_loading(session_id).to_dict()
            
            # Individual components ready
            yield EventEmitter.context_tools_ready(session_id, tools_count).to_dict()
            yield EventEmitter.context_prompts_ready(session_id, prompts_count).to_dict()
            yield EventEmitter.context_resources_ready(session_id, resources_count).to_dict()
            yield EventEmitter.context_memory_ready(session_id, memory_length).to_dict()
            yield EventEmitter.context_knowledge_ready(session_id, files_count).to_dict()
            
            # Context complete
            yield EventEmitter.context_complete(
                session_id,
                duration_ms=context_duration
            ).to_dict()

            first_event_time = time.time()
            self.logger.info(
                f"first_event_sent | "
                f"session_id={session_id} | "
                f"user_id={user_id} | "
                f"time_from_execute_start_ms={int((first_event_time - execute_start) * 1000)}"
            )

            # IMPORTANT: With LangGraph checkpointer + add_messages reducer:
            # - DO NOT manually load/merge checkpoint state
            # - The graph will automatically load state using thread_id from config
            # - Just pass the NEW message in initial_state
            # - The add_messages reducer will append to existing messages
            checkpoint_load_start = time.time()

            # Log checkpoint info for debugging (optional check)
            try:
                checkpointer = getattr(self.graph, 'checkpointer', None)
                if checkpointer:
                    thread_config = {"configurable": {"thread_id": session_id}}
                    state_snapshot = await checkpointer.aget_tuple(thread_config)

                    if state_snapshot:
                        # Extract existing state from checkpoint for logging only
                        checkpoint_data = state_snapshot.checkpoint
                        if 'channel_values' in checkpoint_data:
                            existing_state = checkpoint_data['channel_values']
                            existing_messages_count = len(existing_state.get('messages', []))

                            self.logger.info(
                                f"[PHASE:SERVICE] checkpoint_found | "
                                f"session_id={session_id} | "
                                f"existing_messages={existing_messages_count} | "
                                f"has_summary={bool(existing_state.get('summary'))} | "
                                f"will_auto_merge=True"
                            )
                        else:
                            self.logger.warning(
                                f"checkpoint_no_channel_values | "
                                f"session_id={session_id}"
                            )
                    else:
                        self.logger.info(
                            f"[PHASE:SERVICE] checkpoint_empty | "
                            f"session_id={session_id} | "
                            f"new_session=True"
                        )
                else:
                    self.logger.warning(
                        f"no_checkpointer | "
                        f"session_id={session_id}"
                    )
            except Exception as e:
                self.logger.warning(
                    f"checkpoint_check_error | "
                    f"session_id={session_id} | "
                    f"error={type(e).__name__} | "
                    f"message={str(e)[:200]}"
                )

            checkpoint_load_duration = int((time.time() - checkpoint_load_start) * 1000)
            self.logger.info(
                f"checkpoint_check_complete | "
                f"session_id={session_id} | "
                f"duration_ms={checkpoint_load_duration} | "
                f"note=graph_will_auto_load_via_thread_id"
            )

            # Use StreamProcessor for unified streaming logic
            stream_processor = StreamProcessor(logger=self.logger)
            final_state = None
            stream_completed = False

            graph_stream_start = time.time()
            self.logger.info(
                f"[PHASE:SERVICE] graph_stream_start | "
                f"session_id={session_id} | "
                f"user_id={user_id} | "
                f"time_from_start_event_ms={int((graph_stream_start - first_event_time) * 1000)}"
            )

            # Log graph configuration
            self.logger.info(
                f"[PHASE:SERVICE] graph_stream_config | "
                f"session_id={session_id} | "
                f"user_id={user_id} | "
                f"recursion_limit={config.get('recursion_limit')} | "
                f"thread_id={config.get('configurable', {}).get('thread_id')}"
            )

            # Check checkpoint state BEFORE graph execution
            try:
                pre_state = await self.graph.aget_state(cast(RunnableConfig, config))
                pre_messages = pre_state.values.get("messages", []) if pre_state and pre_state.values else []
                self.logger.info(
                    f"[CHECKPOINT:PRE-EXEC] State before graph execution | "
                    f"session_id={session_id} | "
                    f"thread_id={config.get('configurable', {}).get('thread_id')} | "
                    f"messages_count={len(pre_messages)} | "
                    f"next_nodes={pre_state.next if pre_state else 'N/A'} | "
                    f"checkpoint_id={pre_state.config.get('configurable', {}).get('checkpoint_id') if pre_state else 'N/A'}"
                )
                # Log last few messages
                if pre_messages:
                    for idx, msg in enumerate(pre_messages[-3:]):
                        msg_type = getattr(msg, "type", "unknown")
                        msg_content = str(getattr(msg, "content", ""))[:100]
                        self.logger.info(
                            f"[CHECKPOINT:PRE-EXEC] Recent message {idx+1} | "
                            f"type={msg_type} | "
                            f"content={msg_content}"
                        )
            except Exception as e:
                self.logger.warning(f"[CHECKPOINT:PRE-EXEC] Failed to get pre-execution state: {e}")

            first_graph_event_received = False
            # Use actual EventType enum values for counting
            event_counts = {
                'content.thinking': 0,     # Reasoning/thinking tokens from DeepSeek R1
                'content.token': 0,        # Response tokens
                'tool.call': 0,            # Tool calls
                'content.complete': 0,     # Complete content
                'node.exit': 0             # Node updates
            }
            async for event in stream_processor.process_stream(
                self.graph,
                initial_state,
                cast(RunnableConfig, config),
                session_id,
                stream_context="chat"
            ):
                # Log timing for first event from graph
                if not first_graph_event_received:
                    first_graph_event_time = time.time()
                    self.logger.info(
                        f"[PHASE:SERVICE] first_graph_event_received | "
                        f"session_id={session_id} | "
                        f"user_id={user_id} | "
                        f"event_type={event.get('type')} | "
                        f"time_from_graph_start_ms={int((first_graph_event_time - graph_stream_start) * 1000)}"
                    )
                    first_graph_event_received = True

                # Count event types for debugging
                event_type = event.get("type", "unknown")
                if event_type in event_counts:
                    event_counts[event_type] += 1

                # Track execution state
                if event.get("type") == "paused":
                    self.logger.info(
                        f"[PHASE:SERVICE] stream_paused | "
                        f"session_id={session_id} | "
                        f"user_id={user_id} | "
                        f"event_counts={event_counts}"
                    )
                    # Stream was paused due to interrupt - don't process billing/memory
                    yield event
                    return
                elif event.get("type") == "session.end":
                    stream_completed = True
                    # Capture final_state from event metadata if available
                    final_state = event.get("metadata", {}).get("final_state")
                    self.logger.info(
                        f"[PHASE:SERVICE] stream_end_received | "
                        f"session_id={session_id} | "
                        f"user_id={user_id} | "
                        f"event_counts={event_counts} | "
                        f"final_state_captured={bool(final_state)}"
                    )
                    # Don't yield the end event yet - we'll send our own after business logic
                    continue
                else:
                    yield event
            
            # Handle business logic after streaming completes (if not interrupted)
            if stream_completed:
                # Check checkpoint state AFTER graph execution
                try:
                    post_state = await self.graph.aget_state(cast(RunnableConfig, config))
                    post_messages = post_state.values.get("messages", []) if post_state and post_state.values else []
                    self.logger.info(
                        f"[CHECKPOINT:POST-EXEC] State after graph execution | "
                        f"session_id={session_id} | "
                        f"thread_id={config.get('configurable', {}).get('thread_id')} | "
                        f"messages_count={len(post_messages)} | "
                        f"next_nodes={post_state.next if post_state else 'N/A'} | "
                        f"checkpoint_id={post_state.config.get('configurable', {}).get('checkpoint_id') if post_state else 'N/A'}"
                    )
                    # Log last few messages
                    if post_messages:
                        for idx, msg in enumerate(post_messages[-3:]):
                            msg_type = getattr(msg, "type", "unknown")
                            msg_content = str(getattr(msg, "content", ""))[:100]
                            self.logger.info(
                                f"[CHECKPOINT:POST-EXEC] Recent message {idx+1} | "
                                f"type={msg_type} | "
                                f"content={msg_content}"
                            )
                except Exception as e:
                    self.logger.warning(f"[CHECKPOINT:POST-EXEC] Failed to get post-execution state: {e}")

                # Extract final state for memory and billing processing
                try:
                    # Get the final state from the last graph execution
                    # We need to get this from the graph's checkpointer or final execution state
                    state_snapshot = None
                    try:
                        # Try to get the latest state from checkpointer if available
                        checkpointer = getattr(self.graph, 'checkpointer', None)
                        if checkpointer:
                            # Get the latest checkpoint
                            thread_config = {"configurable": {"thread_id": session_id}}
                            state_snapshot = await checkpointer.aget_tuple(thread_config)
                            if state_snapshot:
                                # Support both old (.values) and new (.checkpoint) API
                                final_state = getattr(state_snapshot, 'checkpoint', getattr(state_snapshot, 'values', None))
                    except Exception as e:
                        self.logger.debug(f"Could not get final state from checkpointer: {e}")
                    
                    # Handle memory update if we have final state and MCP service
                    if final_state and runtime_context.get("mcp_service"):
                        try:
                            memory_start = time.time()
                            # Extract final state values for memory storage
                            state_values = list(final_state.values())[0] if isinstance(final_state, dict) and final_state else final_state

                            # Send memory storing started event
                            self.logger.info(f"DEBUG: Sending memory_storing event for session {session_id}")
                            yield EventEmitter.memory_storing(session_id).to_dict()

                            memory_result = await update_context_after_chat(
                                session_id=session_id,
                                user_id=user_id,
                                final_state=state_values,
                                mcp_service=runtime_context["mcp_service"]
                            )

                            memory_duration = int((time.time() - memory_start) * 1000)
                            self.logger.info(
                                f"memory_update_complete | "
                                f"session_id={session_id} | "
                                f"user_id={user_id} | "
                                f"memories_stored={memory_result.get('memories_stored', 0)} | "
                                f"duration_ms={memory_duration}"
                            )

                            # Send memory stored event
                            if memory_result.get("memory_updated", False):
                                yield EventEmitter.memory_stored(
                                    session_id,
                                    memories_count=memory_result.get('memories_stored', 0)
                                ).to_dict()
                            
                            # Send memory curation events if performed
                            if memory_result.get("curation_performed", False):
                                curation_type = memory_result.get("curation_results", {}).get("curation_type", "auto")
                                yield EventEmitter.memory_curating(session_id, curation_type).to_dict()
                                yield EventEmitter.memory_curated(session_id, curation_type).to_dict()

                        except Exception as e:
                            self.logger.error(
                                f"memory_update_error | "
                                f"session_id={session_id} | "
                                f"user_id={user_id} | "
                                f"error={type(e).__name__} | "
                                f"message={str(e)[:200]}",
                                exc_info=True
                            )
                            # Don't fail the whole chat if memory update fails
                    
                    # Finalize billing
                    try:
                        billing_start = time.time()
                        billing_result = await billing_service.finalize_billing(billing_handler)
                        billing_duration = int((time.time() - billing_start) * 1000)

                        self.logger.info(
                            f"billing_complete | "
                            f"session_id={session_id} | "
                            f"user_id={user_id} | "
                            f"model_calls={billing_result.model_calls} | "
                            f"tool_calls={billing_result.tool_calls} | "
                            f"total_credits={billing_result.total_credits} | "
                            f"credits_remaining={billing_result.credits_remaining} | "
                            f"duration_ms={billing_duration}"
                        )

                        # Stream billing result
                        yield EventEmitter.system_billing(
                            session_id,
                            billing_result.total_credits,
                            success=billing_result.success,
                            model_calls=billing_result.model_calls,
                            tool_calls=billing_result.tool_calls,
                            credits_remaining=billing_result.credits_remaining,
                            error_message=billing_result.error_message
                        ).to_dict()

                    except Exception as e:
                        self.logger.error(
                            f"billing_error | "
                            f"session_id={session_id} | "
                            f"user_id={user_id} | "
                            f"error={type(e).__name__} | "
                            f"message={str(e)[:200]}",
                            exc_info=True
                        )
                        # Stream billing error
                        yield EventEmitter.system_error(
                            session_id,
                            f"Billing error: {str(e)}",
                            success=False,
                            error=str(e),
                            event_type="billing_error"
                        ).to_dict()
                    
                    # Format hardware response if this is a hardware request
                    if processed_request and processed_request.get("is_hardware_request"):
                        try:
                            # Extract the final response content
                            response_content = ""
                            if final_state and hasattr(final_state, 'get'):
                                messages = final_state.get('messages', [])
                                if messages:
                                    last_message = messages[-1]
                                    response_content = getattr(last_message, 'content', str(last_message))
                            
                            # Format response for device
                            device_ctx = processed_request.get("device_context")
                            if device_ctx and response_content:
                                hardware_response = await hardware_service.format_response_for_device(
                                    response_content=response_content,
                                    device_context=device_ctx,
                                    additional_data=processed_request
                                )
                                
                                # Stream hardware response
                                yield EventEmitter.system_info(
                                    session_id,
                                    "Hardware response formatted",
                                    text_response=hardware_response.text_response,
                                    audio_url=hardware_response.audio_url,
                                    display_data=hardware_response.display_data,
                                    device_commands=[cmd.__dict__ for cmd in hardware_response.device_commands] if hardware_response.device_commands else [],
                                    automation_triggered=hardware_response.automation_triggered,
                                    event_type="hardware_response"
                                ).to_dict()
                                
                        except Exception as e:
                            self.logger.warning(f"Hardware response formatting failed: {e}")
                    
                    # Send completion event
                    yield EventEmitter.session_end(
                        session_id,
                        completion_message="Chat processing completed"
                    ).to_dict()
                
                except Exception as e:
                    self.logger.warning(f"Post-processing error: {e}")
                    # Still send completion even if post-processing fails
                    yield EventEmitter.session_end(
                        session_id,
                        completion_message="Chat processing completed with post-processing issues"
                    ).to_dict()
                
        except Exception as e:
            import traceback
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
            self.logger.error(f"Chat processing error: {e}")
            self.logger.error(f"Error details: {error_details}")
            yield EventEmitter.system_error(
                session_id,
                f"Chat processing error: {str(e) or 'Unknown error'}",
                debug=error_details,
                error_type=type(e).__name__
            ).to_dict()
    
    async def resume_execution(
        self,
        session_id: str,
        user_id: str,
        resume_value: Any = None,
        prompt_name: str = None,
        prompt_args: Dict[str, Any] = None,
        auth_token: str = None,
        confidence_threshold: float = 0.7,
        proactive_predictions: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Resume interrupted execution using StreamProcessor
        
        Args:
            session_id: Session/thread ID to resume
            user_id: User ID for context
            resume_value: Value to pass to resume (e.g., authorization result)
            prompt_name: Optional prompt for re-initialization
            prompt_args: Optional prompt arguments
            
        Yields:
            Stream of execution events
        """
        try:
            await self.ensure_initialized()
            
            # Build graph if needed
            if not self.graph:
                self.graph = self.graph_builder.build_graph()
            
            # Get runtime context for resume
            runtime_context = await prepare_runtime_context(
                user_id=user_id,
                thread_id=session_id,
                session_service=self.session_service,
                mcp_url=self.config.get("mcp_url", settings.resolved_mcp_server_url),
                user_query="",  # No new query for resume
                prompt_name=prompt_name,
                prompt_args=prompt_args
            )
            
            # Create billing handler for resume
            billing_handler = create_billing_handler(user_id, session_id, auth_token)
            
            # Configure with same thread_id to resume from checkpoint
            config = {
                "configurable": {
                    "thread_id": session_id,
                    **runtime_context
                },
                "callbacks": [billing_handler],
                "recursion_limit": self.config.get("max_graph_iterations", 50)
            }
            
            # If we have resume_value, invoke with Command(resume=...)
            if resume_value is not None:
                initial_input = Command(resume=resume_value)
            else:
                initial_input = None
            
            # Send resume start event
            yield EventEmitter.session_start(
                session_id,
                resumed=True,
                event_type="resume_start"
            ).to_dict()
            
            # Use StreamProcessor for resume streaming
            stream_processor = StreamProcessor(logger=self.logger)
            final_state = None
            stream_completed = False
            
            # Use the specialized resume stream method if available
            if hasattr(stream_processor, 'process_resume_stream'):
                async for event in stream_processor.process_resume_stream(
                    self.graph,
                    initial_input,
                    cast(RunnableConfig, config),
                    session_id
                ):
                    # Track execution state
                    if event.get("type") == "interrupt":
                        # Another interrupt occurred during resume
                        yield event
                        return
                    elif event.get("type") == "end":
                        stream_completed = True
                        # Don't yield the end event yet - we'll send our own after business logic
                        continue
                    else:
                        # Add resumed flag to all events
                        event["resumed"] = True
                        yield event
            else:
                # Fallback to regular process_stream with resume context
                async for event in stream_processor.process_stream(
                    self.graph,
                    initial_input,
                    cast(RunnableConfig, config),
                    session_id,
                    stream_context="resume"
                ):
                    # Track execution state
                    if event.get("type") == "paused":
                        # Another interrupt occurred during resume
                        yield event
                        return
                    elif event.get("type") == "end":
                        stream_completed = True
                        # Don't yield the end event yet - we'll send our own after business logic
                        continue
                    else:
                        # Add resumed flag to all events
                        event["resumed"] = True
                        yield event
            
            # Handle business logic after streaming completes (if not interrupted)
            if stream_completed:
                try:
                    # Get the final state from the last graph execution (same as execute method)
                    state_snapshot = None
                    try:
                        # Try to get the latest state from checkpointer if available
                        checkpointer = getattr(self.graph, 'checkpointer', None)
                        if checkpointer:
                            # Get the latest checkpoint
                            thread_config = {"configurable": {"thread_id": session_id}}
                            state_snapshot = await checkpointer.aget_tuple(thread_config)
                            if state_snapshot:
                                # Support both old (.values) and new (.checkpoint) API
                                final_state = getattr(state_snapshot, 'checkpoint', getattr(state_snapshot, 'values', None))
                    except Exception as e:
                        self.logger.debug(f"Could not get final state from checkpointer: {e}")
                    
                    # Handle memory update if we have final state and MCP service
                    if final_state and runtime_context.get("mcp_service"):
                        try:
                            state_values = list(final_state.values())[0] if isinstance(final_state, dict) and final_state else final_state
                            
                            memory_result = await update_context_after_chat(
                                session_id=session_id,
                                user_id=user_id,
                                final_state=state_values,
                                mcp_service=runtime_context["mcp_service"]
                            )
                            
                            yield EventEmitter.system_info(
                                session_id,
                                f"Memory updated: {memory_result.get('memories_stored', 0)} memories stored",
                                memories_stored=memory_result.get('memories_stored', 0),
                                memory_result=memory_result,
                                resumed=True,
                                event_type="memory_update"
                            ).to_dict()
                            
                        except Exception as e:
                            self.logger.warning(f"Resume memory update failed: {e}")
                    
                    # Finalize billing for resume
                    try:
                        billing_result = await billing_service.finalize_billing(billing_handler)
                        
                        yield EventEmitter.system_billing(
                            session_id,
                            billing_result.total_credits,
                            success=billing_result.success,
                            model_calls=billing_result.model_calls,
                            tool_calls=billing_result.tool_calls,
                            credits_remaining=billing_result.credits_remaining,
                            error_message=billing_result.error_message,
                            resumed=True
                        ).to_dict()
                        
                    except Exception as e:
                        self.logger.warning(f"Resume billing finalization failed: {e}")
                        yield EventEmitter.system_error(
                            session_id,
                            f"Resume billing error: {str(e)}",
                            success=False,
                            error=str(e),
                            resumed=True,
                            event_type="billing_error"
                        ).to_dict()
                    
                    # Send resume completion event
                    yield EventEmitter.session_end(
                        session_id,
                        completion_message="Resume execution completed"
                    ).to_dict()
                    
                except Exception as e:
                    self.logger.warning(f"Resume post-processing error: {e}")
                    # Still send completion even if post-processing fails
                    yield EventEmitter.session_end(
                        session_id,
                        completion_message="Resume execution completed with post-processing issues"
                    ).to_dict()
                
        except Exception as e:
            self.logger.error(f"Resume execution error: {e}")
            yield EventEmitter.system_error(
                session_id,
                f"Resume execution error: {str(e)}",
                resumed=True,
                error_type=type(e).__name__
            ).to_dict()
    
    async def resume_execution_non_streaming(
        self,
        thread_id: str,
        action: str = "continue",
        resume_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Resume execution non-streaming - wrapper for API compatibility"""
        # Note: action parameter is kept for API compatibility but not currently used
        _ = action  # Suppress unused parameter warning

        try:
            # Collect all events from streaming method
            events = []
            async for event in self.resume_execution(
                session_id=thread_id,
                user_id="system",
                resume_value=resume_data
            ):
                events.append(event)

            # Return success response
            return {
                "success": True,
                "thread_id": thread_id,
                "message": f"Resume completed with {len(events)} events",
                "next_step": "completed"
            }
        except Exception as e:
            return {
                "success": False,
                "thread_id": thread_id,
                "message": f"Resume failed: {str(e)}",
                "next_step": "error"
            }

    async def resume_execution_stream(
        self,
        thread_id: str,
        action: str = "continue",
        resume_data: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Resume execution with streaming - wrapper for existing method"""
        # Note: action parameter is kept for API compatibility but not currently used
        _ = action  # Suppress unused parameter warning

        async for event in self.resume_execution(
            session_id=thread_id,
            user_id="system",
            resume_value=resume_data
        ):
            yield event

    async def get_execution_status(self, thread_id: str) -> Dict[str, Any]:
        """Get execution status for thread"""
        try:
            # Get checkpointer and check for state
            from isa_agent_sdk.services.persistence import get_durable_service
            durable_service = get_durable_service()
            
            # For now, return basic status - could be enhanced with real checkpoint data
            return {
                "status": "ready",
                "thread_id": thread_id,
                "current_node": "unknown",
                "interrupts": [],
                "checkpoints": 0,
                "durable": durable_service.get_service_info()["features"]["durable_execution"]
            }
        except Exception as e:
            return {
                "status": "error",
                "thread_id": thread_id,
                "error": str(e)
            }
    
    async def get_execution_history(
        self, 
        thread_id: str, 
        limit: int = 50
    ) -> Dict[str, Any]:
        """Get execution history for thread"""
        try:
            # For now, return empty history - could be enhanced with real checkpoint data
            return {
                "history": [],
                "total": 0,
                "thread_id": thread_id,
                "limit": limit
            }
        except Exception as e:
            return {
                "error": str(e),
                "thread_id": thread_id
            }
    
    async def rollback_execution(
        self,
        thread_id: str,
        checkpoint_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Rollback execution to checkpoint"""
        try:
            # For now, return success - could be enhanced with real rollback logic
            return {
                "success": True,
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id or "latest",
                "message": "Rollback completed successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "thread_id": thread_id,
                "error": str(e)
            }
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information"""
        try:
            from isa_agent_sdk.services.persistence import get_durable_service
            from isa_agent_sdk.services.human_in_the_loop import get_hil_service
            
            durable_service = get_durable_service()
            hil_service = get_hil_service()
            
            return {
                "interrupt_features": {
                    "human_in_loop": True,
                    "approval_workflow": True,
                    "tool_authorization": True,
                    "total_interrupts": hil_service.get_interrupt_stats()["total"]
                },
                "graph_info": {
                    "nodes": 4,
                    "durable": durable_service.get_service_info()["features"]["durable_execution"],
                    "checkpoints": True,
                    "environment": durable_service.environment
                }
            }
        except Exception as e:
            return {
                "error": str(e),
                "interrupt_features": {},
                "graph_info": {}
            }