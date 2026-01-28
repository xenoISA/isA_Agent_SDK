#!/usr/bin/env python3
"""
Tool Node - Clean tool execution with MCP integration

Professional tool execution node that:
1. Executes tool calls from reason_node
2. Uses base_node MCP integration  
3. Handles plan_tool detection for autonomous execution
4. Provides clean streaming feedback
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt, Command

from isa_agent_sdk.agent_types.agent_state import AgentState
from .base_node import BaseNode
from .utils.tool_hil_detector import ToolHILDetector
from .utils.tool_hil_router import ToolHILRouter
from isa_agent_sdk.services.human_in_the_loop import get_hil_service
from isa_agent_sdk.utils.logger import agent_logger

hil_service = get_hil_service()
logger = agent_logger  # Use centralized logger for Loki integration


class ToolNode(BaseNode):
    """Professional tool execution node with MCP integration"""

    def __init__(self):
        super().__init__("ToolNode")
        print("🔧🔧🔧 ToolNode CONSTRUCTOR called! 🔧🔧🔧")

        # HIL detection and routing (simple, follows architecture)
        self.hil_detector = ToolHILDetector()
        self.hil_router = ToolHILRouter()

        # Tool profiler for execution time tracking
        from isa_agent_sdk.services.auto_detection.tool_profiler import get_tool_profiler
        try:
            self.profiler = get_tool_profiler()
            self.logger.info("✅ Tool profiler initialized with Consul service discovery")
        except Exception as e:
            self.logger.warning(f"⚠️ Tool profiler initialization failed: {e}, profiling disabled")
            self.profiler = None
    
    async def _execute_logic(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """
        Execute tools from reason_node
        
        Args:
            state: Current agent state with messages containing tool_calls
            config: Runtime config with MCP service context
            
        Returns:
            Updated state with tool execution results
        """
        print("🔧🔧🔧 ToolNode._execute_logic started! 🔧🔧🔧")
        
        self.logger.info("🔧 ToolNode executing tools")
        
        messages = state.get("messages", [])
        
        if not messages:
            self.logger.warning("No messages for tool execution")
            return {"next_action": "end"}
        
        last_message = messages[-1]
        tool_calls = getattr(last_message, 'tool_calls', [])
        
        if not tool_calls:
            self.logger.warning("No tool calls found in last message")
            return {"next_action": "end"}
        
        # Extract all tool names for batch authorization check
        tool_info_list = []
        for i, tool_call in enumerate(tool_calls):
            tool_name, tool_args, tool_call_id = self._extract_tool_info(tool_call, i)
            tool_info_list.append((tool_name, tool_args, tool_call_id))
        
        tool_names = [info[0] for info in tool_info_list]
        
        # Batch security check using MCP (nodes layer responsibility)
        self.logger.info(f"🔒 Checking security levels for {len(tool_names)} tools: {tool_names}")
        high_security_tools = await self._check_tool_security_batch(tool_names, config)

        # Request authorization via HIL service if needed (service layer responsibility)
        if high_security_tools:
            self.logger.info(f"🚨 {len(high_security_tools)} high-security tools require authorization")
            user_id = self.get_user_id(config)
            authorized = await hil_service.request_batch_tool_authorization(
                tools=high_security_tools,
                user_id=user_id,
                node_source="tool_node"
            )
            if not authorized:
                self.logger.warning("❌ Tool authorization denied by user")
                return {
                    "messages": [ToolMessage(
                        content="Tool authorization denied by user",
                        tool_call_id=tool_info_list[0][2]
                    )],
                    "next_action": "end"
                }
            self.logger.info(f"✅ Tool authorization approved for {len(high_security_tools)} tools")

        # PHASE 1: Detect long-running tasks and offer HIL choice
        long_running_detected = self._detect_long_running_task(tool_info_list)
        if long_running_detected:
            execution_choice = await self._offer_execution_choice(long_running_detected, config)

            if execution_choice == "background":
                # PHASE 2: Queue background job and return job_id
                job_result = await self._queue_background_job(tool_info_list, state, config)
                tool_messages = [ToolMessage(
                    content=json.dumps(job_result),
                    tool_call_id=tool_info_list[0][2]  # Use first tool's call_id
                )]
                return {
                    "messages": tool_messages,
                    "next_action": "call_model",
                    "background_job": job_result
                }
            elif execution_choice == "quick":
                # Limit to fast execution (3 sources max, 30s timeout)
                tool_info_list = self._optimize_for_quick_execution(tool_info_list)

        # Collect all tool messages
        tool_messages = []

        # Execute each tool call (authorization already handled)
        for i, (tool_name, tool_args, tool_call_id) in enumerate(tool_info_list):
            self.stream_tool(tool_name, f"Starting execution ({i+1}/{len(tool_calls)})")

            # Structured logging: tool start
            tool_start = time.time()
            session_id = state.get("session_id", "unknown")
            self.logger.info(
                f"tool_call_start | "
                f"session_id={session_id} | "
                f"tool={tool_name} | "
                f"args_length={len(str(tool_args))}"
            )

            # Execute tool via base_node MCP integration (authorization already done)
            try:
                # Log tool execution start
                args_preview = str(tool_args)[:200]
                self.logger.info(
                    f"[PHASE:NODE_TOOL] tool_execution_start | "
                    f"session_id={session_id} | "
                    f"tool={tool_name} | "
                    f"args={args_preview}"
                )

                # Create progress callback to stream MCP progress messages as tool.executing events
                def progress_callback(progress_message: str) -> None:
                    """
                    Callback to handle MCP progress messages from SSE stream
                    Converts MCP progress notifications to tool.executing events
                    """
                    # Stream progress message via stream_tool (which sends type="progress" events)
                    # stream_processor.py will convert these to tool.executing events
                    self.stream_tool(tool_name, progress_message)
                    
                    # Also log for debugging
                    self.logger.debug(
                        f"MCP Progress | "
                        f"session_id={session_id} | "
                        f"tool={tool_name} | "
                        f"message={progress_message[:100]}"
                    )

                # Normalize tool arguments for MCP compatibility
                # Some tools expect string parameters but LLM may generate lists
                normalized_args = self._normalize_tool_args(tool_name, tool_args)

                result = await self.mcp_call_tool(tool_name, normalized_args, config, progress_callback=progress_callback)

                # Structured logging: tool success
                duration_ms = int((time.time() - tool_start) * 1000)
                result_preview = str(result)[:300] if result else "None"

                # Enhanced logging for web_crawl to track timeout issues
                if tool_name == "web_crawl":
                    url = tool_args.get("url", "unknown")
                    self.logger.info(
                        f"[PHASE:NODE_TOOL] web_crawl_complete | "
                        f"session_id={session_id} | "
                        f"url={url} | "
                        f"duration_ms={duration_ms} | "
                        f"result_length={len(str(result))}"
                    )

                self.logger.info(
                    f"[PHASE:NODE_TOOL] tool_execution_complete | "
                    f"session_id={session_id} | "
                    f"tool={tool_name} | "
                    f"status=success | "
                    f"duration_ms={duration_ms} | "
                    f"result_length={len(str(result))} | "
                    f"result_preview='{result_preview}'"
                )

                # Record execution time to profiler for auto-detection
                if self.profiler:
                    try:
                        self.profiler.record_execution(
                            tool_name=tool_name,
                            execution_time_ms=duration_ms,
                            tool_args=tool_args,
                            session_id=session_id,
                            success=True
                        )
                    except Exception as profiler_error:
                        self.logger.warning(f"Failed to record tool execution to profiler: {profiler_error}")

            except Exception as e:
                # Structured logging: tool error
                duration_ms = int((time.time() - tool_start) * 1000)
                error_msg = f"Tool execution failed: {str(e)}"
                self.logger.error(
                    f"tool_call_error | "
                    f"session_id={session_id} | "
                    f"tool={tool_name} | "
                    f"duration_ms={duration_ms} | "
                    f"error={type(e).__name__} | "
                    f"message={str(e)[:200]}",
                    exc_info=True
                )

                # Record failed execution to profiler
                if self.profiler:
                    try:
                        self.profiler.record_execution(
                            tool_name=tool_name,
                            execution_time_ms=duration_ms,
                            tool_args=tool_args,
                            session_id=session_id,
                            success=False
                        )
                    except Exception as profiler_error:
                        self.logger.warning(f"Failed to record failed execution to profiler: {profiler_error}")

                # Create error tool message
                error_message = ToolMessage(
                    content=error_msg,
                    tool_call_id=tool_call_id
                )
                tool_messages.append(error_message)
                self.stream_tool(tool_name, f"Failed - {str(e)}")
                continue  # Skip to next tool
            
            # Check if MCP tool returned HIL request (using new detector)
            # DEBUG: Log result type and preview for HIL detection debugging
            self.logger.debug(f"[HIL_DEBUG] Tool {tool_name} result type: {type(result)}, preview: {str(result)[:200]}")

            if self.hil_detector.is_hil_response(result):
                self.logger.info(f"🤖 Detected HIL response from {tool_name}")

                # Detect HIL type and extract data
                hil_type = self.hil_detector.detect_hil_type(result)
                hil_data = self.hil_detector.extract_hil_data(result)

                self.logger.info(f"📋 HIL Type: {hil_type}")

                # Route to HIL service (this will call interrupt() which raises Interrupt exception)
                # DON'T catch the Interrupt exception - let it propagate to LangGraph!
                human_response = await self.hil_router.route(
                    hil_type=hil_type,
                    hil_data=hil_data,
                    tool_name=tool_name,
                    tool_args=tool_args
                )

                # This line is only reached if interrupt() was resumed
                self.logger.info(f"✅ HIL completed: {hil_type}")
                result = str(human_response) if human_response is not None else "No human response received"

            # Process normal tool results (only reached if no HIL interrupt)
            # All HIL responses (including plan reviews) are handled by the HIL detector above
            try:
                # Create tool message
                tool_message = ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call_id
                )
                tool_messages.append(tool_message)
                
                self.stream_tool(tool_name, f"Completed - {len(str(result))} chars result")
                
            except Exception as e:
                # Check if this is a LangGraph interrupt (not a real error)
                # Interrupts contain 'Interrupt(' in their string representation
                error_str = str(e)
                if 'Interrupt(' in error_str and 'resumable=True' in error_str:
                    # This is a LangGraph interrupt - re-raise it to pause graph execution
                    self.logger.info(f"HIL interrupt detected for {tool_name} - re-raising to pause graph")
                    raise
                
                # Real error - handle normally
                error_msg = f"Tool execution failed: {str(e)}"
                self.logger.error(f"Tool {tool_name} failed: {e}")
                
                # Create error tool message
                error_message = ToolMessage(
                    content=error_msg,
                    tool_call_id=tool_call_id
                )
                tool_messages.append(error_message)
                
                self.stream_tool(tool_name, f"Failed - {str(e)}")
        
        # Return the state update including tool messages
        # The add_messages reducer will automatically append tool responses to existing conversation
        result = {
            "messages": tool_messages,
            "next_action": "call_model"
        }
        
        # Override next action if already set by plan_tool
        if "next_action" in state and state["next_action"] == "agent_executor":
            result["next_action"] = "agent_executor"
            
        return result
    
    def _extract_tool_info(self, tool_call, index: int) -> tuple:
        """Extract tool information from tool_call object or dict"""
        if hasattr(tool_call, 'name'):
            # LangChain ToolCall object
            return (
                tool_call.name,
                tool_call.args,
                getattr(tool_call, 'id', f'call_{index}')
            )
        elif isinstance(tool_call, dict):
            # Dictionary format
            return (
                tool_call.get('name', 'unknown'),
                tool_call.get('args', {}),
                tool_call.get('id', f'call_{index}')
            )
        else:
            # Fallback for unknown formats
            return ('unknown', {}, f'call_{index}')
    
    def _normalize_tool_args(self, tool_name: str, tool_args: dict) -> dict:
        """
        Normalize tool arguments for MCP compatibility

        Some MCP tools expect string parameters (e.g., CSV or JSON strings)
        but LLMs may generate list/dict types. This method converts them.

        Args:
            tool_name: Name of the tool
            tool_args: Original tool arguments from LLM

        Returns:
            Normalized tool arguments
        """
        normalized = tool_args.copy()

        # Plan tools: convert available_tools list to string
        if tool_name in ['create_execution_plan', 'replan_execution', 'adjust_execution_plan']:
            if 'available_tools' in normalized:
                tools = normalized['available_tools']
                if isinstance(tools, list):
                    # Convert list to JSON string for MCP
                    normalized['available_tools'] = json.dumps(tools)
                    self.logger.debug(f"[ARG_NORMALIZE] Converted available_tools list to JSON string for {tool_name}")

        return normalized

    async def _check_tool_security_batch(
        self,
        tool_names: List[str],
        config: RunnableConfig
    ) -> List[Tuple[str, str]]:
        """
        Check security levels for multiple tools and return high-security ones

        Uses MCP to query tool security levels.
        Returns tools with HIGH or CRITICAL security levels.

        Args:
            tool_names: List of tool names to check
            config: Runtime config

        Returns:
            List of (tool_name, security_level) tuples for HIGH/CRITICAL tools
        """
        high_security_tools = []

        try:
            security_data = await self.mcp_get_tool_security_levels(config)
            tools_info = security_data.get("tools", {})

            for tool_name in tool_names:
                tool_data = tools_info.get(tool_name, {})
                security_level = tool_data.get('security_level', 'LOW')

                if security_level in ['HIGH', 'CRITICAL']:
                    high_security_tools.append((tool_name, security_level))
                    self.logger.info(f"[ToolSecurity] {tool_name}: {security_level} - requires authorization")
                else:
                    self.logger.debug(f"[ToolSecurity] {tool_name}: {security_level} - no authorization needed")

        except Exception as e:
            self.logger.error(f"[ToolSecurity] Failed to check security levels: {e}")
            # Fail-safe: require authorization for all tools when check fails
            self.logger.warning("[ToolSecurity] Defaulting to HIGH for safety")
            high_security_tools = [(name, "HIGH") for name in tool_names]

        return high_security_tools
    
    # HIL detection and routing moved to ToolHILDetector and ToolHILRouter
    # (See app/nodes/utils/tool_hil_detector.py and tool_hil_router.py)
    
    # Removed all mode-related methods - keeping tool execution simple

    # =============================================================================
    # PHASE 1 & 2: LONG-RUNNING TASK DETECTION + BACKGROUND JOB QUEUE
    # =============================================================================

    def _detect_long_running_task(self, tool_info_list: List[Tuple[str, dict, str]]) -> Optional[dict]:
        """
        Detect if the tool execution will be long-running using BackgroundHILDetector

        Returns:
            Full execution_choice data structure from BackgroundHILDetector if long-running, None otherwise

            Data structure includes:
            - estimated_time_seconds: float
            - tool_count: int
            - task_type: str (web_crawling, web_searching, mixed)
            - task_composition: dict
            - recommendation: str (quick, comprehensive, background)
            - options: list of dicts
            - prompt: str
            - context: dict
        """
        # Use BackgroundHILDetector for intelligent detection
        from .utils.background_hil_detector import BackgroundHILDetector

        # Debug logging
        tool_names = [tool_name for tool_name, _, _ in tool_info_list]
        self.logger.info(f"🔍 [BackgroundHIL] Checking {len(tool_info_list)} tools: {tool_names}")

        detector = BackgroundHILDetector()
        execution_choice_data = detector.should_offer_execution_choice(tool_info_list)

        if execution_choice_data:
            self.logger.info(
                f"🔔 [BackgroundHIL] Long-running task detected! "
                f"Estimated: {execution_choice_data['estimated_time_seconds']:.1f}s, "
                f"Tools: {execution_choice_data['tool_count']}, "
                f"Type: {execution_choice_data['task_type']}, "
                f"Recommendation: {execution_choice_data['recommendation']}"
            )
        else:
            self.logger.info(f"✅ [BackgroundHIL] No long-running task detected (below threshold)")

        # Return the full data structure (needed for proper execution_choice HIL)
        return execution_choice_data

    async def _offer_execution_choice(self, task_info: dict, config: RunnableConfig) -> str:
        """
        Offer user choice: quick (limited), comprehensive (wait), or background (async)

        Uses proper execution_choice HIL method (5th HIL method).

        Args:
            task_info: Full execution_choice data from BackgroundHILDetector containing:
                - prompt: User-facing prompt
                - estimated_time_seconds: Estimated execution time
                - tool_count: Number of tools
                - task_type: Type (web_crawling, web_searching, mixed)
                - recommendation: Recommended choice
                - options: List of option dicts
                - context: Additional context
            config: RunnableConfig with user_id

        Returns:
            "quick" | "comprehensive" | "background"
        """
        from isa_agent_sdk.services.human_in_the_loop import get_hil_service
        hil_service = get_hil_service()

        # Determine scenario based on task_type
        task_type = task_info.get("task_type", "unknown")
        scenario_map = {
            "web_crawling": "long_running_web_crawl",
            "web_searching": "long_running_web_search",
            "mixed": "long_running_mixed",
            "unknown": "long_running_task"
        }
        scenario = scenario_map.get(task_type, "long_running_task")

        self.logger.info(
            f"🔔 [ExecutionChoice] Triggering HIL for scenario '{scenario}' - "
            f"{task_info['tool_count']} tools, {task_info['estimated_time_seconds']:.1f}s, "
            f"recommendation: {task_info['recommendation']}"
        )

        try:
            # Use proper execution_choice HIL method (creates type="execution_choice" event)
            choice = await hil_service.request_execution_choice(
                scenario=scenario,
                data=task_info,  # Full data structure from BackgroundHILDetector
                user_id=self.get_user_id(config),
                node_source="tool_node"
            )

            self.logger.info(f"✅ [ExecutionChoice] User chose {choice.upper()} execution mode")
            return choice

        except Exception as e:
            self.logger.error(f"❌ [ExecutionChoice] HIL failed: {e}, defaulting to comprehensive", exc_info=True)
            return "comprehensive"

    def _optimize_for_quick_execution(self, tool_info_list: List[Tuple[str, dict, str]]) -> List[Tuple[str, dict, str]]:
        """
        Optimize tool list for quick execution (limit web_crawls to 3)
        """
        web_crawls = [t for t in tool_info_list if t[0] == "web_crawl"]
        other_tools = [t for t in tool_info_list if t[0] != "web_crawl"]

        # Keep only first 3 web_crawls (usually the most reliable sources)
        limited_crawls = web_crawls[:3]

        self.logger.info(f"🚀 Quick mode: Limited from {len(web_crawls)} to {len(limited_crawls)} web_crawls")

        return other_tools + limited_crawls

    async def _queue_background_job(self, tool_info_list: List[Tuple[str, dict, str]], state: dict, config: RunnableConfig) -> dict:
        """
        Queue tools as background job using NATS + Redis and return job_id

        Returns:
            dict with job_id and status URL
        """
        import uuid
        from datetime import datetime

        job_id = f"job_{uuid.uuid4().hex[:12]}"
        session_id = state.get("session_id", "unknown")
        user_id = self.get_user_id(config)

        # Prepare job data
        job_data = {
            "job_id": job_id,
            "session_id": session_id,
            "user_id": user_id,
            "tools": [
                {
                    "tool_name": t[0],
                    "tool_args": t[1],
                    "tool_call_id": t[2]
                }
                for t in tool_info_list
            ],
            "created_at": datetime.now().isoformat(),
            "status": "queued"
        }

        try:
            # Use new NATS + Redis background job system
            from isa_agent_sdk.services.background_jobs import submit_tool_execution_task

            # Submit task to NATS queue with high priority
            task_result = await submit_tool_execution_task(
                task_data={
                    "job_id": job_id,
                    "session_id": session_id,
                    "user_id": user_id,
                    "tools": job_data["tools"],
                    "config": self._serialize_config(config)
                },
                priority="high",  # Tool execution is high priority
                max_retries=2
            )

            self.logger.info(
                f"background_job_queued | "
                f"job_id={job_id} | "
                f"session_id={session_id} | "
                f"nats_task_id={task_result['task_id']} | "
                f"tool_count={len(tool_info_list)} | "
                f"queue=nats"
            )

            return {
                "status": "queued",
                "job_id": job_id,
                "task_id": task_result["task_id"],
                "message": f"Background job queued with {len(tool_info_list)} tools",
                "poll_url": f"/api/v1/jobs/{job_id}",
                "sse_url": f"/api/v1/jobs/{job_id}/stream",
                "estimated_completion": f"{len(tool_info_list) * 12}s",
                "queue_system": "nats+redis"
            }

        except ImportError as e:
            self.logger.warning(f"NATS background job system not available: {e}")

            # Fallback: Execute synchronously
            return {
                "status": "fallback_to_sync",
                "job_id": job_id,
                "message": "Background queue unavailable, will execute synchronously",
                "note": "NATS+Redis system not available"
            }

        except Exception as queue_error:
            self.logger.error(f"Failed to queue background job: {queue_error}")

            # Final fallback: return synchronous execution
            return {
                "status": "fallback_to_sync",
                "job_id": job_id,
                "message": f"Background queue error: {str(queue_error)}, executing synchronously",
                "note": "Queue submission failed"
            }

    def _serialize_config(self, config: RunnableConfig) -> dict:
        """
        Serialize RunnableConfig for background job

        Args:
            config: Runtime configuration

        Returns:
            Serializable config dict
        """
        try:
            configurable = config.get("configurable", {})

            # Extract serializable fields
            serialized = {
                "user_id": configurable.get("user_id"),
                "session_id": configurable.get("session_id"),
                "thread_id": configurable.get("thread_id"),
                # MCP service info will be reconstructed by worker
                "mcp_service_name": "mcp_service"
            }

            return serialized

        except Exception as e:
            self.logger.error(f"Failed to serialize config: {e}")
            return {}