#!/usr/bin/env python3
"""
Agent Executor Node - Modern LangGraph autonomous task execution

Advanced task execution node using LangGraph patterns:
- Loop control with conditional routing
- RemainingSteps management for recursion limits
- Parallel and sequential task execution
- Intelligent termination conditions
- BaseNode inheritance with modern streaming
"""

from typing import Dict, List, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from isa_agent_sdk.agent_types.agent_state import AgentState
from .base_node import BaseNode
from isa_agent_sdk.services.human_in_the_loop import get_hil_service
from isa_agent_sdk.utils.logger import agent_logger

hil_service = get_hil_service()

logger = agent_logger  # Use centralized logger for Loki integration


class AgentExecutorNode(BaseNode):
    """
    Modern autonomous agent executor with LangGraph loop patterns
    
    Features:
    - LangGraph loop control with conditional routing
    - RemainingSteps management for safe recursion
    - Parallel task execution capabilities
    - Intelligent termination conditions
    - Task failure recovery and retry mechanisms
    """
    
    def __init__(self):
        super().__init__("AgentExecutorNode")
        
        # Execution configuration
        self.max_parallel_tasks = 3
        self.max_task_retries = 2
        self.task_timeout = 300  # 5 minutes per task
    
    async def _execute_logic(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """
        Execute autonomous tasks with unified mode support (Reactive/Collaborative/Proactive)
        
        This method uses LangGraph's conditional routing to control task execution:
        - Checks remaining steps to avoid infinite loops
        - Determines next action based on task completion status
        - Handles both sequential and parallel execution modes
        - Applies execution mode enhancements based on state flags
        
        Args:
            state: Current agent state with task information
            config: Runtime config with MCP service and context
            
        Returns:
            Updated state with execution results and next_action routing
        """
        # Get context for mode determination
        context = self.get_runtime_context(config)
        
        # Determine and apply execution mode enhancements
        execution_mode_type = self._determine_execution_mode(state, context)
        
        if execution_mode_type == "proactive":
            state = await self._apply_proactive_execution_enhancements(state, context)
            self.logger.info("AgentExecutorNode executing in PROACTIVE mode")
        elif execution_mode_type == "collaborative":
            state = await self._apply_collaborative_execution_enhancements(state, context)
            self.logger.info("AgentExecutorNode executing in COLLABORATIVE mode")
        else:
            self.logger.info("AgentExecutorNode executing in REACTIVE mode")
        
        # Get remaining steps for LangGraph loop control (official pattern)
        remaining_steps = state["remaining_steps"]
        
        # Extract task execution state first
        task_list = self._get_task_list(state)
        current_index = state.get("current_task_index") or 0  # Handle None case
        execution_mode = state.get("execution_mode") or "sequential"  # sequential or parallel
        
        # Stream execution start
        self.stream_custom({
            "agent_execution": {
                "status": "starting",
                "remaining_steps": remaining_steps
            }
        })
        
        # Stream detailed task state with task names/titles
        if task_list:
            current_task = task_list[current_index] if current_index < len(task_list) else None
            current_task_name = current_task.get("title", f"Task {current_index + 1}") if current_task else "No current task"
            
            # Get all task names for overview
            task_names = [task.get("title", f"Task {i+1}") for i, task in enumerate(task_list)]
            
            self.stream_custom({
                "task_state": {
                    "total_tasks": len(task_list),
                    "current_task_index": current_index,
                    "current_task_name": current_task_name,
                    "task_names": task_names,
                    "completed_tasks": state.get("completed_task_count", 0),
                    "failed_tasks": state.get("failed_task_count", 0),
                    "execution_mode": execution_mode,
                    "status": "executing"
                }
            })
        
        # Check termination conditions
        if not task_list:
            return self._handle_no_tasks(state)
        
        if current_index >= len(task_list):
            return self._handle_all_tasks_completed(state, task_list)
        
        # LangGraph official recursion limit pattern
        if remaining_steps <= 2:  # Leave 2 steps: one for current node, one for termination
            return self._handle_recursion_limit_reached(state, current_index, len(task_list))
        
        # Execute tasks based on mode
        if execution_mode == "parallel":
            return await self._execute_parallel_tasks(state, config, task_list, current_index)
        else:
            return await self._execute_sequential_task(state, config, task_list, current_index)
    
    def _get_task_list(self, state: AgentState) -> List[Dict[str, Any]]:
        """
        Extract task list from state using priority order
        
        Args:
            state: Current agent state
            
        Returns:
            List of task dictionaries
        """
        # Priority order: direct task_list > execution_plan.tasks > autonomous_tasks > messages
        task_sources = [
            ("task_list", state.get("task_list")),
            ("execution_plan.tasks", state.get("execution_plan", {}).get("tasks")),
            ("autonomous_tasks", state.get("autonomous_tasks")),
            ("messages", self._extract_tasks_from_messages(state.get("messages", [])))
        ]
        
        for source_name, tasks in task_sources:
            if tasks and isinstance(tasks, list) and len(tasks) > 0:
                # Validate and clean the task list
                cleaned_tasks = self._validate_and_clean_tasks(tasks)
                if cleaned_tasks:
                    self.logger.debug(f"Using {len(cleaned_tasks)} tasks from {source_name}")
                    return cleaned_tasks
        
        self.logger.warning("No valid task list found in state")
        return []
    
    def _extract_tasks_from_messages(self, messages) -> List[Dict[str, Any]]:
        """
        Extract tasks from tool call messages with simplified JSON parsing
        
        Args:
            messages: List of conversation messages
            
        Returns:
            List of task dictionaries or empty list
        """
        if not messages:
            return []
        
        import json
        
        # Look for the most recent tool call message containing tasks
        for message in reversed(messages):
            if not (hasattr(message, 'content') and hasattr(message, 'tool_call_id')):
                continue
                
            content = str(message.content)
            
            # Quick filter: skip messages that don't contain task-related keywords
            if not any(keyword in content.lower() for keyword in ['tasks', 'plan', 'execution']):
                continue
            
            # Try to extract and parse JSON
            tasks = self._parse_tasks_from_content(content)
            if tasks:
                self.logger.debug(f"Extracted {len(tasks)} tasks from message content")
                return tasks
        
        return []
    
    def _parse_tasks_from_content(self, content: str) -> List[Dict[str, Any]]:
        """Parse tasks from message content with robust JSON extraction"""
        import json
        
        # Find JSON boundaries
        json_start = content.find('{')
        json_end = content.rfind('}')
        
        if json_start == -1 or json_end == -1 or json_start >= json_end:
            return []
        
        json_str = content[json_start:json_end + 1]
        
        try:
            data = json.loads(json_str)
            
            # Extract tasks from common JSON structures
            task_paths = [
                data.get('data', {}).get('tasks'),  # MCP execution plan format
                data.get('tasks'),                   # Direct tasks format
                data if isinstance(data, list) else None  # Direct list format
            ]
            
            for tasks in task_paths:
                if isinstance(tasks, list) and tasks and all(isinstance(task, dict) for task in tasks):
                    return tasks
                    
        except (json.JSONDecodeError, ValueError, AttributeError):
            self.logger.debug("Failed to parse JSON from message content")
        
        return []
    
    def _validate_and_clean_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and clean task list, ensuring all tasks have required fields
        
        Args:
            tasks: Raw task list
            
        Returns:
            Cleaned and validated task list
        """
        if not isinstance(tasks, list):
            return []
        
        cleaned_tasks = []
        for i, task in enumerate(tasks):
            if not isinstance(task, dict):
                self.logger.warning(f"Task {i} is not a dictionary, skipping")
                continue
            
            # Ensure required fields exist with defaults
            cleaned_task = {
                'id': task.get('id', f'task_{i+1}'),
                'title': task.get('title', f'Task {i+1}'),
                'description': task.get('description', task.get('title', '')),
                'priority': task.get('priority', 'medium'),
                'tools': task.get('tools', []),
                **{k: v for k, v in task.items() if k not in ['id', 'title', 'description', 'priority', 'tools']}
            }
            
            cleaned_tasks.append(cleaned_task)
        
        if len(cleaned_tasks) != len(tasks):
            self.logger.info(f"Cleaned task list: {len(tasks)} -> {len(cleaned_tasks)} valid tasks")
        
        return cleaned_tasks
    
    def _filter_task_tools(self, task_tools: List[str], all_tools: List[Any]) -> List[Dict[str, Any]]:
        """
        Filter and format tools based on task requirements
        
        Args:
            task_tools: List of tool names requested by task
            all_tools: List of all available tools from MCP
            
        Returns:
            List of filtered tool dictionaries
        """
        if "functions" in task_tools:
            # Special case: use all available tools
            return all_tools if isinstance(all_tools, list) else []
        
        # Clean tool names (remove 'functions.' prefix if present)
        clean_tool_names = set()
        for tool_name in task_tools:
            if tool_name.startswith("functions."):
                clean_tool_names.add(tool_name[10:])
            else:
                clean_tool_names.add(tool_name)
        
        # Filter tools based on format
        if not all_tools:
            return []
        
        if isinstance(all_tools[0], dict):
            # Tools are dictionaries with 'name' field
            return [tool for tool in all_tools if tool.get("name") in clean_tool_names]
        elif isinstance(all_tools[0], str):
            # Tools are just strings - convert to dict format
            return [
                {"name": tool, "type": "function"} 
                for tool in all_tools 
                if tool in clean_tool_names
            ]
        else:
            self.logger.warning(f"Unexpected tool format: {type(all_tools[0])}")
            return []
    
    def _handle_no_tasks(self, state: AgentState) -> AgentState:
        """Handle case when no tasks are available"""
        self.stream_custom({
            "agent_execution": {
                "status": "no_tasks",
                "message": "No tasks found for execution"
            }
        })
        
        # Add result message to conversation (add_messages reducer handles appending)
        result_message = SystemMessage(
            content="[AGENT_EXECUTOR] No autonomous tasks found for execution"
        )
        return {
            "messages": [result_message],
            "next_action": "end"
        }
    
    def _handle_all_tasks_completed(self, state: AgentState, task_list: List[Dict]) -> AgentState:
        """Handle case when all tasks are completed"""
        total_tasks = len(task_list)
        completed_tasks = state.get("completed_task_count", total_tasks)
        failed_tasks = state.get("failed_task_count", 0)
        
        self.stream_custom({
            "agent_execution": {
                "status": "all_completed",
                "total_tasks": total_tasks,
                "completed": completed_tasks,
                "failed": failed_tasks
            }
        })
        
        success_message = f"🎨 **Autonomous Execution Complete!**\n\n"
        success_message += f"📊 **Summary:**\n"
        success_message += f"- Total Tasks: {total_tasks}\n"
        success_message += f"- Successfully Completed: {completed_tasks}\n"
        success_message += f"- Failed: {failed_tasks}\n"
        success_message += f"- Success Rate: {(completed_tasks/total_tasks)*100:.1f}%\n\n"
        success_message += f"🤔 **Next:** ReasonNode will evaluate results and determine if more actions are needed"
        
        # Add completion message to conversation (add_messages reducer handles appending)
        result_message = SystemMessage(content=f"[AGENT_EXECUTOR] {success_message}")
        return {
            "messages": [result_message],
            "next_action": "call_model"  # Go to ReasonNode for re-evaluation of results
        }
    
    def _handle_recursion_limit_reached(self, state: AgentState, current_index: int, total_tasks: int) -> AgentState:
        """Handle case when recursion limit is reached"""
        remaining_tasks = total_tasks - current_index
        
        self.stream_custom({
            "agent_execution": {
                "status": "recursion_limit",
                "completed_tasks": current_index,
                "remaining_tasks": remaining_tasks
            }
        })
        
        limit_message = f"⏱️ **Execution Paused - Recursion Limit Reached**\n\n"
        limit_message += f"📊 **Progress:**\n"
        limit_message += f"- Tasks Completed: {current_index}/{total_tasks}\n"
        limit_message += f"- Remaining Tasks: {remaining_tasks}\n\n"
        limit_message += f"🔄 **Note:** Execution can be resumed to continue remaining tasks"
        
        # Add recursion limit message to conversation (add_messages reducer handles appending)
        result_message = SystemMessage(content=f"[AGENT_EXECUTOR] {limit_message}")
        
        return {
            "messages": [result_message],
            "next_action": "call_model",  # Generate summary of current progress
            "remaining_steps": state["remaining_steps"]  # Preserve remaining_steps
        }
    
    async def _execute_sequential_task(
        self, 
        state: AgentState, 
        config: RunnableConfig, 
        task_list: List[Dict], 
        current_index: int
    ) -> AgentState:
        """
        Execute single task in sequential mode
        
        Args:
            state: Current agent state
            config: Runtime configuration
            task_list: List of all tasks
            current_index: Index of current task to execute
            
        Returns:
            Updated state with task execution results
        """
        current_task = task_list[current_index]
        task_title = current_task.get("title", f"Task {current_index + 1}")
        
        # Stream task start
        self.stream_tool("task_execution", f"Starting task {current_index + 1}/{len(task_list)}: {task_title}")
        
        # 发射task开始执行事件
        self.stream_custom({
            "task_step_start": {
                "task_index": current_index + 1,
                "task_title": task_title,
                "task_id": current_task.get("id", f"task_{current_index + 1}"),
                "total_tasks": len(task_list),
                "status": "started"
            }
        })
        
        # CO-EXECUTE: Check if this task requires human intervention checkpoint
        if self._should_trigger_progress_checkpoint(state, current_task, current_index, task_list):
            checkpoint_result = await self._trigger_progress_checkpoint(state, current_task, current_index, task_list)
            
            # Handle checkpoint result
            if checkpoint_result == "pause":
                return {
                    "execution_paused": True,
                    "pause_reason": "Human requested execution pause",
                    "next_action": "call_model"  # Return to reasoning
                }
            elif checkpoint_result == "redirect":
                # Human wants to redirect execution
                return {
                    "execution_redirected": True,
                    "redirect_reason": "Human redirected execution flow", 
                    "next_action": "call_model"  # Return to reasoning for new instructions
                }
            # If "continue", proceed with normal execution
        
        try:
            # Execute single task
            execution_result = await self._execute_single_task(
                current_task, current_index + 1, len(task_list), config
            )
            
            # Update state with success
            state["current_task_index"] = current_index + 1
            completed_count = state.get("completed_task_count", 0) + 1
            state["completed_task_count"] = completed_count
            
            # Add execution result to messages (add_messages reducer handles appending)
            result_message = SystemMessage(content=f"[TASK_RESULT] {execution_result}")
            state["messages"] = [result_message]
            
            # Determine next action - let ReasonNode evaluate results and decide
            if current_index + 1 >= len(task_list):
                # All tasks in current plan completed - let ReasonNode re-evaluate
                state["next_action"] = "call_model"  # Go to ReasonNode for re-evaluation
            else:
                # Continue to next task in current plan
                state["next_action"] = "agent_executor"
            
            # Stream task completion
            self.stream_custom({
                "task_step_complete": {
                    "task_index": current_index + 1,
                    "task_title": task_title,
                    "task_id": current_task.get("id", f"task_{current_index + 1}"),
                    "status": "success",
                    "remaining_tasks": len(task_list) - (current_index + 1),
                    "completed_count": completed_count,
                    "total_tasks": len(task_list),
                    "execution_result": execution_result[:200] if execution_result else ""
                }
            })
            
            # Stream updated task state
            self.stream_custom({
                "task_state": {
                    "total_tasks": len(task_list),
                    "current_task_index": current_index + 1,
                    "current_task_name": task_list[current_index + 1].get("title", f"Task {current_index + 2}") if current_index + 1 < len(task_list) else "All tasks completed",
                    "task_names": [task.get("title", f"Task {i+1}") for i, task in enumerate(task_list)],
                    "completed_tasks": completed_count,
                    "failed_tasks": state.get("failed_task_count", 0),
                    "execution_mode": state.get("execution_mode", "sequential"),
                    "status": "task_completed",
                    "last_completed_task": task_title
                }
            })
            
        except Exception as e:
            # Handle task failure
            logger.error(f"Task execution failed: {e}")
            
            error_result = f"❌ **Task {current_index + 1} Failed: {task_title}**\n\n"
            error_result += f"🚨 **Error:** {str(e)}\n"
            error_result += f"⏱️ **Status:** Execution failed"
            
            # Update state with failure
            state["current_task_index"] = current_index + 1
            failed_count = state.get("failed_task_count", 0) + 1
            state["failed_task_count"] = failed_count
            
            # Add error result to messages (add_messages reducer handles appending)
            result_message = SystemMessage(content=f"[TASK_ERROR] {error_result}")
            state["messages"] = [result_message]
            
            # Continue to next task or end
            if current_index + 1 >= len(task_list):
                state["next_action"] = "call_model"
            else:
                state["next_action"] = "agent_executor"
            
            # Stream task failure
            self.stream_custom({
                "task_step_complete": {
                    "task_index": current_index + 1,
                    "task_title": task_title,
                    "task_id": current_task.get("id", f"task_{current_index + 1}"),
                    "status": "failed",
                    "error": str(e),
                    "remaining_tasks": len(task_list) - (current_index + 1),
                    "failed_count": failed_count,
                    "total_tasks": len(task_list)
                }
            })
            
            # Stream updated task state for failure
            self.stream_custom({
                "task_state": {
                    "total_tasks": len(task_list),
                    "current_task_index": current_index + 1,
                    "current_task_name": task_list[current_index + 1].get("title", f"Task {current_index + 2}") if current_index + 1 < len(task_list) else "All tasks processed",
                    "task_names": [task.get("title", f"Task {i+1}") for i, task in enumerate(task_list)],
                    "completed_tasks": state.get("completed_task_count", 0),
                    "failed_tasks": failed_count,
                    "execution_mode": state.get("execution_mode", "sequential"),
                    "status": "task_failed",
                    "last_failed_task": task_title,
                    "last_error": str(e)
                }
            })
        
        return state
    
    async def _execute_parallel_tasks(
        self, 
        state: AgentState, 
        config: RunnableConfig, 
        task_list: List[Dict], 
        current_index: int
    ) -> AgentState:
        """
        Execute multiple tasks in parallel (fan-out execution)
        
        Args:
            state: Current agent state
            config: Runtime configuration
            task_list: List of all tasks
            current_index: Starting index for parallel execution
            
        Returns:
            Updated state with parallel execution results
        """
        import asyncio
        
        # Determine how many tasks to execute in parallel
        remaining_tasks = task_list[current_index:]
        parallel_batch = remaining_tasks[:self.max_parallel_tasks]
        
        self.stream_tool("parallel_execution", f"Starting parallel execution of {len(parallel_batch)} tasks")
        
        # Execute tasks concurrently
        tasks = []
        for i, task in enumerate(parallel_batch):
            task_index = current_index + i + 1
            coroutine = self._execute_single_task(task, task_index, len(task_list), config)
            tasks.append(coroutine)
        
        try:
            # Wait for all tasks to complete with timeout
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.task_timeout
            )
            
            # Process results and collect messages to add
            completed_count = state.get("completed_task_count", 0)
            failed_count = state.get("failed_task_count", 0)
            result_messages = []
            
            for i, result in enumerate(results):
                task_index = current_index + i + 1
                task_title = parallel_batch[i].get("title", f"Task {task_index}")
                
                if isinstance(result, Exception):
                    failed_count += 1
                    error_msg = f"❌ **Parallel Task {task_index} Failed: {task_title}**\n🚨 **Error:** {str(result)}"
                    result_message = SystemMessage(content=f"[PARALLEL_TASK_ERROR] {error_msg}")
                else:
                    completed_count += 1
                    result_message = SystemMessage(content=f"[PARALLEL_TASK_RESULT] {result}")
                
                result_messages.append(result_message)
            
            # Add all parallel task results to conversation (add_messages reducer handles appending)
            state["messages"] = result_messages
            
            # Update state
            state["current_task_index"] = current_index + len(parallel_batch)
            state["completed_task_count"] = completed_count
            state["failed_task_count"] = failed_count
            
            # Determine next action
            if state["current_task_index"] >= len(task_list):
                state["next_action"] = "call_model"  # All tasks completed - let ReasonNode re-evaluate
            else:
                state["next_action"] = "agent_executor"  # More tasks remain
            
            # Stream parallel completion
            self.stream_custom({
                "parallel_execution_completed": {
                    "batch_size": len(parallel_batch),
                    "completed": completed_count,
                    "failed": failed_count,
                    "remaining_tasks": len(task_list) - state["current_task_index"]
                }
            })
            
        except asyncio.TimeoutError:
            # Handle timeout
            timeout_msg = f"⏱️ **Parallel Execution Timeout**\n"
            timeout_msg += f"Batch of {len(parallel_batch)} tasks exceeded {self.task_timeout}s timeout"
            
            # Add timeout message to conversation (add_messages reducer handles appending)
            result_message = SystemMessage(content=f"[PARALLEL_TIMEOUT] {timeout_msg}")
            state["messages"] = [result_message]
            
            # Skip failed batch and continue
            state["current_task_index"] = current_index + len(parallel_batch)
            state["failed_task_count"] = state.get("failed_task_count", 0) + len(parallel_batch)
            
            if state["current_task_index"] >= len(task_list):
                state["next_action"] = "call_model"
            else:
                return {"next_action": "agent_executor"}
    
    async def _execute_single_task(
        self, 
        task: Dict[str, Any], 
        task_index: int, 
        total_tasks: int, 
        config: RunnableConfig
    ) -> str:
        """
        Execute a single task using ReactAgent
        
        Args:
            task: Task dictionary with title, description, tools, etc.
            task_index: Current task number (1-based)
            total_tasks: Total number of tasks
            config: Runtime configuration with MCP service
            
        Returns:
            Formatted execution result string
        """
        try:
            # Validate task format (should already be cleaned by _validate_and_clean_tasks)
            if not isinstance(task, dict):
                # Fallback conversion for edge cases
                if isinstance(task, str):
                    task = {
                        "title": f"Task {task_index}",
                        "description": task,
                        "priority": "medium",
                        "tools": []
                    }
                else:
                    return f"❌ **Task {task_index}/{total_tasks} Failed: Invalid Task Format**\n🚨 **Error:** Task must be a dictionary, got {type(task)}"
            
            # Get task details
            task_title = task.get("title", f"Task {task_index}")
            task_description = task.get("description", "")
            task_priority = task.get("priority", "medium")
            task_tools = task.get("tools", [])
            
            # Get MCP service from config
            mcp_service = config.get("configurable", {}).get("mcp_service")
            if not mcp_service:
                return f"❌ **Task {task_index}/{total_tasks} Failed: {task_title}**\n🚨 **Error:** MCP service not available"
            
            # Get available tools
            tools = []
            if task_tools:
                try:
                    all_tools = await mcp_service.list_tools()
                    tools = self._filter_task_tools(task_tools, all_tools or [])
                    self.logger.debug(f"Filtered {len(tools)} tools from {len(task_tools)} requested")
                except Exception as e:
                    self.logger.error(f"Failed to get tools for task: {e}")
                    tools = []
            
            # Handle tasks without tools (analysis/design tasks)
            if not tools:
                analysis_result = f"📋 **Analysis/Design Task**\n{task_description}"
                return f"✅ **Task {task_index}/{total_tasks} Completed: {task_title}**\n\n{analysis_result}\n\n⏱️ **Status:** Completed through analysis (no tools required)"
            
            # Create and execute ReactAgent
            from isa_agent_sdk.nodes.agent_nodes.react_agent import create_react_agent
            agent = create_react_agent(tools, mcp_service)
            
            # Build execution prompt
            execution_prompt = f"""
Task: {task_title}
Description: {task_description}
Priority: {task_priority}
Available Tools: {len(tools)} tools

Execute this task step by step. Provide detailed feedback on your progress and results.
"""
            
            # Execute task with agent
            result = await agent.execute(execution_prompt)
            
            # Format success result
            success_result = f"✅ **Task {task_index}/{total_tasks} Completed: {task_title}**\n\n"
            success_result += f"📋 **Result:**\n{result}\n\n"
            success_result += f"🔧 **Tools Used:** {len(tools)} tools available\n"
            success_result += f"⏱️ **Status:** Successfully executed using autonomous agent"
            
            return success_result
            
        except Exception as e:
            # Format error result
            task_title = task.get("title", f"Task {task_index}")
            error_result = f"❌ **Task {task_index}/{total_tasks} Failed: {task_title}**\n\n"
            error_result += f"🚨 **Error:** {str(e)}\n"
            error_result += f"⏱️ **Status:** Execution failed - manual intervention may be required"
            
            return error_result
    
    def _should_trigger_progress_checkpoint(
        self, 
        state: AgentState, 
        current_task: Dict[str, Any], 
        current_index: int, 
        task_list: List[Dict[str, Any]]
    ) -> bool:
        """
        Determine if a progress checkpoint should be triggered for human intervention
        
        Args:
            state: Current agent state
            current_task: Task about to be executed
            current_index: Index of current task
            task_list: Full list of tasks
            
        Returns:
            True if checkpoint should be triggered
        """
        # Trigger checkpoints based on various criteria
        
        # 1. Every N tasks (configurable milestone checkpoints)
        checkpoint_frequency = state.get("checkpoint_frequency", 3)  # Default: every 3 tasks
        if current_index > 0 and (current_index % checkpoint_frequency) == 0:
            return True
            
        # 2. High-priority tasks always get checkpoints
        if current_task.get("priority") == "high":
            return True
            
        # 3. Tasks with certain sensitive tools
        sensitive_tools = ["system_admin", "file_operations", "database_operations", "external_communications"]
        task_tools = current_task.get("tools", [])
        if any(tool in sensitive_tools for tool in task_tools):
            return True
            
        # 4. Tasks that are estimated to take long time
        if current_task.get("estimated_duration_minutes", 0) > 10:
            return True
            
        # 5. User has requested checkpoint on all tasks
        if state.get("checkpoint_all_tasks", False):
            return True
            
        # 6. Previous task failed - check before continuing
        if state.get("last_task_failed", False):
            return True
        
        return False
    
    async def _trigger_progress_checkpoint(
        self,
        state: AgentState,
        current_task: Dict[str, Any], 
        current_index: int,
        task_list: List[Dict[str, Any]]
    ) -> str:
        """
        Trigger a progress checkpoint for human intervention
        
        Args:
            state: Current agent state
            current_task: Task about to be executed
            current_index: Index of current task
            task_list: Full list of tasks
            
        Returns:
            Human decision: "continue", "pause", or "redirect"
        """
        # Create progress summary
        completed_tasks = state.get("completed_task_count", current_index)
        total_tasks = len(task_list)
        failed_tasks = state.get("failed_task_count", 0)
        
        progress_summary = self._create_progress_summary(
            completed_tasks, total_tasks, failed_tasks, current_task, task_list[current_index:]
        )
        
        # Use HIL service for checkpoint interaction
        checkpoint_question = f"""🚀 **Execution Progress Checkpoint**

{progress_summary}

**Next Task to Execute:**
• **{current_task.get('title', f'Task {current_index + 1}')}**
• Description: {current_task.get('description', 'No description')}
• Tools: {', '.join(current_task.get('tools', []))}
• Priority: {current_task.get('priority', 'normal')}

**Options:**
• Type 'continue' to proceed with execution
• Type 'pause' to stop execution and return control
• Type 'redirect' to change direction with new instructions

Your choice:"""

        try:
            human_response = hil_service.ask_human_with_interrupt(
                question=checkpoint_question,
                context=f"Execution checkpoint at task {current_index + 1}/{total_tasks}",
                node_source="agent_executor"
            )
            
            response_str = str(human_response).lower().strip() if human_response else "continue"
            
            # Validate response
            if response_str in ["continue", "pause", "redirect"]:
                return response_str
            else:
                # Default to continue for invalid responses
                return "continue"
                
        except Exception as e:
            self.logger.error(f"Progress checkpoint failed: {e}")
            # Default to continue on error
            return "continue"
    
    def _create_progress_summary(
        self,
        completed_tasks: int,
        total_tasks: int,
        failed_tasks: int,
        current_task: Dict[str, Any],
        remaining_tasks: List[Dict[str, Any]]
    ) -> str:
        """
        Create human-friendly progress summary for checkpoint
        
        Args:
            completed_tasks: Number of completed tasks
            total_tasks: Total number of tasks
            failed_tasks: Number of failed tasks
            current_task: Current task about to execute
            remaining_tasks: List of remaining tasks after current
            
        Returns:
            Formatted progress summary
        """
        success_rate = (completed_tasks / max(completed_tasks + failed_tasks, 1)) * 100
        
        summary_parts = []
        summary_parts.append("📊 **Current Progress:**")
        summary_parts.append(f"• Completed: {completed_tasks}/{total_tasks} tasks")
        summary_parts.append(f"• Failed: {failed_tasks} tasks")
        summary_parts.append(f"• Success Rate: {success_rate:.1f}%")
        summary_parts.append(f"• Remaining: {len(remaining_tasks)} tasks")
        summary_parts.append("")
        
        if remaining_tasks:
            summary_parts.append("📋 **Remaining Tasks:**")
            for i, task in enumerate(remaining_tasks[:3]):  # Show next 3 tasks
                task_num = completed_tasks + failed_tasks + i + 1
                task_title = task.get('title', f'Task {task_num}')
                summary_parts.append(f"   {task_num}. {task_title}")
            
            if len(remaining_tasks) > 3:
                summary_parts.append(f"   ... and {len(remaining_tasks) - 3} more tasks")
            summary_parts.append("")
        
        return "\n".join(summary_parts)
    
    # =============================================================================
    # UNIFIED MODE ENHANCEMENTS (Reactive/Collaborative/Proactive)
    # =============================================================================
    
    def _determine_execution_mode(self, state: AgentState, context: dict) -> str:
        """
        Determine execution mode based on state flags and prediction confidence
        
        Args:
            state: Current agent state
            context: Runtime context
            
        Returns:
            Execution mode: 'reactive', 'collaborative', or 'proactive'
        """
        try:
            # Check for proactive mode first (most advanced) - requires high confidence predictions
            if context.get('proactive_enabled', False) or state.get('proactive_enabled', False):
                predictions = state.get('proactive_predictions', {})
                if predictions and self._has_high_confidence_predictions(predictions):
                    return "proactive"
            
            # Check for collaborative mode
            if context.get('collaborative_enabled', False) or state.get('collaborative_enabled', False):
                return "collaborative"
            
            # Default to reactive mode (including low-confidence predictions)
            return "reactive"
            
        except Exception as e:
            self.logger.warning(f"Failed to determine execution mode: {e}")
            return "reactive"  # Safe default
    
    async def _apply_proactive_execution_enhancements(self, state: AgentState, context: dict) -> AgentState:
        """
        Apply proactive enhancements to task execution
        
        Args:
            state: Current agent state
            context: Runtime context with predictions
            
        Returns:
            Enhanced state with proactive execution optimizations
        """
        try:
            # Note: context parameter available for future enhancements
            _ = context  # Suppress unused parameter warning
            
            predictions = state.get('proactive_predictions', {})
            
            if not self._has_high_confidence_predictions(predictions):
                return state  # No high-confidence predictions
            
            # Extract execution insights from predictions
            task_outcomes = predictions.get('task_outcomes', {})
            resource_requirements = predictions.get('resource_requirements', {})
            execution_patterns = predictions.get('execution_patterns', {})
            
            enhancements_applied = []
            
            # Apply task scheduling optimizations
            if task_outcomes.get('confidence', 0) > 0.7:
                success_probability = task_outcomes.get('success_probability', 0)
                risk_factors = task_outcomes.get('risk_factors', [])
                
                if success_probability > 0.8:
                    # High success probability - optimize for speed
                    state['execution_optimization'] = 'speed_optimized'
                    state['parallel_execution_preferred'] = True
                    enhancements_applied.append(f"Speed optimization (success rate: {success_probability:.1%})")
                elif risk_factors:
                    # Risk factors detected - optimize for safety
                    state['execution_optimization'] = 'safety_optimized'
                    state['checkpoint_frequency'] = 1  # Checkpoint after every task
                    enhancements_applied.append(f"Safety optimization (risks: {', '.join(risk_factors[:2])})")
            
            # Apply resource management optimizations
            if resource_requirements.get('confidence', 0) > 0.7:
                cpu_intensive = resource_requirements.get('cpu_intensive', False)
                io_intensive = resource_requirements.get('io_intensive', False)
                
                if cpu_intensive:
                    state['max_parallel_tasks'] = max(1, self.max_parallel_tasks // 2)  # Reduce parallelism
                    enhancements_applied.append("CPU-aware parallelism reduction")
                elif io_intensive:
                    state['max_parallel_tasks'] = self.max_parallel_tasks * 2  # Increase for I/O bound
                    enhancements_applied.append("I/O-optimized parallelism increase")
            
            # Apply execution pattern optimizations
            if execution_patterns.get('confidence', 0) > 0.7:
                preferred_mode = execution_patterns.get('preferred_execution_mode', 'sequential')
                if preferred_mode in ['parallel', 'sequential']:
                    state['execution_mode'] = preferred_mode
                    enhancements_applied.append(f"Pattern-based mode: {preferred_mode}")
            
            # Add metadata for monitoring
            state['proactive_execution_enhancements'] = {
                'enhancement_count': len(enhancements_applied),
                'applied_enhancements': enhancements_applied,
                'prediction_confidence_scores': {
                    'task_outcomes': task_outcomes.get('confidence', 0),
                    'resource_requirements': resource_requirements.get('confidence', 0),
                    'execution_patterns': execution_patterns.get('confidence', 0)
                }
            }
            
            if enhancements_applied:
                self.logger.debug(f"Applied {len(enhancements_applied)} proactive execution enhancements")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Failed to apply proactive execution enhancements: {e}")
            return state
    
    async def _apply_collaborative_execution_enhancements(self, state: AgentState, context: dict) -> AgentState:
        """
        Apply collaborative enhancements to task execution
        
        Args:
            state: Current agent state
            context: Runtime context
            
        Returns:
            Enhanced state with collaborative execution features
        """
        try:
            # Note: context parameter available for future enhancements
            _ = context  # Suppress unused parameter warning
            
            enhancements_applied = []
            
            # Enable human checkpoints for collaborative execution
            if not state.get('checkpoint_frequency'):
                state['checkpoint_frequency'] = 2  # Checkpoint every 2 tasks by default
                enhancements_applied.append("Enabled collaborative checkpoints")
            
            # Check for Co-execute opportunities in task list
            task_list = self._get_task_list(state)
            if task_list and len(task_list) > 3:
                state['collaborative_execution_suggested'] = True
                enhancements_applied.append("Suggested collaborative execution for complex workflow")
            
            # Enable progress monitoring for transparency
            state['detailed_progress_reporting'] = True
            enhancements_applied.append("Enhanced progress reporting for collaboration")
            
            # Adjust timeout for collaborative decision-making
            original_timeout = self.task_timeout
            state['collaborative_timeout'] = int(original_timeout * 1.5)  # 50% longer for collaboration
            enhancements_applied.append("Extended timeouts for collaborative decision-making")
            
            # Check for tasks requiring human expertise
            if task_list:
                high_priority_tasks = [task for task in task_list if task.get('priority') == 'high']
                if high_priority_tasks:
                    state['checkpoint_all_tasks'] = True  # Checkpoint all tasks if any are high priority
                    enhancements_applied.append(f"Enabled checkpoints for {len(high_priority_tasks)} high-priority tasks")
            
            # Add metadata for monitoring
            state['collaborative_execution_enhancements'] = {
                'enhancement_count': len(enhancements_applied),
                'applied_enhancements': enhancements_applied,
                'collaboration_features': {
                    'checkpoints_enabled': bool(state.get('checkpoint_frequency')),
                    'progress_reporting': state.get('detailed_progress_reporting', False),
                    'extended_timeouts': bool(state.get('collaborative_timeout')),
                    'human_oversight': state.get('checkpoint_all_tasks', False)
                }
            }
            
            if enhancements_applied:
                self.logger.debug(f"Applied {len(enhancements_applied)} collaborative execution enhancements")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Failed to apply collaborative execution enhancements: {e}")
            return state
    
    def _has_high_confidence_predictions(self, predictions: dict) -> bool:
        """Check if predictions have sufficient confidence for enhancement"""
        try:
            # Check all available prediction types for maximum confidence
            user_needs_confidence = predictions.get('user_needs', {}).get('confidence', 0.0)
            task_outcomes_confidence = predictions.get('task_outcomes', {}).get('confidence', 0.0)
            resource_requirements_confidence = predictions.get('resource_requirements', {}).get('confidence', 0.0)
            execution_patterns_confidence = predictions.get('execution_patterns', {}).get('confidence', 0.0)
            user_patterns_confidence = predictions.get('user_patterns', {}).get('confidence', 0.0)
            
            max_confidence = max(user_needs_confidence, task_outcomes_confidence, resource_requirements_confidence, execution_patterns_confidence, user_patterns_confidence)
            return max_confidence > 0.7
        except:
            return False


# Factory function for compatibility
def create_agent_executor_node() -> AgentExecutorNode:
    """Create AgentExecutorNode instance for compatibility"""
    return AgentExecutorNode()