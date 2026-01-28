#!/usr/bin/env python3
"""
Execution Manager for LangGraph-based AI Agent Systems
Handles execution control, state management, and thread operations
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional, AsyncIterator, cast
from langchain_core.runnables import RunnableConfig

from isa_agent_sdk.utils.logger import api_logger


class ExecutionManager:
    """
    Manages execution control for LangGraph instances
    
    Features:
    - Execution resumption with Command pattern
    - Status monitoring and history tracking
    - State snapshots and rollback functionality
    - Thread/session management
    - Works with any LangGraph instance
    """
    
    def __init__(self, logger=None):
        """
        Initialize ExecutionManager
        
        Args:
            logger: Optional logger instance, defaults to api_logger
        """
        self.logger = logger or api_logger
        self.logger.info("🔧 ExecutionManager initialized")
    
    def get_session_config(self, session_id: str) -> Dict:
        """
        Get session configuration for LangGraph
        
        Args:
            session_id: Session/thread identifier
            
        Returns:
            Configuration dictionary for LangGraph
        """
        return {
            "configurable": {
                "thread_id": session_id
            }
        }
    
    async def resume_execution(
        self,
        graph,
        thread_id: str,
        action: str = "continue",
        resume_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Resume interrupted execution using LangGraph's native Command pattern
        
        Args:
            graph: LangGraph instance to operate on
            thread_id: Thread identifier for the execution
            action: Action to perform ('continue', 'modify', 'skip', 'pause')
            resume_data: Optional data for resumption (human input, state modifications)
            
        Returns:
            Result dictionary with success status and details
        """
        try:
            if not graph:
                raise ValueError("Graph instance is required")
                
            # Prepare config for thread resumption
            config = self.get_session_config(thread_id)
            
            # Use LangGraph's native Command pattern for resumption
            from langgraph.types import Command
            
            # Extract human response from resume_data for interrupt responses
            human_response = None
            if resume_data:
                # For ask_human scenarios - direct text answer
                if 'human_answer' in resume_data:
                    human_response = resume_data['human_answer']
                # For authorization scenarios - structured decision
                elif 'action' in resume_data:
                    human_response = resume_data
                # Fallback - use entire resume_data
                else:
                    human_response = resume_data
            
            if action == "continue" and human_response:
                # Resume with human input for interrupt
                command = Command(resume=human_response)
            elif action == "continue":
                # Simple continue without specific input
                command = Command(resume=None)
            elif action == "modify" and resume_data:
                # Modify state using Command pattern
                command = Command(resume=resume_data)
            else:
                # For skip/pause, use update_state
                values = {"action": action}
                if resume_data:
                    values.update(resume_data)
                
                await graph.aupdate_state(
                    cast(RunnableConfig, config),
                    values=values,
                    as_node=None
                )
                
                return {
                    "success": True,
                    "thread_id": thread_id,
                    "message": f"Action '{action}' applied successfully",
                    "next_step": f"{action}_applied"
                }
            
            # Apply the command (for continue/modify cases)
            await graph.aupdate_state(
                cast(RunnableConfig, config),
                values=command,
                as_node=None
            )
            
            return {
                "success": True,
                "thread_id": thread_id,
                "message": f"Execution resumed with action: {action}",
                "next_step": "resumed"
            }
                
        except Exception as e:
            self.logger.error(f"❌ Resume execution error: {e}")
            return {
                "success": False,
                "thread_id": thread_id,
                "message": f"Resume failed: {str(e)}",
                "next_step": "error"
            }
    
    async def resume_execution_stream(
        self,
        graph,
        stream_handler,
        thread_id: str,
        action: str = "continue",
        resume_data: Optional[Dict] = None,
        trace_id: Optional[str] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Resume interrupted execution and stream the continued execution
        
        Args:
            graph: LangGraph instance to operate on
            stream_handler: Stream handler for processing execution events
            thread_id: Thread identifier for the execution
            action: Action to perform ('continue', 'modify', etc.)
            resume_data: Optional data for resumption
            trace_id: Optional trace ID for lifecycle management
            
        Yields:
            Streaming event dictionaries
        """
        try:
            if not graph:
                raise ValueError("Graph instance is required")
            if not stream_handler:
                raise ValueError("Stream handler is required")
                
            # Prepare config for thread resumption
            config = self.get_session_config(thread_id)
            
            # Use LangGraph's native Command pattern for resumption
            from langgraph.types import Command
            
            # Extract human response from resume_data for interrupt responses
            human_response = None
            if resume_data:
                # For ask_human scenarios - direct text answer
                if 'human_answer' in resume_data:
                    human_response = resume_data['human_answer']
                # For authorization scenarios - structured decision
                elif 'action' in resume_data:
                    human_response = resume_data
                # Fallback - use entire resume_data
                else:
                    human_response = resume_data
            
            yield {
                "type": "resume_start",
                "content": f"🔄 Resuming execution with human input: {human_response}",
                "timestamp": datetime.now().isoformat(),
                "session_id": thread_id
            }
            
            if action == "continue" and human_response:
                # Resume with human input for interrupt
                command = Command(resume=human_response)
            elif action == "continue":
                # Simple continue without specific input
                command = Command(resume=None)
            else:
                command = Command(resume=resume_data)
            
            # Use the stream handler's specialized resume stream processing
            async for event in stream_handler.process_resume_stream(
                graph=graph,
                command=command,
                config=cast(RunnableConfig, config),
                session_id=thread_id,
                trace_id=trace_id  # Pass trace_id for lifecycle management
            ):
                yield event
            
        except Exception as e:
            self.logger.error(f"❌ Resume execution stream error: {e}")
            yield {
                "type": "error",
                "content": f"Resume execution error: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "session_id": thread_id
            }
    
    async def get_execution_status(self, graph, thread_id: str) -> Dict[str, Any]:
        """
        Get current execution status for a thread
        
        Args:
            graph: LangGraph instance to query
            thread_id: Thread identifier
            
        Returns:
            Dictionary containing execution status and details
        """
        try:
            if not graph:
                raise ValueError("Graph instance is required")
                
            # Get current state
            config = self.get_session_config(thread_id)
            
            try:
                state_snapshot = await graph.aget_state(cast(RunnableConfig, config))
                
                if not state_snapshot:
                    return {
                        "status": "not_found",
                        "message": "No execution state found for this thread"
                    }
                
                current_state = state_snapshot.values
                next_node = state_snapshot.next
                
                # Debug logging
                self.logger.info(f"🔍 Execution status debug - thread: {thread_id}")
                self.logger.info(f"   next_node: {next_node}")
                self.logger.info(f"   next_action: {current_state.get('next_action')}")
                self.logger.info(f"   messages_count: {len(current_state.get('messages', []))}")
                self.logger.info(f"   execution_paused: {current_state.get('execution_paused')}")
                
                # Determine execution status
                if current_state.get("execution_paused"):
                    status = "paused"
                    reason = current_state.get("pause_reason", "unknown")
                elif next_node and len(next_node) > 0:
                    # Check if next_node tuple has content
                    status = "interrupted"
                    reason = f"Waiting for approval at {next_node[0] if next_node else 'unknown'}"
                elif current_state.get("next_action") == "end":
                    status = "completed"
                    reason = "Execution completed successfully"
                elif (not next_node or len(next_node) == 0) and len(current_state.get("messages", [])) == 0:
                    # Empty next_node and no messages - probably no execution or completed
                    status = "completed"
                    reason = "Execution completed (no active state)"
                else:
                    status = "running"
                    reason = "Execution in progress"
                
                return {
                    "status": status,
                    "reason": reason,
                    "current_task_index": current_state.get("current_task_index", 0),
                    "total_tasks": len(current_state.get("task_list", [])),
                    "next_action": current_state.get("next_action"),
                    "execution_strategy": current_state.get("execution_strategy"),
                    "credits_used": current_state.get("credits_used", 0.0),
                    "session_validated": current_state.get("session_validated", True),
                    "next_node": next_node[0] if next_node else None
                }
                
            except Exception as state_error:
                return {
                    "status": "error",
                    "reason": f"Failed to get state: {str(state_error)}"
                }
                
        except Exception as e:
            self.logger.error(f"❌ Get execution status error: {e}")
            return {
                "status": "error",
                "reason": f"Status check failed: {str(e)}"
            }
    
    async def get_execution_history(
        self, 
        graph, 
        thread_id: str, 
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Get execution history for a thread
        
        Args:
            graph: LangGraph instance to query
            thread_id: Thread identifier
            limit: Maximum number of history entries to retrieve
            
        Returns:
            Dictionary containing history entries and metadata
        """
        try:
            if not graph:
                raise ValueError("Graph instance is required")
                
            # Get state history
            config = self.get_session_config(thread_id)
            
            try:
                # Get state history using LangGraph's history API
                history = []
                async for state_snapshot in graph.aget_state_history(
                    cast(RunnableConfig, config), 
                    limit=limit
                ):
                    history.append({
                        "timestamp": (
                            state_snapshot.created_at.isoformat() 
                            if hasattr(state_snapshot.created_at, 'isoformat') and state_snapshot.created_at 
                            else None
                        ),
                        "node": (
                            state_snapshot.metadata.get("source", "unknown") 
                            if state_snapshot.metadata 
                            else "unknown"
                        ),
                        "next_action": state_snapshot.values.get("next_action"),
                        "credits_used": state_snapshot.values.get("credits_used", 0.0),
                        "task_index": state_snapshot.values.get("current_task_index", 0),
                        "execution_strategy": state_snapshot.values.get("execution_strategy"),
                        "config": state_snapshot.config
                    })
                
                return {
                    "history": history,
                    "total_entries": len(history)
                }
                
            except Exception as history_error:
                return {
                    "history": [],
                    "error": f"Failed to get history: {str(history_error)}"
                }
                
        except Exception as e:
            self.logger.error(f"❌ Get execution history error: {e}")
            return {
                "history": [],
                "error": f"History retrieval failed: {str(e)}"
            }
    
    async def rollback_execution(
        self, 
        graph, 
        thread_id: str, 
        checkpoint_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Rollback execution to a previous checkpoint
        
        Args:
            graph: LangGraph instance to operate on
            thread_id: Thread identifier
            checkpoint_id: Optional specific checkpoint to rollback to
            
        Returns:
            Dictionary indicating success/failure and details
        """
        try:
            if not graph:
                raise ValueError("Graph instance is required")
                
            config = self.get_session_config(thread_id)
            
            if checkpoint_id:
                # Rollback to specific checkpoint
                # Note: This would require implementing custom checkpoint selection
                return {
                    "success": False,
                    "message": "Specific checkpoint rollback not yet implemented",
                    "suggestion": "Use state modification through resume endpoint instead"
                }
            else:
                # Rollback to last known good state
                try:
                    # Get previous state from history
                    history = []
                    async for state_snapshot in graph.aget_state_history(
                        cast(RunnableConfig, config), 
                        limit=5
                    ):
                        history.append(state_snapshot)
                    
                    if len(history) < 2:
                        return {
                            "success": False,
                            "message": "No previous state available for rollback"
                        }
                    
                    # Use second-to-last state (skip current state)
                    previous_state = history[1]
                    
                    # Update to previous state
                    await graph.aupdate_state(
                        cast(RunnableConfig, config),
                        values=previous_state.values,
                        as_node=None
                    )
                    
                    return {
                        "success": True,
                        "message": "Rolled back to previous state",
                        "rolled_back_to": (
                            str(previous_state.created_at) 
                            if previous_state.created_at 
                            else "unknown"
                        )
                    }
                    
                except Exception as rollback_error:
                    return {
                        "success": False,
                        "message": f"Rollback failed: {str(rollback_error)}"
                    }
                
        except Exception as e:
            self.logger.error(f"❌ Rollback execution error: {e}")
            return {
                "success": False,
                "message": f"Rollback operation failed: {str(e)}"
            }
    
    def get_manager_info(self) -> Dict[str, Any]:
        """
        Get information about the ExecutionManager
        
        Returns:
            Dictionary containing manager capabilities and features
        """
        return {
            "name": "ExecutionManager",
            "version": "1.0",
            "description": "Manages execution control for LangGraph instances",
            "features": [
                "Execution resumption with Command pattern",
                "Status monitoring and history tracking", 
                "State snapshots and rollback functionality",
                "Thread/session management",
                "Works with any LangGraph instance",
                "Stream-based resumption support"
            ],
            "supported_actions": [
                "continue",    # Continue execution
                "modify",      # Modify state and continue
                "skip",        # Skip current step
                "pause"        # Pause execution
            ],
            "capabilities": {
                "resume_execution": "Resume interrupted workflows with human input",
                "execution_status": "Get current status and progress information",
                "execution_history": "Retrieve historical state snapshots",
                "rollback_support": "Rollback to previous execution states",
                "streaming_resume": "Stream execution continuation in real-time"
            }
        }