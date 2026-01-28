#!/usr/bin/env python3
"""
Task Worker - Background task execution worker

Pulls tasks from NATS queue and executes them with progress tracking
"""

import asyncio
import json
import time
import signal
import sys
from typing import Dict, Any, Optional
from datetime import datetime
from langchain_core.runnables import RunnableConfig

from isa_agent_sdk.utils.logger import setup_logger
from .nats_task_queue import NATSTaskQueue
from .redis_state_manager import RedisStateManager
from .task_models import (
    TaskDefinition,
    TaskProgress,
    TaskResult,
    TaskStatus,
    ToolExecutionResult,
    ProgressEvent,
    ToolCallInfo
)

logger = setup_logger("isA_Agent.TaskWorker")


class TaskWorker:
    """Background task worker that processes tasks from NATS queue"""

    def __init__(self, worker_name: str = "worker-1", priority_filter: Optional[str] = None):
        """
        Initialize task worker

        Args:
            worker_name: Unique worker name
            priority_filter: Process only specific priority ('high', 'normal', 'low') or None for all
        """
        self.worker_name = worker_name
        self.priority_filter = priority_filter
        self.running = False

        # Initialize components
        self.task_queue = NATSTaskQueue(user_id=f"agent-worker-{worker_name}")
        self.state_manager = RedisStateManager(user_id=f"agent-worker-{worker_name}")

    def start(self):
        """Start worker and begin processing tasks"""
        try:
            logger.info(
                f"🚀 Starting task worker | "
                f"worker={self.worker_name} | "
                f"priority_filter={self.priority_filter or 'all'}"
            )

            # Connect to NATS and Redis
            self.task_queue.connect()
            self.state_manager.connect()

            # Create consumer for this worker
            self.task_queue.create_worker_consumer(
                worker_name=self.worker_name,
                priority_filter=self.priority_filter
            )

            # Set up signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._shutdown_handler)
            signal.signal(signal.SIGTERM, self._shutdown_handler)

            self.running = True
            logger.info(f"✅ Worker {self.worker_name} is ready")

            # Start processing loop
            self._processing_loop()

        except Exception as e:
            logger.error(f"Failed to start worker: {e}", exc_info=True)
            raise

    def stop(self):
        """Stop worker gracefully"""
        logger.info(f"🛑 Stopping worker {self.worker_name}")
        self.running = False

        # Disconnect from services
        self.task_queue.disconnect()
        self.state_manager.disconnect()

        # Close MCP service if initialized
        if hasattr(self, '_mcp_service'):
            try:
                asyncio.run(self._mcp_service.close())
                logger.info("✅ MCP service closed")
            except Exception as e:
                logger.warning(f"Failed to close MCP service: {e}")

        logger.info(f"✅ Worker {self.worker_name} stopped")

    def _shutdown_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)

    def _processing_loop(self):
        """Main processing loop - pulls and processes tasks"""
        poll_interval = 1  # Poll every 1 second
        batch_size = 1  # Process one task at a time

        while self.running:
            try:
                # Pull tasks from queue
                messages = self.task_queue.pull_tasks(
                    worker_name=self.worker_name,
                    batch_size=batch_size
                )

                if not messages:
                    # No tasks available, wait and retry
                    time.sleep(poll_interval)
                    continue

                # Process each task
                for message in messages:
                    if not self.running:
                        break

                    try:
                        # Process the task
                        self._process_task(message)

                        # Acknowledge task completion
                        sequence = message.get('sequence')
                        self.task_queue.acknowledge_task(self.worker_name, sequence)

                    except Exception as task_error:
                        logger.error(
                            f"Task processing failed: {task_error}",
                            exc_info=True
                        )
                        # NACK task for retry (will requeue)
                        sequence = message.get('sequence')
                        self.task_queue.nack_task(self.worker_name, sequence, delay_seconds=30)

            except Exception as e:
                logger.error(f"Processing loop error: {e}", exc_info=True)
                time.sleep(poll_interval)

    def _process_task(self, message: Dict[str, Any]):
        """
        Process a single task

        Args:
            message: NATS message containing task definition
        """
        try:
            # Parse task definition
            task_data = json.loads(message['data'].decode('utf-8'))
            task = TaskDefinition(**task_data)

            logger.info(
                f"📝 Processing task | "
                f"job_id={task.job_id} | "
                f"session_id={task.session_id} | "
                f"tools={len(task.tools)}"
            )

            # Update task status to RUNNING
            progress = TaskProgress(
                job_id=task.job_id,
                status=TaskStatus.RUNNING,
                total_tools=len(task.tools),
                completed_tools=0,
                failed_tools=0,
                progress_percent=0.0,
                started_at=datetime.now()
            )
            self.state_manager.store_task_status(task.job_id, progress)

            # Publish task start event
            self.state_manager.publish_progress_event(ProgressEvent(
                type="task_start",
                job_id=task.job_id,
                data={
                    "total_tools": len(task.tools),
                    "started_at": datetime.now().isoformat()
                }
            ))

            # Execute tools
            task_start_time = time.time()
            results = asyncio.run(self._execute_tools(task, progress))
            task_end_time = time.time()

            # Calculate final statistics
            successful_tools = sum(1 for r in results if r.status == "success")
            failed_tools = len(results) - successful_tools

            # Determine final status
            final_status = TaskStatus.COMPLETED if failed_tools == 0 else TaskStatus.FAILED

            # Store final result
            final_result = TaskResult(
                job_id=task.job_id,
                session_id=task.session_id,
                status=final_status,
                total_tools=len(task.tools),
                successful_tools=successful_tools,
                failed_tools=failed_tools,
                results=results,
                started_at=progress.started_at,
                completed_at=datetime.now(),
                execution_time_seconds=task_end_time - task_start_time
            )

            self.state_manager.store_task_result(task.job_id, final_result)

            # Update final status
            progress.status = final_status
            progress.completed_tools = successful_tools
            progress.failed_tools = failed_tools
            progress.progress_percent = 100.0
            progress.completed_at = datetime.now()
            self.state_manager.store_task_status(task.job_id, progress)

            # Publish completion event
            self.state_manager.publish_progress_event(ProgressEvent(
                type="task_complete",
                job_id=task.job_id,
                data={
                    "status": final_status.value,
                    "successful_tools": successful_tools,
                    "failed_tools": failed_tools,
                    "execution_time_seconds": final_result.execution_time_seconds
                }
            ))

            # Update counters
            if final_status == TaskStatus.COMPLETED:
                self.state_manager.increment_task_counter("tasks_completed")
            else:
                self.state_manager.increment_task_counter("tasks_failed")

            logger.info(
                f"✅ Task completed | "
                f"job_id={task.job_id} | "
                f"status={final_status.value} | "
                f"successful={successful_tools}/{len(task.tools)} | "
                f"time={final_result.execution_time_seconds:.2f}s"
            )

        except Exception as e:
            logger.error(f"Failed to process task: {e}", exc_info=True)
            raise

    async def _execute_tools(
        self,
        task: TaskDefinition,
        progress: TaskProgress
    ) -> list[ToolExecutionResult]:
        """
        Execute all tools in the task

        Args:
            task: Task definition
            progress: Task progress object

        Returns:
            List of tool execution results
        """
        results = []

        for i, tool_info_dict in enumerate(task.tools):
            tool_info = ToolCallInfo(**tool_info_dict)
            tool_name = tool_info.tool_name
            tool_args = tool_info.tool_args
            tool_call_id = tool_info.tool_call_id

            # Update progress
            progress_percent = int((i / len(task.tools)) * 100)
            progress.current_tool = tool_name
            progress.progress_percent = progress_percent
            self.state_manager.store_task_status(task.job_id, progress)

            # Publish tool start event
            self.state_manager.publish_progress_event(ProgressEvent(
                type="tool_start",
                job_id=task.job_id,
                data={
                    "tool_name": tool_name,
                    "tool_index": i,
                    "total_tools": len(task.tools),
                    "progress_percent": progress_percent
                }
            ))

            logger.info(
                f"🔧 Executing tool | "
                f"job_id={task.job_id} | "
                f"tool={tool_name} | "
                f"progress={progress_percent}%"
            )

            # Execute tool
            tool_start = time.time()
            try:
                result = await self._execute_single_tool(
                    tool_name,
                    tool_args,
                    task.config
                )
                tool_end = time.time()

                tool_result = ToolExecutionResult(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    status="success",
                    result=result,
                    result_length=len(str(result)),
                    execution_time_ms=(tool_end - tool_start) * 1000
                )

                results.append(tool_result)
                progress.completed_tools += 1

                # Publish tool complete event
                self.state_manager.publish_progress_event(ProgressEvent(
                    type="tool_complete",
                    job_id=task.job_id,
                    data={
                        "tool_name": tool_name,
                        "result_preview": str(result)[:500],  # Truncate large results
                        "completed_tools": progress.completed_tools,
                        "progress_percent": int((progress.completed_tools / len(task.tools)) * 100)
                    }
                ))

                logger.info(
                    f"✅ Tool completed | "
                    f"job_id={task.job_id} | "
                    f"tool={tool_name} | "
                    f"time={tool_result.execution_time_ms:.0f}ms"
                )

            except Exception as tool_error:
                tool_end = time.time()

                tool_result = ToolExecutionResult(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    status="error",
                    error=str(tool_error),
                    execution_time_ms=(tool_end - tool_start) * 1000
                )

                results.append(tool_result)
                progress.failed_tools += 1

                # Publish tool error event
                self.state_manager.publish_progress_event(ProgressEvent(
                    type="tool_error",
                    job_id=task.job_id,
                    data={
                        "tool_name": tool_name,
                        "error": str(tool_error),
                        "completed_tools": progress.completed_tools,
                        "progress_percent": int((i / len(task.tools)) * 100)
                    }
                ))

                logger.error(
                    f"❌ Tool failed | "
                    f"job_id={task.job_id} | "
                    f"tool={tool_name} | "
                    f"error={str(tool_error)}"
                )

        return results

    async def _execute_single_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Any:
        """
        Execute a single tool via MCP

        Args:
            tool_name: Tool name
            tool_args: Tool arguments
            config: Serialized runtime config dict

        Returns:
            Tool execution result
        """
        try:
            # Get MCP service instance
            mcp_service = await self._get_mcp_service()

            logger.debug(
                f"Calling MCP tool | "
                f"tool={tool_name} | "
                f"args_keys={list(tool_args.keys())}"
            )

            # Execute tool via MCP directly
            result = await mcp_service.call_tool(tool_name, tool_args)

            return result

        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {e}")
            raise

    async def _get_mcp_service(self):
        """
        Get or create MCP service instance with Consul discovery

        Returns:
            MCP service instance
        """
        # Cache MCP service instance
        if not hasattr(self, '_mcp_service'):
            try:
                # Import MCP service
                from isa_agent_sdk.components.mcp_service import MCPService

                # Create MCP service instance with Consul discovery
                self._mcp_service = MCPService(user_id=f"worker-{self.worker_name}")

                # Initialize service (connect to MCP servers via Consul)
                await self._mcp_service.initialize()

                logger.info(f"✅ MCP service initialized for worker {self.worker_name}")

            except Exception as e:
                logger.error(f"Failed to initialize MCP service: {e}", exc_info=True)
                raise

        return self._mcp_service


def main():
    """Main entry point for worker"""
    import argparse

    parser = argparse.ArgumentParser(description="isA_Agent Task Worker")
    parser.add_argument("--name", default="worker-1", help="Worker name")
    parser.add_argument(
        "--priority",
        choices=["high", "normal", "low"],
        help="Process only specific priority tasks"
    )
    args = parser.parse_args()

    worker = TaskWorker(
        worker_name=args.name,
        priority_filter=args.priority
    )

    try:
        worker.start()
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
        worker.stop()
    except Exception as e:
        logger.error(f"Worker error: {e}", exc_info=True)
        worker.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
