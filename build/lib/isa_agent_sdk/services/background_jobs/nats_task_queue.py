#!/usr/bin/env python3
"""
NATS Task Queue - JetStream-based task queue using isA_common AsyncNATSClient

Handles:
- Task enqueueing to NATS JetStream
- Stream and consumer management
- Priority queue support
- Message acknowledgment
"""

import json
from typing import Any, Dict, List, Optional

from isa_common import AsyncNATSClient

from isa_agent_sdk.utils.logger import setup_logger
from .task_models import TaskDefinition

logger = setup_logger("isA_Agent.NATSTaskQueue")


class NATSTaskQueue:
    """NATS JetStream-based task queue (Celery replacement) with async support"""

    # Stream and subject configuration
    STREAM_NAME = "ISA_AGENT_TASKS"
    SUBJECT_PREFIX = "isa.agent.tasks"

    # Priority subjects
    SUBJECT_HIGH = f"{SUBJECT_PREFIX}.high"
    SUBJECT_NORMAL = f"{SUBJECT_PREFIX}.normal"
    SUBJECT_LOW = f"{SUBJECT_PREFIX}.low"

    def __init__(self, user_id: str = "agent-service"):
        """
        Initialize NATS task queue

        Args:
            user_id: Service user ID for NATS
        """
        self.user_id = user_id
        self.nats_client: Optional[AsyncNATSClient] = None
        self._connected = False

    async def connect(self):
        """Establish NATS connection and initialize stream"""
        try:
            # Use ConsulRegistry for service discovery
            import os

            from isa_common.consul_client import ConsulRegistry

            # Use environment variables for Consul connection
            consul_host = os.getenv("CONSUL_HOST", "localhost")
            consul_port = int(os.getenv("CONSUL_PORT", "8500"))

            consul_registry = ConsulRegistry(
                consul_host=consul_host, consul_port=consul_port
            )

            self.nats_client = AsyncNATSClient(
                user_id=self.user_id,
                consul_registry=consul_registry,
                service_name_override="nats_grpc_service",  # Use correct Consul service name (underscore)
            )

            self._connected = True
            logger.info(
                "NATS task queue connected via Consul discovery (nats_grpc_service)"
            )

            # Initialize JetStream
            await self._initialize_stream()

        except Exception as e:
            logger.error(f"Failed to connect to NATS: {e}")
            raise

    async def disconnect(self):
        """Close NATS connection"""
        if self.nats_client and self._connected:
            await self.nats_client.close()
            self._connected = False
            logger.info("NATS task queue disconnected")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()

    async def _initialize_stream(self):
        """Initialize JetStream stream for tasks"""
        try:
            # Create stream with priority subjects
            async with self.nats_client:
                await self.nats_client.create_stream(
                    name=self.STREAM_NAME,
                    subjects=[f"{self.SUBJECT_PREFIX}.>"],  # Catch all priority levels
                    max_msgs=100000,  # Keep last 100k tasks
                    max_bytes=500 * 1024 * 1024,  # 500MB max storage
                )
            logger.info(f"JetStream initialized: {self.STREAM_NAME}")

        except Exception as e:
            # Stream might already exist, that's okay
            logger.warning(f"Stream initialization: {e} (may already exist)")

    def _get_subject_for_priority(self, priority: str) -> str:
        """Get NATS subject based on task priority"""
        priority_map = {
            "high": self.SUBJECT_HIGH,
            "normal": self.SUBJECT_NORMAL,
            "low": self.SUBJECT_LOW,
        }
        return priority_map.get(priority, self.SUBJECT_NORMAL)

    # ============================================
    # Task Enqueue Operations
    # ============================================

    async def enqueue_task(self, task: TaskDefinition) -> Optional[int]:
        """
        Enqueue task to NATS JetStream

        Args:
            task: Task definition

        Returns:
            Sequence number if successful, None otherwise
        """
        try:
            # Serialize task to JSON
            task_json = task.model_dump_json()
            task_bytes = task_json.encode("utf-8")

            # Get subject based on priority
            subject = self._get_subject_for_priority(task.priority)

            # Publish to JetStream
            async with self.nats_client:
                result = await self.nats_client.publish_to_stream(
                    stream_name=self.STREAM_NAME, subject=subject, data=task_bytes
                )

            if result and "sequence" in result:
                sequence = result["sequence"]
                logger.info(
                    f"task_enqueued | "
                    f"job_id={task.job_id} | "
                    f"sequence={sequence} | "
                    f"priority={task.priority} | "
                    f"tools={len(task.tools)}"
                )
                return sequence
            else:
                logger.error(
                    f"Failed to enqueue task {task.job_id}: No sequence returned"
                )
                return None

        except Exception as e:
            logger.error(f"Failed to enqueue task {task.job_id}: {e}", exc_info=True)
            return None

    async def enqueue_task_dict(self, task_dict: Dict[str, Any]) -> Optional[int]:
        """
        Enqueue task from dictionary

        Args:
            task_dict: Task definition as dictionary

        Returns:
            Sequence number if successful
        """
        try:
            task = TaskDefinition(**task_dict)
            return await self.enqueue_task(task)
        except Exception as e:
            logger.error(f"Failed to parse task definition: {e}")
            return None

    # ============================================
    # Consumer Management
    # ============================================

    async def create_worker_consumer(
        self, worker_name: str, priority_filter: Optional[str] = None
    ):
        """
        Create consumer for a worker

        Args:
            worker_name: Unique worker name
            priority_filter: Filter by priority ('high', 'normal', 'low') or None for all
        """
        try:
            # Build filter subject
            if priority_filter:
                filter_subject = self._get_subject_for_priority(priority_filter)
            else:
                filter_subject = f"{self.SUBJECT_PREFIX}.>"

            async with self.nats_client:
                await self.nats_client.create_consumer(
                    stream_name=self.STREAM_NAME,
                    consumer_name=worker_name,
                    filter_subject=filter_subject,
                )

            logger.info(f"Consumer created: {worker_name} | filter={filter_subject}")

        except Exception as e:
            logger.error(f"Failed to create consumer {worker_name}: {e}")
            raise

    async def pull_tasks(self, worker_name: str, batch_size: int = 1) -> List[Dict[str, Any]]:
        """
        Pull tasks from queue for processing

        Args:
            worker_name: Worker consumer name
            batch_size: Number of tasks to pull

        Returns:
            List of task messages with metadata
        """
        try:
            async with self.nats_client:
                messages = await self.nats_client.pull_messages(
                    stream_name=self.STREAM_NAME,
                    consumer_name=worker_name,
                    batch_size=batch_size,
                )

            if messages:
                logger.info(
                    f"tasks_pulled | worker={worker_name} | count={len(messages)}"
                )

            return messages

        except Exception as e:
            logger.error(f"Failed to pull tasks for {worker_name}: {e}")
            return []

    async def acknowledge_task(self, worker_name: str, sequence: int):
        """
        Acknowledge task completion

        Args:
            worker_name: Worker consumer name
            sequence: Message sequence number
        """
        try:
            async with self.nats_client:
                await self.nats_client.ack_message(
                    stream_name=self.STREAM_NAME,
                    consumer_name=worker_name,
                    sequence=sequence,
                )

            logger.debug(
                f"task_acknowledged | worker={worker_name} | sequence={sequence}"
            )

        except Exception as e:
            logger.error(f"Failed to acknowledge task {sequence}: {e}")

    async def nack_task(self, worker_name: str, sequence: int, delay_seconds: int = 30):
        """
        Negative acknowledge task (requeue for retry)

        Args:
            worker_name: Worker consumer name
            sequence: Message sequence number
            delay_seconds: Delay before retry
        """
        try:
            # NATS doesn't have explicit NACK in our client yet
            # For now, just don't ACK and let it requeue
            logger.warning(
                f"task_nacked | "
                f"worker={worker_name} | "
                f"sequence={sequence} | "
                f"delay={delay_seconds}s | "
                f"will_requeue"
            )
            # TODO: Implement NACK with delay in AsyncNATSClient if needed

        except Exception as e:
            logger.error(f"Failed to NACK task {sequence}: {e}")

    # ============================================
    # Queue Statistics
    # ============================================

    async def get_stream_info(self) -> Optional[Dict[str, Any]]:
        """Get stream information and statistics"""
        try:
            async with self.nats_client:
                streams = await self.nats_client.list_streams()

            # Find our stream
            for stream in streams:
                if stream.get("name") == self.STREAM_NAME:
                    logger.info(
                        f"stream_info | "
                        f"messages={stream.get('messages', 0)} | "
                        f"bytes={stream.get('bytes', 0)}"
                    )
                    return stream

            logger.warning(f"Stream {self.STREAM_NAME} not found")
            return None

        except Exception as e:
            logger.error(f"Failed to get stream info: {e}")
            return None

    async def get_queue_statistics(self) -> Dict[str, Any]:
        """Get queue statistics"""
        try:
            async with self.nats_client:
                health = await self.nats_client.health_check()
                streams = await self.nats_client.list_streams()

            return {
                "total_streams": len(streams),
                "jetstream_enabled": health.get("jetstream_enabled", False) if health else False,
                "stream_info": await self.get_stream_info(),
            }
        except Exception as e:
            logger.error(f"Failed to get queue statistics: {e}")
            return {}

    async def health_check(self) -> bool:
        """Check if NATS and JetStream are healthy"""
        try:
            async with self.nats_client:
                health = await self.nats_client.health_check()
            is_healthy = health.get("healthy", False) and health.get(
                "jetstream_enabled", False
            ) if health else False

            if is_healthy:
                logger.info("NATS task queue is healthy")
            else:
                logger.warning("NATS task queue health check failed")

            return is_healthy

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# Singleton instance for easy access
_task_queue: Optional[NATSTaskQueue] = None


async def get_task_queue() -> NATSTaskQueue:
    """Get singleton NATS task queue instance"""
    global _task_queue
    if _task_queue is None:
        _task_queue = NATSTaskQueue()
        await _task_queue.connect()
    return _task_queue
