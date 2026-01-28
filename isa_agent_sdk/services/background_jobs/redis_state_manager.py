#!/usr/bin/env python3
"""
Redis State Manager - Task status and progress tracking using isA_common AsyncRedisClient

Handles:
- Task status storage and retrieval
- Progress event publishing (pub/sub)
- Task result caching
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from isa_common import AsyncRedisClient

from isa_agent_sdk.utils.logger import setup_logger
from .task_models import ProgressEvent, TaskProgress, TaskResult, TaskStatus

logger = setup_logger("isA_Agent.RedisStateManager")


class RedisStateManager:
    """Manages task state in Redis using isA_common AsyncRedisClient"""

    def __init__(
        self, user_id: str = "agent-service", organization_id: str = "default-org"
    ):
        """
        Initialize Redis state manager

        Args:
            user_id: Service user ID for Redis scoping
            organization_id: Organization ID for Redis scoping
        """
        self.user_id = user_id
        self.organization_id = organization_id
        self.redis_client: Optional[AsyncRedisClient] = None
        self._connected = False

    async def connect(self):
        """Establish Redis connection"""
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

            self.redis_client = AsyncRedisClient(
                user_id=self.user_id,
                organization_id=self.organization_id,
                consul_registry=consul_registry,
                service_name_override="redis_grpc_service",  # Use correct Consul service name (underscore)
                lazy_connect=False,  # Connect immediately
            )
            self._connected = True
            logger.info(
                "Redis state manager connected via Consul discovery (redis_grpc_service)"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            self._connected = False
            logger.info("Redis state manager disconnected")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()

    async def _get_redis_async(self):
        """
        Get the underlying async Redis client for direct operations (e.g., pub/sub).

        Returns:
            The internal Redis client from AsyncRedisClient
        """
        if not self._connected or not self.redis_client:
            await self.connect()

        # Access the underlying redis client from AsyncRedisClient
        # AsyncRedisClient wraps a redis.asyncio.Redis instance
        if hasattr(self.redis_client, '_redis') and self.redis_client._redis:
            return self.redis_client._redis
        elif hasattr(self.redis_client, 'redis') and self.redis_client.redis:
            return self.redis_client.redis
        else:
            # Fallback: return the client itself if it supports pubsub
            return self.redis_client

    # ============================================
    # Task Status Operations
    # ============================================

    async def store_task_status(
        self, job_id: str, progress: TaskProgress, ttl_seconds: int = 3600
    ):
        """
        Store task status in Redis

        Args:
            job_id: Job ID
            progress: Task progress object
            ttl_seconds: Time to live (default 1 hour)
        """
        try:
            key = f"task_status:{job_id}"
            value = progress.model_dump_json()

            async with self.redis_client:
                await self.redis_client.set(key, value, ttl_seconds=ttl_seconds)

            logger.info(
                f"task_status_stored | "
                f"job_id={job_id} | "
                f"status={progress.status} | "
                f"progress={progress.progress_percent}%"
            )
        except Exception as e:
            logger.error(f"Failed to store task status for {job_id}: {e}")

    async def get_task_status(self, job_id: str) -> Optional[TaskProgress]:
        """
        Get task status from Redis

        Args:
            job_id: Job ID

        Returns:
            TaskProgress object or None if not found
        """
        try:
            key = f"task_status:{job_id}"

            async with self.redis_client:
                value = await self.redis_client.get(key)

            if value:
                progress_dict = json.loads(value)
                return TaskProgress(**progress_dict)
            else:
                logger.warning(f"Task status not found: {job_id}")
                return None
        except Exception as e:
            logger.error(f"Failed to get task status for {job_id}: {e}")
            return None

    async def delete_task_status(self, job_id: str):
        """Delete task status from Redis"""
        try:
            key = f"task_status:{job_id}"
            async with self.redis_client:
                await self.redis_client.delete(key)
            logger.info(f"Task status deleted: {job_id}")
        except Exception as e:
            logger.error(f"Failed to delete task status for {job_id}: {e}")

    # ============================================
    # Task Result Operations
    # ============================================

    async def store_task_result(
        self, job_id: str, result: TaskResult, ttl_seconds: int = 7200
    ):
        """
        Store final task result in Redis

        Args:
            job_id: Job ID
            result: Task result object
            ttl_seconds: Time to live (default 2 hours)
        """
        try:
            key = f"task_result:{job_id}"
            value = result.model_dump_json()

            async with self.redis_client:
                await self.redis_client.set(key, value, ttl_seconds=ttl_seconds)

            logger.info(
                f"task_result_stored | "
                f"job_id={job_id} | "
                f"status={result.status} | "
                f"successful={result.successful_tools}/{result.total_tools}"
            )
        except Exception as e:
            logger.error(f"Failed to store task result for {job_id}: {e}")

    async def get_task_result(self, job_id: str) -> Optional[TaskResult]:
        """
        Get task result from Redis

        Args:
            job_id: Job ID

        Returns:
            TaskResult object or None if not found
        """
        try:
            key = f"task_result:{job_id}"

            async with self.redis_client:
                value = await self.redis_client.get(key)

            if value:
                result_dict = json.loads(value)
                return TaskResult(**result_dict)
            else:
                logger.warning(f"Task result not found: {job_id}")
                return None
        except Exception as e:
            logger.error(f"Failed to get task result for {job_id}: {e}")
            return None

    # ============================================
    # Progress Event Publishing (Pub/Sub)
    # ============================================

    async def publish_progress_event(self, event: ProgressEvent):
        """
        Publish progress event to Redis pub/sub channel

        Args:
            event: Progress event object
        """
        try:
            channel = f"job_progress:{event.job_id}"
            message = event.model_dump_json()

            async with self.redis_client:
                subscriber_count = await self.redis_client.publish(channel, message)

            logger.debug(
                f"progress_event_published | "
                f"job_id={event.job_id} | "
                f"type={event.type} | "
                f"subscribers={subscriber_count}"
            )
        except Exception as e:
            logger.error(f"Failed to publish progress event for {event.job_id}: {e}")

    async def subscribe_progress_events(self, job_id: str):
        """
        Subscribe to progress events for a job (async generator)

        Args:
            job_id: Job ID

        Yields:
            Progress event messages
        """
        try:
            channel = f"job_progress:{job_id}"

            logger.info(f"Subscribing to progress events for job: {job_id}")
            async with self.redis_client:
                async for message in self.redis_client.subscribe([channel]):
                    yield message

        except Exception as e:
            logger.error(f"Failed to subscribe to progress events for {job_id}: {e}")

    # ============================================
    # Task Queue Metadata
    # ============================================

    async def increment_task_counter(self, counter_name: str) -> int:
        """
        Increment task counter

        Args:
            counter_name: Counter name (e.g., 'tasks_queued', 'tasks_completed')

        Returns:
            New counter value
        """
        try:
            key = f"task_counter:{counter_name}"
            async with self.redis_client:
                value = await self.redis_client.incr(key, delta=1)
            return value or 0
        except Exception as e:
            logger.error(f"Failed to increment counter {counter_name}: {e}")
            return 0

    async def get_task_counter(self, counter_name: str) -> int:
        """Get task counter value"""
        try:
            key = f"task_counter:{counter_name}"
            async with self.redis_client:
                value = await self.redis_client.get(key)
            return int(value) if value else 0
        except Exception as e:
            logger.error(f"Failed to get counter {counter_name}: {e}")
            return 0

    async def list_active_tasks(
        self, pattern: str = "task_status:*", limit: int = 100
    ) -> List[str]:
        """
        List active task IDs

        Args:
            pattern: Key pattern (default: all task statuses)
            limit: Maximum number of results

        Returns:
            List of job IDs
        """
        try:
            async with self.redis_client:
                keys = await self.redis_client.list_keys(pattern=pattern, limit=limit)
            # Extract job_id from keys (task_status:job_123 -> job_123)
            job_ids = [key.replace("task_status:", "") for key in keys]
            return job_ids
        except Exception as e:
            logger.error(f"Failed to list active tasks: {e}")
            return []

    async def get_task_statistics(self) -> Dict[str, Any]:
        """Get overall task statistics"""
        try:
            stats = {
                "tasks_queued": await self.get_task_counter("tasks_queued"),
                "tasks_completed": await self.get_task_counter("tasks_completed"),
                "tasks_failed": await self.get_task_counter("tasks_failed"),
                "active_tasks": len(await self.list_active_tasks(limit=1000)),
            }
            return stats
        except Exception as e:
            logger.error(f"Failed to get task statistics: {e}")
            return {}


# Singleton instance for easy access
_state_manager: Optional[RedisStateManager] = None


async def get_state_manager() -> RedisStateManager:
    """Get singleton Redis state manager instance"""
    global _state_manager
    if _state_manager is None:
        _state_manager = RedisStateManager()
        await _state_manager.connect()
    return _state_manager
