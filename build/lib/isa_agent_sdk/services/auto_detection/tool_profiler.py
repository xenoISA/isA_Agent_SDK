#!/usr/bin/env python3
"""
Tool Profiler - Records and estimates tool execution times
Uses isA_common AsyncRedisClient with sorted sets for time-series data
"""

import time
import json
import logging
from typing import Dict, Any, List, Optional
from isa_common import AsyncRedisClient
from isa_agent_sdk.core.config import settings
from isa_agent_sdk.core.config.consul_config import consul_config

logger = logging.getLogger(__name__)


class ToolProfiler:
    """
    Records tool execution times to Redis sorted sets and estimates future execution times

    Uses centralized AsyncRedisClient from isA_common with unified config management
    Storage pattern: Sorted sets with key `tool_profile:{tool_name}`
    """

    def __init__(
        self,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None
    ):
        """
        Initialize Tool Profiler with Redis connection using unified config management

        Args:
            user_id: User ID for multi-tenant support
            organization_id: Organization ID for multi-tenant support
        """
        # Use unified infrastructure config for Redis
        redis_host = settings.infra.redis_host
        redis_port = settings.infra.redis_port

        logger.info(f"[ToolProfiler] Initializing with Redis: {redis_host}:{redis_port}")

        # Try Consul service discovery if enabled
        if consul_config.is_enabled:
            try:
                redis_url = consul_config.discover_service(
                    'redis-service',
                    fallback_url=f'http://{redis_host}:{redis_port}'
                )
                # Parse host and port from URL
                redis_url = redis_url.replace('http://', '').replace('https://', '')
                if ':' in redis_url:
                    redis_host, redis_port_str = redis_url.split(':')
                    redis_port = int(redis_port_str)
                logger.info(f"[ToolProfiler] Discovered Redis via Consul: {redis_host}:{redis_port}")
            except Exception as e:
                logger.warning(f"[ToolProfiler] Consul discovery failed, using config: {e}")

        # Initialize async Redis client
        try:
            self.redis = AsyncRedisClient(
                host=redis_host,
                port=redis_port,
                user_id=user_id or 'agent',
                organization_id=organization_id or 'default-org'
            )
            logger.info(f"[ToolProfiler] AsyncRedisClient initialized successfully")
        except Exception as e:
            logger.error(f"[ToolProfiler] Failed to initialize AsyncRedisClient: {e}")
            self.redis = None

        self.ttl_seconds = 604800  # 7 days
        self.max_records = 100  # Keep last 100 executions per tool

    async def record_execution(
        self,
        tool_name: str,
        execution_time_ms: int,
        tool_args: Dict,
        session_id: str,
        success: bool
    ) -> bool:
        """
        Record tool execution to Redis sorted set

        Args:
            tool_name: Name of the tool executed
            execution_time_ms: Execution time in milliseconds
            tool_args: Tool arguments (for complexity estimation)
            session_id: Session ID for tracking
            success: Whether execution was successful

        Returns:
            True if recorded successfully, False otherwise
        """
        # Skip if Redis client not initialized
        if self.redis is None:
            logger.debug(f"[ToolProfiler] Redis not available, skipping record for {tool_name}")
            return False

        try:
            key = f"tool_profile:{tool_name}"

            # Create execution record
            record = {
                "time_ms": execution_time_ms,
                "args_size": len(str(tool_args)),
                "session_id": session_id,
                "timestamp": time.time(),
                "success": success
            }

            # Add to sorted set (score = timestamp)
            timestamp = time.time()
            async with self.redis:
                await self.redis.zadd(key, {json.dumps(record): timestamp})

                # Keep only last N records (remove old ones)
                total_count = await self.redis.zcard(key)
                if total_count and total_count > self.max_records:
                    # Get the oldest members to remove
                    to_remove = total_count - self.max_records
                    oldest_members = await self.redis.zrange(key, 0, to_remove - 1)
                    if oldest_members:
                        await self.redis.zrem(key, oldest_members)

                # Set TTL on the key
                await self.redis.expire(key, self.ttl_seconds)

            return True

        except Exception as e:
            logger.warning(f"[ToolProfiler] Failed to record execution for {tool_name}: {e}")
            return False

    async def estimate_time(self, tool_name: str, tool_args: Dict) -> int:
        """
        Estimate execution time in milliseconds using 90th percentile

        Args:
            tool_name: Name of the tool
            tool_args: Tool arguments (for complexity-based estimation)

        Returns:
            Estimated time in milliseconds
        """
        # Skip if Redis client not initialized
        if self.redis is None:
            logger.debug(f"[ToolProfiler] Redis not available, using default estimate for {tool_name}")
            return self._get_default_estimate(tool_name)

        try:
            key = f"tool_profile:{tool_name}"

            # Get recent records (last 50)
            async with self.redis:
                records_raw = await self.redis.zrange(key, -50, -1)

            if not records_raw:
                # Return default estimate if no history
                return self._get_default_estimate(tool_name)

            # Parse records
            records = []
            for r in records_raw:
                try:
                    records.append(json.loads(r))
                except json.JSONDecodeError:
                    continue

            if not records:
                return self._get_default_estimate(tool_name)

            # Filter by similar argument size (complexity heuristic)
            args_size = len(str(tool_args))
            similar_times = []

            for record in records:
                if not record.get("success", True):
                    continue  # Skip failed executions

                record_args_size = record.get("args_size", 0)
                if record_args_size > 0:
                    size_ratio = record_args_size / max(args_size, 1)
                    if 0.5 < size_ratio < 2.0:  # Similar complexity
                        similar_times.append(record["time_ms"])
                else:
                    similar_times.append(record["time_ms"])

            # Need at least 3 samples for reliable estimate
            if len(similar_times) < 3:
                # Fall back to all successful executions
                similar_times = [r["time_ms"] for r in records if r.get("success", True)]

            if len(similar_times) < 3:
                return self._get_default_estimate(tool_name)

            # Return 90th percentile (pessimistic estimate)
            similar_times.sort()
            p90_index = int(len(similar_times) * 0.9)
            return similar_times[p90_index]

        except Exception as e:
            logger.warning(f"[ToolProfiler] Failed to estimate time for {tool_name}: {e}")
            return self._get_default_estimate(tool_name)

    async def get_batch_estimate(self, tool_info_list: List[Dict[str, Any]]) -> int:
        """
        Estimate total execution time for a batch of tools

        Args:
            tool_info_list: List of dicts with keys: tool_name, tool_args

        Returns:
            Total estimated time in milliseconds
        """
        total_time = 0

        for tool_info in tool_info_list:
            tool_name = tool_info.get("tool_name", "")
            tool_args = tool_info.get("tool_args", {})

            if not tool_name:
                continue

            estimated = await self.estimate_time(tool_name, tool_args)
            total_time += estimated

        return total_time

    def _get_default_estimate(self, tool_name: str) -> int:
        """
        Default estimates for cold start (no historical data)

        Args:
            tool_name: Name of the tool

        Returns:
            Default estimated time in milliseconds
        """
        # Default estimates based on common tool types
        defaults = {
            # Web operations (slow)
            "web_search": 3000,
            "web_crawl": 12000,
            "web_fetch": 2000,

            # File operations (fast)
            "file_read": 500,
            "file_write": 800,
            "file_list": 300,

            # Database operations (medium)
            "database_query": 2000,

            # AI operations (slow)
            "image_generation": 15000,
            "text_to_speech": 5000,

            # System operations (fast)
            "bash_command": 1000,
            "python_exec": 2000,
        }

        # Try exact match first
        if tool_name in defaults:
            return defaults[tool_name]

        # Try partial match (e.g., "web" in "web_search_advanced")
        for key, value in defaults.items():
            if key in tool_name or tool_name in key:
                return value

        # Default fallback: 5 seconds
        return 5000

    async def get_statistics(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get execution statistics for a tool

        Args:
            tool_name: Name of the tool

        Returns:
            Dict with statistics or None if no data
        """
        # Skip if Redis client not initialized
        if self.redis is None:
            logger.debug(f"[ToolProfiler] Redis not available, cannot get statistics for {tool_name}")
            return None

        try:
            key = f"tool_profile:{tool_name}"

            # Get all records
            async with self.redis:
                records_raw = await self.redis.zrange(key, 0, -1)

            if not records_raw:
                return None

            # Parse records
            records = []
            for r in records_raw:
                try:
                    records.append(json.loads(r))
                except json.JSONDecodeError:
                    continue

            if not records:
                return None

            # Calculate statistics
            times = [r["time_ms"] for r in records if r.get("success", True)]

            if not times:
                return None

            times.sort()

            stats = {
                "tool_name": tool_name,
                "total_executions": len(records),
                "successful_executions": len(times),
                "failed_executions": len(records) - len(times),
                "min_time_ms": min(times),
                "max_time_ms": max(times),
                "avg_time_ms": int(sum(times) / len(times)),
                "median_time_ms": times[len(times) // 2],
                "p90_time_ms": times[int(len(times) * 0.9)],
                "p95_time_ms": times[int(len(times) * 0.95)],
            }

            return stats

        except Exception as e:
            logger.warning(f"[ToolProfiler] Failed to get statistics for {tool_name}: {e}")
            return None

    async def close(self):
        """Close Redis connection"""
        if self.redis is not None:
            try:
                await self.redis.close()
            except Exception as e:
                logger.debug(f"[ToolProfiler] Error closing Redis: {e}")

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - close Redis connection"""
        await self.close()


# Singleton instance for easy access
_profiler_instance = None


async def get_tool_profiler(
    user_id: Optional[str] = None,
    organization_id: Optional[str] = None
) -> ToolProfiler:
    """
    Get or create singleton ToolProfiler instance

    Args:
        user_id: User ID
        organization_id: Organization ID

    Returns:
        ToolProfiler instance
    """
    global _profiler_instance

    if _profiler_instance is None:
        _profiler_instance = ToolProfiler(
            user_id=user_id,
            organization_id=organization_id
        )

    return _profiler_instance
