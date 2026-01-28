#!/usr/bin/env python3
"""
Agent Lightning Service - Wrapper for RL-based training data collection
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis.asyncio as aioredis

from isa_agent_sdk.core.config.lightning_config import LightningConfig
from isa_agent_sdk.utils.logger import api_logger


class LightningService:
    """
    Service for collecting agent execution data for RL training.

    In API mode (lightweight):
    - Collects prompts, responses, tool calls, and rewards
    - Stores data in Redis for later analysis
    - Minimal overhead on agent execution

    In Full mode:
    - Additionally supports direct model training
    - Integrates with Agent Lightning training server
    """

    def __init__(self, config: LightningConfig):
        self.config = config
        self.logger = api_logger
        self._redis_client: Optional[aioredis.Redis] = None
        self._initialized = False

    async def initialize(self):
        """Initialize Lightning service and connections"""
        if self._initialized:
            return

        if not self.config.is_enabled:
            self.logger.info("Agent Lightning is disabled")
            return

        try:
            # Initialize Redis connection for data storage
            if self.config.store_type == "redis":
                self._redis_client = await aioredis.from_url(
                    self.config.store_url, encoding="utf-8", decode_responses=True
                )
                # Test connection
                await self._redis_client.ping()
                self.logger.info(
                    f"Agent Lightning initialized in {self.config.mode} mode "
                    f"with Redis storage at {self.config.store_url}"
                )
            else:
                self.logger.warning(
                    f"Storage type {self.config.store_type} not yet implemented"
                )

            self._initialized = True

        except Exception as e:
            self.logger.error(f"Failed to initialize Agent Lightning: {e}")
            # Don't fail - just disable Lightning
            self.config.enabled = False

    async def close(self):
        """Close connections"""
        if self._redis_client:
            await self._redis_client.close()

    @property
    def is_enabled(self) -> bool:
        """Check if Lightning is enabled and initialized"""
        return self.config.is_enabled and self._initialized

    # ========================================
    # Data Collection Methods
    # ========================================

    async def emit_state(
        self,
        thread_id: str,
        state_data: Dict[str, Any],
        node_name: Optional[str] = None,
    ):
        """
        Emit agent state for trajectory tracking

        Args:
            thread_id: Conversation thread ID
            state_data: Current agent state
            node_name: Name of the node emitting state
        """
        if not self.is_enabled or not self.config.enable_trajectory_tracking:
            return

        try:
            event = {
                "type": "state",
                "thread_id": thread_id,
                "node_name": node_name,
                "state": state_data,
                "timestamp": datetime.utcnow().isoformat(),
            }

            await self._store_event(thread_id, event)

        except Exception as e:
            self.logger.error(f"Failed to emit state: {e}")

    async def emit_action(
        self,
        thread_id: str,
        action_type: str,
        action_data: Dict[str, Any],
        node_name: Optional[str] = None,
    ):
        """
        Emit agent action (LLM call, tool execution, etc.)

        Args:
            thread_id: Conversation thread ID
            action_type: Type of action (llm_call, tool_execution, etc.)
            action_data: Action details
            node_name: Node performing the action
        """
        if not self.is_enabled:
            return

        try:
            event = {
                "type": "action",
                "thread_id": thread_id,
                "node_name": node_name,
                "action_type": action_type,
                "action_data": action_data,
                "timestamp": datetime.utcnow().isoformat(),
            }

            await self._store_event(thread_id, event)

        except Exception as e:
            self.logger.error(f"Failed to emit action: {e}")

    async def emit_llm_call(
        self,
        thread_id: str,
        prompt: str,
        response: str,
        model: str,
        node_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Emit LLM call for prompt optimization training

        Args:
            thread_id: Conversation thread ID
            prompt: Input prompt
            response: LLM response
            model: Model name
            node_name: Node making the call
            metadata: Additional metadata
        """
        if not self.is_enabled or not self.config.collect_prompts:
            return

        try:
            event = {
                "type": "llm_call",
                "thread_id": thread_id,
                "node_name": node_name,
                "prompt": prompt,
                "response": response,
                "model": model,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow().isoformat(),
            }

            await self._store_event(thread_id, event)

        except Exception as e:
            self.logger.error(f"Failed to emit LLM call: {e}")

    async def emit_tool_call(
        self,
        thread_id: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_result: Any,
        success: bool,
        node_name: Optional[str] = None,
        execution_time: Optional[float] = None,
    ):
        """
        Emit tool execution for tool selection training

        Args:
            thread_id: Conversation thread ID
            tool_name: Name of the tool
            tool_args: Tool arguments
            tool_result: Tool execution result
            success: Whether execution succeeded
            node_name: Node executing tool
            execution_time: Execution time in seconds
        """
        if not self.is_enabled or not self.config.collect_tool_calls:
            return

        try:
            event = {
                "type": "tool_call",
                "thread_id": thread_id,
                "node_name": node_name,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "tool_result": str(tool_result)[:1000],  # Limit size
                "success": success,
                "execution_time": execution_time,
                "timestamp": datetime.utcnow().isoformat(),
            }

            await self._store_event(thread_id, event)

        except Exception as e:
            self.logger.error(f"Failed to emit tool call: {e}")

    async def emit_reward(
        self,
        thread_id: str,
        reward: float,
        reward_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Emit reward signal for RL training

        Args:
            thread_id: Conversation thread ID
            reward: Reward value (typically -1.0 to 1.0)
            reward_type: Type of reward (human_feedback, metric, success, etc.)
            metadata: Additional context
        """
        if not self.is_enabled or not self.config.collect_rewards:
            return

        try:
            event = {
                "type": "reward",
                "thread_id": thread_id,
                "reward": reward,
                "reward_type": reward_type,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow().isoformat(),
            }

            await self._store_event(thread_id, event)

            self.logger.info(
                f"Reward collected for thread {thread_id}: {reward_type}={reward}"
            )

        except Exception as e:
            self.logger.error(f"Failed to emit reward: {e}")

    # ========================================
    # Data Storage & Retrieval
    # ========================================

    async def _store_event(self, thread_id: str, event: Dict[str, Any]):
        """Store event to Redis"""
        if not self._redis_client:
            return

        try:
            # Store in thread-specific list
            key = f"lightning:trajectory:{thread_id}"
            await self._redis_client.rpush(key, json.dumps(event))

            # Set expiration (30 days)
            await self._redis_client.expire(key, 30 * 24 * 3600)

            # Also add to global event stream
            stream_key = f"lightning:stream:{event['type']}"
            await self._redis_client.rpush(stream_key, json.dumps(event))
            await self._redis_client.expire(stream_key, 30 * 24 * 3600)

        except Exception as e:
            self.logger.error(f"Failed to store event: {e}")

    async def get_trajectory(self, thread_id: str) -> List[Dict[str, Any]]:
        """
        Get complete trajectory for a thread

        Args:
            thread_id: Thread ID

        Returns:
            List of events in chronological order
        """
        if not self._redis_client:
            return []

        try:
            key = f"lightning:trajectory:{thread_id}"
            events_json = await self._redis_client.lrange(key, 0, -1)
            return [json.loads(e) for e in events_json]

        except Exception as e:
            self.logger.error(f"Failed to get trajectory: {e}")
            return []

    async def get_trajectories_count(self) -> int:
        """Get total number of trajectories collected"""
        if not self._redis_client:
            return 0

        try:
            keys = await self._redis_client.keys("lightning:trajectory:*")
            return len(keys)
        except Exception as e:
            self.logger.error(f"Failed to count trajectories: {e}")
            return 0

    async def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics"""
        if not self._redis_client:
            return {"enabled": False}

        try:
            total_trajectories = await self.get_trajectories_count()

            # Count events by type
            event_counts = {}
            for event_type in ["state", "action", "llm_call", "tool_call", "reward"]:
                key = f"lightning:stream:{event_type}"
                count = await self._redis_client.llen(key)
                event_counts[event_type] = count

            return {
                "enabled": True,
                "mode": self.config.mode,
                "total_trajectories": total_trajectories,
                "event_counts": event_counts,
                "storage": {
                    "type": self.config.store_type,
                    "url": self.config.store_url,
                },
            }

        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {"enabled": True, "error": str(e)}


# Global singleton instance
_lightning_service: Optional[LightningService] = None


def get_lightning_service(config: Optional[LightningConfig] = None) -> LightningService:
    """Get or create Lightning service singleton"""
    global _lightning_service

    if _lightning_service is None:
        if config is None:
            from isa_agent_sdk.core.config.lightning_config import LightningConfig

            config = LightningConfig.from_env()
        _lightning_service = LightningService(config)

    return _lightning_service
