#!/usr/bin/env python3
"""
Pool Manager Client - Manages agent execution in isolated VMs

Provides interface to pool_manager service for:
- Acquiring agent VMs from the pool
- Executing agent queries in isolated environments
- Streaming responses from agent VMs
- Releasing VMs back to pool
"""

import logging
import asyncio
import json
from typing import Dict, Any, Optional, AsyncIterator
import aiohttp

logger = logging.getLogger(__name__)


class PoolManagerClient:
    """
    Client for pool_manager service to run agents in isolated VMs

    Supports:
    - VM acquisition with timeout
    - Query execution with streaming
    - Session management (reset, get_state)
    - VM release/cleanup
    """

    def __init__(
        self,
        pool_manager_url: str = "http://localhost:8090",
        acquire_timeout: int = 30,
        request_timeout: int = 300
    ):
        """
        Initialize pool manager client

        Args:
            pool_manager_url: Pool manager service URL
            acquire_timeout: Seconds to wait for VM acquisition
            request_timeout: Seconds to wait for query execution
        """
        self.pool_manager_url = pool_manager_url.rstrip('/')
        self.acquire_timeout = acquire_timeout
        self.request_timeout = request_timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()

        logger.info(f"PoolManagerClient initialized: {pool_manager_url}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            async with self._lock:
                if self._session is None or self._session.closed:
                    timeout = aiohttp.ClientTimeout(total=self.request_timeout)
                    self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def acquire_agent_vm(
        self,
        agent_config: Dict[str, Any],
        session_id: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Acquire an agent VM from the pool

        Args:
            agent_config: Agent configuration (tools, system_prompt, etc.)
            session_id: User session ID for isolation
            user_id: Optional user ID for billing/tracking

        Returns:
            VM instance info with instance_id
        """
        session = await self._get_session()

        payload = {
            "pool_type": "agent",
            "config": agent_config,
            "session_id": session_id,
            "user_id": user_id,
            "timeout": self.acquire_timeout
        }

        try:
            async with session.post(
                f"{self.pool_manager_url}/api/v1/pools/agent/acquire",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.acquire_timeout + 5)
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Failed to acquire agent VM: {resp.status} - {error_text}")
                    raise RuntimeError(f"VM acquisition failed: {error_text}")

                result = await resp.json()
                logger.info(f"Acquired agent VM: {result.get('instance_id')}")
                return result

        except asyncio.TimeoutError:
            logger.error(f"VM acquisition timeout after {self.acquire_timeout}s")
            raise RuntimeError("Agent VM acquisition timeout")
        except aiohttp.ClientError as e:
            logger.error(f"VM acquisition network error: {e}")
            raise RuntimeError(f"Agent VM acquisition failed: {e}")

    async def execute_query(
        self,
        instance_id: str,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a query on an agent VM (non-streaming)

        Args:
            instance_id: VM instance ID from acquire
            query: User query to execute
            context: Optional additional context

        Returns:
            Query result with response and metadata
        """
        session = await self._get_session()

        payload = {
            "action": "query",
            "params": {
                "query": query,
                "context": context or {}
            }
        }

        try:
            async with session.post(
                f"{self.pool_manager_url}/api/v1/pools/agent/{instance_id}/execute",
                json=payload
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Query execution failed: {resp.status} - {error_text}")
                    raise RuntimeError(f"Query execution failed: {error_text}")

                return await resp.json()

        except asyncio.TimeoutError:
            logger.error(f"Query execution timeout after {self.request_timeout}s")
            raise RuntimeError("Query execution timeout")
        except aiohttp.ClientError as e:
            logger.error(f"Query execution network error: {e}")
            raise RuntimeError(f"Query execution failed: {e}")

    async def stream_query(
        self,
        instance_id: str,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream a query response from an agent VM

        Args:
            instance_id: VM instance ID from acquire
            query: User query to execute
            context: Optional additional context

        Yields:
            Streaming response chunks (tokens, tool calls, etc.)
        """
        session = await self._get_session()

        payload = {
            "query": query,
            "context": context or {}
        }

        try:
            async with session.post(
                f"{self.pool_manager_url}/api/v1/pools/agent/{instance_id}/stream",
                json=payload
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Stream query failed: {resp.status} - {error_text}")
                    raise RuntimeError(f"Stream query failed: {error_text}")

                # Parse SSE stream
                async for line in resp.content:
                    line = line.decode('utf-8').strip()

                    if not line:
                        continue

                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix

                        if data == '[DONE]':
                            break

                        try:
                            chunk = json.loads(data)
                            yield chunk
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in stream: {data}")
                            continue

        except asyncio.TimeoutError:
            logger.error(f"Stream query timeout after {self.request_timeout}s")
            raise RuntimeError("Stream query timeout")
        except aiohttp.ClientError as e:
            logger.error(f"Stream query network error: {e}")
            raise RuntimeError(f"Stream query failed: {e}")

    async def get_state(self, instance_id: str) -> Dict[str, Any]:
        """
        Get current state of an agent VM

        Args:
            instance_id: VM instance ID

        Returns:
            Agent state including conversation history, memory, etc.
        """
        session = await self._get_session()

        try:
            async with session.get(
                f"{self.pool_manager_url}/api/v1/pools/agent/{instance_id}/state"
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"Get state failed: {error_text}")

                return await resp.json()

        except aiohttp.ClientError as e:
            logger.error(f"Get state network error: {e}")
            raise RuntimeError(f"Get state failed: {e}")

    async def reset_session(self, instance_id: str) -> Dict[str, Any]:
        """
        Reset an agent VM session (clear conversation history)

        Args:
            instance_id: VM instance ID

        Returns:
            Reset confirmation
        """
        session = await self._get_session()

        payload = {
            "action": "reset",
            "params": {}
        }

        try:
            async with session.post(
                f"{self.pool_manager_url}/api/v1/pools/agent/{instance_id}/execute",
                json=payload
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"Reset failed: {error_text}")

                logger.info(f"Reset agent session: {instance_id}")
                return await resp.json()

        except aiohttp.ClientError as e:
            logger.error(f"Reset network error: {e}")
            raise RuntimeError(f"Reset failed: {e}")

    async def release_vm(self, instance_id: str) -> bool:
        """
        Release an agent VM back to the pool

        Args:
            instance_id: VM instance ID

        Returns:
            True if released successfully
        """
        session = await self._get_session()

        try:
            async with session.post(
                f"{self.pool_manager_url}/api/v1/pools/agent/{instance_id}/release"
            ) as resp:
                if resp.status not in (200, 204):
                    error_text = await resp.text()
                    logger.warning(f"VM release warning: {error_text}")
                    return False

                logger.info(f"Released agent VM: {instance_id}")
                return True

        except aiohttp.ClientError as e:
            logger.warning(f"VM release network error: {e}")
            return False

    async def health_check(self) -> bool:
        """Check if pool_manager service is available"""
        session = await self._get_session()

        try:
            async with session.get(
                f"{self.pool_manager_url}/health",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                return resp.status == 200
        except Exception as e:
            logger.warning(f"Pool manager health check failed: {e}")
            return False

    async def close(self):
        """Clean shutdown"""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("PoolManagerClient closed")


# Global singleton
_pool_client: Optional[PoolManagerClient] = None
_pool_lock = asyncio.Lock()


async def get_pool_manager_client(
    pool_manager_url: str = None,
    acquire_timeout: int = 30,
    request_timeout: int = 300
) -> PoolManagerClient:
    """Get thread-safe singleton pool manager client"""
    global _pool_client

    if _pool_client is None:
        async with _pool_lock:
            if _pool_client is None:
                if pool_manager_url is None:
                    from isa_agent_sdk.core.config import settings
                    pool_manager_url = settings.pool_manager_url
                    acquire_timeout = settings.pool_acquire_timeout
                    request_timeout = settings.pool_request_timeout

                _pool_client = PoolManagerClient(
                    pool_manager_url=pool_manager_url,
                    acquire_timeout=acquire_timeout,
                    request_timeout=request_timeout
                )
                logger.info("Global PoolManagerClient initialized")

    return _pool_client
