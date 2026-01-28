#!/usr/bin/env python3
"""
Base Client - Foundation for dual-mode clients (Local/Cloud)

Provides:
1. Backend protocol definition
2. Mode switching (local/cloud/auto)
3. Graceful fallback from cloud to local
4. Unified configuration interface

All clients (session, storage, model) inherit from this pattern.
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Protocol, TypeVar, Generic
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ClientMode(str, Enum):
    """Client operation mode"""
    LOCAL = "local"      # Use local backend (file/SQLite)
    CLOUD = "cloud"      # Use cloud backend (external service)
    AUTO = "auto"        # Try cloud first, fallback to local


@dataclass
class BackendConfig:
    """Configuration for backend selection"""
    mode: ClientMode = ClientMode.AUTO

    # Local backend settings
    local_storage_path: str = ".isa_agent"
    local_db_name: str = "isa_agent.db"

    # Cloud backend settings
    service_url: Optional[str] = None
    auth_token: Optional[str] = None
    timeout: float = 30.0

    # Fallback behavior
    fallback_to_local: bool = True

    # Additional settings
    extra: Dict[str, Any] = field(default_factory=dict)


class Backend(Protocol):
    """Protocol for backend implementations"""

    async def initialize(self) -> bool:
        """Initialize the backend, return True if successful"""
        ...

    async def health_check(self) -> bool:
        """Check if backend is healthy and accessible"""
        ...

    async def close(self) -> None:
        """Close backend connections"""
        ...


T = TypeVar('T', bound=Backend)


class BaseClient(ABC, Generic[T]):
    """
    Base client with dual-mode support (local/cloud)

    Features:
    - Automatic mode selection based on config
    - Graceful fallback from cloud to local
    - Unified interface regardless of backend

    Usage:
        client = SessionClient(config=BackendConfig(mode=ClientMode.AUTO))
        await client.initialize()
        # Use client...
        await client.close()
    """

    def __init__(self, config: Optional[BackendConfig] = None):
        self.config = config or BackendConfig()
        self._backend: Optional[T] = None
        self._is_initialized = False
        self._active_mode: Optional[ClientMode] = None

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def active_mode(self) -> Optional[ClientMode]:
        """Returns the currently active mode (local or cloud)"""
        return self._active_mode

    @property
    def backend(self) -> T:
        """Get the active backend, raises if not initialized"""
        if self._backend is None:
            raise RuntimeError(
                f"{self.__class__.__name__} not initialized. Call initialize() first."
            )
        return self._backend

    @abstractmethod
    def _create_local_backend(self) -> T:
        """Create local backend instance"""
        ...

    @abstractmethod
    def _create_cloud_backend(self) -> T:
        """Create cloud backend instance"""
        ...

    async def initialize(self) -> bool:
        """
        Initialize client with appropriate backend based on config mode

        Returns:
            True if initialization successful
        """
        if self._is_initialized:
            logger.debug(f"{self.__class__.__name__} already initialized")
            return True

        mode = self.config.mode

        if mode == ClientMode.LOCAL:
            return await self._init_local()

        elif mode == ClientMode.CLOUD:
            return await self._init_cloud()

        elif mode == ClientMode.AUTO:
            # Try cloud first, fallback to local
            if self.config.service_url:
                if await self._init_cloud():
                    return True

                if self.config.fallback_to_local:
                    logger.warning(
                        f"{self.__class__.__name__}: Cloud backend unavailable, "
                        "falling back to local"
                    )
                    return await self._init_local()
                else:
                    return False
            else:
                # No cloud URL configured, use local
                return await self._init_local()

        return False

    async def _init_local(self) -> bool:
        """Initialize with local backend"""
        try:
            self._backend = self._create_local_backend()
            if await self._backend.initialize():
                self._is_initialized = True
                self._active_mode = ClientMode.LOCAL
                logger.info(f"{self.__class__.__name__} initialized with LOCAL backend")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to initialize local backend: {e}")
            return False

    async def _init_cloud(self) -> bool:
        """Initialize with cloud backend"""
        try:
            self._backend = self._create_cloud_backend()
            if await self._backend.initialize():
                # Verify cloud is accessible
                if await self._backend.health_check():
                    self._is_initialized = True
                    self._active_mode = ClientMode.CLOUD
                    logger.info(f"{self.__class__.__name__} initialized with CLOUD backend")
                    return True
                else:
                    logger.warning(f"Cloud backend health check failed")
                    return False
            return False
        except Exception as e:
            logger.error(f"Failed to initialize cloud backend: {e}")
            return False

    async def close(self) -> None:
        """Close client and cleanup resources"""
        if self._backend:
            await self._backend.close()
            self._backend = None
        self._is_initialized = False
        self._active_mode = None
        logger.info(f"{self.__class__.__name__} closed")

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
