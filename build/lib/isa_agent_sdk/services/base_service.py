#!/usr/bin/env python3
"""
Base Service - Unified service management base class
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncIterator
from isa_agent_sdk.utils.logger import api_logger


class BaseService(ABC):
    """Base service class for unified service management"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = api_logger
        self._initialized = False
    
    @abstractmethod
    async def service_init(self):
        """Service initialization - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    async def execute(self, *args, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """Service execution - must be implemented by subclasses"""
        pass
    
    async def ensure_initialized(self):
        """Ensure service is initialized"""
        if not self._initialized:
            await self.service_init()
            self._initialized = True