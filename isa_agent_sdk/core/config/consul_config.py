"""Consul Service Discovery Configuration"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ConsulConfig:
    """Consul configuration and service discovery helper"""

    def __init__(self):
        self.host = os.getenv("CONSUL_HOST", "localhost")
        self.port = int(os.getenv("CONSUL_PORT", "8500"))
        self.enabled = os.getenv("CONSUL_ENABLED", "true").lower() == "true"

    def discover_service(self, service_name: str, fallback_url: str) -> str:
        """
        Discover service URL from Consul

        Args:
            service_name: Service name in Consul registry
            fallback_url: Fallback URL if discovery fails

        Returns:
            Service URL
        """
        if not self.enabled:
            logger.debug(f"Consul disabled, using fallback for {service_name}")
            return fallback_url

        try:
            from isa_common.consul_client import ConsulRegistry
            consul = ConsulRegistry(self.host, self.port)
            url = consul.get_service_address(service_name, fallback_url, max_retries=1)
            logger.debug(f"Discovered {service_name}: {url}")
            return url
        except Exception as e:
            logger.warning(f"Consul discovery failed for {service_name}: {e}")
            return fallback_url

    @property
    def is_enabled(self) -> bool:
        return self.enabled


consul_config = ConsulConfig()
