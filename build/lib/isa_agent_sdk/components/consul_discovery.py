#!/usr/bin/env python3
"""
Consul Service Discovery Client
"""

import logging
import consul
from typing import Optional, Dict, List
import os

logger = logging.getLogger(__name__)

class ConsulServiceDiscovery:
    """Consul service discovery client"""
    
    def __init__(self, consul_host: str = "localhost", consul_port: int = 8500):
        """Initialize Consul client"""
        self.consul_host = consul_host
        self.consul_port = consul_port
        self._consul = consul.Consul(host=consul_host, port=consul_port)
        
    def get_service_url(self, service_name: str, default_url: Optional[str] = None) -> Optional[str]:
        """
        Get service URL from Consul by service name

        Args:
            service_name: Name of the service to discover (e.g., 'mcp', 'models')
            default_url: Fallback URL if service not found

        Returns:
            Service URL or default_url if not found
        """
        try:
            # Get all services (temporarily disabled health check for staging)
            services = self._consul.health.service(service_name, passing=False)[1]

            if not services:
                logger.warning(f"No service instances found for service '{service_name}'")
                return default_url

            # Use first available service instance (health check disabled for staging)
            service = services[0]
            service_info = service['Service']
            node_info = service.get('Node', {})

            # Get address and port
            service_address = service_info.get('Address', '')
            port = service_info['Port']

            # In Docker environments, service address might be container ID
            # Extract service name from service ID (format: servicename-containerid-port)
            service_id = service_info.get('ID', '')

            # Try to extract container name from service tags or metadata
            # Docker Consul registration typically uses pattern: servicename-containerid-port
            if service_id and '-' in service_id:
                # Extract potential container name (everything before last two hyphens)
                parts = service_id.rsplit('-', 2)
                if len(parts) == 3:
                    potential_container_name = parts[0]
                    # Use container name as hostname for Docker networking
                    address = potential_container_name
                    logger.info(f"Using Docker service name '{address}' from service ID '{service_id}'")
                else:
                    address = service_address
            else:
                address = service_address

            # Fallback: if address still looks like container ID (12 hex chars), log warning
            if address and len(address) == 12 and all(c in '0123456789abcdef' for c in address.lower()):
                logger.warning(f"Service address '{address}' appears to be container ID but couldn't resolve to service name")

            service_url = f"http://{address}:{port}"
            logger.info(f"Discovered service '{service_name}' at {service_url}")
            return service_url

        except Exception as e:
            logger.error(f"Failed to discover service '{service_name}': {e}")
            return default_url
    
    def get_all_services(self) -> Dict[str, List[str]]:
        """Get all registered services"""
        try:
            services = self._consul.agent.services()
            result = {}
            for service_id, service_info in services.items():
                service_name = service_info['Service']
                if service_name not in result:
                    result[service_name] = []
                service_url = f"http://{service_info['Address']}:{service_info['Port']}"
                result[service_name].append(service_url)
            return result
        except Exception as e:
            logger.error(f"Failed to get all services: {e}")
            return {}

# Global discovery instance
_discovery_instance = None

def get_consul_discovery() -> ConsulServiceDiscovery:
    """Get global Consul discovery instance"""
    global _discovery_instance
    if _discovery_instance is None:
        consul_host = os.getenv("CONSUL_HOST", "localhost")
        consul_port = int(os.getenv("CONSUL_PORT", "8500"))
        _discovery_instance = ConsulServiceDiscovery(consul_host, consul_port)
    return _discovery_instance

def discover_service_url(service_name: str, default_url: Optional[str] = None) -> Optional[str]:
    """Convenient function to discover service URL"""
    discovery = get_consul_discovery()
    return discovery.get_service_url(service_name, default_url)