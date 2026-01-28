#!/usr/bin/env python3
"""
Graph Configuration Service

Manages SmartAgentGraphBuilder configuration with persistence and dynamic updates.
Provides CRUD operations for graph builder settings.
"""

import json
import logging
import base64
from datetime import datetime
from typing import Dict, Any, Optional, List, AsyncIterator
from pydantic import BaseModel, Field, validator
from enum import Enum

from .smart_agent_graph import SmartAgentGraphBuilder
from isa_agent_sdk.services.base_service import BaseService

logger = logging.getLogger(__name__)


class GuardrailMode(str, Enum):
    """Supported guardrail modes"""
    LENIENT = "lenient"
    MODERATE = "moderate" 
    STRICT = "strict"
    ENTERPRISE = "enterprise"


class GraphBuilderConfig(BaseModel):
    """Graph Builder Configuration Schema"""
    
    # Core configuration
    config_name: str = Field(..., description="Configuration name/identifier")
    config_version: str = Field(default="1.0", description="Configuration version")
    is_active: bool = Field(default=True, description="Whether this config is active")
    
    # Guardrail settings
    guardrail_enabled: bool = Field(default=False, description="Enable guardrail node")
    guardrail_mode: GuardrailMode = Field(default=GuardrailMode.MODERATE, description="Guardrail strictness level")
    
    # Failsafe settings
    failsafe_enabled: bool = Field(default=True, description="Enable failsafe node")
    confidence_threshold: float = Field(default=0.7, description="Confidence threshold for failsafe", ge=0.0, le=1.0)
    
    # Performance limits
    max_graph_iterations: int = Field(default=50, description="Maximum graph iterations", ge=1, le=1000)
    max_agent_loops: int = Field(default=10, description="Maximum agent loops", ge=1, le=100) 
    max_tool_loops: int = Field(default=5, description="Maximum tool loops", ge=1, le=50)
    
    # Cache policies
    llm_cache_ttl: int = Field(default=300, description="LLM cache TTL in seconds", ge=60, le=3600)
    tool_cache_ttl: int = Field(default=120, description="Tool cache TTL in seconds", ge=30, le=1800)
    
    # Retry policies
    max_retry_attempts: int = Field(default=3, description="Maximum retry attempts", ge=1, le=10)
    
    # Metadata
    description: Optional[str] = Field(None, description="Configuration description")
    created_by: Optional[str] = Field(None, description="Configuration creator")
    tags: List[str] = Field(default_factory=list, description="Configuration tags")
    
    # Timestamps
    created_at: Optional[datetime] = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = Field(default_factory=datetime.now)
    
    @validator('confidence_threshold')
    def validate_confidence_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence threshold must be between 0.0 and 1.0')
        return v
    
    def to_builder_config(self) -> Dict[str, Any]:
        """Convert to SmartAgentGraphBuilder config format"""
        return {
            "guardrail_enabled": self.guardrail_enabled,
            "guardrail_mode": self.guardrail_mode.value,
            "failsafe_enabled": self.failsafe_enabled,
            "confidence_threshold": self.confidence_threshold,
            "max_graph_iterations": self.max_graph_iterations,
            "max_agent_loops": self.max_agent_loops,
            "max_tool_loops": self.max_tool_loops,
            "llm_cache_ttl": self.llm_cache_ttl,
            "tool_cache_ttl": self.tool_cache_ttl,
            "max_retry_attempts": self.max_retry_attempts
        }


class GraphConfigService(BaseService):
    """Service for managing Graph Builder configurations"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
        # In-memory storage for now (can be replaced with database later)
        self._configurations: Dict[str, GraphBuilderConfig] = {}
        self._active_config: Optional[str] = None
        self._graph_builder_instance: Optional[SmartAgentGraphBuilder] = None
        
        # Load default configuration
        self._load_default_configuration()
    
    async def service_init(self):
        """Initialize service"""
        self.logger.info("GraphConfigService initialized")
    
    async def execute(self, *args, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """Service execution interface - not used for configuration service"""
        yield {"status": "ready", "service": "GraphConfigService"}
    
    def _load_default_configuration(self):
        """Load default graph builder configuration"""
        default_config = GraphBuilderConfig(
            config_name="default",
            config_version="1.0",
            description="Default graph builder configuration",
            created_by="system",
            tags=["default", "system"]
        )
        
        self._configurations["default"] = default_config
        self._active_config = "default"
        self.logger.info("Default configuration loaded")
    
    async def create_configuration(self, config: GraphBuilderConfig) -> GraphBuilderConfig:
        """
        Create a new graph builder configuration
        
        Args:
            config: Graph builder configuration
            
        Returns:
            Created configuration
        """
        try:
            if config.config_name in self._configurations:
                raise ValueError(f"Configuration '{config.config_name}' already exists")
            
            # Update timestamps
            config.created_at = datetime.now()
            config.updated_at = datetime.now()
            
            # Store configuration
            self._configurations[config.config_name] = config
            
            self.logger.info(f"Configuration '{config.config_name}' created")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to create configuration: {e}")
            raise
    
    async def get_configuration(self, config_name: str) -> Optional[GraphBuilderConfig]:
        """
        Get configuration by name
        
        Args:
            config_name: Configuration name
            
        Returns:
            Configuration or None if not found
        """
        return self._configurations.get(config_name)
    
    async def list_configurations(self, include_inactive: bool = False) -> List[GraphBuilderConfig]:
        """
        List all configurations
        
        Args:
            include_inactive: Whether to include inactive configurations
            
        Returns:
            List of configurations
        """
        configs = list(self._configurations.values())
        
        if not include_inactive:
            configs = [config for config in configs if config.is_active]
        
        # Sort by creation date
        configs.sort(key=lambda x: x.created_at or datetime.min, reverse=True)
        
        return configs
    
    async def update_configuration(self, config_name: str, updates: Dict[str, Any]) -> GraphBuilderConfig:
        """
        Update existing configuration
        
        Args:
            config_name: Configuration name
            updates: Fields to update
            
        Returns:
            Updated configuration
        """
        try:
            if config_name not in self._configurations:
                raise ValueError(f"Configuration '{config_name}' not found")
            
            config = self._configurations[config_name]
            
            # Update fields
            for field, value in updates.items():
                if hasattr(config, field):
                    setattr(config, field, value)
            
            # Update timestamp
            config.updated_at = datetime.now()
            
            # If this config was active, invalidate the builder instance
            if config_name == self._active_config:
                self._graph_builder_instance = None
            
            self.logger.info(f"Configuration '{config_name}' updated")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            raise
    
    async def delete_configuration(self, config_name: str) -> bool:
        """
        Delete configuration
        
        Args:
            config_name: Configuration name
            
        Returns:
            True if deleted, False if not found
        """
        try:
            if config_name == "default":
                raise ValueError("Cannot delete default configuration")
            
            if config_name not in self._configurations:
                return False
            
            # If this was the active config, switch to default
            if config_name == self._active_config:
                self._active_config = "default"
                self._graph_builder_instance = None
            
            del self._configurations[config_name]
            
            self.logger.info(f"Configuration '{config_name}' deleted")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete configuration: {e}")
            raise
    
    async def set_active_configuration(self, config_name: str) -> GraphBuilderConfig:
        """
        Set active configuration
        
        Args:
            config_name: Configuration name to activate
            
        Returns:
            Activated configuration
        """
        try:
            if config_name not in self._configurations:
                raise ValueError(f"Configuration '{config_name}' not found")
            
            config = self._configurations[config_name]
            
            if not config.is_active:
                raise ValueError(f"Configuration '{config_name}' is inactive")
            
            # Set as active
            old_active = self._active_config
            self._active_config = config_name
            
            # Invalidate builder instance to force rebuild
            self._graph_builder_instance = None
            
            self.logger.info(f"Active configuration changed from '{old_active}' to '{config_name}'")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to set active configuration: {e}")
            raise
    
    async def get_active_configuration(self) -> GraphBuilderConfig:
        """
        Get currently active configuration
        
        Returns:
            Active configuration
        """
        if not self._active_config or self._active_config not in self._configurations:
            self._active_config = "default"
        
        return self._configurations[self._active_config]
    
    async def build_graph_with_config(self, config_name: Optional[str] = None) -> Any:
        """
        Build graph using specified or active configuration
        
        Args:
            config_name: Configuration name (uses active if None)
            
        Returns:
            Compiled LangGraph
        """
        try:
            # Get configuration
            if config_name:
                if config_name not in self._configurations:
                    raise ValueError(f"Configuration '{config_name}' not found")
                config = self._configurations[config_name]
            else:
                config = await self.get_active_configuration()
            
            # Build graph
            builder_config = config.to_builder_config()
            graph_builder = SmartAgentGraphBuilder(builder_config)
            
            # Configure additional settings
            graph_builder.configure_guardrails(
                enabled=config.guardrail_enabled,
                mode=config.guardrail_mode.value
            )
            
            graph_builder.configure_failsafe(
                enabled=config.failsafe_enabled,
                confidence_threshold=config.confidence_threshold
            )
            
            # Build and return graph
            graph = graph_builder.build_graph()
            
            self.logger.info(f"Graph built using configuration '{config.config_name}'")
            return graph
            
        except Exception as e:
            self.logger.error(f"Failed to build graph with config: {e}")
            raise
    
    async def get_graph_builder(self, config_name: Optional[str] = None, force_rebuild: bool = False) -> SmartAgentGraphBuilder:
        """
        Get graph builder instance with specified or active configuration
        
        Args:
            config_name: Configuration name (uses active if None)
            force_rebuild: Force rebuild even if cached instance exists
            
        Returns:
            SmartAgentGraphBuilder instance
        """
        try:
            # Determine which config to use
            target_config = config_name or self._active_config
            
            # Check if we need to rebuild
            if (force_rebuild or 
                self._graph_builder_instance is None or 
                getattr(self._graph_builder_instance, '_config_name', None) != target_config):
                
                # Get configuration
                if target_config not in self._configurations:
                    raise ValueError(f"Configuration '{target_config}' not found")
                
                config = self._configurations[target_config]
                builder_config = config.to_builder_config()
                
                # Create new builder instance
                self._graph_builder_instance = SmartAgentGraphBuilder(builder_config)
                self._graph_builder_instance._config_name = target_config  # Track which config was used
                
                # Configure additional settings
                self._graph_builder_instance.configure_guardrails(
                    enabled=config.guardrail_enabled,
                    mode=config.guardrail_mode.value
                )
                
                self._graph_builder_instance.configure_failsafe(
                    enabled=config.failsafe_enabled,
                    confidence_threshold=config.confidence_threshold
                )
                
                self.logger.info(f"Graph builder created with configuration '{target_config}'")
            
            return self._graph_builder_instance
            
        except Exception as e:
            self.logger.error(f"Failed to get graph builder: {e}")
            raise
    
    async def validate_configuration(self, config: GraphBuilderConfig) -> Dict[str, Any]:
        """
        Validate configuration by attempting to build a graph
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validation result with success status and any warnings
        """
        try:
            # Create temporary builder
            builder_config = config.to_builder_config()
            temp_builder = SmartAgentGraphBuilder(builder_config)
            
            # Configure settings
            temp_builder.configure_guardrails(
                enabled=config.guardrail_enabled,
                mode=config.guardrail_mode.value
            )
            
            temp_builder.configure_failsafe(
                enabled=config.failsafe_enabled,
                confidence_threshold=config.confidence_threshold
            )
            
            # Attempt to build graph
            temp_graph = temp_builder.build_graph()
            
            # Get graph info for validation details
            graph_info = temp_builder.get_graph_info()
            
            return {
                "valid": True,
                "message": "Configuration is valid",
                "graph_info": graph_info,
                "warnings": []
            }
            
        except Exception as e:
            return {
                "valid": False,
                "message": f"Configuration validation failed: {str(e)}",
                "graph_info": None,
                "warnings": []
            }
    
    async def visualize_graph(self, config_name: str, format: str = "mermaid") -> Dict[str, Any]:
        """
        Generate graph visualization in specified format
        
        Args:
            config_name: Configuration name to visualize
            format: Visualization format ("mermaid", "png", "ascii")
            
        Returns:
            Visualization data with metadata
        """
        try:
            # Get configuration
            if config_name not in self._configurations:
                raise ValueError(f"Configuration '{config_name}' not found")
            
            config = self._configurations[config_name]
            
            # Build graph for visualization
            builder_config = config.to_builder_config()
            temp_builder = SmartAgentGraphBuilder(builder_config)
            
            # Configure settings
            temp_builder.configure_guardrails(
                enabled=config.guardrail_enabled,
                mode=config.guardrail_mode.value
            )
            
            temp_builder.configure_failsafe(
                enabled=config.failsafe_enabled,
                confidence_threshold=config.confidence_threshold
            )
            
            # Get visualization content
            content = await temp_builder.get_visualization(format)
            
            # Generate metadata
            graph_info = temp_builder.get_graph_info()
            metadata = {
                "config_name": config_name,
                "nodes_count": len(graph_info["nodes"]),
                "features": graph_info["features"],
                "guardrail_enabled": config.guardrail_enabled,
                "failsafe_enabled": config.failsafe_enabled,
                "generated_at": datetime.now().isoformat()
            }
            
            self.logger.info(f"Generated {format} visualization for configuration '{config_name}'")
            
            return {
                "config_name": config_name,
                "format": format,
                "content": content,
                "metadata": metadata
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate visualization: {e}")
            raise
    
    async def get_graph_structure(self, config_name: str) -> Dict[str, Any]:
        """
        Get detailed graph structure information
        
        Args:
            config_name: Configuration name
            
        Returns:
            Detailed graph structure data
        """
        try:
            if config_name not in self._configurations:
                raise ValueError(f"Configuration '{config_name}' not found")
            
            config = self._configurations[config_name]
            builder = await self.get_graph_builder(config_name)
            graph_info = builder.get_graph_info()
            
            # Get detailed node information
            nodes_detail = await builder.get_nodes_detail()
            edges_detail = await builder.get_edges_detail()
            
            return {
                "config_name": config_name,
                "architecture": graph_info["architecture"],
                "nodes": {
                    "list": graph_info["nodes"],
                    "details": nodes_detail
                },
                "edges": edges_detail,
                "routing_logic": {
                    "guardrail_enabled": config.guardrail_enabled,
                    "failsafe_enabled": config.failsafe_enabled,
                    "confidence_threshold": config.confidence_threshold
                },
                "limits": graph_info["recursion_limits"],
                "features": graph_info["features"]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get graph structure: {e}")
            raise

    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            "total_configurations": len(self._configurations),
            "active_configuration": self._active_config,
            "configurations": [
                {
                    "name": config.config_name,
                    "version": config.config_version,
                    "active": name == self._active_config,
                    "created_at": config.created_at.isoformat() if config.created_at else None
                }
                for name, config in self._configurations.items()
            ],
            "builder_cached": self._graph_builder_instance is not None,
            "visualization_supported": True,
            "supported_formats": ["mermaid", "png", "ascii"]
        }


# Global service instance
_graph_config_service = None


async def get_graph_config_service() -> GraphConfigService:
    """Get global graph configuration service instance"""
    global _graph_config_service
    
    if _graph_config_service is None:
        _graph_config_service = GraphConfigService()
        await _graph_config_service.service_init()
    
    return _graph_config_service