#!/usr/bin/env python3
"""
Graph Registry with Authorization Integration

Manages multiple graph implementations with permission-based access control.
Integrates with the authorization service to control who can use which graphs.
"""

import logging
import httpx
import requests  # Add requests for sync HTTP calls
from typing import Dict, Optional, Any, List, Type
from enum import Enum
from datetime import datetime

from .base_graph import BaseGraph
from .smart_agent_graph import SmartAgentGraphBuilder
from .research_graph import ResearchGraph
from .coding_graph import CodingGraph
from .conversation_graph import ConversationGraph

logger = logging.getLogger(__name__)

# Authorization service configuration
from isa_agent_sdk.core.config import settings
AUTH_SERVICE_URL = settings.auth_service_url


class GraphType(str, Enum):
    """Available graph types (treated as resources in auth system)"""
    DEFAULT = "graph:default"           # Standard SmartAgent graph
    RESEARCH = "graph:research"         # Research-focused graph
    CODING = "graph:coding"            # Code generation graph
    CONVERSATION = "graph:conversation" # Simple conversation graph
    HARDWARE = "graph:hardware"        # Hardware integration graph
    CUSTOM = "graph:custom"            # User-defined custom graph


class GraphAccessLevel(str, Enum):
    """Access levels for graphs"""
    NONE = "none"
    READ_ONLY = "read_only"      # Can use the graph
    READ_WRITE = "read_write"    # Can use and configure the graph
    ADMIN = "admin"              # Can manage graph permissions
    OWNER = "owner"              # Full control


class GraphRegistryWithAuth:
    """
    Graph registry with authorization integration
    """

    # User-friendly graph type name mapping
    GRAPH_TYPE_ALIASES = {
        "smart_agent": GraphType.DEFAULT,
        "default": GraphType.DEFAULT,
        "conversation": GraphType.CONVERSATION,
        "chat": GraphType.CONVERSATION,
        "research": GraphType.RESEARCH,
        "coding": GraphType.CODING,
        "code": GraphType.CODING,
        "hardware": GraphType.HARDWARE,
        "custom": GraphType.CUSTOM,
    }

    def __init__(self):
        """Initialize the graph registry with auth"""
        self._graph_classes: Dict[str, Type[BaseGraph]] = {}
        self._graph_instances: Dict[str, Any] = {}
        self._compiled_graphs: Dict[str, Any] = {}
        self._active_graphs: Dict[str, str] = {}  # user_id -> active graph
        self._graph_configs: Dict[str, Dict] = {}  # Graph configurations

        # Register built-in graphs
        self._register_builtin_graphs()

        # Initialize graph permissions in auth service
        self._initialize_graph_permissions()

        logger.info("GraphRegistryWithAuth initialized")
    
    def _register_builtin_graphs(self):
        """Register all built-in graph implementations"""
        
        # Define graph configurations with required permissions
        self._graph_configs = {
            GraphType.DEFAULT: {
                "class": SmartAgentGraphWrapper,
                "name": "Default Smart Agent",
                "description": "Standard agent with full feature set",
                "subscription_required": "free",
                "access_level": "read_only",
                "features": ["tools", "agents", "failsafe", "checkpointing"]
            },
            GraphType.RESEARCH: {
                "class": ResearchGraph,
                "name": "Research Graph",
                "description": "Deep research with enhanced search",
                "subscription_required": "pro",
                "access_level": "read_only",
                "features": ["deep_search", "multi_source", "citations"]
            },
            GraphType.CODING: {
                "class": CodingGraph,
                "name": "Coding Graph",
                "description": "Code generation and debugging",
                "subscription_required": "pro",
                "access_level": "read_write",
                "features": ["code_gen", "testing", "debugging"]
            },
            GraphType.CONVERSATION: {
                "class": ConversationGraph,
                "name": "Conversation Graph",
                "description": "Simple chat without tools",
                "subscription_required": "free",
                "access_level": "read_only",
                "features": ["fast_response", "minimal"]
            }
        }
        
        # Register graph classes
        for graph_type, config in self._graph_configs.items():
            self._graph_classes[graph_type] = config["class"]
        
        logger.info(f"Registered {len(self._graph_classes)} built-in graphs")
    
    def _initialize_graph_permissions(self):
        """Initialize graph permissions in authorization service"""
        try:
            # This would be called during service startup to register graphs as resources
            for graph_type, config in self._graph_configs.items():
                resource_config = {
                    "resource_type": "ai_model",  # Fixed: auth service uses ai_model
                    "resource_name": graph_type,
                    "subscription_required": config["subscription_required"],
                    "default_access_level": config["access_level"],
                    "description": config["description"],
                    "metadata": {
                        "features": config["features"]
                    }
                }
                # In production, this would register with auth service
                logger.info(f"Graph permission registered: {graph_type}")
        except Exception as e:
            logger.error(f"Failed to initialize graph permissions: {e}")

    def _normalize_graph_type(self, graph_type: str) -> str:
        """
        Normalize user-friendly graph type names to canonical registry values

        Args:
            graph_type: User-provided graph type (e.g., "smart_agent", "conversation")

        Returns:
            Canonical graph type from GraphType enum (e.g., "graph:default")
        """
        # If already a canonical GraphType value, return as-is
        if graph_type in [gt.value for gt in GraphType]:
            return graph_type

        # Try to map from user-friendly alias
        normalized = self.GRAPH_TYPE_ALIASES.get(graph_type.lower())
        if normalized:
            logger.info(f"Normalized graph type: '{graph_type}' → '{normalized}'")
            return normalized

        # Unknown type, log warning and return original
        logger.warning(f"Unknown graph type '{graph_type}', using as-is")
        return graph_type
    
    async def check_graph_access(
        self, 
        user_id: str, 
        graph_type: str, 
        required_level: str = "read_only"
    ) -> Dict[str, Any]:
        """
        Check if user has access to a specific graph
        
        Args:
            user_id: User ID
            graph_type: Graph type to check
            required_level: Required access level
            
        Returns:
            Access check result from auth service
        """
        try:
            # Use synchronous requests instead of async httpx
            response = requests.post(
                f"{AUTH_SERVICE_URL}/check-access",
                json={
                    "user_id": user_id,
                    "resource_type": "ai_model",  # Fixed: auth service uses ai_model, not ai_graph
                    "resource_name": graph_type,
                    "required_access_level": required_level
                },
                timeout=3.0  # 3 second timeout
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"Access check for {user_id} to {graph_type}: {result['has_access']}")
                return result
            else:
                logger.error(f"Auth service error: {response.status_code}")
                return {
                    "has_access": False,
                    "reason": "Authorization service unavailable"
                }
                    
        except Exception as e:
            logger.error(f"Failed to check graph access: {e}")
            # Fallback to default graph for free users
            if graph_type == GraphType.DEFAULT:
                return {
                    "has_access": True,
                    "reason": "Default graph always available",
                    "fallback": True
                }
            return {
                "has_access": False,
                "reason": f"Authorization check failed: {str(e)}"
            }
    
    async def get_available_graphs(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get list of graphs available to a specific user
        
        Args:
            user_id: User ID
            
        Returns:
            List of available graphs with access levels
        """
        available = []
        
        for graph_type, config in self._graph_configs.items():
            access_result = await self.check_graph_access(user_id, graph_type)
            
            if access_result["has_access"]:
                available.append({
                    "type": graph_type,
                    "name": config["name"],
                    "description": config["description"],
                    "access_level": access_result.get("user_access_level", "read_only"),
                    "features": config["features"],
                    "subscription_tier": access_result.get("subscription_tier"),
                    "is_active": self._active_graphs.get(user_id) == graph_type
                })
        
        return available
    
    async def get_graph_instance(
        self,
        user_id: str,
        graph_type: str,
        config: Optional[Dict] = None,
        force_new: bool = False
    ) -> Optional[BaseGraph]:
        """
        Get or create a graph instance with permission check

        Args:
            user_id: User ID requesting the graph
            graph_type: Type of graph to get
            config: Optional configuration
            force_new: Force creation of new instance

        Returns:
            Graph instance if authorized, None otherwise
        """
        import time

        # Check access
        access_start = time.time()
        access_result = await self.check_graph_access(user_id, graph_type)
        access_duration = int((time.time() - access_start) * 1000)
        logger.info(f"graph_access_check | user_id={user_id} | graph_type={graph_type} | has_access={access_result['has_access']} | duration_ms={access_duration}")
        
        if not access_result["has_access"]:
            logger.warning(f"User {user_id} denied access to {graph_type}: {access_result['reason']}")
            return None
        
        # Create cache key with user context
        cache_key = f"{user_id}:{graph_type}"
        
        if force_new or cache_key not in self._graph_instances:
            if graph_type not in self._graph_classes:
                logger.error(f"Unknown graph type: {graph_type}")
                return None
            
            graph_class = self._graph_classes[graph_type]
            
            # Add user context to config
            user_config = config or {}
            user_config["user_id"] = user_id
            user_config["access_level"] = access_result.get("user_access_level", "read_only")
            
            # Apply access level restrictions
            if user_config["access_level"] == "read_only":
                # Restrict certain features for read-only users
                user_config["allow_modifications"] = False
                user_config["max_iterations"] = min(
                    user_config.get("max_iterations", 50), 
                    50
                )
            
            self._graph_instances[cache_key] = graph_class(user_config)
            logger.info(f"Created graph instance for {user_id}: {graph_type}")
        
        return self._graph_instances[cache_key]
    
    async def build_graph(
        self,
        user_id: str,
        graph_type: str,
        config: Optional[Dict] = None,
        force_rebuild: bool = False
    ) -> Optional[Any]:
        """
        Build and compile a graph with permission check

        Args:
            user_id: User ID
            graph_type: Type of graph to build (can be user-friendly name or canonical type)
            config: Optional configuration
            force_rebuild: Force rebuild even if cached

        Returns:
            Compiled graph if authorized, None otherwise
        """
        import time
        build_start = time.time()

        # Normalize graph type to canonical value
        normalized_type = self._normalize_graph_type(graph_type)
        cache_key = f"{user_id}:{normalized_type}"

        if force_rebuild or cache_key not in self._compiled_graphs:
            instance_start = time.time()
            instance = await self.get_graph_instance(user_id, normalized_type, config)
            instance_duration = int((time.time() - instance_start) * 1000)
            logger.info(f"graph_instance_created | user_id={user_id} | graph_type={graph_type} | normalized={normalized_type} | duration_ms={instance_duration}")

            if not instance:
                return None

            compile_start = time.time()
            self._compiled_graphs[cache_key] = instance.build_graph()
            compile_duration = int((time.time() - compile_start) * 1000)
            logger.info(f"graph_compiled | user_id={user_id} | graph_type={normalized_type} | duration_ms={compile_duration}")

        build_duration = int((time.time() - build_start) * 1000)
        logger.info(f"graph_build_complete | user_id={user_id} | graph_type={graph_type} | normalized={normalized_type} | cached={cache_key in self._compiled_graphs} | duration_ms={build_duration}")

        return self._compiled_graphs[cache_key]
    
    async def get_active_graph(
        self,
        user_id: str,
        config: Optional[Dict] = None
    ) -> Any:
        """
        Get the currently active graph for a user
        
        Args:
            user_id: User ID
            config: Optional configuration
            
        Returns:
            Active compiled graph
        """
        active_type = self._active_graphs.get(user_id, GraphType.DEFAULT)
        
        # Try to get the active graph
        graph = await self.build_graph(user_id, active_type, config)
        
        # Fallback to default if necessary
        if graph is None and active_type != GraphType.DEFAULT:
            logger.info(f"Falling back to default graph for {user_id}")
            graph = await self.build_graph(user_id, GraphType.DEFAULT, config)
            self._active_graphs[user_id] = GraphType.DEFAULT
        
        return graph
    
    async def set_active_graph(
        self,
        user_id: str,
        graph_type: str
    ) -> Optional[str]:
        """
        Set the active graph for a user
        
        Args:
            user_id: User ID
            graph_type: Graph type to activate
            
        Returns:
            The activated graph type if successful, None otherwise
        """
        # Check access
        access_result = await self.check_graph_access(user_id, graph_type)
        
        if not access_result["has_access"]:
            logger.warning(f"User {user_id} cannot activate {graph_type}: {access_result['reason']}")
            return None
        
        old_active = self._active_graphs.get(user_id)
        self._active_graphs[user_id] = graph_type
        
        logger.info(f"User {user_id} active graph: {old_active} -> {graph_type}")
        return graph_type
    
    async def select_graph_for_task(
        self,
        user_id: str,
        task_description: str,
        context: Optional[Dict] = None
    ) -> str:
        """
        Automatically select the best graph for a task (with permission check)
        
        Args:
            user_id: User ID
            task_description: Description of the task
            context: Optional context information
            
        Returns:
            Selected graph type that user has access to
        """
        task_lower = task_description.lower()
        
        # Determine ideal graph based on task
        ideal_graph = GraphType.DEFAULT
        
        if any(word in task_lower for word in ["research", "search", "find", "investigate"]):
            ideal_graph = GraphType.RESEARCH
        elif any(word in task_lower for word in ["code", "program", "debug", "implement"]):
            ideal_graph = GraphType.CODING
        elif any(word in task_lower for word in ["chat", "talk", "conversation"]):
            ideal_graph = GraphType.CONVERSATION
        
        # Check if user has access to ideal graph
        access_result = await self.check_graph_access(user_id, ideal_graph)
        
        if access_result["has_access"]:
            logger.info(f"Selected {ideal_graph} for user {user_id}")
            return ideal_graph
        else:
            # Fallback logic based on user's subscription
            subscription = access_result.get("subscription_tier", "free")
            
            if subscription == "free":
                # Free users get conversation or default
                if ideal_graph == GraphType.RESEARCH or ideal_graph == GraphType.CODING:
                    logger.info(f"User {user_id} ({subscription}) fallback to default graph")
                    return GraphType.DEFAULT
                return GraphType.CONVERSATION
            else:
                # Pro/Enterprise users should have access to most graphs
                logger.warning(f"Unexpected access denial for {user_id} to {ideal_graph}")
                return GraphType.DEFAULT
    
    async def grant_graph_access(
        self,
        admin_id: str,
        user_id: str,
        graph_type: str,
        access_level: str = "read_only",
        expires_in_days: Optional[int] = None,
        reason: Optional[str] = None
    ) -> bool:
        """
        Grant user access to a specific graph (admin only)
        
        Args:
            admin_id: Admin user granting access
            user_id: User receiving access
            graph_type: Graph type
            access_level: Access level to grant
            expires_in_days: Optional expiration
            reason: Reason for grant
            
        Returns:
            True if successful
        """
        try:
            # Use synchronous requests
            response = requests.post(
                f"{AUTH_SERVICE_URL}/grant",
                json={
                    "user_id": user_id,
                    "resource_type": "ai_model",  # Fixed: use ai_model
                    "resource_name": graph_type,
                    "access_level": access_level,
                    "permission_source": "admin_grant",
                    "granted_by": admin_id,
                    "expires_in_days": expires_in_days,
                    "reason": reason or f"Graph access granted by {admin_id}"
                },
                timeout=3.0
            )

            success = response.status_code == 200
            if success:
                logger.info(f"Granted {user_id} access to {graph_type} by {admin_id}")
            else:
                logger.error(f"Failed to grant access: {response.text}")

            return success
                
        except Exception as e:
            logger.error(f"Failed to grant graph access: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            "total_graph_types": len(self._graph_classes),
            "instantiated_graphs": len(self._graph_instances),
            "compiled_graphs": len(self._compiled_graphs),
            "active_users": len(self._active_graphs),
            "graph_types": list(self._graph_configs.keys()),
            "user_sessions": len(self._active_graphs)
        }


class SmartAgentGraphWrapper(BaseGraph):
    """Wrapper for existing SmartAgentGraphBuilder"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.builder = SmartAgentGraphBuilder(self.config)
        self.graph_name = "default_smart_agent"
        self.graph_type = "default"
        logger.info("SmartAgentGraphWrapper initialized")
    
    def get_graph_name(self) -> str:
        return self.graph_name
    
    def get_graph_type(self) -> str:
        return self.graph_type
    
    def get_description(self) -> str:
        return "Standard SmartAgent graph with full feature set"
    
    def _initialize_nodes(self):
        pass  # Nodes initialized in SmartAgentGraphBuilder
    
    def _build_workflow(self, workflow):
        pass  # Not used
    
    def build_graph(self):
        return self.builder.build_graph()
    
    def get_runtime_config(self, session_id: str, custom_limits: Optional[Dict] = None):
        return self.builder.get_runtime_config(session_id, custom_limits)
    
    def get_graph_info(self) -> Dict:
        return self.builder.get_graph_info()
    
    def get_node_list(self) -> List[str]:
        return self.builder.get_graph_info().get("nodes", [])
    
    def get_features(self) -> List[str]:
        return self.builder.get_graph_info().get("features", [])


# Global registry instance
_graph_registry: Optional[GraphRegistryWithAuth] = None


def get_graph_registry() -> GraphRegistryWithAuth:
    """Get global graph registry instance with auth"""
    global _graph_registry
    
    if _graph_registry is None:
        _graph_registry = GraphRegistryWithAuth()
    
    return _graph_registry