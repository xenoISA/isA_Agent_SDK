#!/usr/bin/env python3
"""
Base Graph Class

Abstract base class for all graph implementations.
Provides common functionality and interface for different graph types.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List

from langgraph.graph import StateGraph, START, END
# RetryPolicy removed in newer LangGraph versions
try:
    from langgraph.types import CachePolicy
except ImportError:
    CachePolicy = None

from isa_agent_sdk.agent_types import AgentState
from .utils.context_schema import ContextSchema
from isa_agent_sdk.services.persistence import durable_service

logger = logging.getLogger(__name__)


class BaseGraph(ABC):
    """
    Abstract base class for all graph implementations
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the base graph
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.graph_name = self.get_graph_name()
        self.graph_type = self.get_graph_type()
        
        # Common configuration
        self.max_graph_iterations = self.config.get("max_graph_iterations", 50)
        self.max_agent_loops = self.config.get("max_agent_loops", 10)
        self.max_tool_loops = self.config.get("max_tool_loops", 5)
        
        # Retry and cache policies (if available)
        # self.retry_policy = RetryPolicy(max_attempts=3)  # Removed in newer versions
        if CachePolicy:
            self.llm_cache_policy = CachePolicy(ttl=self.config.get("llm_cache_ttl", 300))
            self.tool_cache_policy = CachePolicy(ttl=self.config.get("tool_cache_ttl", 120))
        else:
            self.llm_cache_policy = None
            self.tool_cache_policy = None
        
        # Initialize nodes (to be implemented by subclasses)
        self.nodes = {}
        self._initialize_nodes()
        
        # Setup checkpointer
        self._setup_checkpointer()
        
        logger.info(f"{self.graph_name} graph initialized (type: {self.graph_type})")
    
    @abstractmethod
    def get_graph_name(self) -> str:
        """Get the unique name of this graph"""
        pass
    
    @abstractmethod
    def get_graph_type(self) -> str:
        """Get the type/category of this graph"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get a description of what this graph does"""
        pass
    
    @abstractmethod
    def _initialize_nodes(self):
        """Initialize the nodes specific to this graph"""
        pass
    
    @abstractmethod
    def _build_workflow(self, workflow: StateGraph) -> StateGraph:
        """
        Build the workflow by adding nodes and edges
        
        Args:
            workflow: The StateGraph to build upon
            
        Returns:
            The configured StateGraph
        """
        pass
    
    def _setup_checkpointer(self):
        """Setup durable service checkpointer"""
        try:
            setup_success = durable_service.setup_postgres_tables()
            if setup_success:
                logger.info(f"✅ {self.graph_name}: Checkpointer ready")
                service_info = durable_service.get_service_info()
                logger.info(f"📊 {self.graph_name}: Using {service_info['checkpointer_type']}")
            else:
                logger.warning(f"⚠️ {self.graph_name}: PostgreSQL setup failed, using fallback")
        except Exception as e:
            logger.warning(f"⚠️ {self.graph_name}: Checkpointer setup failed: {e}")
    
    def build_graph(self):
        """
        Build and compile the complete graph
        
        Returns:
            Compiled LangGraph with checkpointer
        """
        # Get checkpointer
        checkpointer = durable_service.get_checkpointer()
        
        # Create workflow
        workflow = StateGraph(AgentState, config_schema=ContextSchema)
        
        # Build workflow (implemented by subclasses)
        workflow = self._build_workflow(workflow)
        
        # Compile with checkpointer
        graph = workflow.compile(checkpointer=checkpointer)
        
        logger.info(f"{self.graph_name} graph compiled successfully")
        return graph
    
    def get_runtime_config(self, session_id: str, custom_limits: Optional[Dict] = None):
        """Get runtime configuration with recursion limits"""
        from langchain_core.runnables.config import RunnableConfig
        
        recursion_limit = self.max_graph_iterations
        if custom_limits:
            recursion_limit = custom_limits.get("max_iterations", recursion_limit)
        
        return RunnableConfig(
            recursion_limit=recursion_limit,
            configurable={"thread_id": session_id}
        )
    
    def get_graph_info(self) -> Dict:
        """Get graph structure information"""
        return {
            "name": self.graph_name,
            "type": self.graph_type,
            "description": self.get_description(),
            "recursion_limits": {
                "max_graph_iterations": self.max_graph_iterations,
                "max_agent_loops": self.max_agent_loops,
                "max_tool_loops": self.max_tool_loops
            },
            "nodes": self.get_node_list(),
            "features": self.get_features()
        }
    
    @abstractmethod
    def get_node_list(self) -> List[str]:
        """Get list of nodes in this graph"""
        pass
    
    @abstractmethod
    def get_features(self) -> List[str]:
        """Get list of features this graph supports"""
        pass
    
    def get_nodes_detail(self) -> Dict[str, Dict]:
        """Get detailed information about each node"""
        # Override in subclasses for specific node details
        return {}
    
    def get_edges_detail(self) -> List[Dict]:
        """Get detailed information about graph edges"""
        # Override in subclasses for specific edge details
        return []