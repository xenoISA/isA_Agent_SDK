#!/usr/bin/env python3
"""
SmartAgent Graph Builder

A clean, professional LangGraph workflow builder for intelligent agent execution.
This module creates and configures the agent execution graph with proper state management.
"""

from isa_agent_sdk.agent_types.agent_state import AgentState
import os
import base64
from typing import Dict, Optional, List

from langgraph.graph import StateGraph, START, END
# RetryPolicy removed in newer LangGraph versions
# from langgraph.pregel import RetryPolicy
try:
    from langgraph.types import CachePolicy
except ImportError:
    # Fallback for older versions
    CachePolicy = None
from langgraph.cache.memory import InMemoryCache

# Import nodes
from isa_agent_sdk.nodes.sense_node import SenseNode
from isa_agent_sdk.nodes.reason_node import ReasonNode
from isa_agent_sdk.nodes.response_node import ResponseNode
from isa_agent_sdk.nodes.summarization_node import SummarizationNode, should_summarize
from isa_agent_sdk.nodes import (
    ToolNode,
    AgentExecutorNode, GuardrailNode, FailsafeNode
)

# Import types
from isa_agent_sdk.agent_types import AgentState
from .utils.context_schema import ContextSchema

# Import durable service (official LangGraph pattern)
from isa_agent_sdk.services.persistence import durable_service

# Import lifecycle manager for async initialization
from .utils.graph_lifecycle import GraphLifecycleManager

# Use centralized logger with Loki integration
from isa_agent_sdk.utils.logger import agent_logger
logger = agent_logger


class SmartAgentGraphBuilder:
    """
    SmartAgent Graph Builder
    
    Builds and configures LangGraph workflows for intelligent agent execution.
    Supports PostgreSQL checkpointing with MemorySaver fallback.
    
    Architecture: Start → Reason → Tool/Agent → Response → End (with conditional routing)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the graph builder

        Args:
            config: Optional configuration dictionary with keys:
                - guardrail_enabled: Enable guardrail node
                - guardrail_mode: Guardrail strictness
                - failsafe_enabled: Enable failsafe node
                - confidence_threshold: Failsafe confidence threshold
                - summarization_enabled: Enable conversation summarization
                - summarization_threshold: Messages before summarization
                - preserve_last_n: Recent messages to keep verbatim
                - summary_model: Model for summarization
                - max_graph_iterations: Maximum graph iterations
                - trigger_config: TriggerConfig for event triggers (optional)
        """
        self.config = config or {}

        # Lifecycle manager for async initialization
        self._lifecycle_manager = GraphLifecycleManager()
        
        # Guardrail configuration - DISABLED
        self.guardrail_enabled = self.config.get("guardrail_enabled", False)  # Default to disabled
        self.guardrail_mode = self.config.get("guardrail_mode", "moderate")

        # Failsafe configuration - DISABLED
        self.failsafe_enabled = self.config.get("failsafe_enabled", False)  # Default to disabled (was True)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)

        # Summarization configuration
        self.summarization_enabled = self.config.get("summarization_enabled", True)  # Default to enabled
        self.summarization_threshold = self.config.get("summarization_threshold", 20)  # Messages before summarization
        self.preserve_last_n = self.config.get("preserve_last_n", 5)  # Recent messages to keep
        self.summary_model = self.config.get("summary_model", "gpt-4o-mini")  # Fast & cheap model
        
        # Recursion limits
        self.max_graph_iterations = self.config.get("max_graph_iterations", 50)
        self.max_agent_loops = self.config.get("max_agent_loops", 10)
        self.max_tool_loops = self.config.get("max_tool_loops", 5)
        
        # Initialize nodes
        print("🏗️🏗️🏗️ GRAPH BUILDER: Creating nodes! 🏗️🏗️🏗️")

        # Extract trigger config if provided
        trigger_config = self.config.get("trigger_config")
        event_service_url = None
        task_service_url = None
        event_bus = None

        if trigger_config:
            event_service_url = trigger_config.get("event_service_url")
            task_service_url = trigger_config.get("task_service_url")
            event_bus = trigger_config.get("event_bus")

        # Create SenseNode with optional trigger configuration
        self.sense_node = SenseNode(
            event_service_url=event_service_url,
            task_service_url=task_service_url,
            event_bus=event_bus
        )

        # Register sense_node for lifecycle management (async initialization)
        self._lifecycle_manager.register_node("sense", self.sense_node)

        self.reason_node = ReasonNode()
        self.response_node = ResponseNode()
        self.tool_node = ToolNode()
        print(f"🏗️ GRAPH BUILDER: ToolNode created - type: {type(self.tool_node)}")
        self.agent_executor_node = AgentExecutorNode()
        self.guardrail_node = None
        self.failsafe_node = None
        self.summarization_node = None

        # Initialize failsafe node if enabled
        if self.failsafe_enabled:
            self.failsafe_node = FailsafeNode(confidence_threshold=self.confidence_threshold)

        # Initialize summarization node if enabled
        if self.summarization_enabled:
            self.summarization_node = SummarizationNode(
                preserve_last_n=self.preserve_last_n,
                summary_model=self.summary_model
            )
        
        # Retry and cache policies (if available)
        # self.retry_policy = RetryPolicy(max_attempts=3)  # Removed in newer versions
        if CachePolicy:
            self.llm_cache_policy = CachePolicy(ttl=300)  # 5 min cache for LLM calls
            self.tool_cache_policy = CachePolicy(ttl=120)  # 2 min cache for tool calls
        else:
            self.llm_cache_policy = None
            self.tool_cache_policy = None
        
        # Setup checkpointer
        self._setup_checkpointer()
        
        logger.info("SmartAgent Graph Builder initialized")
    
    
    def _setup_checkpointer(self):
        """Setup durable service checkpointer (official LangGraph pattern)"""
        try:
            # Setup PostgreSQL tables if needed
            setup_success = durable_service.setup_postgres_tables()
            if setup_success:
                logger.info("✅ Durable service checkpointer ready")

                # Log service info
                service_info = durable_service.get_service_info()
                logger.info(f"📊 Checkpointer backend: {service_info['checkpointer_backend']}")
                logger.info(f"📊 Actual checkpointer: {service_info['actual_checkpointer']}")
                logger.info(f"🔄 Durable execution: {service_info['features']['durable_execution']}")
            else:
                logger.warning("⚠️ PostgreSQL setup failed, using MemorySaver fallback")
        except Exception as e:
            logger.warning(f"⚠️ Checkpointer setup failed: {e}")

    async def initialize(self, runtime_context: Optional[Dict] = None) -> bool:
        """
        Initialize async components of the graph (e.g., EventTriggerManager)

        This method must be called before streaming graph execution if event triggers
        are enabled. It initializes:
        - SenseNode's EventTriggerManager (NATS subscriptions, event service connections)
        - Any other nodes requiring async initialization

        Args:
            runtime_context: Optional runtime context with configuration

        Returns:
            True if initialization succeeded, False otherwise
        """
        runtime_context = runtime_context or {}

        logger.info("SmartAgentGraphBuilder async initialization starting")

        try:
            success = await self._lifecycle_manager.initialize(runtime_context)

            if success:
                logger.info("✅ SmartAgentGraphBuilder async initialization complete")
            else:
                logger.warning("⚠️ SmartAgentGraphBuilder initialization had failures")

            return success

        except Exception as e:
            logger.error(f"SmartAgentGraphBuilder initialization failed: {e}")
            return False

    async def shutdown(self) -> bool:
        """
        Shutdown async components of the graph gracefully

        This method should be called during application shutdown to ensure
        clean resource cleanup (close HTTP clients, unsubscribe from NATS, etc.)

        Returns:
            True if shutdown succeeded, False otherwise
        """
        logger.info("SmartAgentGraphBuilder async shutdown starting")

        try:
            success = await self._lifecycle_manager.shutdown()

            if success:
                logger.info("✅ SmartAgentGraphBuilder async shutdown complete")
            else:
                logger.warning("⚠️ SmartAgentGraphBuilder shutdown had failures")

            return success

        except Exception as e:
            logger.error(f"SmartAgentGraphBuilder shutdown failed: {e}")
            return False

    @property
    def is_initialized(self) -> bool:
        """Check if async components have been initialized"""
        return self._lifecycle_manager.is_initialized

    def build_graph(self):
        """
        Build the complete agent graph
        
        Returns:
            Compiled LangGraph with checkpointer
        """
        return self._build_graph_with_checkpointer()
    
    def _build_graph_with_checkpointer(self):
        """Build graph with durable service checkpointer (official LangGraph pattern)"""
        # Get checkpointer from durable service
        checkpointer = durable_service.get_checkpointer()
        
        # Log checkpointer type for debugging
        service_info = durable_service.get_service_info()
        logger.info(f"🔧 Building graph with {service_info['actual_checkpointer']} checkpointer")
        
        return self._compile_workflow(checkpointer)
    
    
    def _compile_workflow(self, checkpointer):
        """Compile the workflow with given checkpointer, retry policies, and caching"""
        workflow = StateGraph[AgentState, AgentState, AgentState](AgentState, config_schema=ContextSchema)

        # Add nodes (retry policies not supported in LangGraph 0.6.0a1)
        print(f"🏗️ GRAPH BUILDER: Adding nodes...")
        workflow.add_node("sense", self.sense_node.execute)
        workflow.add_node("reason_model", self.reason_node.execute)
        workflow.add_node("call_tool", self.tool_node.execute)
        workflow.add_node("agent_executor", self.agent_executor_node.execute)
        workflow.add_node("format_response", self.response_node.execute)

        # Add summarization node if enabled
        if self.summarization_enabled and self.summarization_node:
            workflow.add_node("summarize", self.summarization_node.execute)

        # Add failsafe node if enabled
        if self.failsafe_enabled and self.failsafe_node:
            workflow.add_node("failsafe_check", self.failsafe_node.execute)

        # Add guardrail node if enabled
        if self.guardrail_enabled and self.guardrail_node:
            workflow.add_node("guardrail_check", self.guardrail_node.execute)
        
        # Set entry point with optional summarization check
        if self.summarization_enabled:
            # Add conditional edge: START -> check if summarization needed -> summarize/sense
            workflow.add_conditional_edges(
                START,
                lambda state: should_summarize(
                    state,
                    message_threshold=self.summarization_threshold,
                    enable_summarization=self.summarization_enabled
                ),
                {
                    "summarize": "summarize",
                    "continue": "sense"
                }
            )
            # After summarization, go to sense node
            workflow.add_edge("summarize", "sense")
        else:
            # Direct entry to sense node (new intelligent entry point)
            workflow.add_edge(START, "sense")

        # Sense node routing: simple -> response, complex -> reason
        workflow.add_conditional_edges(
            "sense",
            lambda state: state.get("next_action", "reason_model"),
            {
                "reason_model": "reason_model",
                "format_response": "format_response"
            }
        )
        
        # Direct conditional routing from reason_model with smart failsafe routing
        if self.failsafe_enabled and self.guardrail_enabled:
            # Both failsafe and guardrail enabled: reason -> tool/agent -> smart_routing -> guardrail/failsafe -> response
            workflow.add_conditional_edges(
                "reason_model",
                lambda state: self._route_end_action(state),
                {
                    "call_tool": "call_tool",
                    "agent_executor": "agent_executor", 
                    "failsafe_check": "failsafe_check",
                    "guardrail_check": "guardrail_check",
                    "format_response": "format_response"
                }
            )
            workflow.add_edge("failsafe_check", "guardrail_check")
            workflow.add_edge("guardrail_check", "format_response")
        elif self.failsafe_enabled:
            # Only failsafe enabled: reason -> tool/agent -> smart_routing -> failsafe/response
            workflow.add_conditional_edges(
                "reason_model",
                lambda state: self._route_end_action(state),
                {
                    "call_tool": "call_tool",
                    "agent_executor": "agent_executor", 
                    "failsafe_check": "failsafe_check",
                    "format_response": "format_response"
                }
            )
            workflow.add_edge("failsafe_check", "format_response")
        elif self.guardrail_enabled:
            # Only guardrail enabled: reason -> tool/agent -> guardrail -> response
            workflow.add_conditional_edges(
                "reason_model",
                lambda state: state["next_action"],
                {
                    "call_tool": "call_tool",
                    "agent_executor": "agent_executor", 
                    "end": "guardrail_check"
                }
            )
            workflow.add_edge("guardrail_check", "format_response")
        else:
            # Neither enabled: reason -> tool/agent -> response
            workflow.add_conditional_edges(
                "reason_model",
                lambda state: state["next_action"],
                {
                    "call_tool": "call_tool",
                    "agent_executor": "agent_executor", 
                    "end": "format_response"
                }
            )
        
        # Tool routing - use smart routing
        if self.failsafe_enabled:
            workflow.add_conditional_edges(
                "call_tool",
                lambda state: self._route_tool_end_action(state),
                {
                    "agent_executor": "agent_executor",
                    "call_model": "reason_model",
                    "failsafe_check": "failsafe_check",
                    "guardrail_check": "guardrail_check" if self.guardrail_enabled else "format_response",
                    "format_response": "format_response"
                }
            )
        else:
            workflow.add_conditional_edges(
                "call_tool",
                lambda state: state.get("next_action", "reason_model"),
                {
                    "agent_executor": "agent_executor",
                    "call_model": "reason_model",
                    "end": "format_response"
                }
            )
        
        # Agent executor routing - use smart routing
        if self.failsafe_enabled:
            workflow.add_conditional_edges(
                "agent_executor",
                lambda state: self._route_tool_end_action(state),
                {
                    "call_model": "reason_model",
                    "agent_executor": "agent_executor",
                    "failsafe_check": "failsafe_check",
                    "guardrail_check": "guardrail_check" if self.guardrail_enabled else "format_response",
                    "format_response": "format_response"
                }
            )
        else:
            workflow.add_conditional_edges(
                "agent_executor",
                lambda state: state.get("next_action", "reason_model"),
                {
                    "call_model": "reason_model",
                    "agent_executor": "agent_executor",
                    "end": "format_response"
                }
            )
        
        # Final edges using LangGraph END
        workflow.add_edge("format_response", END)
        
        # Compile graph with checkpointer (HIL interrupts happen within tools via interrupt() calls)
        graph = workflow.compile(checkpointer=checkpointer)
        
        logger.info(f"SmartAgent graph compiled with {type(checkpointer).__name__}")
        logger.info(f"Flow: Start → {'Summarize (conditional) → ' if self.summarization_enabled else ''}Sense (intent) → Reason/Response → Tool/Agent → Response → End")
        logger.info(f"Intelligence: Sense node classifies intent for optimal routing")
        logger.info(f"HIL: Interrupt support via tool-level interrupt() calls")
        logger.info(f"Retry policies: LLM/Tool max_attempts=3, Agent max_attempts=5")
        logger.info(f"Cache policies: LLM ttl=300s, Tool ttl=120s")
        logger.info(f"Summarization: {'enabled' if self.summarization_enabled else 'disabled'} (threshold: {self.summarization_threshold} messages, preserve: {self.preserve_last_n})")
        logger.info(f"Guardrails: {'enabled' if self.guardrail_enabled else 'disabled'}")
        logger.info(f"Failsafe: {'enabled' if self.failsafe_enabled else 'disabled'} (threshold: {self.confidence_threshold})")
        logger.info(f"Max iterations: {self.max_graph_iterations}")
        
        return graph
    
    def configure_guardrails(self, enabled: bool = False, mode: str = "moderate"):
        """Configure guardrail settings"""
        self.guardrail_enabled = enabled
        self.guardrail_mode = mode
        
        if enabled and not self.guardrail_node:
            self.guardrail_node = GuardrailNode(mode)
        elif enabled and self.guardrail_node:
            self.guardrail_node.guardrail_mode = mode
        
        logger.info(f"Guardrails {'enabled' if enabled else 'disabled'} (mode: {mode})")
    
    def configure_failsafe(self, enabled: bool = True, confidence_threshold: float = 0.7):
        """Configure failsafe settings"""
        self.failsafe_enabled = enabled
        self.confidence_threshold = confidence_threshold
        
        if enabled and not self.failsafe_node:
            self.failsafe_node = FailsafeNode(confidence_threshold)
        elif enabled and self.failsafe_node:
            self.failsafe_node.confidence_threshold = confidence_threshold
        
        logger.info(f"Failsafe {'enabled' if enabled else 'disabled'} (threshold: {confidence_threshold})")
    
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
            "architecture": f"Start → {'Summarize (conditional) → ' if self.summarization_enabled else ''}Sense (intent) → Reason/Response → Tool/Agent → Response → End",
            "summarization_enabled": self.summarization_enabled,
            "summarization_threshold": self.summarization_threshold,
            "preserve_last_n": self.preserve_last_n,
            "summary_model": self.summary_model,
            "guardrail_enabled": self.guardrail_enabled,
            "guardrail_mode": self.guardrail_mode,
            "failsafe_enabled": self.failsafe_enabled,
            "confidence_threshold": self.confidence_threshold,
            "recursion_limits": {
                "max_graph_iterations": self.max_graph_iterations,
                "max_agent_loops": self.max_agent_loops,
                "max_tool_loops": self.max_tool_loops
            },
            "nodes": (["summarize"] if self.summarization_enabled else []) + [
                "sense",
                "reason_model",
                "call_tool",
                "agent_executor",
                "format_response"
            ] + (["failsafe_check"] if self.failsafe_enabled else []) + (["guardrail_check"] if self.guardrail_enabled else []),
            "features": [
                "Intelligent intent classification (Sense node)",
                "Optimized routing for simple vs complex requests",
                "PostgreSQL/Memory checkpointing",
                "Conditional routing (no router node)",
                "Retry policies for resilience",
                "In-memory caching for performance",
                "Configurable recursion limits",
                "Automatic conversation summarization" if self.summarization_enabled else None,
                "Optional guardrails",
                "AI confidence assessment and failsafe mechanisms",
                "Event-driven architecture ready (future)"
            ]
        }
    
    def _route_end_action(self, state: AgentState) -> str:
        """Route end action from reason_model"""
        session_id = state.get("session_id", "unknown")
        next_action = state.get("next_action", "format_response")

        logger.info(
            f"[PHASE:GRAPH] route_end_action | "
            f"session_id={session_id} | "
            f"next_action={next_action} | "
            f"from_node=reason_model"
        )

        if next_action != "end":
            logger.info(
                f"[PHASE:GRAPH] route_direct | "
                f"session_id={session_id} | "
                f"target_node={next_action}"
            )
            return next_action

        # For "end" action, use smart routing
        routed_node = self._smart_failsafe_router(state)
        logger.info(
            f"[PHASE:GRAPH] route_smart_failsafe | "
            f"session_id={session_id} | "
            f"target_node={routed_node}"
        )
        return routed_node
    
    def _route_tool_end_action(self, state: AgentState) -> str:
        """Route end action from tool/agent nodes"""
        session_id = state.get("session_id", "unknown")
        next_action = state.get("next_action", "reason_model")

        logger.info(
            f"[PHASE:GRAPH] route_tool_end_action | "
            f"session_id={session_id} | "
            f"next_action={next_action} | "
            f"from_node=tool/agent"
        )

        if next_action != "end":
            logger.info(
                f"[PHASE:GRAPH] route_tool_direct | "
                f"session_id={session_id} | "
                f"target_node={next_action}"
            )
            return next_action

        # For "end" action, use smart routing
        routed_node = self._smart_failsafe_router(state)
        logger.info(
            f"[PHASE:GRAPH] route_tool_smart_failsafe | "
            f"session_id={session_id} | "
            f"target_node={routed_node}"
        )
        return routed_node

    def _smart_failsafe_router(self, state: AgentState) -> str:
        """
        Smart failsafe router that only triggers failsafe when actually needed

        This replaces the old forced failsafe routing with intelligent assessment:
        - Checks for actual errors, low confidence, or failure indicators
        - Only routes to failsafe_check if intervention is truly needed
        - Otherwise routes directly to format_response (guardrail_check if enabled)

        Args:
            state: Current agent state

        Returns:
            Next node name: "failsafe_check", "guardrail_check", or "format_response"
        """
        try:
            session_id = state.get("session_id", "unknown")
            messages = state.get("messages", [])

            if not messages:
                logger.info(
                    f"[PHASE:GRAPH] smart_failsafe_router | "
                    f"session_id={session_id} | "
                    f"decision=failsafe_check | "
                    f"reason=no_messages"
                )
                return "failsafe_check"

            last_message = messages[-1]

            # Check for explicit error indicators in state
            if state.get("error") or state.get("hil_error"):
                logger.info(
                    f"[PHASE:GRAPH] smart_failsafe_router | "
                    f"session_id={session_id} | "
                    f"decision=failsafe_check | "
                    f"reason=error_in_state"
                )
                return "failsafe_check"

            # Check if last message indicates failure or uncertainty
            if hasattr(last_message, 'content') and last_message.content:
                content = str(last_message.content).lower()

                # Check for explicit failure indicators
                failure_indicators = [
                    "error", "failed", "exception", "timeout", "unable", "cannot",
                    "i don't know", "i'm not sure", "uncertain", "unclear"
                ]

                if any(indicator in content for indicator in failure_indicators):
                    logger.info(
                        f"[PHASE:GRAPH] smart_failsafe_router | "
                        f"session_id={session_id} | "
                        f"decision=failsafe_check | "
                        f"reason=failure_indicators | "
                        f"content_preview='{content[:100]}'"
                    )
                    return "failsafe_check"

                # Check for very short responses (might indicate issues)
                if len(content.strip()) < 20:
                    logger.info(
                        f"[PHASE:GRAPH] smart_failsafe_router | "
                        f"session_id={session_id} | "
                        f"decision=failsafe_check | "
                        f"reason=short_response | "
                        f"content_length={len(content.strip())}"
                    )
                    return "failsafe_check"

            # Check for tool execution failures
            if any(hasattr(msg, 'content') and msg.content and
                   ("tool execution failed" in str(msg.content).lower() or
                    "mcp service not available" in str(msg.content).lower())
                   for msg in messages[-3:]):  # Check last 3 messages
                logger.info(
                    f"[PHASE:GRAPH] smart_failsafe_router | "
                    f"session_id={session_id} | "
                    f"decision=failsafe_check | "
                    f"reason=tool_execution_failure"
                )
                return "failsafe_check"

            # Success case - route to appropriate next step
            if self.guardrail_enabled:
                logger.info(
                    f"[PHASE:GRAPH] smart_failsafe_router | "
                    f"session_id={session_id} | "
                    f"decision=guardrail_check | "
                    f"reason=success_with_guardrail"
                )
                return "guardrail_check"
            else:
                logger.info(
                    f"[PHASE:GRAPH] smart_failsafe_router | "
                    f"session_id={session_id} | "
                    f"decision=format_response | "
                    f"reason=success_no_guardrail"
                )
                return "format_response"

        except Exception as e:
            # If router fails, be safe and use failsafe
            logger.warning(f"FailsafeRouter error: {e}, defaulting to failsafe_check")
            return "failsafe_check"
    
    async def get_visualization(self, format: str = "mermaid") -> str:
        """
        Generate graph visualization in specified format
        
        Args:
            format: Visualization format ("mermaid", "png", "ascii")
            
        Returns:
            Visualization content as string
        """
        try:
            # Build graph to get visualization
            graph = self.build_graph()
            graph_obj = graph.get_graph()
            
            if format == "mermaid":
                return graph_obj.draw_mermaid()
            elif format == "png":
                try:
                    png_bytes = graph_obj.draw_mermaid_png()
                    return base64.b64encode(png_bytes).decode('utf-8')
                except Exception as e:
                    logger.warning(f"PNG generation failed: {e}, falling back to mermaid")
                    return graph_obj.draw_mermaid()
            elif format == "ascii":
                try:
                    return graph_obj.draw_ascii()
                except Exception as e:
                    logger.warning(f"ASCII generation failed: {e}, falling back to mermaid")
                    return graph_obj.draw_mermaid()
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            # Return fallback visualization
            return self._get_fallback_visualization(format)
    
    def _get_fallback_visualization(self, format: str) -> str:
        """Generate fallback visualization when graph visualization fails"""
        nodes = self.get_graph_info()["nodes"]
        
        if format in ["mermaid", "png"]:
            # Generate simple mermaid diagram
            mermaid = "graph TD\n"
            mermaid += "    START([START])\n"
            
            for i, node in enumerate(nodes):
                mermaid += f"    {node}[{node}]\n"
                if i == 0:
                    mermaid += f"    START --> {node}\n"
                elif i < len(nodes) - 1:
                    next_node = nodes[i + 1]
                    mermaid += f"    {node} --> {next_node}\n"
                else:
                    mermaid += f"    {node} --> END([END])\n"
            
            return mermaid
            
        elif format == "ascii":
            # Generate simple ASCII flow
            ascii_flow = "Graph Flow:\n"
            ascii_flow += "START\n"
            for node in nodes:
                ascii_flow += f"  ↓\n{node}\n"
            ascii_flow += "  ↓\nEND"
            return ascii_flow
        
        return f"Visualization format '{format}' not supported"
    
    async def get_nodes_detail(self) -> Dict[str, Dict]:
        """Get detailed information about each node"""
        return {
            "sense": {
                "type": "SenseNode",
                "description": "Intelligent entry point - classifies user intent (simple vs complex)",
                "inputs": ["messages", "user_input"],
                "outputs": ["next_action", "intent_classification"],
                "conditional_routes": ["reason_model", "format_response"],
                "features": ["Rule-based fast classification", "LLM-based uncertain cases", "Event-driven ready (future)"]
            },
            "reason_model": {
                "type": "ReasonNode",
                "description": "Analyzes input and determines next action",
                "inputs": ["messages", "user_input"],
                "outputs": ["next_action", "reasoning"],
                "conditional_routes": ["call_tool", "agent_executor", "format_response"]
            },
            "call_tool": {
                "type": "ToolNode", 
                "description": "Executes tool calls and API requests",
                "inputs": ["tool_calls", "context"],
                "outputs": ["tool_results", "next_action"],
                "conditional_routes": ["agent_executor", "reason_model", "format_response"]
            },
            "agent_executor": {
                "type": "AgentExecutorNode",
                "description": "Handles complex agent execution workflows",
                "inputs": ["agent_input", "context"],
                "outputs": ["agent_result", "next_action"],
                "conditional_routes": ["reason_model", "agent_executor", "format_response"]
            },
            "format_response": {
                "type": "ResponseNode",
                "description": "Formats final response for user",
                "inputs": ["messages", "context"],
                "outputs": ["formatted_response"],
                "conditional_routes": ["END"]
            },
            "failsafe_check": {
                "type": "FailsafeNode",
                "description": f"Confidence assessment and error handling (threshold: {self.confidence_threshold})",
                "inputs": ["messages", "confidence_score"],
                "outputs": ["failsafe_result", "next_action"],
                "conditional_routes": ["format_response", "guardrail_check"],
                "enabled": self.failsafe_enabled
            } if self.failsafe_enabled else None,
            "guardrail_check": {
                "type": "GuardrailNode", 
                "description": f"Content safety and compliance check (mode: {self.guardrail_mode})",
                "inputs": ["content", "context"],
                "outputs": ["guardrail_result", "approved"],
                "conditional_routes": ["format_response"],
                "enabled": self.guardrail_enabled
            } if self.guardrail_enabled else None
        }
    
    async def get_edges_detail(self) -> List[Dict]:
        """Get detailed information about graph edges"""
        edges = []

        # Start edge -> Sense node (new intelligent entry)
        edges.append({
            "from": "START",
            "to": "sense",
            "type": "direct",
            "description": "Entry point to intelligent intent classification"
        })

        # Sense node conditional edges
        edges.append({
            "from": "sense",
            "to": ["reason_model", "format_response"],
            "type": "conditional",
            "description": "Intent-based routing: simple -> response, complex -> reasoning",
            "condition": "next_action"
        })

        # Reason model conditional edges
        if self.failsafe_enabled:
            edges.append({
                "from": "reason_model",
                "to": ["call_tool", "agent_executor", "failsafe_check", "format_response"],
                "type": "conditional",
                "description": "Smart routing with failsafe support",
                "condition": "_route_end_action"
            })
        else:
            edges.append({
                "from": "reason_model", 
                "to": ["call_tool", "agent_executor", "format_response"],
                "type": "conditional",
                "description": "Direct routing to action nodes",
                "condition": "next_action"
            })
        
        # Tool node edges
        if self.failsafe_enabled:
            edges.append({
                "from": "call_tool",
                "to": ["agent_executor", "reason_model", "failsafe_check", "format_response"],
                "type": "conditional",
                "description": "Tool result routing with failsafe",
                "condition": "_route_tool_end_action"
            })
        else:
            edges.append({
                "from": "call_tool",
                "to": ["agent_executor", "reason_model", "format_response"],
                "type": "conditional", 
                "description": "Tool result routing",
                "condition": "next_action"
            })
        
        # Agent executor edges
        if self.failsafe_enabled:
            edges.append({
                "from": "agent_executor",
                "to": ["reason_model", "agent_executor", "failsafe_check", "format_response"],
                "type": "conditional",
                "description": "Agent result routing with failsafe",
                "condition": "_route_tool_end_action"
            })
        else:
            edges.append({
                "from": "agent_executor",
                "to": ["reason_model", "agent_executor", "format_response"],
                "type": "conditional",
                "description": "Agent result routing",
                "condition": "next_action"
            })
        
        # Failsafe and guardrail edges
        if self.failsafe_enabled and self.guardrail_enabled:
            edges.append({
                "from": "failsafe_check",
                "to": "guardrail_check",
                "type": "direct",
                "description": "Failsafe to guardrail check"
            })
            edges.append({
                "from": "guardrail_check", 
                "to": "format_response",
                "type": "direct",
                "description": "Guardrail to response formatting"
            })
        elif self.failsafe_enabled:
            edges.append({
                "from": "failsafe_check",
                "to": "format_response", 
                "type": "direct",
                "description": "Failsafe to response formatting"
            })
        elif self.guardrail_enabled:
            edges.append({
                "from": "guardrail_check",
                "to": "format_response",
                "type": "direct",
                "description": "Guardrail to response formatting"
            })
        
        # End edge
        edges.append({
            "from": "format_response",
            "to": "END",
            "type": "direct", 
            "description": "Final output"
        })
        
        return edges