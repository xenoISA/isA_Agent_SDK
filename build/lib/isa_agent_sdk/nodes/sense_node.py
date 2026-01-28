#!/usr/bin/env python3
"""
Sense Node - Intelligent Entry Point with Intent Classification

This node serves as the smart entry point for the agent graph, providing:
1. Intent classification: Simple vs Complex requests
2. Smart routing: Direct response vs Reasoning required
3. Event-driven triggers (future): Proactive agent activation

Design Philosophy:
- Fast classification using lightweight models
- Rule-based shortcuts for obvious cases
- Minimize latency for simple requests
- Extensible for event-driven architecture
"""

import time
from typing import Dict, Any, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from isa_agent_sdk.agent_types.agent_state import AgentState
from .base_node import BaseNode
from isa_agent_sdk.utils.logger import agent_logger
from .utils.event_trigger_manager import EventTriggerManager, TriggerType

logger = agent_logger


class SenseNode(BaseNode):
    """
    Intelligent sensing node for request classification and routing

    Responsibilities:
    1. Classify user intent (simple/complex)
    2. Route to appropriate node (response_node/reason_node)
    3. Monitor events for proactive triggers (future)
    """

    def __init__(self, event_service_url: Optional[str] = None, task_service_url: Optional[str] = None, event_bus=None):
        super().__init__("SenseNode")

        # Initialize event trigger manager for proactive triggers
        self.event_trigger_manager: Optional[EventTriggerManager] = None
        if event_service_url and task_service_url:
            self.event_trigger_manager = EventTriggerManager(
                event_service_url=event_service_url,
                task_service_url=task_service_url,
                event_bus=event_bus
            )
            # Set callback for triggering workflows
            self.event_trigger_manager.set_workflow_callback(self._handle_proactive_trigger)

        # Simple request patterns (rule-based shortcuts)
        self.simple_patterns = [
            # Greetings
            "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
            # Gratitude
            "thanks", "thank you", "appreciate",
            # Confirmations
            "ok", "okay", "yes", "no", "sure",
            # Simple questions
            "what time", "what date", "who are you", "what can you do",
        ]

        # Complex request indicators
        self.complex_indicators = [
            # Tool requirements
            "analyze", "calculate", "search", "find data", "query", "fetch",
            # Weather and real-time data queries (require tools)
            "weather", "temperature", "forecast", "climate",
            # Planning requirements
            "plan", "strategy", "approach", "steps", "how to",
            # Multi-step tasks
            "first", "then", "after that", "finally", "multiple",
            # Execution keywords
            "execute", "run", "perform", "do the following",
            # File operations
            "upload", "file", "document", "process data",
        ]

        # Classification confidence threshold
        self.confidence_threshold = 0.7

    async def initialize(self):
        """Initialize sense node and event trigger manager"""
        if self.event_trigger_manager:
            await self.event_trigger_manager.initialize()
            self.logger.info("Event trigger manager initialized in SenseNode")

    async def _execute_logic(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """
        Execute intent sensing and routing logic

        Flow:
        1. Extract user message
        2. Quick rule-based classification
        3. LLM-based classification for uncertain cases
        4. Route to appropriate node
        5. (Future) Check for event triggers

        Args:
            state: Current agent state
            config: Runtime config

        Returns:
            Updated state with routing decision
        """
        node_start = time.time()
        session_id = config.get("configurable", {}).get("thread_id", "unknown") if config else "unknown"

        # Log incoming state
        messages = state.get("messages", [])
        message_types = [type(msg).__name__ for msg in messages]

        self.logger.info(
            f"[PHASE:NODE_SENSE] state_received | "
            f"session_id={session_id} | "
            f"messages_count={len(messages)} | "
            f"message_types={message_types}"
        )

        # Extract user's latest message
        user_message = self._extract_latest_user_message(messages)
        if not user_message:
            self.logger.warning(
                f"[PHASE:NODE_SENSE] no_user_message | "
                f"session_id={session_id} | "
                f"routing=response_node | "
                f"reason=no_user_input"
            )
            return {
                "next_action": "format_response"
            }

        self.logger.info(
            f"[PHASE:NODE_SENSE] user_message_extracted | "
            f"session_id={session_id} | "
            f"message_preview='{user_message[:100]}'"
        )

        try:
            # Step 1: Fast rule-based classification
            rule_classification = self._rule_based_classification(user_message)

            if rule_classification == "simple_certain":
                # Definitely simple - route directly
                self.logger.info(
                    f"[PHASE:NODE_SENSE] intent_classified | "
                    f"session_id={session_id} | "
                    f"method=rule_based | "
                    f"intent=simple | "
                    f"confidence=high | "
                    f"routing=response_node"
                )
                return {
                    "next_action": "format_response"
                }

            elif rule_classification == "complex_certain":
                # Definitely complex - route to reasoning
                self.logger.info(
                    f"[PHASE:NODE_SENSE] intent_classified | "
                    f"session_id={session_id} | "
                    f"method=rule_based | "
                    f"intent=complex | "
                    f"confidence=high | "
                    f"routing=reason_node"
                )
                return {
                    "next_action": "reason_model"
                }

            # Step 2: Uncertain cases - use LLM classification
            llm_classification = await self._llm_based_classification(user_message, config)

            if llm_classification == "simple":
                self.logger.info(
                    f"[PHASE:NODE_SENSE] intent_classified | "
                    f"session_id={session_id} | "
                    f"method=llm_based | "
                    f"intent=simple | "
                    f"routing=response_node"
                )
                return {
                    "next_action": "format_response"
                }
            else:
                self.logger.info(
                    f"[PHASE:NODE_SENSE] intent_classified | "
                    f"session_id={session_id} | "
                    f"method=llm_based | "
                    f"intent=complex | "
                    f"routing=reason_node"
                )
                return {
                    "next_action": "reason_model"
                }

        except Exception as e:
            # On error, default to reason_node (safer)
            self.logger.error(
                f"[PHASE:NODE_SENSE] classification_error | "
                f"session_id={session_id} | "
                f"error={type(e).__name__} | "
                f"message={str(e)[:200]} | "
                f"routing=reason_node | "
                f"reason=error_fallback",
                exc_info=True
            )
            return {
                "next_action": "reason_model"
            }

        finally:
            duration = int((time.time() - node_start) * 1000)
            self.logger.info(
                f"[PHASE:NODE_SENSE] node_complete | "
                f"session_id={session_id} | "
                f"duration_ms={duration}"
            )

    def _extract_latest_user_message(self, messages) -> Optional[str]:
        """
        Extract the latest user message from conversation

        Args:
            messages: List of conversation messages

        Returns:
            Latest user message content or None
        """
        for message in reversed(messages):
            if type(message).__name__ == "HumanMessage":
                content = getattr(message, 'content', '')
                if content:
                    return str(content)
        return None

    def _rule_based_classification(self, user_message: str) -> str:
        """
        Fast rule-based classification using pattern matching

        Args:
            user_message: User's message content

        Returns:
            "simple_certain", "complex_certain", or "uncertain"
        """
        message_lower = user_message.lower().strip()

        # Check message length (very short messages are usually simple)
        if len(message_lower) < 20:
            # Check if it matches simple patterns
            for pattern in self.simple_patterns:
                if pattern in message_lower:
                    return "simple_certain"

        # Check for complex indicators
        complex_match_count = sum(1 for indicator in self.complex_indicators if indicator in message_lower)
        if complex_match_count >= 2:
            return "complex_certain"

        # Check for tool-related keywords
        tool_keywords = ["tool", "function", "call", "execute", "run"]
        if any(keyword in message_lower for keyword in tool_keywords):
            return "complex_certain"

        # Check for question complexity
        question_words = message_lower.count("?")
        if question_words > 1:
            return "complex_certain"

        # Uncertain - need LLM classification
        return "uncertain"

    async def _llm_based_classification(self, user_message: str, config: RunnableConfig) -> str:
        """
        LLM-based classification for uncertain cases

        Uses Cerebras gpt-oss-120b for ultra-fast classification (fastest inference in the world)

        Args:
            user_message: User's message content
            config: Runtime config

        Returns:
            "simple" or "complex"
        """
        classification_start = time.time()
        session_id = config.get("configurable", {}).get("thread_id", "unknown") if config else "unknown"

        # Construct classification prompt
        system_prompt = """You are an intent classifier. Classify user messages as SIMPLE or COMPLEX.

SIMPLE requests:
- Greetings, thanks, confirmations
- General knowledge questions (facts from training data)
- No tool usage needed
- Can be answered directly from existing knowledge

COMPLEX requests:
- Multi-step tasks
- Requires tools/functions (web search, weather API, calculations, etc.)
- Real-time data queries (weather, news, stock prices, current events)
- Needs planning or reasoning
- Data analysis, file processing
- Multiple sub-tasks

IMPORTANT: If the question requires real-time or external data (weather, news, current prices, etc.), classify as COMPLEX.

Respond with ONLY one word: "SIMPLE" or "COMPLEX"."""

        user_prompt = f"Classify this message:\n\n{user_message}"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        try:
            # Use Cerebras gpt-oss-120b for ultra-fast classification
            # Cerebras provides the fastest inference in the world
            response = await self.call_model(
                messages=messages,
                stream_tokens=False,
                model="gpt-oss-120b",  # Cerebras ultra-fast model
                provider="cerebras"
            )

            classification_duration = int((time.time() - classification_start) * 1000)

            # Parse response
            result = response.content.strip().upper() if hasattr(response, 'content') else "COMPLEX"

            self.logger.info(
                f"[PHASE:NODE_SENSE] llm_classification | "
                f"session_id={session_id} | "
                f"model=gpt-oss-120b | "
                f"provider=cerebras | "
                f"result={result} | "
                f"duration_ms={classification_duration}"
            )

            # Default to complex if unclear
            if "SIMPLE" in result:
                return "simple"
            else:
                return "complex"

        except Exception as e:
            self.logger.error(
                f"[PHASE:NODE_SENSE] llm_classification_error | "
                f"session_id={session_id} | "
                f"error={str(e)[:100]} | "
                f"defaulting=complex",
                exc_info=True
            )
            # On error, default to complex (safer)
            return "complex"

    # ========================================
    # Event-Driven Architecture
    # ========================================

    async def _handle_proactive_trigger(
        self,
        user_id: str,
        trigger_data: Dict[str, Any],
        action_config: Dict[str, Any]
    ) -> None:
        """
        Handle proactive trigger workflow execution.

        This is the callback invoked by EventTriggerManager when a trigger fires.
        It creates a NEW session for clean isolation and executes the agent workflow
        with the trigger context.

        Args:
            user_id: User who owns the trigger
            trigger_data: Information about the fired trigger
            action_config: Configuration for the agent action
        """
        import uuid

        trigger_id = trigger_data.get("trigger_id", "unknown")

        self.logger.info(
            f"[PHASE:NODE_SENSE] proactive_trigger | "
            f"user_id={user_id} | "
            f"trigger_id={trigger_id} | "
            f"description={trigger_data.get('description', 'Unknown')}"
        )

        try:
            # Import SDK query lazily to avoid circular imports
            from isa_agent_sdk.query import query
            from isa_agent_sdk.options import ISAAgentOptions, ExecutionMode

            # Generate new session ID for proactive execution
            session_id = f"proactive_{uuid.uuid4().hex[:12]}"

            # Build prompt from action config
            prompt = action_config.get(
                "prompt",
                f"Trigger fired: {trigger_data.get('description', 'Unknown trigger')}"
            )

            # Include trigger context in the prompt
            trigger_context = (
                f"\n\n[Proactive Trigger Context]\n"
                f"Trigger: {trigger_data.get('description', 'Unknown')}\n"
                f"Type: {trigger_data.get('trigger_type', 'Unknown')}\n"
                f"Event Data: {trigger_data.get('event_data', {})}\n"
                f"Triggered At: {trigger_data.get('triggered_at', 'Unknown')}"
            )
            full_prompt = prompt + trigger_context

            # Build options for proactive execution
            options = ISAAgentOptions(
                user_id=user_id,
                session_id=session_id,
                execution_mode=ExecutionMode.PROACTIVE,
                allowed_tools=action_config.get("allowed_tools"),
                model=action_config.get("model", "gpt-5-nano"),
                metadata={
                    "trigger_id": trigger_id,
                    "proactive": True,
                    "trigger_source": action_config.get("source", "event_trigger")
                }
            )

            # Execute query and stream results
            self.logger.info(
                f"[PHASE:NODE_SENSE] proactive_execution_start | "
                f"session_id={session_id} | "
                f"trigger_id={trigger_id}"
            )

            async for msg in query(prompt=full_prompt, options=options):
                # Log significant events
                if msg.is_complete:
                    self.logger.info(
                        f"[PHASE:NODE_SENSE] proactive_execution_complete | "
                        f"session_id={session_id} | "
                        f"trigger_id={trigger_id}"
                    )
                elif msg.is_error:
                    self.logger.error(
                        f"[PHASE:NODE_SENSE] proactive_execution_error | "
                        f"session_id={session_id} | "
                        f"error={msg.content}"
                    )

            # TODO: Store results, send notifications, etc.

        except Exception as e:
            self.logger.error(
                f"[PHASE:NODE_SENSE] proactive_trigger_failed | "
                f"trigger_id={trigger_id} | "
                f"error={str(e)[:200]}",
                exc_info=True
            )

    async def _check_event_triggers(self, state: AgentState, config: RunnableConfig) -> Optional[Dict[str, Any]]:
        """
        Check for event triggers that require proactive agent action.

        This method checks the EventTriggerManager for any pending triggers
        that should be processed. Used during graph execution to handle
        any queued proactive triggers.

        Args:
            state: Current agent state
            config: Runtime config

        Returns:
            Event trigger info or None if no triggers pending
        """
        if not self.event_trigger_manager:
            return None

        # Check for any queued triggers from event history
        # The actual trigger firing is handled by the EventTriggerManager
        # via NATS subscriptions - this is for any additional checks
        try:
            # Get recent events that might have queued triggers
            recent_events = self.event_trigger_manager.get_event_history(
                limit=10
            )

            # For now, return None - triggers are processed asynchronously
            # via the workflow callback set in __init__
            return None

        except Exception as e:
            self.logger.warning(f"Error checking event triggers: {e}")
            return None

    async def shutdown(self) -> None:
        """
        Shutdown the sense node and its resources.

        This method is called by the GraphLifecycleManager during shutdown.
        It ensures the EventTriggerManager is properly cleaned up.
        """
        self.logger.info("[PHASE:NODE_SENSE] Shutting down SenseNode")

        if self.event_trigger_manager:
            try:
                await self.event_trigger_manager.shutdown()
                self.logger.info("[PHASE:NODE_SENSE] EventTriggerManager shutdown complete")
            except Exception as e:
                self.logger.error(f"[PHASE:NODE_SENSE] EventTriggerManager shutdown error: {e}")
