#!/usr/bin/env python3
"""
Trigger Service - Orchestrator for event-driven agent triggers

This service coordinates:
1. EventTriggerManager for trigger registration and matching
2. NATSEventBus for event delivery
3. SDK query() for proactive workflow execution

When a trigger fires, it creates a NEW session for clean isolation
and executes the agent workflow with the trigger context.
"""

import asyncio
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional

from isa_agent_sdk.nodes.utils.event_trigger_manager import (
    EventTriggerManager,
    TriggerCondition,
    TriggerType,
)
from .nats_event_bus import NATSEventBus

logger = logging.getLogger(__name__)


class TriggerService:
    """
    Orchestrator for event-driven trigger management and workflow execution

    Responsibilities:
    1. Initialize and manage EventTriggerManager and NATSEventBus
    2. Set up workflow callback for trigger execution
    3. Create isolated sessions for proactive agent execution
    4. Coordinate event subscriptions and trigger matching

    Example:
        service = TriggerService(
            event_service_url="http://localhost:8080/events",
            task_service_url="http://localhost:8080/tasks"
        )
        await service.initialize()

        # Register a trigger
        trigger_id = await service.register_trigger(
            user_id="user123",
            trigger_type=TriggerType.THRESHOLD,
            description="Alert when BTC drops > 5%",
            conditions={"threshold_value": 5.0, "direction": "down"},
            action_config={"prompt": "Analyze the price drop and suggest actions"}
        )

        # When a matching event occurs, the service will:
        # 1. Create a new session for the user
        # 2. Execute query() with the trigger context
        # 3. Store results in the session
    """

    def __init__(
        self,
        event_service_url: Optional[str] = None,
        task_service_url: Optional[str] = None,
        event_bus: Optional[NATSEventBus] = None,
        use_nats_bus: bool = True,
        max_triggers_per_user: int = 50,
        trigger_cooldown_seconds: int = 60
    ):
        """
        Initialize the trigger service

        Args:
            event_service_url: URL of external Event Service API
            task_service_url: URL of external Task Service API
            event_bus: Optional pre-configured NATSEventBus
            use_nats_bus: Whether to use NATS for event delivery
            max_triggers_per_user: Maximum triggers per user
            trigger_cooldown_seconds: Minimum seconds between trigger fires
        """
        self.event_service_url = event_service_url
        self.task_service_url = task_service_url
        self.use_nats_bus = use_nats_bus
        self.max_triggers_per_user = max_triggers_per_user
        self.trigger_cooldown_seconds = trigger_cooldown_seconds

        # Event bus
        self._event_bus = event_bus
        self._owns_event_bus = event_bus is None and use_nats_bus

        # Event trigger manager
        self._trigger_manager: Optional[EventTriggerManager] = None

        # Tracking
        self._initialized = False
        self._trigger_stats: Dict[str, Dict[str, Any]] = {}
        self._event_processing_task: Optional[asyncio.Task] = None

    async def initialize(self) -> bool:
        """
        Initialize the trigger service and its components

        Returns:
            True if initialization succeeded
        """
        if self._initialized:
            logger.debug("TriggerService already initialized")
            return True

        logger.info("TriggerService initialization starting")

        try:
            # Initialize NATS event bus if configured
            if self.use_nats_bus and self._owns_event_bus:
                self._event_bus = NATSEventBus()
                connected = await self._event_bus.connect()
                if not connected:
                    logger.warning("Failed to connect NATS event bus, continuing without")
                    self._event_bus = None

            # Initialize EventTriggerManager
            self._trigger_manager = EventTriggerManager(
                event_service_url=self.event_service_url or "",
                task_service_url=self.task_service_url or "",
                event_bus=self._event_bus
            )

            # Set workflow callback
            self._trigger_manager.set_workflow_callback(self._handle_workflow)

            # Initialize trigger manager
            await self._trigger_manager.initialize()

            # Start event processing loop if using NATS
            if self._event_bus and self._event_bus.is_connected:
                await self._subscribe_to_events()
                self._start_event_processing()

            self._initialized = True
            logger.info("TriggerService initialization complete")
            return True

        except Exception as e:
            logger.error(f"TriggerService initialization failed: {e}")
            return False

    async def shutdown(self) -> bool:
        """
        Shutdown the trigger service gracefully

        Returns:
            True if shutdown succeeded
        """
        if not self._initialized:
            return True

        logger.info("TriggerService shutdown starting")

        try:
            # Stop event processing
            if self._event_processing_task:
                self._event_processing_task.cancel()
                try:
                    await self._event_processing_task
                except asyncio.CancelledError:
                    pass

            # Shutdown trigger manager
            if self._trigger_manager:
                await self._trigger_manager.shutdown()

            # Disconnect event bus if we own it
            if self._event_bus and self._owns_event_bus:
                await self._event_bus.disconnect()

            self._initialized = False
            logger.info("TriggerService shutdown complete")
            return True

        except Exception as e:
            logger.error(f"TriggerService shutdown failed: {e}")
            return False

    @property
    def is_initialized(self) -> bool:
        """Check if the service is initialized"""
        return self._initialized

    # ==================== Trigger Management ====================

    async def register_trigger(
        self,
        user_id: str,
        trigger_type: TriggerType,
        description: str,
        conditions: Dict[str, Any],
        action_config: Dict[str, Any]
    ) -> Optional[str]:
        """
        Register a new event trigger

        Args:
            user_id: User ID who owns the trigger
            trigger_type: Type of trigger (THRESHOLD, EVENT_PATTERN, etc.)
            description: Human-readable description
            conditions: Trigger conditions (depends on type)
            action_config: Configuration for agent action when triggered

        Returns:
            trigger_id if successful, None otherwise
        """
        if not self._initialized or not self._trigger_manager:
            logger.error("TriggerService not initialized")
            return None

        # Check user trigger limit
        user_triggers = await self.get_user_triggers(user_id)
        if len(user_triggers) >= self.max_triggers_per_user:
            logger.warning(f"User {user_id} has reached trigger limit ({self.max_triggers_per_user})")
            return None

        try:
            trigger_id = await self._trigger_manager.register_trigger(
                user_id=user_id,
                trigger_type=trigger_type,
                description=description,
                conditions=conditions,
                action_config=action_config
            )

            # Initialize stats
            self._trigger_stats[trigger_id] = {
                "user_id": user_id,
                "created_at": asyncio.get_event_loop().time(),
                "fire_count": 0,
                "last_fired_at": None,
                "workflow_executions": 0
            }

            logger.info(f"Trigger registered: {trigger_id} for user {user_id}")
            return trigger_id

        except Exception as e:
            logger.error(f"Failed to register trigger: {e}")
            return None

    async def unregister_trigger(self, trigger_id: str) -> bool:
        """
        Unregister an event trigger

        Args:
            trigger_id: Trigger ID to remove

        Returns:
            True if successful
        """
        if not self._initialized or not self._trigger_manager:
            return False

        try:
            result = await self._trigger_manager.unregister_trigger(trigger_id)

            if result and trigger_id in self._trigger_stats:
                del self._trigger_stats[trigger_id]

            return result

        except Exception as e:
            logger.error(f"Failed to unregister trigger {trigger_id}: {e}")
            return False

    async def get_user_triggers(self, user_id: str) -> List[TriggerCondition]:
        """
        Get all triggers for a user

        Args:
            user_id: User ID

        Returns:
            List of TriggerCondition objects
        """
        if not self._initialized or not self._trigger_manager:
            return []

        return await self._trigger_manager.get_user_triggers(user_id)

    async def get_trigger_stats(self) -> Dict[str, Any]:
        """
        Get trigger service statistics

        Returns:
            Statistics dictionary
        """
        return {
            "initialized": self._initialized,
            "event_bus_connected": self._event_bus.is_connected if self._event_bus else False,
            "total_triggers": len(self._trigger_stats),
            "trigger_stats": self._trigger_stats
        }

    # ==================== Event Processing ====================

    async def _subscribe_to_events(self) -> None:
        """Subscribe to relevant event subjects"""
        if not self._event_bus:
            return

        # Subscribe to all agent events
        subjects = [
            NATSEventBus.SUBJECT_PRICE,
            NATSEventBus.SUBJECT_IOT,
            NATSEventBus.SUBJECT_SCHEDULE,
            NATSEventBus.SUBJECT_CUSTOM,
        ]

        for subject in subjects:
            await self._event_bus.subscribe(
                subject=subject,
                callback=self._handle_event,
                consumer_name=f"trigger_service_{subject.replace('.', '_')}"
            )

    def _start_event_processing(self) -> None:
        """Start background event processing loop"""
        self._event_processing_task = asyncio.create_task(
            self._event_processing_loop()
        )

    async def _event_processing_loop(self) -> None:
        """Background loop for processing events"""
        while self._initialized:
            try:
                # Process events from each subject
                for subject in self._event_bus._subscriptions.keys():
                    await self._event_bus.process_events(subject, batch_size=10)

                # Small delay between processing cycles
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Event processing error: {e}")
                await asyncio.sleep(1)  # Back off on error

    async def _handle_event(self, event_data: Dict[str, Any]) -> None:
        """
        Handle incoming event from NATS

        Args:
            event_data: Event data dictionary
        """
        if not self._trigger_manager:
            return

        try:
            # Let trigger manager check all triggers against this event
            await self._trigger_manager._check_triggers(event_data)

        except Exception as e:
            logger.error(f"Error handling event: {e}")

    # ==================== Workflow Execution ====================

    async def _handle_workflow(
        self,
        user_id: str,
        trigger_data: Dict[str, Any],
        action_config: Dict[str, Any]
    ) -> None:
        """
        Handle workflow execution when a trigger fires

        This method creates a NEW session for proactive execution,
        ensuring clean isolation from the user's existing sessions.

        Args:
            user_id: User who owns the trigger
            trigger_data: Information about the fired trigger
            action_config: Configuration for the agent action
        """
        trigger_id = trigger_data.get("trigger_id", "unknown")

        logger.info(
            f"Trigger workflow starting | "
            f"trigger_id={trigger_id} | "
            f"user_id={user_id}"
        )

        # Check cooldown
        if trigger_id in self._trigger_stats:
            stats = self._trigger_stats[trigger_id]
            last_fired = stats.get("last_fired_at")

            if last_fired:
                elapsed = asyncio.get_event_loop().time() - last_fired
                if elapsed < self.trigger_cooldown_seconds:
                    logger.info(f"Trigger {trigger_id} in cooldown ({elapsed:.1f}s < {self.trigger_cooldown_seconds}s)")
                    return

            # Update stats
            stats["fire_count"] = stats.get("fire_count", 0) + 1
            stats["last_fired_at"] = asyncio.get_event_loop().time()

        try:
            # Generate new session ID for proactive execution
            session_id = f"proactive_{uuid.uuid4().hex[:12]}"

            # Build prompt from action config
            prompt = action_config.get("prompt", f"Trigger fired: {trigger_data.get('description', 'Unknown trigger')}")

            # Include trigger context in the prompt
            trigger_context = (
                f"\n\n[Proactive Trigger Context]\n"
                f"Trigger: {trigger_data.get('description', 'Unknown')}\n"
                f"Type: {trigger_data.get('trigger_type', 'Unknown')}\n"
                f"Event Data: {trigger_data.get('event_data', {})}\n"
                f"Triggered At: {trigger_data.get('triggered_at', 'Unknown')}"
            )
            full_prompt = prompt + trigger_context

            # Execute workflow via SDK query
            await self._execute_proactive_query(
                user_id=user_id,
                session_id=session_id,
                prompt=full_prompt,
                trigger_id=trigger_id,
                action_config=action_config
            )

            # Update workflow execution count
            if trigger_id in self._trigger_stats:
                self._trigger_stats[trigger_id]["workflow_executions"] = (
                    self._trigger_stats[trigger_id].get("workflow_executions", 0) + 1
                )

            logger.info(
                f"Trigger workflow complete | "
                f"trigger_id={trigger_id} | "
                f"session_id={session_id}"
            )

        except Exception as e:
            logger.error(f"Trigger workflow failed: {e}", exc_info=True)

    async def _execute_proactive_query(
        self,
        user_id: str,
        session_id: str,
        prompt: str,
        trigger_id: str,
        action_config: Dict[str, Any]
    ) -> None:
        """
        Execute proactive agent query for triggered workflow

        Args:
            user_id: User ID
            session_id: New session ID for this execution
            prompt: Full prompt with trigger context
            trigger_id: Trigger that fired
            action_config: Action configuration
        """
        try:
            # Import SDK query lazily to avoid circular imports
            from isa_agent_sdk.query import query
            from isa_agent_sdk.options import ISAAgentOptions, ExecutionMode

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
            async for msg in query(prompt=prompt, options=options):
                # Log significant events
                if msg.is_complete:
                    logger.info(
                        f"Proactive execution complete | "
                        f"session_id={session_id} | "
                        f"trigger_id={trigger_id}"
                    )
                elif msg.is_error:
                    logger.error(
                        f"Proactive execution error | "
                        f"session_id={session_id} | "
                        f"error={msg.content}"
                    )

            # TODO: Store results, send notifications, etc.

        except Exception as e:
            logger.error(f"Proactive query execution failed: {e}", exc_info=True)


# Singleton instance
_trigger_service: Optional[TriggerService] = None


async def get_trigger_service(
    event_service_url: Optional[str] = None,
    task_service_url: Optional[str] = None
) -> TriggerService:
    """
    Get singleton TriggerService instance

    Args:
        event_service_url: Event Service URL (only used on first call)
        task_service_url: Task Service URL (only used on first call)

    Returns:
        Initialized TriggerService instance
    """
    global _trigger_service
    if _trigger_service is None:
        _trigger_service = TriggerService(
            event_service_url=event_service_url,
            task_service_url=task_service_url
        )
        await _trigger_service.initialize()
    return _trigger_service


async def shutdown_trigger_service() -> None:
    """Shutdown the singleton trigger service"""
    global _trigger_service
    if _trigger_service is not None:
        await _trigger_service.shutdown()
        _trigger_service = None


__all__ = [
    "TriggerService",
    "get_trigger_service",
    "shutdown_trigger_service",
]
