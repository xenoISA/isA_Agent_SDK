#!/usr/bin/env python3
"""
Event Trigger Manager - Manages event-driven proactive agent triggers

This module provides:
1. Event subscription management
2. Trigger pattern matching
3. Task-based scheduling integration
4. Proactive agent workflow initiation

Integrates with:
- Event Service: For real-time event monitoring
- Task Service: For scheduled recurring tasks
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from enum import Enum
import httpx

logger = logging.getLogger(__name__)


class TriggerType(str, Enum):
    """Types of event triggers"""
    EVENT_PATTERN = "event_pattern"      # Pattern-based event trigger
    SCHEDULED_TASK = "scheduled_task"    # Task-based scheduled trigger
    THRESHOLD = "threshold"              # Threshold-based trigger (e.g., price change > 5%)
    TIME_BASED = "time_based"            # Time-based trigger (e.g., every day at 9am)
    COMPOSITE = "composite"              # Multiple conditions combined


class TriggerCondition:
    """
    Defines conditions for triggering proactive agent actions
    """

    def __init__(
        self,
        trigger_id: str,
        trigger_type: TriggerType,
        user_id: str,
        description: str,
        conditions: Dict[str, Any],
        action_config: Dict[str, Any],
        enabled: bool = True
    ):
        """
        Initialize trigger condition

        Args:
            trigger_id: Unique trigger identifier
            trigger_type: Type of trigger
            user_id: User who owns this trigger
            description: Human-readable description
            conditions: Conditions to match (depends on trigger_type)
            action_config: Configuration for agent action when triggered
            enabled: Whether trigger is active

        Examples:
            # Price threshold trigger
            conditions = {
                "event_type": "price_change",
                "product": "Bitcoin",
                "threshold_type": "percentage",
                "threshold_value": 5.0,
                "direction": "any"  # "up", "down", or "any"
            }

            # Time-based trigger
            conditions = {
                "schedule": {
                    "type": "daily",
                    "time": "09:00",
                    "timezone": "UTC"
                },
                "task_type": "news_summary"
            }

            # Event pattern trigger
            conditions = {
                "event_source": "frontend",
                "event_category": "security",
                "event_type": "failed_login",
                "pattern": {
                    "count": 3,
                    "within_minutes": 5
                }
            }
        """
        self.trigger_id = trigger_id
        self.trigger_type = trigger_type
        self.user_id = user_id
        self.description = description
        self.conditions = conditions
        self.action_config = action_config
        self.enabled = enabled
        self.created_at = datetime.utcnow()
        self.last_triggered_at: Optional[datetime] = None
        self.trigger_count = 0

    def matches(self, event_data: Dict[str, Any]) -> bool:
        """
        Check if event data matches this trigger's conditions

        Args:
            event_data: Event data to check

        Returns:
            True if conditions match
        """
        if not self.enabled:
            return False

        try:
            if self.trigger_type == TriggerType.EVENT_PATTERN:
                return self._match_event_pattern(event_data)
            elif self.trigger_type == TriggerType.THRESHOLD:
                return self._match_threshold(event_data)
            elif self.trigger_type == TriggerType.TIME_BASED:
                return self._match_time_based(event_data)
            elif self.trigger_type == TriggerType.COMPOSITE:
                return self._match_composite(event_data)
            else:
                return False
        except Exception as e:
            logger.error(f"Error matching trigger {self.trigger_id}: {e}")
            return False

    def _match_event_pattern(self, event_data: Dict[str, Any]) -> bool:
        """Match event pattern conditions"""
        # Check basic event attributes
        if "event_type" in self.conditions:
            if event_data.get("event_type") != self.conditions["event_type"]:
                return False

        if "event_source" in self.conditions:
            if event_data.get("event_source") != self.conditions["event_source"]:
                return False

        if "event_category" in self.conditions:
            if event_data.get("event_category") != self.conditions["event_category"]:
                return False

        # Check pattern conditions (e.g., count within time window)
        if "pattern" in self.conditions:
            # This would require event history tracking
            # Simplified implementation for now
            return True

        return True

    def _match_threshold(self, event_data: Dict[str, Any]) -> bool:
        """Match threshold conditions"""
        threshold_value = self.conditions.get("threshold_value", 0)
        threshold_type = self.conditions.get("threshold_type", "absolute")
        direction = self.conditions.get("direction", "any")

        # Get current and previous values
        current_value = event_data.get("current_value")
        previous_value = event_data.get("previous_value")

        if current_value is None:
            return False

        if threshold_type == "percentage" and previous_value is not None:
            change_percent = abs((current_value - previous_value) / previous_value * 100)

            if direction == "up":
                return current_value > previous_value and change_percent >= threshold_value
            elif direction == "down":
                return current_value < previous_value and change_percent >= threshold_value
            else:  # any
                return change_percent >= threshold_value
        elif threshold_type == "absolute":
            if direction == "up":
                return current_value > threshold_value
            elif direction == "down":
                return current_value < threshold_value
            else:  # any
                return abs(current_value) >= threshold_value

        return False

    def _match_time_based(self, event_data: Dict[str, Any]) -> bool:
        """Match time-based conditions"""
        # Time-based triggers are usually handled by task scheduler
        # This is for verification
        return True

    def _match_composite(self, event_data: Dict[str, Any]) -> bool:
        """Match composite conditions (AND/OR logic)"""
        logic = self.conditions.get("logic", "AND")
        sub_conditions = self.conditions.get("conditions", [])

        results = []
        for condition in sub_conditions:
            # Recursively check each condition
            # Simplified implementation
            results.append(True)

        if logic == "AND":
            return all(results)
        elif logic == "OR":
            return any(results)
        else:
            return False


class EventTriggerManager:
    """
    Manages event-driven triggers for proactive agent activation

    Responsibilities:
    1. Subscribe to events from Event Service
    2. Create and manage scheduled tasks via Task Service
    3. Match incoming events against trigger conditions
    4. Initiate agent workflows when triggers fire
    """

    def __init__(self, event_service_url: str, task_service_url: str, event_bus=None):
        """
        Initialize event trigger manager

        Args:
            event_service_url: URL of event service
            task_service_url: URL of task service
            event_bus: Optional NATS event bus for direct subscriptions
        """
        self.event_service_url = event_service_url
        self.task_service_url = task_service_url
        self.event_bus = event_bus

        # Trigger registry: trigger_id -> TriggerCondition
        self.triggers: Dict[str, TriggerCondition] = {}

        # User trigger mapping: user_id -> List[trigger_id]
        self.user_triggers: Dict[str, List[str]] = {}

        # Event subscriptions: subscription_id -> trigger_ids
        self.event_subscriptions: Dict[str, List[str]] = {}

        # Scheduled tasks: task_id -> trigger_id
        self.scheduled_tasks: Dict[str, str] = {}

        # Callback for triggering agent workflow
        self.workflow_callback: Optional[Callable] = None

        # HTTP client for service communication
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # Event history for pattern matching
        self.event_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000

    async def initialize(self):
        """Initialize the trigger manager"""
        logger.info("EventTriggerManager initializing...")

        # Subscribe to NATS events if event_bus is available
        if self.event_bus:
            try:
                await self.event_bus.subscribe(
                    subject="events.>",
                    callback=self._handle_nats_event
                )
                logger.info("Subscribed to NATS events")
            except Exception as e:
                logger.warning(f"Failed to subscribe to NATS events: {e}")

        logger.info("EventTriggerManager initialized")

    def set_workflow_callback(self, callback: Callable):
        """
        Set callback function for triggering agent workflow

        Args:
            callback: Async function(user_id, trigger_data, action_config) -> result
        """
        self.workflow_callback = callback

    # ==================== Trigger Management ====================

    async def register_trigger(
        self,
        user_id: str,
        trigger_type: TriggerType,
        description: str,
        conditions: Dict[str, Any],
        action_config: Dict[str, Any]
    ) -> str:
        """
        Register a new event trigger

        Args:
            user_id: User ID
            trigger_type: Type of trigger
            description: Human-readable description
            conditions: Trigger conditions
            action_config: Agent action configuration

        Returns:
            trigger_id: Unique trigger ID
        """
        import uuid
        trigger_id = str(uuid.uuid4())

        trigger = TriggerCondition(
            trigger_id=trigger_id,
            trigger_type=trigger_type,
            user_id=user_id,
            description=description,
            conditions=conditions,
            action_config=action_config,
            enabled=True
        )

        # Store trigger
        self.triggers[trigger_id] = trigger

        # Add to user mapping
        if user_id not in self.user_triggers:
            self.user_triggers[user_id] = []
        self.user_triggers[user_id].append(trigger_id)

        # Set up event subscription or scheduled task
        if trigger_type == TriggerType.SCHEDULED_TASK or trigger_type == TriggerType.TIME_BASED:
            await self._create_scheduled_task(trigger)
        else:
            await self._create_event_subscription(trigger)

        logger.info(
            f"Registered trigger {trigger_id} for user {user_id}: {description}"
        )

        return trigger_id

    async def unregister_trigger(self, trigger_id: str) -> bool:
        """
        Unregister an event trigger

        Args:
            trigger_id: Trigger ID to remove

        Returns:
            True if successful
        """
        if trigger_id not in self.triggers:
            return False

        trigger = self.triggers[trigger_id]

        # Remove from user mapping
        if trigger.user_id in self.user_triggers:
            self.user_triggers[trigger.user_id].remove(trigger_id)

        # Clean up event subscription or scheduled task
        if trigger.trigger_type in [TriggerType.SCHEDULED_TASK, TriggerType.TIME_BASED]:
            await self._delete_scheduled_task(trigger_id)
        else:
            await self._delete_event_subscription(trigger_id)

        # Remove trigger
        del self.triggers[trigger_id]

        logger.info(f"Unregistered trigger {trigger_id}")
        return True

    async def get_user_triggers(self, user_id: str) -> List[TriggerCondition]:
        """
        Get all triggers for a user

        Args:
            user_id: User ID

        Returns:
            List of trigger conditions
        """
        trigger_ids = self.user_triggers.get(user_id, [])
        return [self.triggers[tid] for tid in trigger_ids if tid in self.triggers]

    # ==================== Event Subscription ====================

    async def _create_event_subscription(self, trigger: TriggerCondition):
        """
        Create event subscription in Event Service

        Args:
            trigger: Trigger condition to create subscription for
        """
        try:
            # Extract event types from conditions
            event_types = []
            if "event_type" in trigger.conditions:
                event_types = [trigger.conditions["event_type"]]

            # Extract event sources
            event_sources = []
            if "event_source" in trigger.conditions:
                event_sources = [trigger.conditions["event_source"]]

            # Extract event categories
            event_categories = []
            if "event_category" in trigger.conditions:
                event_categories = [trigger.conditions["event_category"]]

            # Create subscription via Event Service API
            subscription_data = {
                "subscriber_name": f"agent_trigger_{trigger.trigger_id}",
                "subscriber_type": "agent_service",
                "event_types": event_types or ["*"],
                "event_sources": event_sources,
                "event_categories": event_categories,
                "callback_url": None,  # We handle via NATS, not webhook
                "enabled": True
            }

            response = await self.http_client.post(
                f"{self.event_service_url}/api/events/subscriptions",
                json=subscription_data
            )

            if response.status_code == 200:
                subscription = response.json()
                subscription_id = subscription["subscription_id"]

                # Store mapping
                if subscription_id not in self.event_subscriptions:
                    self.event_subscriptions[subscription_id] = []
                self.event_subscriptions[subscription_id].append(trigger.trigger_id)

                logger.info(
                    f"Created event subscription {subscription_id} for trigger {trigger.trigger_id}"
                )
            else:
                logger.error(f"Failed to create event subscription: {response.text}")

        except Exception as e:
            logger.error(f"Error creating event subscription: {e}")

    async def _delete_event_subscription(self, trigger_id: str):
        """Delete event subscription for trigger"""
        # Find subscription for this trigger
        subscription_id_to_delete = None
        for subscription_id, trigger_ids in self.event_subscriptions.items():
            if trigger_id in trigger_ids:
                trigger_ids.remove(trigger_id)
                if not trigger_ids:  # No more triggers use this subscription
                    subscription_id_to_delete = subscription_id
                break

        if subscription_id_to_delete:
            try:
                response = await self.http_client.delete(
                    f"{self.event_service_url}/api/events/subscriptions/{subscription_id_to_delete}"
                )

                if response.status_code == 200:
                    del self.event_subscriptions[subscription_id_to_delete]
                    logger.info(f"Deleted event subscription {subscription_id_to_delete}")
            except Exception as e:
                logger.error(f"Error deleting event subscription: {e}")

    # ==================== Task Scheduling ====================

    async def _create_scheduled_task(self, trigger: TriggerCondition):
        """
        Create scheduled task in Task Service

        Args:
            trigger: Trigger condition to create task for
        """
        try:
            # Extract schedule from conditions
            schedule = trigger.conditions.get("schedule", {})
            task_type = trigger.conditions.get("task_type", "custom")

            # Create task via Task Service API
            task_data = {
                "name": trigger.description,
                "description": f"Proactive agent trigger: {trigger.trigger_id}",
                "task_type": task_type,
                "priority": "medium",
                "config": {
                    "trigger_id": trigger.trigger_id,
                    "agent_action": trigger.action_config
                },
                "schedule": schedule,
                "metadata": {
                    "trigger_id": trigger.trigger_id,
                    "trigger_type": trigger.trigger_type.value
                }
            }

            # Need to use user's auth token here
            # For now, using internal API
            response = await self.http_client.post(
                f"{self.task_service_url}/api/v1/tasks",
                json=task_data,
                headers={"x-api-key": "internal_agent_service_key"}  # TODO: Use proper auth
            )

            if response.status_code == 200:
                task = response.json()
                task_id = task["task_id"]

                # Store mapping
                self.scheduled_tasks[task_id] = trigger.trigger_id

                logger.info(
                    f"Created scheduled task {task_id} for trigger {trigger.trigger_id}"
                )
            else:
                logger.error(f"Failed to create scheduled task: {response.text}")

        except Exception as e:
            logger.error(f"Error creating scheduled task: {e}")

    async def _delete_scheduled_task(self, trigger_id: str):
        """Delete scheduled task for trigger"""
        # Find task for this trigger
        task_id_to_delete = None
        for task_id, tid in self.scheduled_tasks.items():
            if tid == trigger_id:
                task_id_to_delete = task_id
                break

        if task_id_to_delete:
            try:
                response = await self.http_client.delete(
                    f"{self.task_service_url}/api/v1/tasks/{task_id_to_delete}",
                    headers={"x-api-key": "internal_agent_service_key"}
                )

                if response.status_code == 200:
                    del self.scheduled_tasks[task_id_to_delete]
                    logger.info(f"Deleted scheduled task {task_id_to_delete}")
            except Exception as e:
                logger.error(f"Error deleting scheduled task: {e}")

    # ==================== Event Handling ====================

    async def _handle_nats_event(self, event: Any):
        """
        Handle incoming NATS event

        Args:
            event: NATS event message
        """
        try:
            # Parse event data
            event_data = event.data if hasattr(event, 'data') else {}

            # Add to history
            self._add_to_event_history(event_data)

            # Check all triggers
            await self._check_triggers(event_data)

        except Exception as e:
            logger.error(f"Error handling NATS event: {e}")

    async def handle_task_completion(self, task_data: Dict[str, Any]):
        """
        Handle task completion event from Task Service

        Args:
            task_data: Task completion data
        """
        try:
            task_id = task_data.get("task_id")

            # Check if this is a scheduled trigger task
            if task_id in self.scheduled_tasks:
                trigger_id = self.scheduled_tasks[task_id]
                trigger = self.triggers.get(trigger_id)

                if trigger:
                    # Trigger agent workflow with task results
                    await self._trigger_workflow(trigger, task_data)

        except Exception as e:
            logger.error(f"Error handling task completion: {e}")

    async def _check_triggers(self, event_data: Dict[str, Any]):
        """
        Check all triggers against event data

        Args:
            event_data: Event data to check
        """
        for trigger_id, trigger in self.triggers.items():
            try:
                if trigger.matches(event_data):
                    logger.info(
                        f"Trigger {trigger_id} matched for user {trigger.user_id}"
                    )
                    await self._trigger_workflow(trigger, event_data)
            except Exception as e:
                logger.error(f"Error checking trigger {trigger_id}: {e}")

    async def _trigger_workflow(
        self,
        trigger: TriggerCondition,
        event_data: Dict[str, Any]
    ):
        """
        Trigger agent workflow

        Args:
            trigger: Trigger condition that fired
            event_data: Event data that triggered the workflow
        """
        try:
            # Update trigger stats
            trigger.last_triggered_at = datetime.utcnow()
            trigger.trigger_count += 1

            # Call workflow callback
            if self.workflow_callback:
                await self.workflow_callback(
                    user_id=trigger.user_id,
                    trigger_data={
                        "trigger_id": trigger.trigger_id,
                        "trigger_type": trigger.trigger_type.value,
                        "description": trigger.description,
                        "event_data": event_data,
                        "triggered_at": trigger.last_triggered_at.isoformat()
                    },
                    action_config=trigger.action_config
                )

                logger.info(
                    f"Triggered workflow for trigger {trigger.trigger_id}, "
                    f"user {trigger.user_id}"
                )
            else:
                logger.warning("No workflow callback registered")

        except Exception as e:
            logger.error(f"Error triggering workflow: {e}")

    # ==================== Event History ====================

    def _add_to_event_history(self, event_data: Dict[str, Any]):
        """Add event to history for pattern matching"""
        self.event_history.append({
            **event_data,
            "received_at": datetime.utcnow()
        })

        # Trim history if too large
        if len(self.event_history) > self.max_history_size:
            self.event_history = self.event_history[-self.max_history_size:]

    def get_event_history(
        self,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get event history with optional filters

        Args:
            user_id: Filter by user
            event_type: Filter by event type
            since: Filter events after this time
            limit: Maximum number of events to return

        Returns:
            List of events
        """
        filtered = self.event_history

        if user_id:
            filtered = [e for e in filtered if e.get("user_id") == user_id]

        if event_type:
            filtered = [e for e in filtered if e.get("event_type") == event_type]

        if since:
            filtered = [
                e for e in filtered
                if e.get("received_at", datetime.min) >= since
            ]

        return filtered[-limit:]

    # ==================== Cleanup ====================

    async def shutdown(self):
        """Shutdown the trigger manager"""
        logger.info("EventTriggerManager shutting down...")

        # Close HTTP client
        await self.http_client.aclose()

        logger.info("EventTriggerManager shutdown complete")
