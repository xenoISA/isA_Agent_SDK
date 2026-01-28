#!/usr/bin/env python3
"""
isA Agent SDK - Event Triggers API
==================================

SDK wrapper for event-driven proactive agent activation.

This module provides a clean API for:
1. Registering event triggers (price thresholds, schedules, IoT events, etc.)
2. Managing user triggers (list, update, delete)
3. Querying trigger statistics

Event triggers enable PROACTIVE agent activation - the agent starts working
WITHOUT a user chat message, triggered by external events like:
- Price changes exceeding a threshold
- Scheduled recurring tasks (cron-like)
- IoT device events
- Custom webhook events

Example:
    from isa_agent_sdk import triggers
    from isa_agent_sdk.triggers import TriggerType

    # Register a price alert trigger
    trigger_id = await triggers.register_trigger(
        user_id="user123",
        trigger_type=TriggerType.THRESHOLD,
        description="Alert when BTC drops > 5%",
        conditions={
            "event_type": "price_change",
            "product": "Bitcoin",
            "threshold_type": "percentage",
            "threshold_value": 5.0,
            "direction": "down"
        },
        action_config={
            "prompt": "Analyze the Bitcoin price drop and suggest actions",
            "allowed_tools": ["web_search", "analyze_data"]
        }
    )

    # List user's triggers
    user_triggers = await triggers.get_user_triggers("user123")

    # Unregister a trigger
    await triggers.unregister_trigger(trigger_id)
"""

import logging
from typing import Any, Dict, List, Optional

# Re-export TriggerType for convenience
from .nodes.utils.event_trigger_manager import TriggerCondition, TriggerType

logger = logging.getLogger(__name__)


# ==================== Trigger Registration ====================

async def register_trigger(
    user_id: str,
    trigger_type: TriggerType,
    description: str,
    conditions: Dict[str, Any],
    action_config: Dict[str, Any]
) -> Optional[str]:
    """
    Register an event trigger for proactive agent activation.

    When the trigger conditions are met, a new agent session is created
    and executes the configured action automatically.

    Args:
        user_id: User who owns the trigger
        trigger_type: Type of trigger (THRESHOLD, EVENT_PATTERN, SCHEDULED_TASK, etc.)
        description: Human-readable description
        conditions: Conditions that must be met to fire the trigger
        action_config: Configuration for the agent action when triggered

    Returns:
        trigger_id if successful, None otherwise

    Condition Examples by Type:

        THRESHOLD:
            conditions = {
                "event_type": "price_change",
                "product": "Bitcoin",
                "threshold_type": "percentage",  # or "absolute"
                "threshold_value": 5.0,
                "direction": "down"  # "up", "down", or "any"
            }

        SCHEDULED_TASK / TIME_BASED:
            conditions = {
                "schedule": {
                    "type": "daily",  # or "weekly", "monthly", "cron"
                    "time": "09:00",
                    "timezone": "UTC"
                },
                "task_type": "news_summary"
            }

        EVENT_PATTERN:
            conditions = {
                "event_source": "iot_hub",
                "event_category": "sensor",
                "event_type": "temperature_alert",
                "pattern": {
                    "field": "temperature",
                    "operator": "gt",
                    "value": 100
                }
            }

    Action Config:
        action_config = {
            "prompt": "Analyze the situation and take appropriate action",
            "allowed_tools": ["web_search", "send_notification"],
            "model": "gpt-5-nano",
            "source": "user_trigger"  # for tracking
        }
    """
    try:
        from .services.triggers import get_trigger_service

        service = await get_trigger_service()
        return await service.register_trigger(
            user_id=user_id,
            trigger_type=trigger_type,
            description=description,
            conditions=conditions,
            action_config=action_config
        )

    except ImportError as e:
        logger.error(f"Trigger service not available: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to register trigger: {e}")
        return None


async def unregister_trigger(trigger_id: str) -> bool:
    """
    Unregister an event trigger.

    Args:
        trigger_id: ID of the trigger to remove

    Returns:
        True if successful, False otherwise
    """
    try:
        from .services.triggers import get_trigger_service

        service = await get_trigger_service()
        return await service.unregister_trigger(trigger_id)

    except ImportError as e:
        logger.error(f"Trigger service not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to unregister trigger {trigger_id}: {e}")
        return False


# ==================== Trigger Query ====================

async def get_user_triggers(user_id: str) -> List[TriggerCondition]:
    """
    Get all triggers registered by a user.

    Args:
        user_id: User ID to query triggers for

    Returns:
        List of TriggerCondition objects
    """
    try:
        from .services.triggers import get_trigger_service

        service = await get_trigger_service()
        return await service.get_user_triggers(user_id)

    except ImportError as e:
        logger.error(f"Trigger service not available: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to get triggers for user {user_id}: {e}")
        return []


async def get_trigger_stats() -> Dict[str, Any]:
    """
    Get trigger service statistics.

    Returns:
        Dictionary with trigger statistics:
        - initialized: Whether the service is initialized
        - event_bus_connected: Whether NATS event bus is connected
        - total_triggers: Total number of registered triggers
        - trigger_stats: Per-trigger statistics
    """
    try:
        from .services.triggers import get_trigger_service

        service = await get_trigger_service()
        return await service.get_trigger_stats()

    except ImportError as e:
        logger.error(f"Trigger service not available: {e}")
        return {"initialized": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Failed to get trigger stats: {e}")
        return {"initialized": False, "error": str(e)}


# ==================== Service Management ====================

async def initialize_triggers(
    event_service_url: Optional[str] = None,
    task_service_url: Optional[str] = None,
    use_nats: bool = True
) -> bool:
    """
    Initialize the trigger service.

    Call this early in your application startup if you plan to use triggers.
    If not called explicitly, the service will be initialized lazily on first use.

    Args:
        event_service_url: URL of the Event Service API
        task_service_url: URL of the Task Service API
        use_nats: Whether to use NATS for event delivery

    Returns:
        True if initialization succeeded
    """
    try:
        from .services.triggers import TriggerService

        service = TriggerService(
            event_service_url=event_service_url,
            task_service_url=task_service_url,
            use_nats_bus=use_nats
        )
        return await service.initialize()

    except ImportError as e:
        logger.error(f"Trigger service not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize trigger service: {e}")
        return False


async def shutdown_triggers() -> bool:
    """
    Shutdown the trigger service gracefully.

    Call this during application shutdown to clean up resources.

    Returns:
        True if shutdown succeeded
    """
    try:
        from .services.triggers import shutdown_trigger_service

        await shutdown_trigger_service()
        return True

    except ImportError as e:
        logger.error(f"Trigger service not available: {e}")
        return True  # Not a failure if service wasn't available
    except Exception as e:
        logger.error(f"Failed to shutdown trigger service: {e}")
        return False


# ==================== Event Manager Access ====================

def get_trigger_manager():
    """
    Get the underlying EventTriggerManager for advanced usage.

    Note: For most use cases, prefer the SDK functions above.
    This is for advanced scenarios requiring direct manager access.

    Returns:
        EventTriggerManager instance or None
    """
    try:
        from .services.triggers import _trigger_service

        if _trigger_service and _trigger_service._trigger_manager:
            return _trigger_service._trigger_manager
        return None

    except ImportError:
        return None


# ==================== Convenience Functions ====================

async def register_price_trigger(
    user_id: str,
    product: str,
    threshold_percent: float,
    direction: str = "any",
    prompt: str = None
) -> Optional[str]:
    """
    Convenience function to register a price threshold trigger.

    Args:
        user_id: User ID
        product: Product/asset name (e.g., "Bitcoin", "AAPL")
        threshold_percent: Percentage change threshold
        direction: "up", "down", or "any"
        prompt: Optional custom prompt for the action

    Returns:
        trigger_id if successful
    """
    description = f"Alert when {product} moves {direction} > {threshold_percent}%"

    conditions = {
        "event_type": "price_change",
        "product": product,
        "threshold_type": "percentage",
        "threshold_value": threshold_percent,
        "direction": direction
    }

    action_config = {
        "prompt": prompt or f"Analyze the {product} price change and suggest actions",
        "source": "price_trigger"
    }

    return await register_trigger(
        user_id=user_id,
        trigger_type=TriggerType.THRESHOLD,
        description=description,
        conditions=conditions,
        action_config=action_config
    )


async def register_schedule_trigger(
    user_id: str,
    schedule_type: str,
    time: str,
    task_description: str,
    prompt: str,
    timezone: str = "UTC"
) -> Optional[str]:
    """
    Convenience function to register a scheduled trigger.

    Args:
        user_id: User ID
        schedule_type: "daily", "weekly", "monthly"
        time: Time of day (e.g., "09:00")
        task_description: Description of the scheduled task
        prompt: Prompt for the agent action
        timezone: Timezone (default: UTC)

    Returns:
        trigger_id if successful
    """
    description = f"{schedule_type.capitalize()} task at {time}: {task_description}"

    conditions = {
        "schedule": {
            "type": schedule_type,
            "time": time,
            "timezone": timezone
        },
        "task_type": "scheduled_agent_task"
    }

    action_config = {
        "prompt": prompt,
        "source": "schedule_trigger"
    }

    return await register_trigger(
        user_id=user_id,
        trigger_type=TriggerType.SCHEDULED_TASK,
        description=description,
        conditions=conditions,
        action_config=action_config
    )


async def register_event_pattern_trigger(
    user_id: str,
    event_source: str,
    event_type: str,
    description: str,
    prompt: str,
    event_category: Optional[str] = None,
    pattern: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Convenience function to register an event pattern trigger.

    Args:
        user_id: User ID
        event_source: Source of events (e.g., "iot_hub", "webhook")
        event_type: Type of event to match
        description: Human-readable description
        prompt: Prompt for the agent action
        event_category: Optional event category filter
        pattern: Optional additional pattern matching rules

    Returns:
        trigger_id if successful
    """
    conditions = {
        "event_source": event_source,
        "event_type": event_type
    }

    if event_category:
        conditions["event_category"] = event_category

    if pattern:
        conditions["pattern"] = pattern

    action_config = {
        "prompt": prompt,
        "source": "event_pattern_trigger"
    }

    return await register_trigger(
        user_id=user_id,
        trigger_type=TriggerType.EVENT_PATTERN,
        description=description,
        conditions=conditions,
        action_config=action_config
    )


__all__ = [
    # Types
    "TriggerType",
    "TriggerCondition",

    # Core functions
    "register_trigger",
    "unregister_trigger",
    "get_user_triggers",
    "get_trigger_stats",

    # Service management
    "initialize_triggers",
    "shutdown_triggers",
    "get_trigger_manager",

    # Convenience functions
    "register_price_trigger",
    "register_schedule_trigger",
    "register_event_pattern_trigger",
]
