#!/usr/bin/env python3
"""
Triggers Service Package

Provides event-driven trigger infrastructure for proactive agent activation.

Components:
- NATSEventBus: NATS JetStream adapter for event pub/sub
- TriggerService: Orchestrator for trigger management and workflow execution

Example:
    from isa_agent.services.triggers import TriggerService, NATSEventBus

    # Create event bus and trigger service
    event_bus = NATSEventBus()
    trigger_service = TriggerService(event_bus=event_bus)

    # Initialize
    await trigger_service.initialize()

    # Register a trigger
    trigger_id = await trigger_service.register_trigger(
        user_id="user123",
        trigger_type=TriggerType.THRESHOLD,
        description="Alert when price drops > 5%",
        conditions={"threshold_value": 5.0, "direction": "down"},
        action_config={"prompt": "Analyze the price drop"}
    )
"""

from .nats_event_bus import NATSEventBus, get_event_bus
from .trigger_service import TriggerService, get_trigger_service

__all__ = [
    "NATSEventBus",
    "get_event_bus",
    "TriggerService",
    "get_trigger_service",
]
