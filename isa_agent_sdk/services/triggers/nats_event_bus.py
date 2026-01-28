#!/usr/bin/env python3
"""
NATS Event Bus - JetStream-based event pub/sub for agent triggers

Provides real-time event delivery for proactive agent activation.
Follows the pattern established in nats_task_queue.py.

Features:
- JetStream-based event streaming
- Subject-based event routing
- Consumer management for event processing
- Event acknowledgment and replay support
"""

import json
import os
from typing import Any, Callable, Dict, List, Optional

from isa_common import AsyncNATSClient

from isa_agent_sdk.utils.logger import setup_logger

logger = setup_logger("isA_Agent.NATSEventBus")


class NATSEventBus:
    """
    NATS JetStream-based event bus for agent triggers

    Handles event pub/sub for:
    - Price/threshold alerts
    - IoT device events
    - Scheduled task completions
    - System events and notifications
    """

    # Stream and subject configuration
    STREAM_NAME = "ISA_AGENT_EVENTS"
    SUBJECT_PREFIX = "isa.agent.events"

    # Event category subjects
    SUBJECT_PRICE = f"{SUBJECT_PREFIX}.price"
    SUBJECT_IOT = f"{SUBJECT_PREFIX}.iot"
    SUBJECT_SCHEDULE = f"{SUBJECT_PREFIX}.schedule"
    SUBJECT_SYSTEM = f"{SUBJECT_PREFIX}.system"
    SUBJECT_CUSTOM = f"{SUBJECT_PREFIX}.custom"

    def __init__(self, user_id: str = "event-trigger-service"):
        """
        Initialize NATS event bus

        Args:
            user_id: Service user ID for NATS authentication
        """
        self.user_id = user_id
        self.nats_client: Optional[AsyncNATSClient] = None
        self._connected = False
        self._subscriptions: Dict[str, Any] = {}
        self._callbacks: Dict[str, List[Callable]] = {}

    async def connect(self) -> bool:
        """
        Establish NATS connection and initialize stream

        Returns:
            True if connection succeeded
        """
        try:
            from isa_common.consul_client import ConsulRegistry

            # Use environment variables for Consul connection
            consul_host = os.getenv("CONSUL_HOST", "localhost")
            consul_port = int(os.getenv("CONSUL_PORT", "8500"))

            consul_registry = ConsulRegistry(
                consul_host=consul_host, consul_port=consul_port
            )

            self.nats_client = AsyncNATSClient(
                user_id=self.user_id,
                consul_registry=consul_registry,
                service_name_override="nats_grpc_service",
            )

            self._connected = True
            logger.info("NATS event bus connected via Consul discovery")

            # Initialize JetStream
            await self._initialize_stream()

            return True

        except Exception as e:
            logger.error(f"Failed to connect to NATS: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Close NATS connection and clean up subscriptions"""
        if self.nats_client and self._connected:
            # Clean up subscriptions
            for subject in list(self._subscriptions.keys()):
                try:
                    await self.unsubscribe(subject)
                except Exception as e:
                    logger.warning(f"Error unsubscribing from {subject}: {e}")

            await self.nats_client.close()
            self._connected = False
            logger.info("NATS event bus disconnected")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()

    @property
    def is_connected(self) -> bool:
        """Check if event bus is connected"""
        return self._connected

    async def _initialize_stream(self) -> None:
        """Initialize JetStream stream for events"""
        try:
            async with self.nats_client:
                await self.nats_client.create_stream(
                    name=self.STREAM_NAME,
                    subjects=[f"{self.SUBJECT_PREFIX}.>"],  # Catch all event types
                    max_msgs=100000,  # Keep last 100k events
                    max_bytes=500 * 1024 * 1024,  # 500MB max storage
                )
            logger.info(f"JetStream event stream initialized: {self.STREAM_NAME}")

        except Exception as e:
            # Stream might already exist, that's okay
            logger.warning(f"Stream initialization: {e} (may already exist)")

    # ============================================
    # Event Publishing
    # ============================================

    async def publish(
        self,
        subject: str,
        data: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None
    ) -> Optional[int]:
        """
        Publish event to NATS JetStream

        Args:
            subject: Event subject (e.g., "isa.agent.events.price")
            data: Event data dictionary
            headers: Optional message headers

        Returns:
            Sequence number if successful, None otherwise
        """
        if not self._connected:
            logger.error("Cannot publish: event bus not connected")
            return None

        try:
            # Ensure subject has correct prefix
            if not subject.startswith(self.SUBJECT_PREFIX):
                subject = f"{self.SUBJECT_PREFIX}.{subject}"

            # Serialize event data
            event_bytes = json.dumps(data).encode("utf-8")

            async with self.nats_client:
                result = await self.nats_client.publish_to_stream(
                    stream_name=self.STREAM_NAME,
                    subject=subject,
                    data=event_bytes
                )

            if result and "sequence" in result:
                sequence = result["sequence"]
                logger.debug(
                    f"event_published | "
                    f"subject={subject} | "
                    f"sequence={sequence}"
                )
                return sequence
            else:
                logger.error(f"Failed to publish event: No sequence returned")
                return None

        except Exception as e:
            logger.error(f"Failed to publish event to {subject}: {e}", exc_info=True)
            return None

    async def publish_price_event(
        self,
        product: str,
        current_value: float,
        previous_value: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[int]:
        """
        Publish a price change event

        Args:
            product: Product/asset name
            current_value: Current price value
            previous_value: Previous price value
            metadata: Optional additional metadata

        Returns:
            Sequence number if successful
        """
        change_percent = ((current_value - previous_value) / previous_value * 100) if previous_value else 0

        event_data = {
            "event_type": "price_change",
            "event_category": "price",
            "product": product,
            "current_value": current_value,
            "previous_value": previous_value,
            "change_percent": change_percent,
            "metadata": metadata or {}
        }

        return await self.publish(self.SUBJECT_PRICE, event_data)

    async def publish_iot_event(
        self,
        device_id: str,
        event_type: str,
        data: Dict[str, Any]
    ) -> Optional[int]:
        """
        Publish an IoT device event

        Args:
            device_id: Device identifier
            event_type: Type of IoT event
            data: Event data

        Returns:
            Sequence number if successful
        """
        event_data = {
            "event_type": event_type,
            "event_category": "iot",
            "device_id": device_id,
            **data
        }

        return await self.publish(self.SUBJECT_IOT, event_data)

    async def publish_schedule_event(
        self,
        task_id: str,
        task_type: str,
        result: Dict[str, Any]
    ) -> Optional[int]:
        """
        Publish a scheduled task completion event

        Args:
            task_id: Task identifier
            task_type: Type of scheduled task
            result: Task execution result

        Returns:
            Sequence number if successful
        """
        event_data = {
            "event_type": "task_completed",
            "event_category": "schedule",
            "task_id": task_id,
            "task_type": task_type,
            "result": result
        }

        return await self.publish(self.SUBJECT_SCHEDULE, event_data)

    async def publish_custom_event(
        self,
        event_type: str,
        data: Dict[str, Any]
    ) -> Optional[int]:
        """
        Publish a custom event

        Args:
            event_type: Custom event type
            data: Event data

        Returns:
            Sequence number if successful
        """
        event_data = {
            "event_type": event_type,
            "event_category": "custom",
            **data
        }

        return await self.publish(self.SUBJECT_CUSTOM, event_data)

    # ============================================
    # Event Subscription
    # ============================================

    async def subscribe(
        self,
        subject: str,
        callback: Callable[[Dict[str, Any]], Any],
        consumer_name: Optional[str] = None
    ) -> bool:
        """
        Subscribe to events on a subject

        Args:
            subject: Event subject pattern (e.g., "isa.agent.events.>")
            callback: Async callback function to handle events
            consumer_name: Optional consumer name for durable subscription

        Returns:
            True if subscription succeeded
        """
        if not self._connected:
            logger.error("Cannot subscribe: event bus not connected")
            return False

        try:
            # Ensure subject has correct prefix
            if not subject.startswith(self.SUBJECT_PREFIX) and subject != "events.>":
                subject = f"{self.SUBJECT_PREFIX}.{subject}"

            # Store callback
            if subject not in self._callbacks:
                self._callbacks[subject] = []
            self._callbacks[subject].append(callback)

            # Create consumer if name provided
            consumer = consumer_name or f"trigger_consumer_{subject.replace('.', '_')}"

            async with self.nats_client:
                await self.nats_client.create_consumer(
                    stream_name=self.STREAM_NAME,
                    consumer_name=consumer,
                    filter_subject=subject
                )

            self._subscriptions[subject] = {
                "consumer_name": consumer,
                "callbacks": self._callbacks[subject]
            }

            logger.info(f"Subscribed to {subject} with consumer {consumer}")
            return True

        except Exception as e:
            logger.error(f"Failed to subscribe to {subject}: {e}")
            return False

    async def unsubscribe(self, subject: str) -> bool:
        """
        Unsubscribe from a subject

        Args:
            subject: Event subject to unsubscribe from

        Returns:
            True if unsubscription succeeded
        """
        if subject not in self._subscriptions:
            return True

        try:
            # Remove callbacks
            if subject in self._callbacks:
                del self._callbacks[subject]

            del self._subscriptions[subject]

            logger.info(f"Unsubscribed from {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to unsubscribe from {subject}: {e}")
            return False

    async def pull_events(
        self,
        subject: str,
        batch_size: int = 10,
        timeout_ms: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Pull events from a subject

        Args:
            subject: Event subject
            batch_size: Number of events to pull
            timeout_ms: Timeout in milliseconds

        Returns:
            List of event data dictionaries
        """
        if not self._connected or subject not in self._subscriptions:
            return []

        try:
            consumer_name = self._subscriptions[subject]["consumer_name"]

            async with self.nats_client:
                messages = await self.nats_client.pull_messages(
                    stream_name=self.STREAM_NAME,
                    consumer_name=consumer_name,
                    batch_size=batch_size,
                )

            events = []
            for msg in messages:
                try:
                    if isinstance(msg, dict) and "data" in msg:
                        data = msg["data"]
                        if isinstance(data, bytes):
                            data = json.loads(data.decode("utf-8"))
                        events.append({
                            "data": data,
                            "sequence": msg.get("sequence"),
                            "subject": msg.get("subject")
                        })
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode event message")
                    continue

            return events

        except Exception as e:
            logger.error(f"Failed to pull events from {subject}: {e}")
            return []

    async def process_events(
        self,
        subject: str,
        batch_size: int = 10
    ) -> int:
        """
        Pull and process events using registered callbacks

        Args:
            subject: Event subject
            batch_size: Number of events to process

        Returns:
            Number of events processed
        """
        events = await self.pull_events(subject, batch_size)

        if not events or subject not in self._callbacks:
            return 0

        processed = 0
        for event in events:
            event_data = event.get("data", {})

            for callback in self._callbacks[subject]:
                try:
                    import asyncio
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event_data)
                    else:
                        callback(event_data)
                    processed += 1
                except Exception as e:
                    logger.error(f"Callback error for event: {e}")

        return processed

    # ============================================
    # Health and Statistics
    # ============================================

    async def health_check(self) -> bool:
        """
        Check if NATS and JetStream are healthy

        Returns:
            True if healthy
        """
        if not self._connected or not self.nats_client:
            return False

        try:
            async with self.nats_client:
                health = await self.nats_client.health_check()
            is_healthy = (
                health.get("healthy", False) and
                health.get("jetstream_enabled", False)
            ) if health else False

            if is_healthy:
                logger.debug("NATS event bus is healthy")
            else:
                logger.warning("NATS event bus health check failed")

            return is_healthy

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def get_stream_info(self) -> Optional[Dict[str, Any]]:
        """
        Get event stream information and statistics

        Returns:
            Stream info dictionary or None
        """
        if not self._connected:
            return None

        try:
            async with self.nats_client:
                streams = await self.nats_client.list_streams()

            for stream in streams:
                if stream.get("name") == self.STREAM_NAME:
                    return stream

            return None

        except Exception as e:
            logger.error(f"Failed to get stream info: {e}")
            return None


# Singleton instance for easy access
_event_bus: Optional[NATSEventBus] = None


async def get_event_bus() -> NATSEventBus:
    """
    Get singleton NATS event bus instance

    Returns:
        Connected NATSEventBus instance
    """
    global _event_bus
    if _event_bus is None:
        _event_bus = NATSEventBus()
        await _event_bus.connect()
    return _event_bus


async def shutdown_event_bus() -> None:
    """Shutdown the singleton event bus"""
    global _event_bus
    if _event_bus is not None:
        await _event_bus.disconnect()
        _event_bus = None


__all__ = [
    "NATSEventBus",
    "get_event_bus",
    "shutdown_event_bus",
]
