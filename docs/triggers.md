# Event Triggers

Event triggers enable proactive agent activation based on external events, schedules, or conditions.

## Overview

Triggers allow agents to:
- React to price changes or threshold breaches
- Execute scheduled tasks
- Respond to IoT device events
- Process webhook notifications
- Handle time-based conditions

## Trigger Types

| Type | Description | Use Case |
|------|-------------|----------|
| `THRESHOLD` | Value crosses a threshold | Price alerts, metrics |
| `SCHEDULED_TASK` | Cron-like scheduling | Daily reports, cleanup |
| `EVENT_PATTERN` | Pattern matching on events | IoT, log monitoring |
| `TIME_BASED` | Time conditions | Business hours, deadlines |
| `WEBHOOK` | HTTP webhook events | External integrations |

## Basic Usage

### Register a Trigger

```python
from isa_agent_sdk import register_trigger, TriggerType

trigger_id = await register_trigger(
    user_id="user-123",
    trigger_type=TriggerType.THRESHOLD,
    description="Alert when BTC drops 5%",
    conditions={
        "event_type": "price_change",
        "product": "Bitcoin",
        "threshold_value": 5.0,
        "direction": "down"
    },
    action_config={
        "prompt": "Analyze the Bitcoin price drop and suggest actions",
        "allowed_tools": ["web_search", "send_notification"]
    }
)

print(f"Trigger registered: {trigger_id}")
```

### Unregister a Trigger

```python
from isa_agent_sdk import unregister_trigger

success = await unregister_trigger(trigger_id)
print(f"Unregistered: {success}")
```

### List User Triggers

```python
from isa_agent_sdk import get_user_triggers

triggers = await get_user_triggers("user-123")
for trigger in triggers:
    print(f"- {trigger.trigger_id}: {trigger.description}")
    print(f"  Type: {trigger.trigger_type}")
    print(f"  Active: {trigger.is_active}")
```

## Trigger Examples

### Price Alert Trigger

```python
from isa_agent_sdk import register_trigger, TriggerType

# Alert on price drop
trigger_id = await register_trigger(
    user_id="user-123",
    trigger_type=TriggerType.THRESHOLD,
    description="Stock price alert",
    conditions={
        "event_type": "price_change",
        "product": "AAPL",
        "threshold_type": "percentage",
        "threshold_value": 3.0,
        "direction": "down"
    },
    action_config={
        "prompt": "Apple stock dropped significantly. Analyze the situation and suggest whether to buy, hold, or sell.",
        "allowed_tools": ["web_search", "fetch_url"]
    }
)
```

### Scheduled Task Trigger

```python
# Daily summary at 9 AM
trigger_id = await register_trigger(
    user_id="user-123",
    trigger_type=TriggerType.SCHEDULED_TASK,
    description="Daily news summary",
    conditions={
        "schedule": {
            "type": "daily",
            "time": "09:00",
            "timezone": "America/New_York"
        }
    },
    action_config={
        "prompt": "Summarize today's top tech news and industry trends",
        "allowed_tools": ["web_search"]
    }
)

# Weekly report on Mondays
trigger_id = await register_trigger(
    user_id="user-123",
    trigger_type=TriggerType.SCHEDULED_TASK,
    description="Weekly metrics report",
    conditions={
        "schedule": {
            "type": "weekly",
            "day": "monday",
            "time": "08:00",
            "timezone": "UTC"
        }
    },
    action_config={
        "prompt": "Generate a weekly metrics report from the dashboard data",
        "allowed_tools": ["database_query", "send_email"]
    }
)
```

### IoT Event Trigger

```python
# Temperature alert
trigger_id = await register_trigger(
    user_id="user-123",
    trigger_type=TriggerType.EVENT_PATTERN,
    description="Server room temperature alert",
    conditions={
        "event_type": "iot_reading",
        "device_type": "temperature_sensor",
        "location": "server-room",
        "pattern": {
            "field": "temperature",
            "operator": "greater_than",
            "value": 80
        }
    },
    action_config={
        "prompt": "Server room temperature is critical! Check cooling systems and alert the ops team.",
        "allowed_tools": ["send_notification", "fetch_url"]
    }
)
```

### Webhook Trigger

```python
# GitHub webhook
trigger_id = await register_trigger(
    user_id="user-123",
    trigger_type=TriggerType.WEBHOOK,
    description="PR review request",
    conditions={
        "webhook_source": "github",
        "event_type": "pull_request",
        "action": "review_requested"
    },
    action_config={
        "prompt": "A PR review was requested. Analyze the changes and provide feedback.",
        "allowed_tools": ["fetch_url", "read_file"],
        "skills": ["code-review"]
    }
)
```

### Time-Based Trigger

```python
# Business hours only
trigger_id = await register_trigger(
    user_id="user-123",
    trigger_type=TriggerType.TIME_BASED,
    description="Process during business hours",
    conditions={
        "time_window": {
            "start": "09:00",
            "end": "17:00",
            "days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
            "timezone": "America/New_York"
        },
        "trigger_on": "window_start"  # or "window_end"
    },
    action_config={
        "prompt": "Check the support queue and prioritize tickets"
    }
)
```

## Helper Functions

### Price Trigger Shortcut

```python
from isa_agent_sdk import register_price_trigger

trigger_id = await register_price_trigger(
    user_id="user-123",
    product="ETH",
    threshold_percent=5.0,
    direction="up",
    prompt="Ethereum price increased! Analyze the market conditions."
)
```

### Schedule Trigger Shortcut

```python
from isa_agent_sdk import register_schedule_trigger

trigger_id = await register_schedule_trigger(
    user_id="user-123",
    schedule_type="daily",
    time="09:00",
    timezone="UTC",
    prompt="Generate daily status report"
)
```

### Event Pattern Shortcut

```python
from isa_agent_sdk import register_event_pattern_trigger

trigger_id = await register_event_pattern_trigger(
    user_id="user-123",
    event_type="error_log",
    pattern={"severity": "critical"},
    prompt="Critical error detected! Investigate and suggest fixes."
)
```

## Trigger Statistics

```python
from isa_agent_sdk import get_trigger_stats

stats = await get_trigger_stats()
print(f"Total triggers: {stats.total_triggers}")
print(f"Active triggers: {stats.active_triggers}")
print(f"Total fires: {stats.total_fires}")
print(f"Latest fire: {stats.latest_fire}")
```

**Note:** `get_trigger_stats()` returns global statistics across all triggers.

## Trigger Lifecycle

### Initialization

```python
from isa_agent_sdk import initialize_triggers

# Initialize trigger system (usually done once at startup)
await initialize_triggers()
```

### Shutdown

```python
from isa_agent_sdk import shutdown_triggers

# Graceful shutdown
await shutdown_triggers()
```

### Trigger Manager

```python
from isa_agent_sdk import get_trigger_manager

manager = get_trigger_manager()

# Check if initialized
if manager.is_initialized:
    print("Trigger system ready")

# Get all triggers for processing
all_triggers = manager.get_all_triggers()
```

## Action Configuration

### Basic Action Config

```python
action_config = {
    "prompt": "Your task description here",
    "allowed_tools": ["tool1", "tool2"],
}
```

### Advanced Action Config

```python
action_config = {
    # Task prompt
    "prompt": "Analyze and respond to the event",

    # Tool access
    "allowed_tools": ["web_search", "send_notification"],

    # Model selection
    "model": "gpt-4.1-nano",

    # Skills to activate
    "skills": ["debug", "code-review"],

    # Notification settings
    "notify_on_complete": True,
    "notification_channel": "slack",

    # Execution limits
    "max_iterations": 10,
    "timeout_seconds": 300,
}
```

## Trigger Conditions

### Threshold Conditions

```python
conditions = {
    "event_type": "price_change",
    "product": "AAPL",
    "threshold_type": "percentage",  # or "absolute"
    "threshold_value": 5.0,
    "direction": "down",  # "up", "down", or "any"
}
```

### Schedule Conditions

```python
conditions = {
    "schedule": {
        "type": "daily",       # daily, weekly, monthly, cron
        "time": "09:00",       # HH:MM format
        "timezone": "UTC",
        # For weekly:
        "day": "monday",
        # For monthly:
        "day_of_month": 1,
        # For cron:
        "cron": "0 9 * * 1-5"  # Cron expression
    }
}
```

### Event Pattern Conditions

```python
conditions = {
    "event_type": "log_entry",
    "pattern": {
        "field": "level",
        "operator": "equals",  # equals, contains, greater_than, less_than, regex
        "value": "ERROR"
    },
    # Multiple patterns (AND)
    "patterns": [
        {"field": "level", "operator": "equals", "value": "ERROR"},
        {"field": "service", "operator": "contains", "value": "auth"}
    ]
}
```

## Integration Example

### Complete Trigger Setup

```python
from isa_agent_sdk import (
    initialize_triggers,
    register_trigger,
    get_user_triggers,
    TriggerType
)

async def setup_user_triggers(user_id: str):
    # Initialize trigger system
    await initialize_triggers()

    # Price monitoring
    await register_trigger(
        user_id=user_id,
        trigger_type=TriggerType.THRESHOLD,
        description="Portfolio alert",
        conditions={
            "event_type": "portfolio_change",
            "threshold_type": "percentage",
            "threshold_value": 10.0,
            "direction": "any"
        },
        action_config={
            "prompt": "Portfolio value changed significantly. Analyze and suggest rebalancing.",
            "allowed_tools": ["web_search"]
        }
    )

    # Daily summary
    await register_trigger(
        user_id=user_id,
        trigger_type=TriggerType.SCHEDULED_TASK,
        description="Daily summary",
        conditions={
            "schedule": {
                "type": "daily",
                "time": "18:00",
                "timezone": "UTC"
            }
        },
        action_config={
            "prompt": "Generate end-of-day summary"
        }
    )

    # Verify triggers
    triggers = await get_user_triggers(user_id)
    print(f"Registered {len(triggers)} triggers for {user_id}")
```

## Best Practices

1. **Set appropriate cooldowns** - Prevent trigger spam
2. **Use specific conditions** - Avoid false positives
3. **Limit action scope** - Restrict tools to what's needed
4. **Monitor trigger stats** - Track fire rates
5. **Clean up unused triggers** - Unregister when done

## Next Steps

- [Options](./options.md) - Configure proactive mode
- [Human-in-the-Loop](./human-in-the-loop.md) - Approval for trigger actions
- [Tools](./tools.md) - Tools available for trigger actions
