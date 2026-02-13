# Steward Tools

Steward tools enable the agent to act as a personal steward (管家) - managing tasks, calendar, reminders, and proactive automation on the user's behalf. These tools help users organize their work and automate routine activities.

## Overview

| Category | Tools | Description |
|----------|-------|-------------|
| **Task Management** | 9 tools | TODO lists, reminders, task tracking |
| **Calendar** | 9 tools | Events, scheduling, external sync |
| **Event Triggers** | 6 tools | Alerts, scheduled tasks, automation |

## Task Management Tools

Task tools help users manage TODO items, reminders, and track task completion.

### Available Tools

| Tool | Description |
|------|-------------|
| `create_task` | Create a new task/todo/reminder |
| `get_task` | Get task details by ID |
| `list_tasks` | List tasks with filters |
| `update_task` | Update task properties |
| `delete_task` | Delete a task |
| `complete_task` | Mark a task as completed |
| `get_due_tasks` | Get tasks due today/tomorrow/this week |
| `get_reminders` | Get upcoming reminders |
| `get_task_analytics` | Get task completion statistics |

### Usage Examples

#### Create a Task

```python
# Via agent conversation
"Create a todo task called 'Review PR #123' with high priority"

# The agent will call create_task with:
# - name: "Review PR #123"
# - task_type: "todo"
# - priority: "high"
```

#### List Due Tasks

```python
# Via agent conversation
"What tasks are due today?"
"Show me overdue tasks"
"List my high priority todos"
```

#### Task Analytics

```python
# Via agent conversation
"Show me my task completion stats for this month"
"How many tasks did I complete last week?"
```

### Task Types

| Type | Description |
|------|-------------|
| `todo` | Simple TODO item |
| `reminder` | Time-based reminder |
| `scheduled` | Recurring scheduled task |
| `event` | Event-triggered task |

### Task Priorities

- `low` - Low priority
- `medium` - Medium priority (default)
- `high` - High priority
- `urgent` - Urgent priority

---

## Calendar Tools

Calendar tools manage events, scheduling, and external calendar synchronization.

### Available Tools

| Tool | Description |
|------|-------------|
| `create_calendar_event` | Create a new calendar event |
| `get_calendar_event` | Get event details by ID |
| `list_calendar_events` | List events in a date range |
| `update_calendar_event` | Update event properties |
| `delete_calendar_event` | Delete an event |
| `get_upcoming_events` | Get events in the next N hours |
| `get_today_events` | Get today's events |
| `sync_external_calendar` | Sync with Google/Outlook calendar |
| `get_calendar_sync_status` | Check sync status |

### Usage Examples

#### Create an Event

```python
# Via agent conversation
"Schedule a meeting with John tomorrow at 2pm for 1 hour"
"Add a dentist appointment on Friday at 10am"

# The agent will call create_calendar_event with:
# - title: "Meeting with John"
# - start_time: "2026-01-30T14:00:00"
# - duration_minutes: 60
```

#### Check Schedule

```python
# Via agent conversation
"What's on my calendar today?"
"Do I have any meetings tomorrow morning?"
"Show me my schedule for this week"
```

#### External Calendar Sync

```python
# Via agent conversation
"Sync my Google Calendar"
"Import events from my Outlook calendar"
```

---

## Event Triggers (Proactive Automation)

Event tools enable proactive agent activation based on conditions, schedules, or external events. See [triggers.md](triggers.md) for the SDK API.

### Available Tools

| Tool | Description |
|------|-------------|
| `register_price_alert` | Alert when price crosses threshold |
| `register_scheduled_task` | Schedule recurring task execution |
| `register_event_trigger` | Register custom event trigger |
| `list_triggers` | List user's active triggers |
| `delete_trigger` | Remove a trigger |
| `query_events` | Query historical events |

### Trigger Types

| Type | Use Case |
|------|----------|
| `THRESHOLD` | Price alerts, metric monitoring |
| `SCHEDULED_TASK` | Daily reports, recurring tasks |
| `EVENT_PATTERN` | IoT events, log patterns |
| `TIME_BASED` | Business hours, deadlines |
| `WEBHOOK` | External service notifications |

### Usage Examples

#### Price Alert

```python
# Via agent conversation
"Alert me when Bitcoin drops more than 5%"
"Notify me if AAPL stock goes above $200"

# The agent will call register_price_alert with:
# - product: "Bitcoin"
# - threshold_value: 5.0
# - direction: "down"
```

#### Scheduled Task

```python
# Via agent conversation
"Send me a daily summary of my tasks at 9am"
"Remind me to check emails every 2 hours during work"

# The agent will call register_scheduled_task with:
# - cron_expression: "0 9 * * *"
# - action_prompt: "Summarize today's tasks"
```

#### List Triggers

```python
# Via agent conversation
"What alerts do I have set up?"
"Show me my active triggers"
```

---

## Architecture

### Data Flow

```
User Request
    ↓
Agent (isA_Agent)
    ↓
SDK injects user_id into tool arguments
    ↓
MCP Server (isA_MCP)
    ↓
User Tools (task_tools, calendar_tools, event_tools)
    ↓
Microservices (task_service, calendar_service, trigger_service)
    ↓
Database (PostgreSQL)
```

### User Context

All user tools automatically receive the `user_id` from the agent session:

1. User authenticates with the agent
2. Agent passes `user_id` in request
3. SDK injects `user_id` into tool arguments
4. Tools send `X-User-Id` header to microservices
5. Microservices isolate data by user

### Service Ports

| Service | Port | Description |
|---------|------|-------------|
| task_service | 8211 | Task management |
| calendar_service | 8212 | Calendar events |
| trigger_service | 8213 | Event triggers |
| MCP Server | 8081 | Tool gateway |

---

## Configuration

### Enabling User Tools

User tools are enabled by default in the agent. To customize:

```python
from isa_agent_sdk import ISAAgentOptions

options = ISAAgentOptions(
    user_id="user-123",
    allowed_tools=[
        # Task tools
        "create_task", "get_task", "list_tasks", "update_task",
        "delete_task", "complete_task", "get_due_tasks",
        "get_reminders", "get_task_analytics",
        # Calendar tools
        "create_calendar_event", "list_calendar_events",
        "get_calendar_event", "get_upcoming_events",
        # Trigger tools
        "register_price_alert", "register_scheduled_task",
        "list_triggers", "delete_trigger",
    ]
)
```

### Environment Variables

```bash
# Service URLs (optional, defaults shown)
TASK_SERVICE_URL=http://localhost:8211
CALENDAR_SERVICE_URL=http://localhost:8212
TRIGGER_SERVICE_URL=http://localhost:8213
```

---

## Best Practices

### Task Management

1. **Use specific task types** - Choose `todo`, `reminder`, or `scheduled` appropriately
2. **Set priorities** - Help users focus on important tasks
3. **Add due dates** - Enable deadline tracking and reminders
4. **Use tags** - Organize tasks into categories

### Calendar

1. **Include duration** - Always specify event length
2. **Add descriptions** - Include meeting details, links, etc.
3. **Set reminders** - Help users prepare for events
4. **Sync regularly** - Keep external calendars in sync

### Triggers

1. **Be specific** - Clear conditions prevent false triggers
2. **Test first** - Verify triggers work before relying on them
3. **Clean up** - Remove triggers that are no longer needed
4. **Monitor usage** - Check trigger execution history

---

## Related Documentation

- [Triggers API](triggers.md) - SDK trigger registration API
- [Tools](tools.md) - General tool documentation
- [Options](options.md) - Agent configuration options
