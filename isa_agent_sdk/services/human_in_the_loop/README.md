# Human-in-the-Loop Service (Refactored) 🔄

**Version**: 2.0.0  
**File Location**: `src/services/human_in_the_loop/`

## 📁 Modular Structure

The HIL service has been refactored from a single 1303-line file into a clean, modular architecture:

```
human_in_the_loop/
├── __init__.py                      # Public API exports
├── README.md                        # This file
├── models.py                        # Data models & enums
├── validators.py                    # Input validation logic
├── interrupt_manager.py             # Core interrupt patterns
├── scenario_handlers.py             # Scenario-based HIL methods
├── hil_service_refactored.py        # Main orchestrating service
└── tests/
    ├── __init__.py
    └── test_hil_service.py          # Comprehensive test suite
```

## 🎯 Key Benefits

1. **Modular**: Each module has a single responsibility
2. **Testable**: Clean separation enables easy unit testing
3. **Maintainable**: ~200-400 lines per file vs 1303 lines
4. **Extensible**: Easy to add new scenarios or validators
5. **Type-safe**: Full type hints throughout
6. **Documented**: Comprehensive docstrings with examples

## 📦 Modules Overview

### 1. `models.py` - Data Models
Defines all data structures used by HIL service:
- **Enums**: `InterruptType`, `InterventionType`, `SecurityLevel`
- **Data Classes**: Various interrupt data structures
- **Example**:
```python
from src.services.human_in_the_loop.models import InterruptType, SecurityLevel

interrupt_type = InterruptType.TOOL_AUTHORIZATION
security = SecurityLevel.HIGH
```

### 2. `validators.py` - Validation Logic
Handles all input validation and response processing:
- `HILValidator`: Main validator class
- `ValidationRulesBuilder`: Fluent API for building rules
- **Example**:
```python
from src.services.human_in_the_loop.validators import HILValidator, ValidationRulesBuilder

validator = HILValidator()
result = validator.validate_input("25", {"type": "int", "min": 0, "max": 120})

# Or use builder
rules = (ValidationRulesBuilder()
    .type("int")
    .min(0)
    .max(120)
    .build())
```

### 3. `interrupt_manager.py` - Core Interrupt Patterns
Manages LangGraph interrupts:
- `approve_or_reject()`: Approval/rejection workflow
- `review_and_edit()`: Review and edit content
- `validate_input_with_retry()`: Input validation with retry
- `simple_interrupt()`: Generic interrupt
- **Example**:
```python
from src.services.human_in_the_loop.interrupt_manager import InterruptManager

manager = InterruptManager()
command = manager.approve_or_reject(
    question="Execute this tool?",
    context={"tool": "web_crawl"},
    node_source="tool_node"
)
```

### 4. `scenario_handlers.py` - Scenario-Based Methods
High-level scenario handlers for common use cases:
- `collect_user_input()`: Collect information from user
- `request_tool_permission()`: Request tool execution authorization
- `request_oauth_authorization()`: Request OAuth authorization
- `request_credential_usage()`: Request credential usage
- `request_manual_intervention()`: Request manual user action

### 5. `hil_service_refactored.py` - Main Service
Orchestrates all components and provides unified API:
- Main `HILService` class
- `get_hil_service()`: Singleton accessor
- Backward compatibility methods

## 🚀 Quick Start

### Basic Usage

```python
from src.services.human_in_the_loop import get_hil_service

# Get service instance
hil = get_hil_service()

# Collect user input
email = await hil.collect_user_input("What is your email?")

# Request tool permission
authorized = await hil.request_tool_permission(
    tool_name="web_crawl",
    tool_args={"url": "https://example.com"},
    security_level="HIGH",
    reason="Need to scrape data"
)

if authorized:
    # Execute tool
    pass
```

### With Validation

```python
# Collect validated input
age = await hil.collect_user_input(
    question="What is your age?",
    validation_rules={"type": "int", "min": 0, "max": 120}
)

# Or use validation builder
from src.services.human_in_the_loop import ValidationRulesBuilder

rules = (ValidationRulesBuilder()
    .type("int")
    .min(18)
    .max(100)
    .build())

age = await hil.collect_user_input("Enter age:", validation_rules=rules)
```

### OAuth Authorization

```python
oauth_result = await hil.request_oauth_authorization(
    provider="gmail",
    oauth_url="https://accounts.google.com/o/oauth2/auth?...",
    scopes=["read", "send"]
)

if oauth_result["authorized"]:
    # Use OAuth token
    pass
```

### Manual Intervention

```python
# CAPTCHA handling
result = await hil.request_manual_intervention(
    intervention_type="captcha",
    provider="google",
    instructions="Please solve the reCAPTCHA v2 challenge",
    screenshot_path="/tmp/captcha.png"
)

# Login required
result = await hil.request_manual_intervention(
    intervention_type="login",
    provider="google",
    instructions="Please login to your Google account",
    oauth_url="https://accounts.google.com/signin"
)
```

### Get Statistics

```python
stats = hil.get_interrupt_stats()
print(f"Total interrupts: {stats.total}")
print(f"By type: {stats.by_type}")
print(f"By node: {stats.by_node}")
print(f"Latest: {stats.latest}")
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Using pytest (recommended)
python -m pytest src/services/human_in_the_loop/tests/test_hil_service.py -v

# Or run directly
python src/services/human_in_the_loop/tests/test_hil_service.py
```

The test suite covers:
- ✓ Data models and enums
- ✓ Validators (all validation types)
- ✓ Interrupt manager (patterns & history)
- ✓ HIL service (all scenario methods)
- ✓ Legacy methods (backward compatibility)

## 📊 Comparison: Old vs New

| Aspect | Old (hil_service.py) | New (Refactored) |
|--------|---------------------|------------------|
| **Lines of Code** | 1,303 lines | ~200-400 per file |
| **Files** | 1 monolithic file | 7 focused modules |
| **Testability** | Difficult | Easy (isolated) |
| **Maintainability** | Hard to navigate | Clear structure |
| **Extensibility** | Add to large file | Add new module |
| **Code Duplication** | Yes (lines 899-1303) | None |
| **Type Hints** | Partial | Complete |
| **Documentation** | Inline only | Module + inline |

## 🔄 Migration Guide

The refactored service is **100% backward compatible**. No code changes needed!

```python
# Old import (still works)
from src.services.hil_service import get_hil_service

# New import (recommended)
from src.services.human_in_the_loop import get_hil_service

# All methods work the same
hil = get_hil_service()
result = await hil.collect_user_input("Question?")
```

### Deprecated Methods

These methods still work but will log warnings:

- `ask_human_via_mcp_with_interrupt()` → Use `collect_user_input()`
- `request_tool_authorization()` → Use `request_tool_permission()`
- `ask_human_with_composio_auth()` → Use `request_oauth_authorization()`

## 🔧 Advanced Usage

### Custom Validators

```python
from src.services.human_in_the_loop import HILValidator

validator = HILValidator()

# Custom validation function
def is_valid_email(value):
    return "@" in value and "." in value

rules = {
    "type": "str",
    "validator": is_valid_email
}

result = validator.validate_input("test@example.com", rules)
```

### Direct Component Access

```python
from src.services.human_in_the_loop import InterruptManager, ScenarioHandler

# Use interrupt manager directly
manager = InterruptManager()
command = manager.approve_or_reject("Approve?", {"data": "test"})

# Use scenario handler directly
handler = ScenarioHandler(manager)
result = await handler.collect_user_input("Question?")
```

### Multiple Interrupts

```python
# Resume multiple interrupts at once
results = hil.resume_multiple_interrupts({
    "interrupt_1": "yes",
    "interrupt_2": {"approved": True},
    "interrupt_3": "completed"
})
```

## 📝 Best Practices

1. **Use scenario methods**: Prefer `collect_user_input()` over generic `simple_interrupt()`
2. **Add validation**: Always validate user input when possible
3. **Clear history**: Call `clear_interrupt_history()` between test runs
4. **Check stats**: Monitor interrupt patterns with `get_interrupt_stats()`
5. **Handle errors**: Wrap HIL calls in try/except for robust error handling

## 🤝 Contributing

When adding new functionality:

1. **New interrupt type**: Add to `models.py` → `InterruptType` enum
2. **New validation rule**: Add to `validators.py` → `HILValidator` class
3. **New scenario**: Add to `scenario_handlers.py` → `ScenarioHandler` class
4. **New pattern**: Add to `interrupt_manager.py` → `InterruptManager` class
5. **Always add tests**: Update `tests/test_hil_service.py`

## 📚 Related Documentation

- [Original HIL Service](../hil_service.py) - Legacy implementation
- [LangGraph Interrupts](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)
- [MCP Client](../../clients/mcp_client.py) - MCP integration

## 📞 Support

For questions or issues:
- Check existing tests for usage examples
- Review inline docstrings (all methods documented)
- See [CBKB/HowTos/](../../../../CBKB/HowTos/) for guides

---

**Author**: isA Agent Team  
**License**: MIT  
**Last Updated**: October 28, 2025

