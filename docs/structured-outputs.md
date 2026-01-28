# Structured Outputs

Get validated JSON from agent workflows using JSON Schema, Zod, or Pydantic. Get type-safe, structured data after multi-turn tool use.

## Overview

Structured outputs let you define the exact shape of data you want back from an agent. The agent can use any tools it needs to complete the task, and you still get validated JSON matching your schema at the end.

**Key benefits:**
- **Type safety**: Get strongly-typed objects instead of free-form text
- **Validation**: Automatic schema validation with retry logic
- **Pydantic integration**: Use familiar Python models
- **Claude SDK compatible**: Same patterns as Claude Agent SDK

## Quick Start

### Basic JSON Schema

```python
from isa_agent_sdk import query, ISAAgentOptions, OutputFormat

# Define the shape of data you want
schema = {
    "type": "object",
    "properties": {
        "company_name": {"type": "string"},
        "founded_year": {"type": "number"},
        "headquarters": {"type": "string"}
    },
    "required": ["company_name"]
}

async for msg in query(
    prompt="Research Anthropic and provide key company information",
    options=ISAAgentOptions(
        output_format=OutputFormat.json_schema(schema)
    )
):
    if msg.has_structured_output:
        print(msg.structured_output)
        # {'company_name': 'Anthropic', 'founded_year': 2021, 'headquarters': 'San Francisco'}
```

### With Pydantic Models

For full type safety, use Pydantic models:

```python
from pydantic import BaseModel
from isa_agent_sdk import query, ISAAgentOptions, OutputFormat

class Step(BaseModel):
    step_number: int
    description: str
    complexity: str  # 'low', 'medium', 'high'

class FeaturePlan(BaseModel):
    feature_name: str
    summary: str
    steps: list[Step]
    risks: list[str]

async for msg in query(
    prompt="Plan how to add dark mode support to a React app",
    options=ISAAgentOptions(
        output_format=OutputFormat.from_pydantic(FeaturePlan)
    )
):
    if msg.has_structured_output:
        # Parse into typed Pydantic model
        plan = msg.parse(FeaturePlan)
        print(f"Feature: {plan.feature_name}")
        print(f"Summary: {plan.summary}")
        for step in plan.steps:
            print(f"  {step.step_number}. [{step.complexity}] {step.description}")
```

## OutputFormat Configuration

### Creating OutputFormat

```python
from isa_agent_sdk import OutputFormat

# Method 1: From Pydantic model (recommended)
output_format = OutputFormat.from_pydantic(MyModel)

# Method 2: From JSON schema
output_format = OutputFormat.json_schema({
    "type": "object",
    "properties": {...},
    "required": [...]
})

# Method 3: Simple JSON object mode (no schema validation)
output_format = OutputFormat.json_object()

# Method 4: Direct construction
output_format = OutputFormat(
    type="json_schema",
    schema=my_schema,
    strict=True
)
```

### OutputFormat Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `type` | `str` | `"text"` | Format type: `"text"`, `"json_object"`, or `"json_schema"` |
| `schema` | `dict` | `None` | JSON Schema for validation (required for `json_schema` type) |
| `strict` | `bool` | `True` | Enforce strict schema validation |

## Working with Results

### Checking for Structured Output

```python
async for msg in query(prompt, options=options):
    # Check if structured output is available
    if msg.has_structured_output:
        data = msg.structured_output  # Dict

    # Check success/error status
    if msg.is_structured_output_success:
        print("Validation succeeded!")
    elif msg.is_structured_output_error:
        print("Validation failed after retries")
```

### Parsing to Pydantic

```python
# Safe parsing (returns None on failure)
user = msg.parse_or_none(User)
if user:
    print(f"Got user: {user.name}")

# Strict parsing (raises on failure)
try:
    user = msg.parse(User)
except ValueError as e:
    print(f"No structured output: {e}")
except ValidationError as e:
    print(f"Validation failed: {e}")
```

### Result Subtypes

The `subtype` field indicates the result status:

| Subtype | Meaning |
|---------|---------|
| `success` | Output was generated and validated successfully |
| `error_max_turns` | Agent hit maximum iteration limit |
| `error_max_structured_output_retries` | Couldn't produce valid output after retries |
| `error_tool_execution` | Tool execution failed |
| `error_model` | Model inference error |
| `interrupted` | Execution was interrupted (HIL) |

```python
from isa_agent_sdk import ResultSubtype

if msg.subtype == ResultSubtype.SUCCESS.value:
    print("Success!")
elif msg.subtype == ResultSubtype.ERROR_MAX_STRUCTURED_OUTPUT_RETRIES.value:
    print("Failed to produce valid structured output")
```

## Example: TODO Tracker

This example demonstrates structured outputs with multi-step tool use:

```python
from pydantic import BaseModel
from isa_agent_sdk import query, ISAAgentOptions, OutputFormat

class TodoItem(BaseModel):
    text: str
    file: str
    line: int
    author: str | None = None
    date: str | None = None

class TodoReport(BaseModel):
    todos: list[TodoItem]
    total_count: int

async for msg in query(
    prompt="Find all TODO comments in this codebase and identify who added them",
    options=ISAAgentOptions(
        allowed_tools=["Grep", "Bash", "Read"],
        output_format=OutputFormat.from_pydantic(TodoReport)
    )
):
    if msg.is_tool_use:
        print(f"Using tool: {msg.tool_name}")

    if msg.has_structured_output:
        report = msg.parse(TodoReport)
        print(f"\nFound {report.total_count} TODOs:")
        for todo in report.todos:
            print(f"  {todo.file}:{todo.line} - {todo.text}")
            if todo.author:
                print(f"    Added by {todo.author} on {todo.date}")
```

## Error Handling

```python
async for msg in query(prompt, options=options):
    if msg.type == "result":
        if msg.subtype == "success" and msg.structured_output:
            # Use the validated output
            data = msg.structured_output
        elif msg.subtype == "error_max_structured_output_retries":
            # Handle the failure
            print("Could not produce valid output")
            print(f"Raw content: {msg.content}")
```

### Tips for Avoiding Errors

1. **Keep schemas focused**: Deeply nested schemas with many required fields are harder to satisfy
2. **Match schema to task**: Make fields optional if the task might not have all information
3. **Use clear prompts**: Ambiguous prompts make it harder to know what output to produce
4. **Start simple**: Add complexity to your schema as needed

## Low-Level Parsing Utilities

For advanced use cases, you can use the parsing utilities directly:

```python
from isa_agent_sdk import (
    StructuredOutputParser,
    StructuredOutputResult,
    parse_structured_output,
    parse_json_response
)

# Parse JSON from text (handles markdown code blocks, etc.)
result = parse_json_response(raw_text)
if result.success:
    data = result.data

# Parse with schema validation
parser = StructuredOutputParser(
    schema=my_schema,
    pydantic_model=MyModel
)
result = parser.parse(raw_text)

# Direct parsing function
result = parse_structured_output(
    content=raw_text,
    model=MyModel,
    schema=my_schema
)
```

## Comparison with isA_Model

The isA Agent SDK structured outputs build on isA_Model's capabilities:

| Feature | isA_Model | isA Agent SDK |
|---------|-----------|---------------|
| `response_format={"type": "json_object"}` | Yes | Yes (via `OutputFormat.json_object()`) |
| JSON Schema validation | Yes (prompt-based) | Yes (native + parsing) |
| Pydantic integration | Manual | Built-in (`OutputFormat.from_pydantic()`) |
| `structured_output` field | No | Yes |
| Automatic retry | Manual | Built-in |
| Result subtypes | No | Yes |

## Related

- [Options Reference](./options.md) - Full configuration options
- [Messages Reference](./messages.md) - Message types and properties
- [Streaming](./streaming.md) - Streaming vs single mode
